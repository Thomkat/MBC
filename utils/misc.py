import numpy as np
import torch.distributed as dist
import os
import wandb
import re
from omegaconf import OmegaConf
from datetime import datetime
import sys
import torch
import random
from hydra.utils import to_absolute_path
import shutil
from tqdm.auto import tqdm
from collections import Counter
import string
from contextlib import contextmanager
from models.modules.kv_lora import LoRALinearAdd, LoRAKVForGPT2CAttn
import torch.nn as nn


class Logger(object):
    """Reference: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514"""

    def __init__(self, logdir, cfg, main_process=True, use_wandb=False, wandb_name=None):
            self.main_process = main_process
            self.logdir = logdir
            self.cfg = cfg
            self.use_wandb = use_wandb

            if self.main_process:
                os.makedirs(os.path.join(self.logdir, 'checkpoints'), exist_ok=True)
                self.set_dir(self.logdir)

                if self.use_wandb:
                    wandb.login(key=cfg.wandb_key)
                    wandb.config = OmegaConf.to_container(
                        cfg, resolve=True, throw_on_missing=True
                    )
                    wandb.init(project=cfg.wandb_project, name=wandb_name, dir=self.logdir,
                            entity=cfg.wandb_entity, settings=wandb.Settings(start_method='fork'))

    def set_dir(self, logdir, log_fn='log.txt'):
        self.logdir = logdir
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        self.log_file = open(os.path.join(logdir, log_fn), 'a')

    def close_writer(self):
        if self.main_process and self.use_wandb:
            wandb.finish()

    def log(self, string):
        if self.main_process:
            self.log_file.write('[%s] %s' % (datetime.now(), string) + '\n')
            self.log_file.flush()

            print('[%s] %s' % (datetime.now(), string))
            sys.stdout.flush()

    def log_dirname(self, string):
        if self.main_process:
            self.log_file.write('%s (%s)' % (string, self.logdir) + '\n')
            self.log_file.flush()

            print('%s (%s)' % (string, self.logdir))
            sys.stdout.flush()

    def wandb_log(self, log_dict, commit=None):
        if self.main_process and self.use_wandb:
            wandb.log(log_dict, commit=commit)


def decode_to_clean_text(tokenizer, ids):
    gen_text = tokenizer.batch_decode(
        ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    return list(map(str.strip, gen_text))


@contextmanager
def lora_disabled(module: nn.Module):
    toggled = []
    for m in module.modules():
        if isinstance(m, (LoRALinearAdd, LoRAKVForGPT2CAttn)):
            toggled.append(m)
            m.enabled = False
    try:
        yield
    finally:
        for m in toggled:
            m.enabled = True

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match(prediction, ground_truth, match_length=False):
    norm_pred = normalize_answer(prediction)
    norm_truth = normalize_answer(ground_truth)
    if not match_length:
        norm_pred = norm_pred[:len(norm_truth)]
    return norm_pred == norm_truth

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()

    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def shuffle_groups(df, group_col):
    """
    Shuffles the order of groups in a Pandas DataFrame without shuffling the order of items within each group.

    Parameters:
    - df: the input DataFrame
    - group_col: the name of the column containing the groups to be shuffled

    Returns:
    - a shuffled copy of the input DataFrame
    """
    # Get a list of unique groups
    groups = df[group_col].unique()

    # Shuffle the list of groups
    np.random.shuffle(groups)

    # Define a sorting key that sorts by the shuffled order of groups
    def sort_key(row):
        return np.argwhere(groups == row[group_col])[0][0]

    df['temp'] = df.apply(sort_key, axis=1)
    shuffled_df = df.sort_values('temp', kind='stable').drop('temp', axis=1).reset_index(drop=True)
    return shuffled_df


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def cycle(loader):
    while True:
        for x in loader:
            yield x

def set_random_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def update_path(cfg):
    if 'data_dir' in cfg: cfg.data_dir = to_absolute_path(cfg.data_dir)
    if 'test_path' in cfg: cfg.test_path = to_absolute_path(cfg.test_path)

def logging_path_check(cfg, run_name):
    log_path = './logs/' if cfg.log_path is None else cfg.log_path
    os.makedirs(log_path, exist_ok=True)
    
    # Use the passed run_name to create the specific log directory
    logdir = os.path.join(log_path, run_name) 
    os.makedirs(logdir, exist_ok=True)

def metric_synchronize_between_processes(metrics: dict, accelerator):
    if accelerator.num_processes == 1:
        return

    for k, v in list(metrics.items()):
        if torch.is_tensor(v):
            t = v.detach().to(accelerator.device)
        else:
            t = torch.as_tensor(v, device=accelerator.device)

        t = accelerator.reduce(t, reduction="mean")

        if t.ndim == 0:
            metrics[k] = t.item()
        else:
            metrics[k] = t

    accelerator.wait_for_everyone()


def tqdm_distributed(main_process, iterator, *args, **kwargs):
    if main_process:
        kwargs.setdefault("leave", False)
        kwargs.setdefault("dynamic_ncols", True)
        return tqdm(iterator, *args, **kwargs)
    else:
        # all other processes just return plain iterator
        return iterator
    
    
def return_k_unique(df, k, column): 
    if k >= len(df[column].unique()):
        return df
    else:
        values_to_keep = df[column].unique()[:k]
        return df[df.apply(lambda x: x[column] in values_to_keep, axis=1)]