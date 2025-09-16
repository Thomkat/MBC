import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module=r"transformers")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module=r"hydra\._internal\.hydra")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["NCCL_DEBUG"] = "WARN"
warnings.filterwarnings("ignore", category=FutureWarning, module=r"huggingface_hub\.file_download", message=r".*`resume_download` is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"transformers\.utils\.generic", message=r".*_register_pytree_node.*")
import numpy as np
import hydra
import torch
import torch
from collections import defaultdict
from accelerate import Accelerator, DistributedDataParallelKwargs
from transformers import AutoTokenizer, LlamaTokenizer
from utils.optim import get_optimizer
from utils.dataset_utils import get_trainval_dataloader
from utils.model_utils import get_base_model, get_mbc_model
from utils.misc import Logger, set_random_seed, update_path, metric_synchronize_between_processes, tqdm_distributed
from transformers import get_scheduler
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()


@hydra.main(config_path='conf', config_name='config_train', version_base='1.1')
def main(cfg):
    kwargs = {}
    update_path(cfg)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.model.grad_acc_steps,
        mixed_precision=cfg.mixed_precision,
        kwargs_handlers=[ddp_kwargs]
    )

    cfg.world_size = accelerator.num_processes

    main_process = accelerator.is_main_process
    #if main_process: logging_path_check(cfg, run_id)
    accelerator.wait_for_everyone()

    # For reproducibility
    # Note: torch.backends.cudnn.benchmark = True can sometimes break
    # determinism, if you want to ensure 100% reproducible runs, disable it
    # If you want to maximize speed, keep it enabled
    set_random_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    log_directory = os.getcwd()
    logger = Logger(logdir=log_directory, cfg=cfg, main_process=main_process,
                    use_wandb=cfg.wandb_log, wandb_name=cfg.run_id)
    
    best_val_loss = np.inf
    best_em = 0.0
    best_f1 = 0.0
    early_stopping_counter = 0
    stop_training = False

    # Llama2
    if cfg.model.base_model == 'Llama2_7b':
        tokenizer = LlamaTokenizer.from_pretrained(cfg.model.llama_cache_dir)
    # GPT-2 Family
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_name_base_model, cache_dir=cfg.CACHE_DIR)

    tokenizer_amort = AutoTokenizer.from_pretrained(cfg.model.tokenizer_name_amort, cache_dir=cfg.CACHE_DIR, model_max_length=1024)

    train_loader, val_loader, val_gen_loader = get_trainval_dataloader(cfg, tokenizer=tokenizer, tokenizer_amort=tokenizer_amort)

    # Initialize base model, amortization network, optimizer
    base_lm = get_base_model(cfg,accelerator=accelerator)
    mbc_model = get_mbc_model(cfg, base_lm, tokenizer=tokenizer, tokenizer_amort=tokenizer_amort, accelerator=accelerator)
    amort_optimizer = get_optimizer(cfg, mbc_model)

    train_steps = len(train_loader) * cfg.n_epochs // cfg.model.grad_acc_steps
    scheduler = get_scheduler(cfg.lr_schedule, amort_optimizer, int(train_steps * cfg.warmup_ratio), train_steps)

    mbc_model, amort_optimizer, scheduler, train_loader, val_loader, val_gen_loader = accelerator.prepare(
        mbc_model, amort_optimizer, scheduler, train_loader, val_loader, val_gen_loader
    )
    kwargs['scheduler'] = scheduler
    kwargs['mbc_model'] = mbc_model
    kwargs['amort_optimizer'] = amort_optimizer
    kwargs['accelerator'] = accelerator

    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(mbc_model)
        
        # Trainable params

        # KV-LoRA
        lora_trainable = sum(p.numel() for n, p in mbc_model.named_parameters()
                             if p.requires_grad and ("lora_A" in n or "lora_B" in n))
        
        # Codebook
        # The VQ's only trainable part is the embedding codebook.
        quantizer_trainable = sum(p.numel() for p in unwrapped_model.vq.parameters() if p.requires_grad)

        # Total trainable params
        total_trainable = sum(p.numel() for p in mbc_model.parameters() if p.requires_grad)

        # The rest are the amortization/input encoder/aggregation network parameters
        amort_agg_trainable = total_trainable - lora_trainable - quantizer_trainable

        accelerator.print(f"\n----------------- Trainable Parameters Breakdown ---------------------")
        accelerator.print(f"Amortization & Input enc & Aggregation Network .: {amort_agg_trainable:,}")
        accelerator.print(f"KV-LoRA ........................................: {lora_trainable:,}")
        accelerator.print(f"Codebook .......................................: {quantizer_trainable:,}")
        accelerator.print(f"---------------------------------------------------------------------")
        accelerator.print(f"Total Trainable Parameters .....................: {total_trainable:,}\n")

    logger.log(f"===== Starting Training =====")

    # Train
    for i_epoch in range(0, cfg.n_epochs):
        accelerator.print()
        logger.log(f'Starting training on epoch {i_epoch}...')
        metrics_dic = defaultdict(lambda: [])
        accelerator.wait_for_everyone()
        main_process = accelerator.is_main_process

        for _, batch in tqdm_distributed(
                main_process, enumerate(train_loader), desc=f'Epoch {i_epoch}', position=0, total=len(train_loader)
        ):
            # Compute loss
            with accelerator.accumulate(mbc_model):
                mbc_model.train()
                outer_loss, metrics = mbc_model(batch)
                accelerator.backward(outer_loss)
                    
                if accelerator.sync_gradients:
                    # Clip gradient when using sync gradients
                    grad_norm = accelerator.clip_grad_norm_(mbc_model.parameters(), cfg.grad_clip_thresh)

                    logger.wandb_log({'grad_norm': grad_norm}, commit=False)
                    logger.wandb_log({'outer_lr': amort_optimizer.param_groups[0]['lr']}, commit=False)
                    logger.wandb_log({'train': {f'{k}': np.mean(v) for k, v in metrics_dic.items()}})
                    metrics_dic.clear()
                amort_optimizer.step()
                amort_optimizer.zero_grad()
                scheduler.step()

                metric_synchronize_between_processes(metrics, accelerator)  # sync metrics across processes
                for k, v in metrics.items():
                    metrics_dic[f'[VAL]{k}'].append(v)

                del batch  # free memory
                accelerator.wait_for_everyone()

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        # Validation
        accelerator.print()
        logger.log(f'Starting validation on epoch {i_epoch}...')

        mbc_model.eval()
        mbc_model.zero_grad()
        amort_optimizer.zero_grad()
        unwrapped_model = accelerator.unwrap_model(mbc_model)

        with torch.no_grad():
            val_metrics, memory_bank = unwrapped_model.validate_amort_compress(val_loader, main_process=main_process)

            if accelerator.num_processes > 1:
                memory_bank = accelerator.gather_for_metrics(memory_bank)

            val_qa_metrics = unwrapped_model.validate_aggregate(
                val_gen_loader, context_bank=memory_bank, main_process=main_process
            )
            val_metrics.update(val_qa_metrics)
            del memory_bank

        if accelerator.num_processes > 1:
            metric_synchronize_between_processes(val_metrics, accelerator)  # sync metrics across processes

        if main_process:
            accelerator.print("\nValidation metrics:")
            for k, v in val_metrics.items():
                logger.log(f"  {k}: {v:.4f}")
            accelerator.print()

        logger.wandb_log({'val': val_metrics}, commit=False)
        if val_metrics['[VAL]qa_loss'] < best_val_loss:
            logger.log('New best validation loss found.')
            best_val_loss = float(val_metrics['[VAL]qa_loss'])
            early_stopping_counter = 0  # Reset counter on improvement
            logger.log('Saving best QA loss model')
            unwrapped_model.save(i_epoch, log_dir=logger.logdir, file_name=f'best_val_loss.pt',
                                main_process=main_process)
        else:
            early_stopping_counter += 1
            logger.log(f"Validation loss did not improve. Early stopping counter: {early_stopping_counter}/{cfg.early_stopping_patience}")
            if early_stopping_counter >= cfg.early_stopping_patience:
                logger.log(f"Early stopping triggered after {cfg.early_stopping_patience} epochs with no improvement.")
                stop_training = True  # Signal to stop training

        if val_metrics['[VAL]EM'] > best_em:
            best_em = float(val_metrics['[VAL]EM'])
            logger.log('Saving best EM model')
            unwrapped_model.save(i_epoch, log_dir=logger.logdir, file_name=f'best_em.pt',
                                main_process=main_process)

        if val_metrics['[VAL]F1 Score'] > best_f1:
            logger.log('Saving best F1 model')
            best_f1 = float(val_metrics['[VAL]F1 Score'])
            unwrapped_model.save(i_epoch, log_dir=logger.logdir, file_name=f'best_f1.pt',
                                main_process=main_process)

        mbc_model.zero_grad()
        amort_optimizer.zero_grad()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        logger.log('Saving last epoch model')
        unwrapped_model.save(i_epoch, log_dir=logger.logdir, file_name=f'last_epoch.pt',
                            main_process=main_process)
        accelerator.wait_for_everyone()

        if stop_training:
            logger.log("Early stopping condition met. Terminating training.")
            break

    logger.close_writer()


if __name__ == "__main__":
    main()