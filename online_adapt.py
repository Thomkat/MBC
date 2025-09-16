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
import functools
import hydra
import torch
from transformers import AutoTokenizer, LlamaTokenizer
from utils.dataset_utils import get_online_adapt_dataloader
from utils.online_adapt_utils import online_adapt_qa_eval, context_summarization_compression
from utils.model_utils import get_base_model, get_mbc_model
from utils.misc import set_random_seed
import csv


def setup_evaluation(cfg):
    eval_fns = []
    top_k = 1
    # Compute a valid num_beams given top_k and num_beam_groups (keep constraints consistent)
    num_beams = max(cfg.model.num_beams, top_k - top_k % cfg.model.num_beam_groups + cfg.model.num_beam_groups)
    if num_beams != cfg.model.num_beams:
        print(f'Overwriting argument num_beams from {cfg.model.num_beams} to {num_beams}')
    
    # Partially apply qa_eval with fixed decoding params
    eval_fns.append(functools.partial(
        online_adapt_qa_eval, 
        top_k=top_k, 
        diversity_penalty=cfg.model.diversity_penalty,
        num_beam_groups=cfg.model.num_beam_groups, 
        num_beams=num_beams
    ))
    
    log_dir = os.getcwd()
    cfg.log_dir = log_dir

    return eval_fns


@hydra.main(config_path='conf', config_name='config_online_adapt', version_base='1.1')
def main(cfg):
    eval_fns = setup_evaluation(cfg)
    set_random_seed(cfg.seed) # For reproducible results

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    base_lm = get_base_model(cfg)
    if cfg.model.base_model not in ["Llama2_7b"]:
        base_lm.to(device)

    if cfg.model.base_model == 'Llama2_7b':
        tokenizer = LlamaTokenizer.from_pretrained(cfg.model.llama_cache_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_name_base_model, cache_dir=cfg.CACHE_DIR)

    tokenizer_amort = AutoTokenizer.from_pretrained(cfg.model.tokenizer_name_amort, cache_dir=cfg.CACHE_DIR, model_max_length=1024)

    amort_comp_dataloader, eval_dataloader = get_online_adapt_dataloader(cfg, tokenizer, tokenizer_amort)

    kwargs = {}
    kwargs_eval = {}
    mbc_model = get_mbc_model(cfg, base_lm, tokenizer=tokenizer).to(device)
    mbc_model.eval()
    kwargs['mbc_model'] = mbc_model
    kwargs_eval['mbc_model'] = mbc_model
    
    # Amortize contexts, build the compressed memory bank
    compressed_memory_bank, avg_adaptation_time = context_summarization_compression(amort_comp_dataloader, **kwargs)

    # Report memory footprint of the summary bank
    bank_size_bytes = compressed_memory_bank.nelement() * compressed_memory_bank.element_size()
    bank_size_mb = bank_size_bytes / (1024 ** 2)
    print("\n------------------------------------------------------------")
    print(f"Compressed context memory bank built. Size: {bank_size_mb:.2f} MB")

    # Report VQ codebook footprint and indices footprint
    codebook = mbc_model.vq.embedding.weight
    codebook_size_bytes = codebook.nelement() * codebook.element_size()
    codebook_size_mb = codebook_size_bytes / (1024 ** 2)
    total_memory_mb = bank_size_mb + codebook_size_mb
    print(f"Codebook size: {codebook_size_mb:.2f} MB")
    print(f"Total memory bank footprint (Bank + Codebook): {total_memory_mb:.2f} MB")
    print("------------------------------------------------------------\n")

    print('Evaluating model...')
    base_lm.eval()

    eval_metrics = {}
    log_path = os.path.join(cfg.log_dir, 'metrics_eval.csv')

    with open(log_path, 'w', newline='', encoding='utf-8') as writefile:
        writer = csv.writer(writefile, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["question", "answer", "predicted_answers", "raw_decoding", "pred_f1s", "em"])

        for eval_fn in eval_fns:
            detailed_results, eval_metrics = eval_fn(
                cfg, 
                eval_dataloader, 
                model=base_lm, 
                tokenizer=tokenizer, 
                compressed_memory_bank=compressed_memory_bank, 
                **kwargs_eval
            )
            writer.writerows(detailed_results)

        peak_memory_gb = 0
        if torch.cuda.is_available():
            peak_memory_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        
        writer.writerow([])
        writer.writerow(["SUMMARY"])
        if eval_metrics:
            writer.writerow(["queries_evaluated", eval_metrics['queries_evaluated']])
            writer.writerow(["em_count", eval_metrics['em_correct']])
            writer.writerow(["em_rate", f"{eval_metrics['em_rate']:.4f}"])
            writer.writerow(["avg_f1_mean", f"{eval_metrics['avg_f1_mean']:.4f}"])
            writer.writerow(["avg_f1_std", f"{eval_metrics['avg_f1_std']:.4f}"])
            writer.writerow(["max_f1_mean", f"{eval_metrics['max_f1_mean']:.4f}"])
            writer.writerow(["max_f1_std", f"{eval_metrics['max_f1_std']:.4f}"])
            writer.writerow(["avg_inference_ms_per_query", f"{eval_metrics['avg_inference_ms']:.2f}"])
        
        writer.writerow(["avg_adaptation_ms_per_doc", f"{avg_adaptation_time * 1000:.2f}"])
        if torch.cuda.is_available():
            writer.writerow(["peak_gpu_memory_gb", f"{peak_memory_gb:.2f}"])

    print("\n--- Summary of Performance Metrics ---")
    if eval_metrics:
        print(f"Queries evaluated     : {eval_metrics['queries_evaluated']}")
        print(f"Exact Match (rate)    : {eval_metrics['em_rate']:>5.2%}")
        print(f"Avg F1 (mean ± std)   : {eval_metrics['avg_f1_mean']:.4f} ± {eval_metrics['avg_f1_std']:.4f}")
        print(f"Avg inference / query : {eval_metrics['avg_inference_ms']:.2f} ms")
    print(f"Avg Adaptation / doc  : {avg_adaptation_time * 1000:.2f} ms")
    if torch.cuda.is_available():
        print(f"Peak GPU Memory       : {peak_memory_gb:.2f} GB")
    print("--------------------------------------\n")


if __name__ == "__main__":
    main()