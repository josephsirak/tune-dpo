import os
import json
import time
import torch
import logging
from functools import partial
from typing import Dict, Optional, Any, Union
from tqdm import tqdm
from collections import defaultdict

from omegaconf import DictConfig
from torch.utils.data import DataLoader, DistributedSampler
from torchtune.data import CROSS_ENTROPY_IGNORE_IDX, padded_collate_dpo
from torchtune import config
from torchtune.config._utils import _get_component_from_path
from vllm import LLM, SamplingParams

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def get_tokenizer_path(checkpoint_path: str, custom_tokenizer_path: Optional[str] = None) -> str:
    """
    Get the tokenizer directory path based on TorchTune's standard structure.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        custom_tokenizer_path: Optional custom path to the tokenizer
        
    Returns:
        Path to the tokenizer directory
    """
    if custom_tokenizer_path:
        if os.path.isfile(custom_tokenizer_path):
            return os.path.dirname(custom_tokenizer_path)
        return custom_tokenizer_path
    
    # Standard TorchTune structure: /path/to/model/original/tokenizer.model
    # Return the directory containing the tokenizer
    return os.path.join(checkpoint_path, "original")

def run_eval(
    checkpoint_path: str,

    # 🚨 The next two params should be refactored for custom dataset.
    data_component: str = 'torchtune.datasets.stack_exchange_paired_dataset', 
    data_split: str = 'test',

    # Changes whether DPO or not. 
    collate_fn: str = padded_collate_dpo,

    # For testing, below this can stay constant. Tune hyperparameters for real batch inference use cases.
    tokenizer_component: str = 'torchtune.models.llama3.llama3_tokenizer',
    tokenizer_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    world_size: int = 1,
    rank: int = 0,
    batch_size: int = 2,
    max_tokens: int = 512,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 50,
    max_batches: Optional[int] = None,
    seed: int = 42,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run evaluation on a model checkpoint using vLLM for inference.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        data_split: Partition of the dataset
        tokenizer_path: Optional custom path to the tokenizer
                      (if not provided, uses checkpoint_path/original)
        output_dir: Directory to save detailed results (None for no saving)
        world_size: Number of GPUs for tensor parallelism
        rank: Process rank for distributed evaluation
        batch_size: Batch size for evaluation
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        max_batches: Maximum number of batches to evaluate (None for all)
        seed: Random seed
        verbose: Whether to display progress bar
        
    Returns:
        Dictionary containing evaluation results
    """
    start_time = time.time()
    is_main_process = rank == 0
    
    torch.manual_seed(seed)

    if output_dir and is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Results will be saved to {output_dir}")
    
    tokenizer_dir = get_tokenizer_path(checkpoint_path, tokenizer_path)
    tokenizer_model_path = os.path.join(tokenizer_dir, "tokenizer.model")
    
    if not os.path.exists(tokenizer_model_path):
        logger.warning(f"Expected tokenizer not found at {tokenizer_model_path}")
        if os.path.exists(tokenizer_dir):
            logger.info(f"Files in tokenizer directory: {os.listdir(tokenizer_dir)}")
    
    logger.info(f"Using tokenizer directory: {tokenizer_dir}")
    
    ### Initialization
    cfg_tokenizer = DictConfig({
        '_component_': tokenizer_component,
        'path': tokenizer_model_path,
        'max_seq_len': 'null'
    })
    
    ### Load assets
    logger.info("Loading tokenizer for dataset processing...")
    tokenizer = config.instantiate(cfg_tokenizer)
    
    logger.info("Loading dataset...")
    cfg_dataset = DictConfig({
        '_component_': data_component,
        'split': data_split
    })
    dataset = config.instantiate(cfg_dataset, tokenizer)
    
    # Create sampler for distributed evaluation
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=seed
        )
    
    logger.info("Setting up dataloader...")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=partial(
            collate_fn,
            padding_idx=tokenizer.pad_id,
            ignore_idx=-100,  # CROSS_ENTROPY_IGNORE_IDX
        )
    )
    
    logger.info(f"Loading model from {checkpoint_path}...")
    try:
        llm = LLM(
            model=checkpoint_path,  # Path to the model directory
            trust_remote_code=True,
            tensor_parallel_size=world_size,
            dtype="bfloat16",
            gpu_memory_utilization=0.9
        )
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise
    
    # Get tokenizer special tokens
    if hasattr(tokenizer, "stop_tokens"):
        stop_token_ids = tokenizer.stop_tokens
    else:
        stop_token_ids = [
            tokenizer.eos_id,  # Default to EOS token
        ]
    stop_token_ids = torch.tensor(stop_token_ids)
    pad_id = tokenizer.pad_id
    
    # Setup sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens
    )
    
    ### Run evaluation
    logger.info("Starting evaluation...")
    
    all_results = {
        "metadata": {
            "checkpoint_path": checkpoint_path,
            "tokenizer_path": tokenizer_model_path,
            "data_split": data_split,
            "batch_size": batch_size,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "world_size": world_size,
            "rank": rank,
            "seed": seed,
        },
        "samples": [],  # Will contain detailed per-sample results
    }
    
    # Process batches
    num_batches = len(dataloader) if max_batches is None else min(max_batches, len(dataloader))
    iter_dataloader = enumerate(dataloader)
    
    if verbose and is_main_process:
        iter_dataloader = tqdm(iter_dataloader, total=num_batches, desc="Evaluating batches")
    
    for batch_idx, batch in iter_dataloader:
        if max_batches is not None and batch_idx >= max_batches:
            break
        
        try:
            # Extract tokens and answers
            tokens = batch["tokens"]         # [batch_size, seq_len]
            answers = batch.get("answer", batch.get("answers", None))
            
            if answers is None:
                # Try to extract both era and date if individually present
                era = batch.get("answer_era", None)
                date = batch.get("answer_date", None)
                
                if era is not None and date is not None:
                    # Combine era and date
                    answers = [f"{e} ({d})" for e, d in zip(era, date)]
                else:
                    logger.warning(f"Could not extract answers from batch {batch_idx}")
                    answers = ["unknown"] * tokens.shape[0]
            
            tokens_ls = tokens.tolist()
            
            # Prepare prompts and answers for vLLM
            prompts = []
            batch_answers = []
            
            for i in range(tokens.shape[0]):
                prompt = tokenizer.decode(tokens_ls[i])
                prompts.extend([prompt] * grpo_size) 
                answer = answers[i]
                batch_answers.extend([answer] * grpo_size)
            
            # Generate responses with vLLM
            outputs = llm.generate(prompts, sampling_params)
            
            # Process generated outputs
            response_tokens = []
            response_texts = []
            
            for o in outputs:
                generated_text = o.outputs[0].text
                out_tokens = list(o.outputs[0].token_ids)

                if len(out_tokens) < max_tokens:
                    out_tokens += [pad_id] * (max_tokens - len(out_tokens))
                
                response_tokens.append(out_tokens)
                response_texts.append(generated_text)

            responses_tensor = torch.tensor(response_tokens, dtype=torch.long).reshape(batch_size, grpo_size, max_tokens)
            rewards, successes = reward_server.batch_shaped_correctness_reward(
                tokenizer=tokenizer,
                completions=responses_tensor,
                answers=batch_answers,
                #details_report=True
            )
            advantages = (rewards - rewards.mean(1, keepdim=True)) / (
                rewards.std(1, keepdim=True) + 1e-4
            )
            
            # Store results
            for i in range(batch_size):
                for j in range(grpo_size):
                    sample_idx = i * grpo_size + j
                    # sample_detail = details[i][j] if details else None
                    sample_detail = False

                    sample_data = {
                        "batch_idx": batch_idx,
                        "sample_idx": sample_idx,
                        "prompt": prompts[sample_idx],
                        "answer": batch_answers[sample_idx],
                        "response": response_texts[sample_idx],
                        "reward": float(rewards[i, j]),
                        "success": bool(successes[i, j] > 0.5),
                        "advantage": float(advantages[i, j])
                    }
                    
                    # Add detailed reward components if available
                    if sample_detail:
                        component_data = {}
                        for component in sample_detail.get("reward_components", []):
                            component_name = component.get("component", "unknown")
                            component_value = component.get("value", 0)
                            try:
                                if isinstance(component_value, str) and "overwrites previous" in component_value:
                                    # Handle special values
                                    value = float(component_value.split()[0])
                                else:
                                    value = float(component_value)
                            except (ValueError, TypeError):
                                value = 0
                            
                            component_data[component_name] = value
                            all_results["aggregated"]["reward_components"][component_name].append(value)
                        
                        sample_data["reward_components"] = component_data
                    
                    all_results["samples"].append(sample_data)
                
                # Store batch-level metrics for aggregation
                all_results["aggregated"]["rewards"].extend(rewards[i].tolist())
                all_results["aggregated"]["successes"].extend(successes[i].tolist())
                all_results["aggregated"]["advantages"].extend(advantages[i].tolist())
            
            all_results['html_viz'].append(reward_server.display_responses(
                responses_tensor,
                tokenizer,
                grpo_size,
                advantages=advantages,
                rewards=rewards,
                successes=successes,
                # details=details
                details=None
            ))

        except Exception as e:
            logger.error(f"Error processing batch {batch_idx}: {str(e)}")
            if verbose:
                import traceback
                logger.error(traceback.format_exc())
    
    logger.info("Calculating final statistics...")
    aggregated = all_results["aggregated"]

    for metric in ["rewards", "successes", "advantages"]:
        values = torch.tensor(aggregated[metric])
        aggregated[f"{metric}_mean"] = float(values.mean())
        aggregated[f"{metric}_std"] = float(values.std())
        aggregated[f"{metric}_min"] = float(values.min())
        aggregated[f"{metric}_max"] = float(values.max())
 
    for component, values in aggregated["reward_components"].items():
        values_tensor = torch.tensor(values)
        aggregated["reward_components_stats"] = aggregated.get("reward_components_stats", {})
        aggregated["reward_components_stats"][component] = {
            "mean": float(values_tensor.mean()),
            "std": float(values_tensor.std()),
            "min": float(values_tensor.min()),
            "max": float(values_tensor.max()),
        }
    
    all_results["summary"] = {
        "num_samples": len(all_results["samples"]),
        "mean_reward": aggregated["rewards_mean"],
        "success_rate": aggregated["successes_mean"],
        "mean_advantage": aggregated["advantages_mean"],
        "elapsed_time": time.time() - start_time
    }
    
    if output_dir and is_main_process:
        results_path = os.path.join(output_dir, "eval_results.json")
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        
        # samples_df = pd.DataFrame(all_results["samples"])
        # samples_path = os.path.join(output_dir, "eval_samples.csv")
        # samples_df.to_csv(samples_path, index=False)
        
        # Save summary as separate file
        summary_path = os.path.join(output_dir, "eval_summary.json")
        with open(summary_path, "w") as f:
            json.dump(all_results["summary"], f, indent=2)
            
        logger.info(f"Saved results to {output_dir}")
        logger.info(f"Summary: {json.dumps(all_results['summary'], indent=2)}")

    return all_results


def parse_arguments():
    """Parse command line arguments for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run evaluation on a model checkpoint")
    parser.add_argument("--checkpoint-path", required=True, help="Path to the model checkpoint")
    parser.add_argument("--data-path", required=True, help="Path to the dataset")
    parser.add_argument("--tokenizer-path", help="Optional override path to the tokenizer directory")
    parser.add_argument("--output-dir", help="Directory to save detailed results")
    parser.add_argument("--reward-fn", default="v0", help="Reward function name")
    parser.add_argument("--world-size", type=int, default=1, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--rank", type=int, default=0, help="Process rank for distributed evaluation")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for evaluation")
    parser.add_argument("--grpo-size", type=int, default=2, help="Number of samples per prompt")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling parameter")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--max-batches", type=int, help="Maximum number of batches to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Display progress bar")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    results = run_eval(
        checkpoint_path=args.checkpoint_path,
        data_split=args.data_split,
        tokenizer_path=args.tokenizer_path,
        reward_fn=args.reward_fn,
        output_dir=args.output_dir,
        world_size=args.world_size,
        rank=args.rank,
        batch_size=args.batch_size,
        grpo_size=args.grpo_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_batches=args.max_batches,
        seed=args.seed,
        verbose=args.verbose
    )
    
    if args.rank == 0 and args.output_dir is None:
        logger.info(f"Evaluation summary: {json.dumps(results['summary'], indent=2)}")