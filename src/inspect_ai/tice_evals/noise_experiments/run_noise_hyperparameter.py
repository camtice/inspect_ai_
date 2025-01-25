import os
from pathlib import Path
import yaml
import json
import gc
import time
import subprocess
import sys
from typing import Optional, Tuple
import numpy as np
from scipy.optimize import minimize_scalar
import torch
from inspect_ai.log import list_eval_logs, read_eval_log

def clear_gpu_memory():
    """Helper function to aggressively clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(f'cuda:{i}'):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_task_args(config: dict) -> tuple[str, dict]:
    """Get task path and arguments based on configuration."""
    task_name = config['task'].get('name')
    if task_name:
        task_config = config['task'][task_name]
        
        if task_name == 'gsm8k':
            return (
                task_config['path'],
                {
                    'sandbagging': task_config['sandbagging'],
                    'use_cot': task_config['use_cot'],
                    'fewshot': task_config['fewshot'],
                    'num_problems': task_config['num_problems']
                }
            )
        elif task_name == 'arc_challenge':
            return (
                task_config['path'],
                {'sandbagging': task_config['sandbagging']}
            )
        else:
            raise ValueError(f"Unknown task: {task_name}")
    else:
        return (config['task']['path'], {})

def run_single_eval(std: float, seed: int, config: dict):
    """Run a single evaluation with given standard deviation and seed"""
    task_path, task_args = get_task_args(config)
    
    # Create a unique log file name for this run
    run_log_dir = os.path.join(config['log_dir'], f"std_{std:.6f}_seed_{seed}")
    os.makedirs(run_log_dir, exist_ok=True)
    
    cmd = [
        sys.executable, "-c",
        f"""
import os
import warnings
import transformers
from inspect_ai import eval

warnings.filterwarnings('ignore')
transformers.logging.set_verbosity_error()

os.environ["INSPECT_LOG_DIR"] = "{run_log_dir}"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

eval(
    tasks="{task_path}",
    max_tokens={config['model']['max_tokens']},
    model=["{config['model']['name']}"],
    do_sample={str(config['model']['do_sample']).lower().capitalize()},
    model_args={{
        "noise_percentage": {config['noise']['percentage']},
        "noise_mean": {config['noise']['mean']},
        "noise_std": {std},
        "seed": {seed},
    }},
    task_config={task_args}
)
        """
    ]
    
    subprocess.run(cmd, check=True)
    return run_log_dir

def extract_accuracy(log_dir: str) -> float:
    """Extract accuracy metrics from the evaluation logs"""
    try:
        # Get all logs from the directory (should only be one)
        logs = list_eval_logs(log_dir)
        if not logs:
            return 0.0
            
        # Read the first log file
        log = read_eval_log(logs[0])
        
        # Extract accuracy based on task type
        if log.results and log.results.scores:
            if log.eval.task == "mbpp":
                # For MBPP, get pass@1 accuracy
                for score in log.results.scores:
                    if "accuracy" in score.metrics and score.reducer == "pass_at_1":
                        return score.metrics["accuracy"].value
            else:
                # For other tasks, get regular accuracy
                for score in log.results.scores:
                    if "accuracy" in score.metrics:
                        return score.metrics["accuracy"].value
                        
        return 0.0
            
    except Exception as e:
        print(f"Error extracting accuracy: {e}")
        return 0.0

def optimize_std_for_seed(seed: int, config: dict) -> Tuple[float, float]:
    """Find optimal std for a given seed using Bayesian optimization"""
    std_range = config['noise'].get('std_range', {})
    min_std = std_range.get('min', 0.0001)
    max_std = std_range.get('max', 0.01)
    
    def objective(std: float) -> float:
        """Objective function to minimize (negative accuracy)"""
        print(f"\nTrying std={std:.6f} for seed={seed}")
        run_log_dir = run_single_eval(std=std, seed=seed, config=config)
        accuracy = extract_accuracy(run_log_dir)
        print(f"Achieved accuracy: {accuracy:.4f}")
        
        # Clear GPU memory between runs
        clear_gpu_memory()
        
        # Return negative accuracy since we want to maximize accuracy
        return -accuracy

    # Use scipy's minimize_scalar with bounds
    result = minimize_scalar(
        objective,
        bounds=(min_std, max_std),
        method='bounded',
        options={
            'xatol': config['optimization'].get('tolerance', 1e-5),
            'maxiter': config['optimization'].get('max_iterations', 20)
        }
    )
    
    optimal_std = result.x
    best_accuracy = -result.fun
    
    return optimal_std, best_accuracy

def get_baseline_accuracy(seed: int, config: dict) -> float:
    """Run evaluation with no noise (std=0) to get baseline accuracy"""
    print("\nGetting baseline accuracy (no noise)...")
    run_log_dir = run_single_eval(std=0.0, seed=seed, config=config)
    accuracy = extract_accuracy(run_log_dir)
    print(f"Baseline accuracy (no noise): {accuracy:.4f}")
    
    # Clear GPU memory after baseline run
    clear_gpu_memory()
    
    return accuracy

def main(config_path: str = "configs/hyperparam.yaml"):
    # Load configuration
    config = load_config(config_path)
    
    # Set up logging
    log_dir = config['log_dir']
    os.makedirs(log_dir, exist_ok=True)

    # Get seeds from config
    seeds = config['noise'].get('seeds', [42])
    
    # Store results for each seed
    results = {}
    
    for seed in seeds:
        print(f"\nOptimizing std for seed {seed}")
        # Get baseline accuracy first
        baseline_accuracy = get_baseline_accuracy(seed, config)
        
        optimal_std, best_accuracy = optimize_std_for_seed(seed, config)
        results[seed] = {
            'baseline_accuracy': float(baseline_accuracy),
            'optimal_std': float(optimal_std),  # Convert numpy float to Python float
            'best_accuracy': float(best_accuracy)
        }
        print(f"Optimal std for seed {seed}: {optimal_std:.6f} (accuracy: {best_accuracy:.4f})")
    
    # Save optimization results
    results_file = Path(log_dir) / "optimization_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    # Print summary
    print("\nOptimization Results Summary:")
    print("-" * 50)
    for seed, result in results.items():
        print(f"Seed {seed}:")
        print(f"  Baseline accuracy (no noise): {result['baseline_accuracy']:.4f}")
        print(f"  Optimal std: {result['optimal_std']:.6f}")
        print(f"  Best accuracy: {result['best_accuracy']:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/hyperparam.yaml', help='Path to config file')
    args = parser.parse_args()
    main(args.config)