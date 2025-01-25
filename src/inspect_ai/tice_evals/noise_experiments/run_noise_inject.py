import os
from pathlib import Path
import yaml
from inspect_ai import eval
import torch
import gc
import time
import subprocess
import sys
from tqdm import tqdm

def clear_gpu_memory():
    """Helper function to aggressively clear GPU memory"""
    if torch.cuda.is_available():
        # Clear PyTorch's CUDA memory
        torch.cuda.empty_cache()
        # Force garbage collection
        gc.collect()
        # Additional aggressive memory clearing
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(f'cuda:{i}'):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_task_args(task_path: str, config: dict) -> dict:
    """Get task arguments based on configuration and task path."""
    task_name = config['task'].get('name')
    if not task_name:
        return {}

    task_config = config['task'][task_name]
    
    # Extract the base task name from the path
    task_base_name = Path(task_path).stem  # This will get 'gsm8k' or 'gsm8k_sandbag' from the filename
    
    if 'gsm8k' in task_base_name.lower():
        return {
            'task_name': task_base_name,  # Add the specific task name
            'sandbagging': task_config['sandbagging'],
            'use_cot': task_config['use_cot'],
            'fewshot': task_config['fewshot'],
            'num_problems': task_config['num_problems']
        }
    elif 'arc_challenge' in task_name.lower():
        return {'sandbagging': task_config['sandbagging']}
    else:
        return {}

def run_single_eval(task_path: str, std: float, seed: int, config: dict):
    """Run a single evaluation with given task, standard deviation and seed"""
    task_args = get_task_args(task_path, config)
    
    cmd = [
        sys.executable, "-c",
        f"""
import os
import warnings
import transformers
from inspect_ai import eval

warnings.filterwarnings('ignore')
transformers.logging.set_verbosity_error()

os.environ["INSPECT_LOG_DIR"] = "{config['log_dir']}"
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

def main(config_path: str = "configs/default.yaml"):
    # Load configuration
    config = load_config(config_path)
    
    # Set up logging
    log_dir = config['log_dir']
    os.makedirs(log_dir, exist_ok=True)

    # Get seeds from config, default to [None] if not specified
    seeds = config['noise'].get('seeds', [None])
    
    # Get list of tasks to run
    tasks = config['tasks']
    if not isinstance(tasks, list):
        tasks = [tasks]
    
    # Calculate total iterations for progress bar
    total_iterations = len(tasks) * len(seeds) * len(config['noise']['std_values'])
    
    # Create progress bar
    with tqdm(total=total_iterations, desc="Running evaluations") as pbar:
        # Run eval for each combination of task, seed and std
        for task_path in tasks:
            for seed in seeds:
                for std in config['noise']['std_values']:
                    try:
                        print(f"\nRunning evaluation for task={task_path}, std={std}, seed={seed}")
                        run_single_eval(task_path=task_path, std=std, seed=seed, config=config)
                    except Exception as e:
                        print(f"Error during evaluation of {task_path} with std={std}, seed={seed}: {str(e)}")
                    
                    # Add a small delay between runs
                    time.sleep(2)
                    
                    # Clear GPU memory between runs
                    clear_gpu_memory()
                    
                    # Update progress bar
                    pbar.update(1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml', help='Path to config file')
    args = parser.parse_args()
    main(args.config)
