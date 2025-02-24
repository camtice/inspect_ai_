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

    # Extract the base task name from the path
    task_base_name = Path(task_path).stem
    
    # Get task specific config
    task_config = config['task'].get(task_name, {})
    
    if task_base_name == 'pw_gsm8k':
        return {
            'fewshot': task_config.get('fewshot', 0),
            'fewshot_seed': task_config.get('fewshot_seed', 42),
            'num_problems': task_config.get('num_problems', 3),
            'use_chat_template': task_config.get('use_chat_template', False),
            'add_generation_prompt': task_config.get('add_generation_prompt', True)
        }
    elif 'arc_challenge' in task_name.lower():
        return {'sandbagging': task_config['sandbagging']}
    else:
        return {}

def run_single_eval(task_path: str, std: float, seed: int, config: dict):
    """Run a single evaluation with given task, standard deviation and seed"""
    task_args = get_task_args(task_path, config)
    
    # Get absolute paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tokenizer_path = current_dir  # Update to use current directory where tokenizer files are located
    model_path = "/home/ubuntu/.cache/huggingface/hub/models--redwoodresearch--math_pwd_lock_deepseek_math7b_on_weak_pythia1b/snapshots/4d306de4a0abb5e42143930857e0d1d222b433da"
    
    cmd = [
        sys.executable, "-c",
        f"""
import os
import warnings
import transformers
from inspect_ai import eval

warnings.filterwarnings('ignore')
transformers.logging.set_verbosity_error()

# Debug settings
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["INSPECT_LOG_DIR"] = "{config['log_dir']}"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # Force offline mode

eval(
    tasks="{task_path}",
    max_tokens={config['model']['max_tokens']},
    model=["noise_hf/redwoodresearch/math_pwd_lock_deepseek_math7b_on_weak_pythia1b"],
    do_sample={str(config['model']['do_sample']).lower().capitalize()},
    temperature={config['model']['temperature']},
    model_args={{
        "noise_percentage": {config['noise']['percentage']},
        "noise_mean": {config['noise']['mean']},
        "noise_std": {std},
        "seed": {seed},
        "model_path": "{model_path}",
        "tokenizer_path": "{tokenizer_path}",
    }},
    task_config={task_args},
    log_format="json"
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
