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
import json
import shutil

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

def run_single_eval(config, task_path, std, seed):
    """Run a single evaluation with the given parameters"""
    print(f"\nRunning evaluation with configuration:")
    print(f"  Task: {task_path}")
    print(f"  Noise std: {std}")
    print(f"  Seed: {seed}")
    
    # Get LoRA configuration
    use_lora = config['noise'].get('use_lora', False)
    lora_r = config['noise'].get('lora_r', 8)
    
    # Validate LoRA rank for vLLM compatibility
    if use_lora and lora_r not in [8, 16, 32, 64, 128, 256]:
        print(f"Warning: vLLM only supports LoRA ranks of 8, 16, 32, 64, 128, 256. Adjusting from {lora_r} to 8.")
        lora_r = 8
    
    # Get target modules if specified
    lora_target_modules = config['noise'].get('lora_target_modules', None)
    
    # Get model path and tokenizer path
    model_path = config['noise'].get('model_path', "redwoodresearch/math_pwd_lock_deepseek_math7b_on_weak_pythia1b")
    
    # Get tokenizer path from config, with proper fallbacks
    # 1. First check if tokenizer_path is defined in model config
    # 2. Then check if tokenizer_path is defined in noise config
    # 3. Finally, default to model_path if neither is specified
    tokenizer_path = config['model'].get('tokenizer_path', 
                    config['noise'].get('tokenizer_path', model_path))
    
    print(f"  Use LoRA: {use_lora}")
    if use_lora:
        print(f"  LoRA rank: {lora_r}")
        if lora_target_modules:
            print(f"  LoRA target modules: {lora_target_modules}")
        else:
            print(f"  LoRA target modules: auto-detect all linear layers")
    
    print(f"  Model path: {model_path}")
    print(f"  Tokenizer path: {tokenizer_path}")
    
    if use_lora:
        print("Starting evaluation with LoRA-based noise generation")
    else:
        print("Starting evaluation with traditional noise generation")
    
    # Construct model_args dictionary
    model_args = {
        'model_path': model_path,
        'tokenizer_path': tokenizer_path,
        'noise_std': std,
        'noise_mean': config['noise'].get('mean', 0.0),
        'noise_percentage': config['noise'].get('percentage', 1.0),
        'seed': seed,
        'use_lora': use_lora,
        'lora_r': lora_r
    }
    
    # Add target modules if specified
    if lora_target_modules:
        model_args["lora_target_modules"] = lora_target_modules
    
    # Get absolute paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Use json.dumps to properly format the dictionary as a string
    model_args_str = json.dumps(model_args)
    # Replace JSON booleans with Python booleans
    model_args_str = model_args_str.replace('true', 'True').replace('false', 'False')
    
    # Create a specialized log directory based on parameters for better organization
    # Remove the duplicate model name from the log path
    log_subdir = f"{config['log_dir']}/seed_{seed}_sweep"
    os.makedirs(log_subdir, exist_ok=True)
    
    # Get task args and force use_chat_template to False
    task_args = get_task_args(task_path, config)
    task_args['use_chat_template'] = False  # Force this to be False to avoid chat template errors
    task_args['add_generation_prompt'] = False  # Also disable generation prompt which can cause issues
    
    # Use json.dumps to properly format the dictionary as a string
    task_args_str = json.dumps(task_args)
    # Replace JSON booleans with Python booleans
    task_args_str = task_args_str.replace('true', 'True').replace('false', 'False')
    
    # Use the model_args_str in the eval command
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
os.environ["INSPECT_LOG_DIR"] = "{log_subdir}"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "0"  # Allow online mode to download models if needed

print("Starting evaluation with {'LoRA-based' if use_lora else 'traditional'} noise generation")
eval(
    tasks="{task_path}",
    max_tokens={config['model']['max_tokens']},
    model=["noise_hf/{model_path}"],
    do_sample={str(config['model']['do_sample']).lower().capitalize()},
    temperature={config['model']['temperature']},
    model_args={model_args_str},
    task_config={task_args_str},
    seed=42,
    log_format="json"
)
print("Evaluation completed")
        """
    ]
    
    # Run the command with added memory/process management
    try:
        # First ensure GPU memory is clear
        clear_gpu_memory()
        
        # Run the command
        subprocess.run(cmd, check=True)
        
        # Successful completion - allow a moment for processes to properly exit
        time.sleep(1)
        
        # Clean up again
        clear_gpu_memory()
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running evaluation: {e}")
        # Force more aggressive cleanup in case of errors
        clear_gpu_memory()
        time.sleep(2)
        clear_gpu_memory()
        return False

def main(config_path: str = "configs/default.yaml"):
    # Load configuration
    config = load_config(config_path)
    
    # Set up logging
    log_dir = config['log_dir']
    
    # Check if log directory exists and contains results
    if os.path.exists(log_dir):
        # Check if there are any files in the directory
        if any(os.scandir(log_dir)):
            print(f"\nWARNING: Log directory '{log_dir}' already exists and contains files.")
            print("Running the experiment may overwrite or mix results with existing data.")
            confirmation = input("Are you sure you want to proceed? Type 'yes' to continue: ")
            if confirmation.lower() != 'yes':
                print("Experiment aborted.")
                return
    
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Save a copy of the configuration file to the log directory
    config_filename = os.path.basename(config_path)
    config_dest = os.path.join(log_dir, f"run_config_{config_filename}")
    shutil.copy2(config_path, config_dest)
    print(f"Configuration saved to: {config_dest}")

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
                        success = run_single_eval(config=config, task_path=task_path, std=std, seed=seed)
                        
                        if not success:
                            print(f"Evaluation failed for {task_path} with std={std}, seed={seed}")
                    except Exception as e:
                        print(f"Error during evaluation of {task_path} with std={std}, seed={seed}: {str(e)}")
                    
                    # Add a small delay between runs to allow for resource cleanup
                    time.sleep(2)
                    
                    # Clear GPU memory between runs
                    clear_gpu_memory()
                    
                    # Update progress bar
                    pbar.update(1)
    
    # Final memory cleanup
    clear_gpu_memory()
    print("All evaluations completed!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml', help='Path to config file')
    args = parser.parse_args()
    main(args.config)
