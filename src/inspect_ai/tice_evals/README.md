# Tice - Noise Injection Evals
The current folder holds eval scripts for the implementation of the `noise_hf.py` provider found in src/inspect_ai/model/_providers/noise_hf.py. The `noise_hf.py` script is the base, which reimplements the noise injection technique from https://arxiv.org/abs/2412.01784.

`run_noise_inject.py` implements the experimental setup from the paper, where you can run evals, while increase the std of the gaussian noise added to the model weights. `run_noise_inject.py` can utilyze any eval script from the Inspect Evals respository, which can be found here: https://github.com/UKGovernmentBEIS/inspect_evals.


## Core Components

### NoiseHF Provider (`noise_hf.py`)
The main addition to Inspect AI that enables controlled weight perturbation. It can be used as a provider in the same fashion as `hf.py` would be used in the original Inspect AI repository. However, a simple set up like this is less useful for experimentation, as you often will want to search a range of noise std values to see how performance increases. 

```python
from inspect_ai import eval

# Example usage with noise injection
eval(
    tasks="path/to/eval",
    model=["noise_hf/microsoft/Phi-3-mini-4k-instruct"],
    model_args={
        "noise_percentage": 1.0,  # Percentage of weights to affect
        "noise_mean": 0.0,       # Mean of Gaussian noise
        "noise_std": 0.001,      # Standard deviation of noise
        "seed": 999,            # Optional seed for reproducibility
    }
)
```

Key parameters for noise injection:
- `noise_percentage`: Proportion of weights to perturb (currently only 1.0 is supported)
- `noise_mean`: Center of Gaussian noise distribution
- `noise_std`: Standard deviation of noise
- `seed`: Optional seed for reproducible noise generation

### Experimental Setup

The experiment workflow consists of three main components:
1. Configuration file (`configs/default.yaml`)
2. Experiment runner (`run_noise_inject.py`)
3. Analysis and visualization tools (`analyze_noise.py` and `plot_noise.py`)

#### Configuration

Create a YAML configuration file (e.g., `configs/default.yaml`) to specify your experiment parameters:

```yaml
# Model configuration
model:
  name: "noise_hf/meta-llama/Meta-Llama-3.1-8B-Instruct"
  max_tokens: 10
  do_sample: false

# Noise parameters
noise:
  std_values: [0.0, 0.0005, 0.001, 0.0015, 0.002, 0.0025]
  percentage: 1.0
  mean: 0.0
  seeds: [1,2,3,4,5]

# List of task scripts to run
tasks:
  - "gsm8k.py"
  - "gsm8k_sandbag.py"

# Task configuration 
# Currently task parameters cannot be passed and have to be changed manually in 
# the task script. Feature to be added in the future.
task:
  name: "gsm8k" 
  gsm8k:
    sandbagging: true # Does not work
    use_cot: true # Does not work
    fewshot: 0
    num_problems: 4

# Logging
log_dir: "./logs/experiment_name"
```

### Running Experiments

To run the experiment:

```bash
git clone https://github.com/camtice/inspect_ai_.git
cd inspect_ai_
pip install -e .
python src/inspect_ai/tice_evals/noise_experiments/run_noise_inject.py 
```

### Analyzing Results

1. First, analyze the logs to generate a CSV file:
```python
from inspect_ai.tice_evals.noise_experiments.analyze_noise import analyze_logs

# This will create a CSV file with the results
output_file = analyze_logs("./logs/experiment_name")
```

2. Then, create visualizations using the plotting script:
```python
# Make sure to manually change the input file to the CSV file you want to plot
python src/inspect_ai/tice_evals/noise_experiments/plot_noise.py
```

The plotting script will create a visualization showing:
- Accuracy vs. Noise Standard Deviation
- Different lines for each seed

## Requirements
- GPU with 16GB+ RAM (tested on L4/A100)
- Python 3.8+

