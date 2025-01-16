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

`run_noise_inject.py` is the main script for running experiments. It allows you to specify a range of noise std values to test, and will run the eval script for each noise std value.

At the top of `run_noise_inject.py` you can define the standard deviations to test, the model name, and the task to evaluate. You can pull evals from the Inspect Evals repository, add them to the current folder, and then run the experiment.

Setting the variables: 

```python
    # Define the standard deviations to test
    std_values = [0.0, 0.001, 0.002, 0.003, 0.004, 0.005]

    # Define the model name
    model_name = "noise_hf/microsoft/Phi-3-mini-4k-instruct"

    # Define the task to evaluate
    task = "src/inspect_ai/tice_evals/sandbag_arc_challenge.py"
```

### Running the Experiment 

```bash
git clone https://github.com/camtice/inspect_ai_.git
cd inspect_ai_
pip install -e .
python src/inspect_ai/tice_evals/run_noise_inject.py
```

### Analyzing Results
Use `analyze_noise.py` to process evaluation logs:

```python
from inspect_ai.tice_evals.analyze_noise import analyze_logs

output_file = analyze_logs("./sandbag_experiment_logs")
```

## Requirements
- GPU with 16GB+ RAM (tested on L4/A100)
- Python 3.8+

