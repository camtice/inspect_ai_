import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
import os
from pathlib import Path
from typing import List, Tuple

# Import configuration helpers from analyze_noise.py
from analyze_noise import load_analysis_config, get_default_csv

def get_output_path(config, input_file, suffix=""):
    """
    Determine the output path for saving files based on config and input file.
    
    Args:
        config: Configuration dictionary
        input_file: Input CSV file path
        suffix: Optional suffix to add to the filename
        
    Returns:
        Path to save the output file
    """
    log_dir = config.get('log_dir')
    if not log_dir:
        # Fallback to the directory of the input file
        return os.path.splitext(input_file)[0] + f"{suffix}"
    
    # Use the log directory with the base name of the input file
    base_name = os.path.basename(os.path.splitext(input_file)[0])
    return os.path.join(log_dir, f"{base_name}{suffix}")

def read_csv_data(file_path: str) -> pd.DataFrame:
    """
    Read the CSV file containing experimental results.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame containing the experimental results
    """
    return pd.read_csv(file_path)

def get_max_accuracy(df: pd.DataFrame) -> float:
    """
    Get the maximum accuracy value from the entire dataset.
    
    Args:
        df: DataFrame containing the experimental results
        
    Returns:
        Maximum accuracy value
    """
    return df['accuracy'].max()

def sample_and_get_max(df: pd.DataFrame, sample_size: int, num_trials: int = 1000) -> List[float]:
    """
    For a given sample size, randomly sample that many rows from the DataFrame
    and find the maximum accuracy for each trial.
    
    Args:
        df: DataFrame containing the experimental results
        sample_size: Number of seeds to sample in each trial
        num_trials: Number of trials to run
        
    Returns:
        List of maximum accuracies from each trial
    """
    max_accuracies = []
    
    for _ in range(num_trials):
        # Randomly sample rows with replacement
        sampled_df = df.sample(n=sample_size, replace=True)
        # Get maximum accuracy from the sample
        max_acc = sampled_df['accuracy'].max()
        max_accuracies.append(max_acc)
    
    return max_accuracies

def calculate_recovery_proportions(df: pd.DataFrame, sample_sizes: List[int], 
                                 num_trials: int = 1000) -> Tuple[List[int], List[float], List[List[float]]]:
    """
    Calculate the proportion of maximum accuracy recovered for different sample sizes.
    
    Args:
        df: DataFrame containing the experimental results
        sample_sizes: List of sample sizes to evaluate
        num_trials: Number of trials for each sample size
        
    Returns:
        Tuple containing:
        - List of sample sizes
        - List of average proportions of max accuracy recovered
        - List of all proportions for each trial (for confidence intervals)
    """
    max_accuracy = get_max_accuracy(df)
    avg_proportions = []
    all_proportions = []
    
    for size in sample_sizes:
        print(f"Processing sample size: {size}")
        max_accuracies = sample_and_get_max(df, size, num_trials)
        # Calculate proportion of max accuracy recovered for each trial
        proportions = [acc / max_accuracy if max_accuracy > 0 else 0 for acc in max_accuracies]
        # Calculate average proportion
        avg_proportion = np.mean(proportions)
        
        avg_proportions.append(avg_proportion)
        all_proportions.append(proportions)
    
    return sample_sizes, avg_proportions, all_proportions

def plot_results(sample_sizes: List[int], avg_proportions: List[float], 
                all_proportions: List[List[float]], output_path: str = None):
    """
    Create a plot showing the proportion of maximum accuracy recovered 
    as a function of sample size.
    
    Args:
        sample_sizes: List of sample sizes
        avg_proportions: List of average proportions
        all_proportions: List of all proportions for each trial
        output_path: Path to save the plot (if None, shows the plot)
    """
    plt.figure(figsize=(10, 6))
    
    # Set style
    sns.set_style("whitegrid")
    
    # Plot average proportions
    plt.plot(sample_sizes, avg_proportions, 'o-', linewidth=2, markersize=8, label='Average')
    
    # Set labels and title
    plt.xlabel('Number of Seeds Sampled', fontsize=14)
    plt.ylabel('Proportion of Maximum Accuracy Recovered', fontsize=14)
    plt.title('Recovery of Peak Performance vs. Number of Seeds Sampled', fontsize=16)
    
    # Set x-axis to log scale since we're using powers of 2
    plt.xscale('log', base=10)
    plt.xticks(sample_sizes, [str(size) for size in sample_sizes])
    
    # Set y-axis limits
    plt.ylim(0, 1.05)
    
    plt.legend()
    plt.tight_layout()
    
    if output_path:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

def analyze_scaling(input_file: str, max_accuracy: float = 0.7, num_trials: int = 1000, seed: int = 42, output_file: str = None):
    """
    Analyze how the proportion of maximum accuracy scales with number of seeds sampled.
    Split analysis by noise_std values if multiple are present.
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Read the CSV file
    print(f"Reading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Get unique std values
    std_values = sorted(df['noise_std'].unique())
    print(f"Found {len(std_values)} different std values: {std_values}")
    
    # Create figure for multiple plots if needed
    if len(std_values) > 1:
        fig, axes = plt.subplots(1, len(std_values), figsize=(15, 6))
        if not isinstance(axes, np.ndarray):
            axes = [axes]
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        axes = [ax]
    
    all_results = []
    
    for idx, std_val in enumerate(std_values):
        # Filter data for current std value
        std_df = df[df['noise_std'] == std_val]
        seeds = std_df['seed'].values
        accuracies = std_df['accuracy'].values
        
        # Define sample sizes (powers of 10)
        sample_sizes = [1, 2, 5, 10, 20, 50, 100]  # Using 1-2-5 series
        sample_sizes = [size for size in sample_sizes if size <= len(seeds)]  # Only use available samples
        
        results = []
        all_proportions = []
        
        print(f"\nAnalyzing std={std_val} with {len(seeds)} seeds...")
        for n in tqdm(sample_sizes):
            max_accs = []
            proportions = []
            for _ in range(num_trials):
                # Randomly sample n seeds with replacement
                sample_indices = np.random.choice(len(seeds), size=n, replace=True)
                sample_accs = accuracies[sample_indices]
                
                # Get maximum accuracy in this sample
                max_acc = np.max(sample_accs)
                max_accs.append(max_acc)
                proportions.append(max_acc / max_accuracy)
            
            # Calculate average of maximum accuracies across trials
            avg_max_acc = np.mean(max_accs)
            avg_proportion = avg_max_acc / max_accuracy
            
            results.append({
                'std': std_val,
                'sample_size': n,
                'avg_max_accuracy': avg_max_acc,
                'proportion_recovered': avg_proportion
            })
            all_proportions.append(proportions)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        all_results.append(results_df)
        
        # Plot on corresponding axis
        ax = axes[idx]
        sns.set_style("whitegrid")
        
        # Plot average proportions
        ax.plot(results_df['sample_size'], results_df['proportion_recovered'], 
                'o-', linewidth=2, markersize=8, label=f'std={std_val}')
        
        # Set labels and title
        ax.set_xlabel('Number of Seeds Sampled', fontsize=12)
        ax.set_ylabel('Proportion of Maximum Accuracy Recovered', fontsize=12)
        ax.set_title(f'Recovery vs Seeds (std={std_val})', fontsize=14)
        
        # Set x-axis to log scale
        ax.set_xscale('log', base=10)
        ax.set_xticks(sample_sizes)
        ax.set_xticklabels([str(size) for size in sample_sizes])
        
        # Set y-axis limits
        ax.set_ylim(0, 1.05)
        ax.legend()
    
    plt.tight_layout()
    
    if output_file:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()
    
    # Combine all results
    combined_results = pd.concat(all_results)
    return combined_results

def main():
    parser = argparse.ArgumentParser(description="Analyze accuracy scaling with sample size")
    parser.add_argument('--config', type=str, default=None, help='Path to config file (optional)')
    parser.add_argument('--input-file', type=str, default=None, help='Path to the CSV file containing results (optional)')
    parser.add_argument("--max-accuracy", type=float, default=0.7, 
                        help="Maximum possible accuracy (default: 0.7)")
    parser.add_argument("--num-trials", type=int, default=1000, 
                        help="Number of random sampling trials (default: 1000)")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path for the plot (optional, overrides default location)")
    args = parser.parse_args()
    
    # Load configuration from YAML
    config = load_analysis_config(args.config)
    
    # Determine input file: priority CLI > config 'scaling_input' > default based on log_dir
    if args.input_file is not None:
        input_file = args.input_file
    elif config.get('scaling_input') is not None:
        input_file = config.get('scaling_input')
    else:
        log_dir = config.get('log_dir')
        if not log_dir:
            print("Error: log_dir must be specified in config")
            return
        # get_default_csv now returns the full path
        input_file = get_default_csv(log_dir)
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return
    
    # Determine output paths
    if args.output:
        output_file = args.output
        output_csv = os.path.splitext(args.output)[0] + ".csv"
    else:
        output_file = get_output_path(config, input_file, "_scaling.png")
        output_csv = get_output_path(config, input_file, "_scaling_results.csv")
    
    print("-" * 40)
    print(f"Analyzing scaling with max accuracy = {args.max_accuracy}")
    print("-" * 40)
    
    results = analyze_scaling(input_file, 
                              max_accuracy=args.max_accuracy,
                              num_trials=args.num_trials,
                              seed=args.seed,
                              output_file=output_file)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Save results to CSV
    results.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")
    print("-" * 40)

if __name__ == "__main__":
    main() 