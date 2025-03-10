import pandas as pd
from analyze_noise import load_analysis_config, get_default_csv  # Import config helpers
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse

# Set default matplotlib style
plt.style.use("ggplot")
plt.style.use("seaborn-v0_8-whitegrid")

def get_output_path(config, input_file, suffix=""):
    """
    Determine the output path for saving plots based on config and input file.
    
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
        return os.path.splitext(input_file)[0] + f"{suffix}.png"
    
    # Use the log directory with the base name of the input file
    base_name = os.path.basename(os.path.splitext(input_file)[0])
    return os.path.join(log_dir, f"{base_name}{suffix}.png")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Plot accuracy vs. noise with configurable number of seeds')
    parser.add_argument('--config', type=str, default=None, help='Path to config file (optional)')
    parser.add_argument('--input-file', type=str, default=None,
                        help='Path to the CSV file with results (optional)')
    parser.add_argument('--max-seeds', type=int, default=None, 
                        help='Maximum number of seeds to include (default: use all seeds)')
    parser.add_argument('--seed-variants', type=str, default=None,
                        help='Comma-separated list of seed counts to compare (e.g., "10,20,50,100")')
    parser.add_argument('--output-file', type=str, default=None,
                        help='Path to save the output plot (optional, overrides default location)')
    parser.add_argument('--use-boxplot', action='store_true', default=True,
                        help='Use box plots instead of error bars (default: True)')
    parser.add_argument('--split-by-std', action='store_true', default=False,
                        help='Split analysis by noise_std values if multiple are present')
    args = parser.parse_args()

    # Load configuration from YAML
    config = load_analysis_config(args.config)
    
    # Determine input_file: priority CLI > config 'plot_input' > default based on log_dir
    if args.input_file is not None:
        input_file = args.input_file
    elif config.get('plot_input') is not None:
        input_file = config.get('plot_input')
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

    # Read the CSV file
    print(f"Reading data from {input_file}")
    df = pd.read_csv(input_file)
    
    # Process seed variants if specified
    if args.seed_variants:
        plot_seed_variants(df, args.seed_variants, input_file, args.output_file, config, 
                          use_boxplot=args.use_boxplot, split_by_std=args.split_by_std)
    else:
        # Process with single max_seeds value
        plot_single(df, args.max_seeds, input_file, args.output_file, config, 
                   use_boxplot=args.use_boxplot, split_by_std=args.split_by_std)

def plot_single(df, max_seeds, input_file, output_file=None, config=None, 
               use_boxplot=True, split_by_std=False):
    """Plot accuracy vs. noise with a single max_seeds value."""
    # Filter out duplicate runs if present
    if 'duplicate' in df.columns:
        df = df[df['duplicate'] == 0]
    
    # Check if we need to split by std values
    if split_by_std and 'noise_std' in df.columns:
        std_values = sorted(df['noise_std'].unique())
        if len(std_values) > 1:
            print(f"Found {len(std_values)} different std values: {std_values}")
            
            # Create a figure with subplots for each std value
            fig, axes = plt.subplots(1, len(std_values), figsize=(15, 6), sharey=True)
            if not isinstance(axes, np.ndarray):
                axes = [axes]
                
            for idx, std_val in enumerate(std_values):
                # Filter data for current std value
                std_df = df[df['noise_std'] == std_val]
                
                # Plot for this std value
                _plot_single_std(std_df, max_seeds, axes[idx], use_boxplot, 
                                title_suffix=f" (std={std_val})")
                
            plt.tight_layout()
        else:
            # Only one std value, use regular plotting
            fig, ax = plt.subplots(figsize=(10, 6))
            _plot_single_std(df, max_seeds, ax, use_boxplot)
    else:
        # Regular plotting without splitting
        fig, ax = plt.subplots(figsize=(10, 6))
        _plot_single_std(df, max_seeds, ax, use_boxplot)
    
    # Determine output path
    if output_file:
        save_path = output_file
    elif config:
        suffix = f"_max_seeds_{max_seeds}" if max_seeds else ""
        suffix += "_boxplot" if use_boxplot else "_errorbar"
        suffix += "_split_std" if split_by_std else ""
        save_path = get_output_path(config, input_file, suffix)
    else:
        # Default fallback
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        suffix = f"_max_seeds_{max_seeds}" if max_seeds else ""
        suffix += "_boxplot" if use_boxplot else "_errorbar"
        suffix += "_split_std" if split_by_std else ""
        save_path = f"{base_name}_plot{suffix}.png"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()

def _plot_single_std(df, max_seeds, ax, use_boxplot=True, title_suffix=""):
    """Helper function to plot data for a single std value."""
    # If max_seeds is specified, limit the number of seeds used
    if max_seeds is not None and max_seeds > 0:
        # For each noise level, randomly sample max_seeds seeds
        filtered_rows = []
        for std_val, group in df.groupby('noise_std'):
            if len(group) > max_seeds:
                # Randomly sample max_seeds rows
                sampled = group.sample(n=max_seeds, random_state=42)
                filtered_rows.append(sampled)
            else:
                filtered_rows.append(group)
        
        # Combine the filtered rows
        filtered_df = pd.concat(filtered_rows)
        
        # Update the title to reflect the number of seeds used
        title = f'Accuracy vs. Noise (Seeds: {max_seeds}){title_suffix}'
    else:
        # Use all available seeds
        filtered_df = df
        title = f'Accuracy vs. Noise (All Seeds){title_suffix}'
    
    if use_boxplot:
        # Create box plot
        data_to_plot = []
        labels = []
        
        # Group by noise_std and collect accuracy values for each group
        for std_val, group in filtered_df.groupby('noise_std'):
            data_to_plot.append(group['accuracy'].values)
            labels.append(str(std_val))
        
        # Get colors from the default color cycle
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        
        # Create the box plot
        box_plot = ax.boxplot(data_to_plot, patch_artist=True, showmeans=True,
                             boxprops=dict(facecolor=colors[0], color=colors[0]),
                             capprops=dict(color=colors[0]),
                             whiskerprops=dict(color=colors[0]),
                             flierprops=dict(color=colors[0], markeredgecolor=colors[0]),
                             medianprops=dict(color='white'),
                             meanprops=dict(marker='o', markerfacecolor='white', 
                                           markeredgecolor='black'))
        
        # Add dotted lines connecting the means
        means = [np.mean(data) for data in data_to_plot]
        ax.plot(range(1, len(means) + 1), means, color='black', linestyle='dotted', linewidth=1.5)
        
        # Set x-tick labels to noise_std values
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
    else:
        # Use traditional error bar plot
        grouped = filtered_df.groupby('noise_std')['accuracy'].agg(['mean', 'std']).reset_index()
        ax.errorbar(grouped['noise_std'], grouped['mean'], yerr=grouped['std'], 
                   fmt='o-', capsize=5, linewidth=2, markersize=8)
    
    # Add labels and title
    ax.set_xlabel('Noise Standard Deviation', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def plot_seed_variants(df, seed_variants_str, input_file, output_file=None, config=None,
                      use_boxplot=True, split_by_std=False):
    """Plot accuracy vs. noise with multiple seed variant values for comparison."""
    # Parse the seed variants
    seed_variants = [int(x.strip()) for x in seed_variants_str.split(',')]
    
    # Filter out duplicate runs if present
    if 'duplicate' in df.columns:
        df = df[df['duplicate'] == 0]
    
    # Check if we need to split by std values
    if split_by_std and 'noise_std' in df.columns:
        std_values = sorted(df['noise_std'].unique())
        if len(std_values) > 1:
            print(f"Found {len(std_values)} different std values: {std_values}")
            
            # Create a figure with subplots for each std value
            fig, axes = plt.subplots(len(std_values), 1, figsize=(12, 6*len(std_values)), sharex=True)
            if not isinstance(axes, np.ndarray):
                axes = [axes]
                
            for idx, std_val in enumerate(std_values):
                # Filter data for current std value
                std_df = df[df['noise_std'] == std_val]
                
                # Plot for this std value
                _plot_seed_variants_std(std_df, seed_variants, axes[idx], use_boxplot, 
                                       title_suffix=f" (std={std_val})")
                
            plt.tight_layout()
        else:
            # Only one std value, use regular plotting
            fig, ax = plt.subplots(figsize=(12, 8))
            _plot_seed_variants_std(df, seed_variants, ax, use_boxplot)
    else:
        # Regular plotting without splitting
        fig, ax = plt.subplots(figsize=(12, 8))
        _plot_seed_variants_std(df, seed_variants, ax, use_boxplot)
    
    # Determine output path
    if output_file:
        save_path = output_file
    elif config:
        suffix = f"_seed_variants_{'-'.join(str(s) for s in seed_variants)}"
        suffix += "_boxplot" if use_boxplot else "_errorbar"
        suffix += "_split_std" if split_by_std else ""
        save_path = get_output_path(config, input_file, suffix)
    else:
        # Default fallback
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        suffix = f"_seed_variants_{'-'.join(str(s) for s in seed_variants)}"
        suffix += "_boxplot" if use_boxplot else "_errorbar"
        suffix += "_split_std" if split_by_std else ""
        save_path = f"{base_name}_plot{suffix}.png"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()

def _plot_seed_variants_std(df, seed_variants, ax, use_boxplot=True, title_suffix=""):
    """Helper function to plot seed variants for a single std value."""
    if use_boxplot:
        # For box plots, we need to organize data differently
        all_data = []
        all_positions = []
        all_labels = []
        
        # Get unique noise_std values
        noise_std_values = sorted(df['noise_std'].unique())
        
        # Calculate positions for box plots
        position_offset = 0
        
        # Get colors from the default color cycle
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        
        for i, max_seeds in enumerate(seed_variants):
            data_for_variant = []
            positions_for_variant = []
            
            for j, std_val in enumerate(noise_std_values):
                # Get data for this noise level
                std_group = df[df['noise_std'] == std_val]
                
                if len(std_group) > max_seeds:
                    # Randomly sample max_seeds rows
                    sampled = std_group.sample(n=max_seeds, random_state=42)
                else:
                    sampled = std_group
                
                # Add data and position
                data_for_variant.append(sampled['accuracy'].values)
                positions_for_variant.append(j + position_offset)
                
                # Only add labels for the first variant
                if i == 0:
                    all_labels.append(str(std_val))
            
            # Plot this variant's box plots
            box_plot = ax.boxplot(data_for_variant, positions=positions_for_variant, 
                                 patch_artist=True, widths=0.15,
                                 boxprops=dict(facecolor=colors[i], color=colors[i]),
                                 capprops=dict(color=colors[i]),
                                 whiskerprops=dict(color=colors[i]),
                                 flierprops=dict(color=colors[i], markeredgecolor=colors[i]),
                                 medianprops=dict(color='white'),
                                 meanprops=dict(marker='o', markerfacecolor='white', 
                                              markeredgecolor='black'),
                                 labels=[])
            
            # Add dotted lines connecting the means
            means = [np.mean(data) for data in data_for_variant]
            ax.plot(positions_for_variant, means, color=colors[i], linestyle='dotted', linewidth=1.5)
            
            # Increment position offset for next variant
            position_offset += 0.2
            
            # Add to legend
            ax.plot([], [], color=colors[i], label=f'Seeds: {max_seeds}')
        
        # Set x-tick positions and labels
        ax.set_xticks(np.arange(len(noise_std_values)) + 0.2 * (len(seed_variants) - 1) / 2)
        ax.set_xticklabels(all_labels)
    else:
        # Traditional error bar plots
        # Get colors from the default color cycle
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        
        for i, max_seeds in enumerate(seed_variants):
            # For each noise level, randomly sample max_seeds seeds
            filtered_rows = []
            for std_val, group in df.groupby('noise_std'):
                if len(group) > max_seeds:
                    # Randomly sample max_seeds rows
                    sampled = group.sample(n=max_seeds, random_state=42)
                    filtered_rows.append(sampled)
                else:
                    filtered_rows.append(group)
            
            # Combine the filtered rows
            filtered_df = pd.concat(filtered_rows)
            
            # Calculate statistics
            grouped = filtered_df.groupby('noise_std')['accuracy'].agg(['mean', 'std']).reset_index()
            
            # Plot this variant
            ax.errorbar(grouped['noise_std'], grouped['mean'], yerr=grouped['std'], 
                       fmt='o-', capsize=5, linewidth=2, markersize=8, 
                       label=f'Seeds: {max_seeds}', color=colors[i])
    
    # Add labels and title
    ax.set_xlabel('Noise Standard Deviation', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'Accuracy vs. Noise for Different Seed Counts{title_suffix}', fontsize=14)
    
    # Add legend and grid
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

if __name__ == "__main__":
    main() 