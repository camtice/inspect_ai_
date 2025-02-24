import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_noise_accuracy_plot(csv_path, output_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Filter out duplicate rows (where duplicate == 1)
    df = df[df['duplicate'] != 1]
    
    # Set style
    plt.style.use("seaborn-v0_8-whitegrid")
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Get unique noise levels
    noise_levels = sorted(df['noise_std'].unique())
    positions = range(len(noise_levels))
    
    # Prepare data for boxplot
    accuracies = [df[df['noise_std'] == noise]['accuracy'].values for noise in noise_levels]
    
    # Create boxplot
    bp = plt.boxplot(accuracies, 
                    positions=positions,
                    patch_artist=True,
                    showmeans=True,
                    boxprops=dict(facecolor='#348ABD', color='#348ABD'),
                    capprops=dict(color='#348ABD'),
                    whiskerprops=dict(color='#348ABD'),
                    flierprops=dict(color='#348ABD', markeredgecolor='#348ABD'),
                    medianprops=dict(color='#348ABD'),
                    meanprops=dict(color="k", marker=".", markerfacecolor="k", markeredgecolor="k"),
                    widths=0.3)
    
    # Add connecting line for means
    means = [np.mean(acc) for acc in accuracies]
    plt.plot(positions, means, 'k-.', alpha=0.8, zorder=1)
    
    # Customize plot
    plt.title('Model Accuracy vs. Noise Level\n(DeepSeek Math 7B)', pad=20)
    plt.xlabel('Noise Standard Deviation')
    plt.ylabel('Accuracy')
    
    # Set x-ticks to actual noise values
    plt.xticks(positions, [f'{n:.3f}' for n in noise_levels], rotation=30, ha='right')
    
    # Set y-axis limits
    plt.ylim(0, 0.7)
    
    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Path to your CSV file
    csv_path = "logs/math_pwd_lock_deepseek_math7b_on_weak_pythia1b/0.0055_std_sweep_results.csv"
    
    # Path where you want to save the plot
    output_path = "noise_accuracy_plot.png"
    
    # Create plot and save it
    create_noise_accuracy_plot(csv_path, output_path)