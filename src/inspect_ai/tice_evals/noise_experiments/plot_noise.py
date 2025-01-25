import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the CSV file
input_file = 'logs/No_CoT_25_01_25_v3_results.csv'
df = pd.read_csv(input_file)

# Create the plot with a larger figure size
plt.figure(figsize=(10, 6))

# Get unique seeds and create a color map
unique_seeds = df['seed'].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_seeds)))  # Using tab10 colormap
seed_to_color = dict(zip(unique_seeds, colors))

# Plot each seed-duplicate combination as a separate line
for seed in unique_seeds:
    seed_data = df[df['seed'] == seed]
    color = seed_to_color[seed]
    
    for duplicate in seed_data['duplicate'].unique():
        mask = (seed_data['duplicate'] == duplicate)
        data = seed_data[mask]
        
        # Solid line for first run, dashed for others
        linestyle = '-' if duplicate == 0 else '--'
        label = f'Seed {seed} (Run {duplicate + 1})'
        
        plt.plot(data['noise_std'], data['accuracy'], 
                marker='o', 
                label=label,
                color=color,
                linestyle=linestyle)

# Customize the plot
plt.xlabel('Noise Standard Deviation')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Noise Standard Deviation for Different Seeds')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout to prevent legend cutoff
plt.tight_layout()

# Generate output filename based on input filename
output_file = input_file.replace('.csv', '_plot.png').replace('logs/', '')

# Save the plot
plt.savefig(output_file, bbox_inches='tight', dpi=300)
plt.close()