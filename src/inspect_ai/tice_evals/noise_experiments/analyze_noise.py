import os
import csv
from inspect_ai.log import list_eval_logs, read_eval_log
import argparse
import yaml
from typing import Dict, List, Any


def analyze_logs(log_dir: str):
    # Create output filename based on log directory name
    dir_name = log_dir.strip("./").rstrip("/")
    output_file = f"{dir_name}_results.csv"

    # Get all logs from the directory
    logs = list_eval_logs(log_dir)

    # Dictionary to store results, keyed by (std, seed, task, counter) tuple
    results: Dict[tuple[float, int, str, int], Dict[str, Any]] = {}
    
    # Counter dictionary to track duplicates
    counter_dict: Dict[tuple[float, int, str], int] = {}

    # Process each log
    for log_path in logs:
        # Read the log header only since we don't need sample data
        log = read_eval_log(log_path, header_only=True)

        # Extract required information
        model_args = log.eval.model_args
        std = model_args.get("noise_std", 0.0)
        seed = model_args.get("seed", None)
        task = log.eval.task
        
        # Get counter for this combination
        base_key = (std, seed, task)
        counter = counter_dict.get(base_key, 0)
        counter_dict[base_key] = counter + 1
        
        # Create key for results dictionary with counter
        key = (std, seed, task, counter)
        
        # Store results
        results[key] = {
            "noise_std": std,
            "noise_percentage": model_args.get("noise_percentage", 1.0),
            "noise_mean": model_args.get("noise_mean", 0.0),
            "seed": seed,
            "task": task,
            "run": counter + 1,  # Add run number to output
            "duplicate": 1 if counter > 0 else 0,  # Add duplicate flag
            "accuracy": None,
            "accuracy_pass_at_1": None,
            "accuracy_pass_at_2": None,
            "accuracy_pass_at_5": None,
        }

        # Extract accuracy from results
        if log.results and log.results.scores:
            if task == "mbpp":
                # For MBPP tasks, extract all accuracy metrics
                for score in log.results.scores:
                    if "accuracy" in score.metrics:
                        reducer = score.reducer
                        accuracy = score.metrics["accuracy"].value
                        if reducer == "mean":
                            results[key]["accuracy"] = accuracy
                        elif reducer == "pass_at_1":
                            results[key]["accuracy_pass_at_1"] = accuracy
                        elif reducer == "pass_at_2":
                            results[key]["accuracy_pass_at_2"] = accuracy
                        elif reducer == "pass_at_5":
                            results[key]["accuracy_pass_at_5"] = accuracy
            else:
                # For non-MBPP tasks, just get the regular accuracy
                for score in log.results.scores:
                    if "accuracy" in score.metrics:
                        results[key]["accuracy"] = score.metrics["accuracy"].value
                        break

    # Write results to CSV, sorted by std, seed, task, and run
    fieldnames = [
        "noise_std",
        "noise_percentage",
        "noise_mean",
        "seed",
        "task",
        "run",
        "duplicate",  # Added duplicate field
        "accuracy",
        "accuracy_pass_at_1",
        "accuracy_pass_at_2",
        "accuracy_pass_at_5",
    ]

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Sort results by std, seed, task, and counter before writing
        sorted_keys = sorted(results.keys())
        for key in sorted_keys:
            writer.writerow(results[key])

    return output_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml', help='Path to config file')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Use log directory from config
    log_dir = config['log_dir']
    output_file = analyze_logs(log_dir)
    print(f"Analysis complete. Results written to {output_file}")


if __name__ == "__main__":
    main()
