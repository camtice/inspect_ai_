import os
import csv
from inspect_ai.log import list_eval_logs, read_eval_log
import argparse
import yaml
from typing import Dict, List, Any

# Helper functions inserted for configuration handling

def load_analysis_config(config_path: str = None) -> dict:
    """Load analysis configuration from a YAML file."""
    import yaml, os
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "analysis_config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_default_csv(log_dir: str) -> str:
    """Compute the default CSV filename based on the log directory."""
    dir_name = log_dir.strip("./").rstrip("/")
    # Return the full path within the log directory
    return os.path.join(log_dir, f"{os.path.basename(dir_name)}_results.csv")


def analyze_logs(log_dir: str):
    # Create output filename based on log directory name
    dir_name = log_dir.strip("./").rstrip("/")
    # Save the output file within the log directory itself
    output_file = os.path.join(log_dir, f"{os.path.basename(dir_name)}_results.csv")

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

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

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

    print(f"Analysis complete. Results written to {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Analyze noise experiment logs")
    parser.add_argument('--config', help='Path to config file (optional)')
    parser.add_argument('--log-dir', help='Path to log directory (alternative to config)')
    args = parser.parse_args()

    config = load_analysis_config(args.config)
    log_dir = args.log_dir if args.log_dir else config.get("log_dir")
    
    if not log_dir:
        print("Error: log_dir must be specified in config or as --log-dir")
        return

    if not os.path.isdir(log_dir):
        print(f"Error: Invalid log directory: {log_dir}")
        return

    output_file = analyze_logs(log_dir)
    print(f"Analysis complete. Results written to {output_file}")


if __name__ == "__main__":
    main()
