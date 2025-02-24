from inspect_ai.tice_evals.noise_experiments.analyze_noise import analyze_logs

# This will create a CSV file with the results
output_file = analyze_logs("./logs/math_pwd_lock_deepseek_math7b_on_weak_pythia1b/0.0055_std_sweep")
print(f"Analysis complete. Results written to {output_file}") 