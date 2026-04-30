import subprocess

SEEDS = [1, 2, 3, 4, 5]

COMMANDS = [
    "python generate_novatech_dataset.py --config experiment_config.json --seed {seed}",
    "python baseline_rag.py",
    "python run_poisoned_rag.py",
    "python corpus_defense.py",
    "python run_defended_rag.py",
    "python run_context_sanitized_rag.py",
    "python run_post_generation_defense_rag.py",
    "python run_majority_vote_rag.py",
    "python analyze_attack_results.py"
]

for seed in SEEDS:
    print("=" * 80)
    print(f"Running experiment seed {seed}")
    print("=" * 80)

    for cmd in COMMANDS:
        cmd = cmd.format(seed=seed)
        print(f"\nRunning: {cmd}")
        subprocess.run(cmd, shell=True, check=True)