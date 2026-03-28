"""
Fusion Method & Learning Rate Comparison
Compare: concat, gated, attn fusion methods
"""

import subprocess
import json
import os
from datetime import datetime

BASE_CMD = [
    "python", "train.py",
    "--gnn-model", "gat",
    "--batch-size", "2",
    "--epochs", "100",
    "--patience", "15"
]

EXPERIMENTS = {
    "fusion_comparison": [
        {"fusion-type": "concat", "lr-t1": "1e-4", "lr-fusion": "5e-4"},
        {"fusion-type": "gated", "lr-t1": "1e-4", "lr-fusion": "5e-4"},
        {"fusion-type": "attn", "lr-t1": "1e-4", "lr-fusion": "5e-4"},
    ],
    "lr_comparison": [
        {"fusion-type": "concat", "lr-t1": "5e-5", "lr-fusion": "5e-4"},
        {"fusion-type": "concat", "lr-t1": "1e-4", "lr-fusion": "5e-4"},
        {"fusion-type": "concat", "lr-t1": "2e-4", "lr-fusion": "5e-4"},
        {"fusion-type": "concat", "lr-t1": "1e-4", "lr-fusion": "1e-4"},
        {"fusion-type": "concat", "lr-t1": "1e-4", "lr-fusion": "1e-3"},
    ],
    "best_combinations": [
        {"fusion-type": "gated", "lr-t1": "5e-5", "lr-fusion": "5e-4"},
        {"fusion-type": "attn", "lr-t1": "5e-5", "lr-fusion": "5e-4"},
        {"fusion-type": "attn", "lr-t1": "1e-4", "lr-fusion": "1e-3"},
    ]
}


def run_experiment(config, exp_name, save_base="./experiments"):
    config_str = f"{config['fusion-type']}_t1-{config['lr-t1']}_fu-{config['lr-fusion']}"
    save_dir = os.path.join(save_base, exp_name, config_str)
    os.makedirs(save_dir, exist_ok=True)

    cmd = BASE_CMD.copy()
    cmd.extend(["--fusion-type", config["fusion-type"]])
    cmd.extend(["--lr-t1", config["lr-t1"]])
    cmd.extend(["--lr-fusion", config["lr-fusion"]])
    cmd.extend(["--save-dir", save_dir])

    print(f"\n{'='*70}\nRunning: {config_str}\n{'='*70}")

    subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    results_path = os.path.join(save_dir, "results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            return json.load(f)
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=list(EXPERIMENTS.keys()) + ["all", "quick"],
                       default="quick")
    args = parser.parse_args()

    all_results = []

    if args.exp == "quick":
        experiments = EXPERIMENTS["fusion_comparison"]
        exp_name = "fusion_comparison"
    elif args.exp == "all":
        for exp_name, configs in EXPERIMENTS.items():
            for config in configs:
                result = run_experiment(config, exp_name)
                if result:
                    result["config_name"] = f"{config['fusion-type']}_t1-{config['lr-t1']}_fu-{config['lr-fusion']}"
                    all_results.append(result)
        exp_name = "all"
        experiments = []
    else:
        experiments = EXPERIMENTS[args.exp]
        exp_name = args.exp

    if experiments:
        for config in experiments:
            result = run_experiment(config, exp_name)
            if result:
                result["config_name"] = f"{config['fusion-type']}_t1-{config['lr-t1']}_fu-{config['lr-fusion']}"
                all_results.append(result)

    # Summary
    print(f"\n{'='*70}\nEXPERIMENT SUMMARY\n{'='*70}")
    print(f"\n{'Config':<40}{'AUC':<12}{'Sens':<12}{'Spec':<12}")
    print("-" * 76)

    all_results.sort(key=lambda x: x.get("cv_auc_mean", 0), reverse=True)
    for r in all_results:
        print(f"{r['config_name']:<40}{r.get('cv_auc_mean', 0):<12.4f}"
              f"{r.get('cv_sensitivity_mean', 0):<12.4f}{r.get('cv_specificity_mean', 0):<12.4f}")

    if all_results:
        best = all_results[0]
        print(f"\nBEST: {best['config_name']} (AUC={best.get('cv_auc_mean', 0):.4f})")


if __name__ == "__main__":
    main()
