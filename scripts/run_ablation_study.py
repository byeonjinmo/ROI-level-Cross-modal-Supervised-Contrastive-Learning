"""
Ablation Study: Fusion Methods & Class Imbalance Handling
1. Fusion types: concat, gated, attn
2. Data configs: stratified, all_highrisk
3. Class weight strategies: auto, fixed_9, fixed_7
"""

import subprocess
import json
import os
from datetime import datetime
import pandas as pd

RESULTS_DIR = "./ablation_results"

EXPERIMENTS = [
    {"name": "fusion_concat_stratified", "fusion": "concat", "data_config": "stratified", "pos_weight": "auto"},
    {"name": "fusion_gated_stratified", "fusion": "gated", "data_config": "stratified", "pos_weight": "auto"},
    {"name": "fusion_attn_stratified", "fusion": "attn", "data_config": "stratified", "pos_weight": "auto"},
    {"name": "data_allhighrisk_attn", "fusion": "attn", "data_config": "all_highrisk", "pos_weight": "auto"},
    {"name": "weight_fixed9_allhighrisk", "fusion": "attn", "data_config": "all_highrisk", "pos_weight": "9.0"},
    {"name": "weight_fixed7_allhighrisk", "fusion": "attn", "data_config": "all_highrisk", "pos_weight": "7.0"},
]


def run_experiment(exp_config):
    name = exp_config["name"]
    save_dir = os.path.join(RESULTS_DIR, name)
    os.makedirs(save_dir, exist_ok=True)

    cmd = [
        "python", "train.py",
        "--gnn-model", "gat",
        "--fusion-type", exp_config["fusion"],
        "--batch-size", "2",
        "--epochs", "100",
        "--patience", "15",
        "--save-dir", save_dir
    ]

    print(f"\n{'='*70}\nRunning: {name}\n{'='*70}")

    subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    results_path = os.path.join(save_dir, "results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            return json.load(f)
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=["fusion", "data", "weight", "all"], default="all")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.exp == "fusion":
        exps = [e for e in EXPERIMENTS if e["name"].startswith("fusion_")]
    elif args.exp == "data":
        exps = [e for e in EXPERIMENTS if e["name"].startswith("data_")]
    elif args.exp == "weight":
        exps = [e for e in EXPERIMENTS if e["name"].startswith("weight_")]
    else:
        exps = EXPERIMENTS

    results = []
    for exp in exps:
        result = run_experiment(exp)
        if result:
            result["experiment"] = exp["name"]
            result["fusion"] = exp["fusion"]
            results.append(result)

    print(f"\n{'='*70}\nABLATION STUDY RESULTS\n{'='*70}")

    summary_data = []
    for r in results:
        summary_data.append({
            "Experiment": r["experiment"],
            "Fusion": r["fusion"],
            "AUC": f"{r.get('cv_auc_mean', 0):.4f}+/-{r.get('cv_auc_std', 0):.4f}",
            "Sens": f"{r.get('cv_sensitivity_mean', 0):.4f}",
            "Spec": f"{r.get('cv_specificity_mean', 0):.4f}",
        })

    df = pd.DataFrame(summary_data)
    print(df.to_string(index=False))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(RESULTS_DIR, f"ablation_summary_{timestamp}.csv")
    df.to_csv(summary_path, index=False)
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
