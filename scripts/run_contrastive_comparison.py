#!/usr/bin/env python3
"""
Contrastive Learning Comparison Script
Compare three methods under equal conditions:
1. Baseline: Standard supervised learning (No Contrastive)
2. Self-supervised: Cross-modal contrastive (same subject = positive)
3. Supervised (Ours): Supervised contrastive (same class = positive)

Usage:
    python scripts/run_contrastive_comparison.py
    python scripts/run_contrastive_comparison.py --compile-only
    python scripts/run_contrastive_comparison.py --print-commands
"""

import subprocess
import argparse
import os
import json
import pandas as pd
from datetime import datetime


INTERNAL_CONFIG = {
    "name": "Internal",
    "script": "train.py",
    "args": [],
    "params": {
        "fusion_type": "attn",
        "gnn_hidden": 256,
        "gat_heads": 4,
        "fusion_hidden": 256,
        "fusion_dropout": 0.4,
        "lr_t1": "2e-5",
        "lr_gnn": "5e-5",
        "lr_fusion": "1e-4",
        "patience": 30,
        "epochs": 150,
        "seed": 42,
        "contrastive_tau": 0.05,
        "pretrain_epochs": 50,
    },
}

CONDITIONS = {
    "baseline": {
        "name": "Baseline (No Contrastive)",
        "pretrain_contrastive": False,
        "supervised_contrastive": False,
    },
    "self_supervised": {
        "name": "Self-supervised Contrastive",
        "pretrain_contrastive": True,
        "supervised_contrastive": False,
    },
    "supervised": {
        "name": "Supervised Contrastive (Ours)",
        "pretrain_contrastive": True,
        "supervised_contrastive": True,
    },
}


def build_command(config, condition_key, save_dir):
    condition = CONDITIONS[condition_key]
    params = config["params"]

    cmd = ["python", config["script"]]
    cmd.extend(config["args"])

    cmd.extend([
        "--fusion-type", params["fusion_type"],
        "--gnn-hidden", str(params["gnn_hidden"]),
        "--gat-heads", str(params["gat_heads"]),
        "--fusion-hidden", str(params["fusion_hidden"]),
        "--fusion-dropout", str(params["fusion_dropout"]),
        "--lr-t1", params["lr_t1"],
        "--lr-gnn", params["lr_gnn"],
        "--lr-fusion", params["lr_fusion"],
        "--patience", str(params["patience"]),
        "--epochs", str(params["epochs"]),
        "--seed", str(params["seed"]),
        "--save-dir", save_dir,
    ])

    if condition["pretrain_contrastive"]:
        cmd.append("--pretrain-contrastive")
        cmd.extend(["--pretrain-epochs", str(params["pretrain_epochs"])])
        cmd.extend(["--contrastive-tau", str(params["contrastive_tau"])])
        if condition["supervised_contrastive"]:
            cmd.append("--supervised-contrastive")

    return cmd


def run_experiment(config, condition_key, output_base):
    condition = CONDITIONS[condition_key]
    save_dir = os.path.join(output_base, f"{config['name'].lower()}_{condition_key}")

    print("\n" + "=" * 70)
    print(f"Running: {condition['name']}")
    print(f"Save dir: {save_dir}")
    print("=" * 70)

    cmd = build_command(config, condition_key, save_dir)
    print(f"Command:\n{' '.join(cmd)}")

    try:
        subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return {"status": "success", "save_dir": save_dir}
    except Exception as e:
        print(f"Error: {e}")
        return {"status": "failed", "error": str(e)}


def load_results(save_dir):
    for fname in ["results.json", "cv_results.json", "summary.json"]:
        fpath = os.path.join(save_dir, fname)
        if os.path.exists(fpath):
            with open(fpath, 'r') as f:
                return json.load(f)
    return None


def compile_results(output_base, dataset_name):
    results = []
    for condition_key, condition in CONDITIONS.items():
        save_dir = os.path.join(output_base, f"{dataset_name.lower()}_{condition_key}")
        data = load_results(save_dir)

        if data:
            def get_metric(data, *keys):
                for key in keys:
                    if key in data:
                        val = data[key]
                        if isinstance(val, (int, float)):
                            return f"{val:.4f}"
                        return val
                return "N/A"

            row = {
                "Method": condition["name"],
                "AUC": get_metric(data, "cv_auc_mean"),
                "AUC_std": get_metric(data, "cv_auc_std"),
                "AUPR": get_metric(data, "cv_aupr_mean"),
                "Sensitivity": get_metric(data, "cv_sensitivity_mean"),
                "Specificity": get_metric(data, "cv_specificity_mean"),
                "F1": get_metric(data, "cv_f1_mean"),
            }
        else:
            row = {"Method": condition["name"], "AUC": "N/A", "AUPR": "N/A",
                   "Sensitivity": "N/A", "Specificity": "N/A", "F1": "N/A"}
        results.append(row)
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Compare contrastive learning methods")
    parser.add_argument("--output-base", default="results_contrastive_compare")
    parser.add_argument("--conditions", nargs="+",
                        choices=["baseline", "self_supervised", "supervised"],
                        default=["baseline", "self_supervised", "supervised"])
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--print-commands", action="store_true")
    args = parser.parse_args()

    config = INTERNAL_CONFIG

    if args.print_commands:
        for condition_key in args.conditions:
            save_dir = os.path.join(args.output_base, f"{config['name'].lower()}_{condition_key}")
            cmd = build_command(config, condition_key, save_dir)
            print(f"# {CONDITIONS[condition_key]['name']}")
            print(" ".join(cmd))
            print()
        return

    if not args.compile_only:
        for condition_key in args.conditions:
            run_experiment(config, condition_key, args.output_base)

    os.makedirs(args.output_base, exist_ok=True)
    df = compile_results(args.output_base, config['name'])
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)
    print(df.to_string(index=False))

    csv_path = os.path.join(args.output_base, "comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
