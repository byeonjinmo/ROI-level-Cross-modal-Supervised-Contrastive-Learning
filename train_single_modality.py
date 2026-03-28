"""
Single Modality Depression Classification
T1 MRI only OR rs-fMRI only
5-Fold Cross-Validation with full statistical metrics

Usage:
    # T1 with MedicalNet ResNet-18 (pretrained)
    python train_single_modality.py --modality t1 --t1-model resnet --save-dir results_t1_resnet

    # T1 with Simple 3D CNN (no pretrain)
    python train_single_modality.py --modality t1 --t1-model simple --save-dir results_t1_simple

    # fMRI with GAT
    python train_single_modality.py --modality fmri --gnn-model gat --save-dir results_fmri_gat
"""

import argparse
import copy
import json
import os
from datetime import datetime

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONHASHSEED'] = '42'

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader

from models.resnet3d import MedicalNetResNet18, Simple3DCNN, load_medicalnet_pretrained
from models.gnn import GNNBackbone
from models.multimodal_fusion import SingleModalityModel

# Shared utilities
from utils import (set_seed, fit_feature_normalizer, compute_metrics,
                   find_optimal_threshold, build_dataset)


def train_epoch(model, loader, device, criterion, optimizer) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, batch.y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / max(len(loader.dataset), 1)


def evaluate(model, loader, device) -> tuple:
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            prob = torch.sigmoid(logits)
            y_prob.extend(prob.cpu().tolist())
            y_true.extend(batch.y.cpu().tolist())
    return np.array(y_true), np.array(y_prob)


def main():
    parser = argparse.ArgumentParser(description="Single Modality Depression Classification")

    # Data paths
    parser.add_argument("--label-path", default="./data/label.csv")
    parser.add_argument("--outputs-root", default="./data/outputs")
    parser.add_argument("--t1-root", default="./data/T1_MNI")
    parser.add_argument("--adj-name", default="ADJ_abs_dens10.csv")
    parser.add_argument("--medicalnet-ckpt", default="./pretrain/resnet_18_23dataset.pth")

    # Model settings
    parser.add_argument("--modality", choices=["t1", "fmri"], required=True)
    parser.add_argument("--t1-model", choices=["resnet", "simple"], default="resnet")
    parser.add_argument("--gnn-model", choices=["gcn", "sage", "gat"], default="gat")
    parser.add_argument("--gnn-hidden", type=int, default=256)
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--fusion-hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--fusion-dropout", type=float, default=0.3)

    # Training settings
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-classifier", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)

    # CV settings
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    # Output
    parser.add_argument("--save-dir", default="./results_single")

    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("="*70)
    print(f"Single Modality Depression Classification: {args.modality.upper()}")
    print("="*70)

    # Load data
    graphs, labels = build_dataset(
        args.label_path, args.outputs_root, args.t1_root, args.adj_name,
        modality=args.modality
    )

    if not graphs:
        raise RuntimeError("No data loaded!")

    labels_array = np.array(labels)
    print(f"Total: {len(graphs)} (0={sum(labels_array==0)}, 1={sum(labels_array==1)})")

    kfold = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.arange(len(graphs)), labels_array), 1):
        print(f"\n{'-'*70}\nFold {fold}/{args.folds}\n{'-'*70}")

        train_graphs_raw = [graphs[i] for i in train_idx]
        val_graphs_raw = [graphs[i] for i in val_idx]

        normalizer = fit_feature_normalizer(train_graphs_raw, skip_cols=(1,))
        train_graphs = [normalizer(g) for g in train_graphs_raw]
        val_graphs = [normalizer(g) for g in val_graphs_raw]

        y_train = torch.tensor([g.y.item() for g in train_graphs])
        pos = (y_train == 1).sum().item()
        neg = (y_train == 0).sum().item()

        train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)

        # Build model
        if args.modality == "t1":
            if args.t1_model == "simple":
                backbone = Simple3DCNN(dropout=args.dropout)
            else:
                backbone = MedicalNetResNet18(dropout=args.dropout)
                load_medicalnet_pretrained(backbone, args.medicalnet_ckpt)
                backbone.freeze_bn()
        else:  # fmri
            backbone = GNNBackbone(
                in_dim=train_graphs[0].num_node_features,
                hidden=args.gnn_hidden,
                model_type=args.gnn_model,
                dropout=args.dropout,
                heads=args.gat_heads
            )

        model = SingleModalityModel(
            backbone=backbone,
            modality=args.modality,
            hidden=args.fusion_hidden,
            dropout=args.fusion_dropout
        ).to(device)

        param_groups = [
            {"params": model.backbone.parameters(), "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": model.classifier.parameters(), "lr": args.lr_classifier, "weight_decay": args.weight_decay},
        ]

        pos_weight = torch.tensor([neg / max(pos, 1)], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(param_groups)

        best_state, best_score, patience_counter = None, -1e9, 0

        for epoch in range(1, args.epochs + 1):
            loss = train_epoch(model, train_loader, device, criterion, optimizer)
            y_val_true, y_val_prob = evaluate(model, val_loader, device)
            val_metrics = compute_metrics(y_val_true, y_val_prob)
            score = val_metrics["auc"] if not np.isnan(val_metrics["auc"]) else val_metrics["accuracy"]

            if score > best_score + 1e-4:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: Loss={loss:.4f}, Val AUC={val_metrics['auc']:.4f}")
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

        if best_state:
            model.load_state_dict(best_state)

        y_true, y_prob = evaluate(model, val_loader, device)
        optimal_thr = find_optimal_threshold(y_true, y_prob)
        val_metrics = compute_metrics(y_true, y_prob, threshold=optimal_thr)

        print(f"\n  Fold {fold}: AUC={val_metrics['auc']:.4f}, Sens={val_metrics['sensitivity']:.4f}, Spec={val_metrics['specificity']:.4f}")

        fold_results.append({
            "fold": fold,
            "val_auc": val_metrics["auc"],
            "val_aupr": val_metrics["aupr"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_sensitivity": val_metrics["sensitivity"],
            "val_specificity": val_metrics["specificity"],
            "val_f1": val_metrics["f1"],
            "val_kappa": val_metrics["kappa"],
            "val_mcc": val_metrics["mcc"],
            "threshold": optimal_thr
        })

        torch.save(best_state, os.path.join(args.save_dir, f"fold_{fold}_model.pt"))

    # Summary
    print(f"\n{'='*70}")
    print(f"RESULTS: {args.modality.upper()} only")
    print(f"{'='*70}")
    cv_auc = np.mean([r["val_auc"] for r in fold_results])
    cv_auc_std = np.std([r["val_auc"] for r in fold_results])
    print(f"  AUC: {cv_auc:.4f} +/- {cv_auc_std:.4f}")

    results = {
        "config": vars(args),
        "cv_results": fold_results,
        "cv_auc_mean": float(cv_auc),
        "cv_auc_std": float(cv_auc_std),
        "timestamp": datetime.now().isoformat()
    }

    with open(os.path.join(args.save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Results saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
