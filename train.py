"""
Multimodal Depression Classification
T1 MRI (MedicalNet 3D ResNet) + rs-fMRI (GNN)
5-Fold Cross-Validation with 20% Holdout Test

Target: AUC >= 0.8
"""

import argparse
import copy
import json
import os
from datetime import datetime

# CUDA deterministic settings (set before other imports)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONHASHSEED'] = '42'

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch_geometric.loader import DataLoader

# Model imports
from models.resnet3d import MedicalNetResNet18, Simple3DCNN, load_medicalnet_pretrained
from models.gnn import GNNBackbone
from models.multimodal_fusion import MultimodalFusion, SingleModalityModel

# Shared utilities
from utils import (set_seed, load_labels, load_subject_graph, load_t1_volume,
                   fit_feature_normalizer, compute_metrics, find_optimal_threshold,
                   compute_statistical_tests, build_dataset)


def train_epoch(model, loader, device, criterion, optimizer,
                use_contrastive: bool = False, contrastive_weight: float = 0.1,
                supervised_contrastive: bool = False) -> float:
    """Train for one epoch with optional contrastive loss."""
    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        if use_contrastive:
            contrastive_labels = batch.y.long() if supervised_contrastive else None
            logits, contrastive_loss = model(batch, return_contrastive=True,
                                              contrastive_labels=contrastive_labels)
            cls_loss = criterion(logits, batch.y)
            if contrastive_loss is not None:
                loss = cls_loss + contrastive_weight * contrastive_loss
            else:
                loss = cls_loss
        else:
            logits = model(batch)
            loss = criterion(logits, batch.y)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs

    n_samples = max(len(loader.dataset), 1)
    return total_loss / n_samples


def train_epoch_contrastive_only(model, loader, device, optimizer,
                                  supervised_contrastive: bool = False) -> float:
    """Stage A: Contrastive pretraining only (no classification loss)."""
    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        contrastive_labels = batch.y.long() if supervised_contrastive else None

        _, contrastive_loss = model(batch, return_contrastive=True,
                                     contrastive_labels=contrastive_labels)

        if contrastive_loss is not None:
            contrastive_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += contrastive_loss.item() * batch.num_graphs

    n_samples = max(len(loader.dataset), 1)
    return total_loss / n_samples


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
    parser = argparse.ArgumentParser(description="Multimodal Depression Classification")

    # Data paths
    parser.add_argument("--label-path", default="./data/label.csv")
    parser.add_argument("--outputs-root", default="./data/outputs")
    parser.add_argument("--t1-root", default="./data/T1_MNI")
    parser.add_argument("--adj-name", default="ADJ_abs_dens10.csv")
    parser.add_argument("--medicalnet-ckpt", default="./pretrain/resnet_18_23dataset.pth")

    # Model settings
    parser.add_argument("--modality", choices=["both", "t1", "fmri"], default="both",
                        help="Which modality to use (t1, fmri, or both for multimodal)")
    parser.add_argument("--t1-model", choices=["resnet", "simple"], default="resnet",
                        help="T1 backbone: resnet (MedicalNet pretrained) or simple (vanilla CNN)")
    parser.add_argument("--gnn-model", choices=["gcn", "sage", "gat"], default="gat")
    parser.add_argument("--gnn-hidden", type=int, default=256)
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--fusion-hidden", type=int, default=256)
    parser.add_argument("--fusion-type",
                        choices=["concat", "gated", "attn", "attn_cls", "cross_attn", "cross_attn_uni", "ot"],
                        default="attn")
    parser.add_argument("--dropout", type=float, default=0.25)

    # OT (Sinkhorn) fusion settings
    parser.add_argument("--ot-eps", type=float, default=0.1,
                        help="Sinkhorn regularization (lower = sharper matching)")
    parser.add_argument("--ot-iters", type=int, default=30,
                        help="Number of Sinkhorn iterations")
    parser.add_argument("--ot-proj-dim", type=int, default=128,
                        help="OT projection dimension for cost matrix")
    parser.add_argument("--ot-row-normalize", action="store_true",
                        help="Row-normalize transport plan for stability")
    parser.add_argument("--atlas-path", default="./data/atlas/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii",
                        help="Path to Schaefer200 atlas for ROI pooling")
    parser.add_argument("--ot-t1-layer", choices=["layer2", "layer3"], default="layer2",
                        help="Which T1 CNN layer to extract features from")

    # Contrastive learning settings
    parser.add_argument("--use-contrastive", action="store_true",
                        help="Enable CLIP-style contrastive learning")
    parser.add_argument("--pretrain-contrastive", action="store_true",
                        help="Stage A: pretrain with contrastive loss only")
    parser.add_argument("--pretrain-epochs", type=int, default=20,
                        help="Number of epochs for contrastive pretraining")
    parser.add_argument("--contrastive-weight", type=float, default=0.1,
                        help="Weight for contrastive loss in joint training")
    parser.add_argument("--contrastive-tau", type=float, default=0.07,
                        help="Temperature for InfoNCE loss")
    parser.add_argument("--contrastive-queue", action="store_true",
                        help="Use queue memory for small batch negatives")
    parser.add_argument("--contrastive-queue-size", type=int, default=256,
                        help="Queue size for additional negatives")
    parser.add_argument("--supervised-contrastive", action="store_true",
                        help="Use supervised contrastive (same class = positive)")
    parser.add_argument("--contrastive-mode", choices=["unsupervised", "supervised", "hybrid"],
                        default="unsupervised",
                        help="Contrastive mode: unsupervised, supervised, or hybrid")
    parser.add_argument("--diagonal-weight", type=float, default=2.0,
                        help="Weight for diagonal (same subject) in hybrid mode")

    parser.add_argument("--fusion-dropout", type=float, default=0.3)
    parser.add_argument("--modality-dropout", type=float, default=0.1)

    # Training settings
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr-t1", type=float, default=5e-5)
    parser.add_argument("--lr-gnn", type=float, default=1e-4)
    parser.add_argument("--lr-fusion", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)

    # CV settings
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    # Output
    parser.add_argument("--save-dir", default="./results")

    # Threshold
    parser.add_argument("--fixed-threshold", type=float, default=None,
                        help="Use fixed threshold instead of optimizing (e.g., 0.5)")

    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("="*70)
    print("Multimodal Depression Classification")
    print("T1 (MedicalNet ResNet18) + rs-fMRI (GNN)")
    print("5-Fold CV (80% Train) + 20% Holdout")
    print("="*70)
    print(f"Device: {device}")

    # Load data
    print("\nLoading multimodal data...")
    graphs, labels = build_dataset(
        args.label_path, args.outputs_root, args.t1_root, args.adj_name
    )

    if not graphs:
        raise RuntimeError("No data loaded!")

    labels_array = np.array(labels)
    print(f"Total samples: {len(graphs)}")
    print(f"Class 0 (Normal): {sum(labels_array == 0)}")
    print(f"Class 1 (High-risk): {sum(labels_array == 1)}")

    # Split: 20% holdout test, 80% for CV
    all_indices = np.arange(len(graphs))
    cv_indices, test_indices = train_test_split(
        all_indices, test_size=args.test_size,
        stratify=labels_array, random_state=args.seed
    )

    cv_labels = labels_array[cv_indices]

    print(f"\nData Split:")
    print(f"  Train: {len(cv_indices)} (0={sum(cv_labels==0)}, 1={sum(cv_labels==1)})")
    print(f"  Test:  {len(test_indices)} (0={sum(labels_array[test_indices]==0)}, 1={sum(labels_array[test_indices]==1)})")

    # 5-Fold CV
    kfold = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    fold_results = []
    fold_models = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(cv_indices, cv_labels), 1):
        print(f"\n{'-'*70}")
        print(f"Fold {fold}/{args.folds}")
        print(f"{'-'*70}")

        # Map to original indices
        fold_train_idx = cv_indices[train_idx]
        fold_val_idx = cv_indices[val_idx]

        train_graphs_raw = [graphs[i] for i in fold_train_idx]
        val_graphs_raw = [graphs[i] for i in fold_val_idx]

        # Class distribution per fold
        y_train_labels = [g.y.item() for g in train_graphs_raw]
        train_pos = sum(y_train_labels)
        train_neg = len(y_train_labels) - train_pos

        y_val_labels = [g.y.item() for g in val_graphs_raw]
        val_pos = sum(y_val_labels)
        val_neg = len(y_val_labels) - val_pos

        print(f"  Train: {len(train_graphs_raw)} (pos={train_pos}, neg={train_neg}, ratio={train_pos/len(train_graphs_raw)*100:.1f}%)")
        print(f"  Val:   {len(val_graphs_raw)} (pos={val_pos}, neg={val_neg}, ratio={val_pos/len(val_graphs_raw)*100:.1f}%)")

        # Normalize graph features (fit on train only)
        norm_save_path = os.path.join(args.save_dir, "feat_norm.npz") if fold == 1 else None
        normalizer = fit_feature_normalizer(train_graphs_raw, skip_cols=(1,), save_path=norm_save_path)
        train_graphs = [normalizer(g) for g in train_graphs_raw]
        val_graphs = [normalizer(g) for g in val_graphs_raw]

        # Save ReHo mean per ROI on first fold for external validation
        if fold == 1:
            reho_per_subject = torch.stack([g.x[:, 0] for g in train_graphs_raw])
            reho_mean_per_roi = reho_per_subject.mean(dim=0).numpy()
            reho_save_path = os.path.join(args.save_dir, "train_reho_mean_200.npy")
            np.save(reho_save_path, reho_mean_per_roi)
            print(f"[ReHo] Saved train mean per ROI to {reho_save_path}")

        # Count classes for pos_weight
        y_train = torch.tensor([g.y.item() for g in train_graphs])
        pos = (y_train == 1).sum().item()
        neg = (y_train == 0).sum().item()

        train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)

        # Build model based on modality
        if args.modality == "t1":
            if args.t1_model == "simple":
                t1_backbone = Simple3DCNN(dropout=args.dropout)
                print("[T1-only] Using Simple3DCNN (no pretrain)")
            else:
                t1_backbone = MedicalNetResNet18(dropout=args.dropout)
                load_medicalnet_pretrained(t1_backbone, args.medicalnet_ckpt)
                t1_backbone.freeze_bn()
                print("[T1-only] Using MedicalNet ResNet-18 (pretrained)")

            model = SingleModalityModel(
                backbone=t1_backbone,
                modality="t1",
                hidden=args.fusion_hidden,
                dropout=args.fusion_dropout
            ).to(device)
            param_groups = [
                {"params": model.backbone.parameters(),
                 "lr": args.lr_t1, "weight_decay": args.weight_decay},
                {"params": model.classifier.parameters(),
                 "lr": args.lr_fusion, "weight_decay": args.weight_decay},
            ]
            use_contrastive_training = False

        elif args.modality == "fmri":
            gnn = GNNBackbone(
                in_dim=train_graphs[0].num_node_features,
                hidden=args.gnn_hidden,
                model_type=args.gnn_model,
                dropout=args.dropout,
                heads=args.gat_heads
            )
            model = SingleModalityModel(
                backbone=gnn,
                modality="fmri",
                hidden=args.fusion_hidden,
                dropout=args.fusion_dropout
            ).to(device)
            param_groups = [
                {"params": model.backbone.parameters(),
                 "lr": args.lr_gnn, "weight_decay": args.weight_decay},
                {"params": model.classifier.parameters(),
                 "lr": args.lr_fusion, "weight_decay": args.weight_decay},
            ]
            use_contrastive_training = False

        else:  # both - multimodal
            gnn = GNNBackbone(
                in_dim=train_graphs[0].num_node_features,
                hidden=args.gnn_hidden,
                model_type=args.gnn_model,
                dropout=args.dropout,
                heads=args.gat_heads
            )

            t1_backbone = MedicalNetResNet18(dropout=args.dropout)
            load_medicalnet_pretrained(t1_backbone, args.medicalnet_ckpt)
            t1_backbone.freeze_bn()

            # Determine contrastive mode
            if args.supervised_contrastive:
                effective_contrastive_mode = args.contrastive_mode if args.contrastive_mode != "unsupervised" else "supervised"
            else:
                effective_contrastive_mode = args.contrastive_mode

            model = MultimodalFusion(
                gnn=gnn,
                t1=t1_backbone,
                fusion_hidden=args.fusion_hidden,
                fusion_dropout=args.fusion_dropout,
                fusion_type=args.fusion_type,
                modality_dropout=args.modality_dropout,
                ot_eps=args.ot_eps,
                ot_iters=args.ot_iters,
                ot_proj_dim=args.ot_proj_dim,
                ot_row_normalize=args.ot_row_normalize,
                atlas_path=args.atlas_path,
                ot_t1_layer=args.ot_t1_layer,
                use_contrastive=args.use_contrastive or args.pretrain_contrastive,
                contrastive_tau=args.contrastive_tau,
                contrastive_queue=args.contrastive_queue,
                contrastive_queue_size=args.contrastive_queue_size,
                contrastive_mode=effective_contrastive_mode,
                diagonal_weight=args.diagonal_weight,
            ).to(device)

            param_groups = [
                {"params": model.t1.parameters(),
                 "lr": args.lr_t1, "weight_decay": args.weight_decay},
                {"params": model.gnn.parameters(),
                 "lr": args.lr_gnn, "weight_decay": args.weight_decay},
                {"params": model.classifier.parameters(),
                 "lr": args.lr_fusion, "weight_decay": args.weight_decay},
            ]

            if args.fusion_type == "ot":
                param_groups.extend([
                    {"params": model.sinkhorn_ot.parameters(),
                     "lr": args.lr_fusion, "weight_decay": args.weight_decay},
                    {"params": model.ot_fusion_mlp.parameters(),
                     "lr": args.lr_fusion, "weight_decay": args.weight_decay},
                ])

            if args.use_contrastive or args.pretrain_contrastive:
                param_groups.append({
                    "params": model.contrastive.parameters(),
                    "lr": args.lr_fusion, "weight_decay": args.weight_decay
                })

            use_contrastive_training = args.use_contrastive or args.pretrain_contrastive

        # Loss with class weight
        pos_weight = torch.tensor([neg / max(pos, 1)], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.AdamW(param_groups)

        # Stage A: Contrastive Pretraining (optional, multimodal only)
        if args.pretrain_contrastive and args.modality == "both":
            if effective_contrastive_mode == "hybrid":
                mode_str = f"Hybrid (diag_weight={args.diagonal_weight})"
            elif effective_contrastive_mode == "supervised" or args.supervised_contrastive:
                mode_str = "Supervised"
            else:
                mode_str = "Unsupervised"

            use_labels_for_contrastive = effective_contrastive_mode in ["supervised", "hybrid"] or args.supervised_contrastive

            print(f"\n  [Stage A] {mode_str} Contrastive Pretraining ({args.pretrain_epochs} epochs)")
            for epoch in range(1, args.pretrain_epochs + 1):
                contrast_loss = train_epoch_contrastive_only(
                    model, train_loader, device, optimizer,
                    supervised_contrastive=use_labels_for_contrastive
                )
                if epoch % 5 == 0:
                    print(f"    Pretrain Epoch {epoch}: Contrastive Loss={contrast_loss:.4f}")
            print("  [Stage A] Contrastive pretraining complete")

        # Stage B: Supervised Finetuning
        if args.pretrain_contrastive and args.modality == "both":
            print(f"\n  [Stage B] Supervised Finetuning ({args.epochs} epochs)")

        best_state = None
        best_score = -1e9
        patience_counter = 0

        for epoch in range(1, args.epochs + 1):
            use_joint_contrastive = (use_contrastive_training and
                                     args.use_contrastive and not args.pretrain_contrastive)
            use_labels_for_joint = (effective_contrastive_mode in ["supervised", "hybrid"]
                                    if args.modality == "both" else False) or args.supervised_contrastive
            loss = train_epoch(
                model, train_loader, device, criterion, optimizer,
                use_contrastive=use_joint_contrastive,
                contrastive_weight=args.contrastive_weight,
                supervised_contrastive=use_labels_for_joint
            )

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
                print(f"    Epoch {epoch}: Loss={loss:.4f}, Val AUC={val_metrics['auc']:.4f}")

            if patience_counter >= args.patience:
                print(f"    Early stopping at epoch {epoch}")
                break

        # Load best model
        if best_state:
            model.load_state_dict(best_state)

        # Evaluate on VAL set
        y_true, y_prob = evaluate(model, val_loader, device)
        optimal_thr = find_optimal_threshold(y_true, y_prob)
        val_metrics = compute_metrics(y_true, y_prob, threshold=optimal_thr)

        print(f"\n  Fold {fold} Val Results (threshold={optimal_thr:.3f}):")
        print(f"    AUC={val_metrics['auc']:.4f}, AUPR={val_metrics['aupr']:.4f}, BAcc={val_metrics['balanced_accuracy']:.4f}")
        print(f"    Sens={val_metrics['sensitivity']:.4f}, Spec={val_metrics['specificity']:.4f}")
        print(f"    PPV={val_metrics['ppv']:.4f}, NPV={val_metrics['npv']:.4f}")
        print(f"    Kappa={val_metrics['kappa']:.4f}, MCC={val_metrics['mcc']:.4f}")

        fold_results.append({
            "fold": fold,
            "val_auc": val_metrics["auc"],
            "val_aupr": val_metrics["aupr"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_sensitivity": val_metrics["sensitivity"],
            "val_specificity": val_metrics["specificity"],
            "val_ppv": val_metrics["ppv"],
            "val_npv": val_metrics["npv"],
            "val_f1": val_metrics["f1"],
            "val_f1_weighted": val_metrics["f1_weighted"],
            "val_kappa": val_metrics["kappa"],
            "val_mcc": val_metrics["mcc"],
            "threshold": optimal_thr
        })
        fold_models.append(best_state)

        # Save fold model
        torch.save(best_state, os.path.join(args.save_dir, f"fold_{fold}_model.pt"))

    # Results Summary
    print(f"\n{'='*100}")
    print("5-FOLD CROSS-VALIDATION RESULTS")
    print(f"{'='*100}")

    print(f"\n{'Fold':<6}{'AUC':<8}{'AUPR':<8}{'BAcc':<8}{'Sens':<8}{'Spec':<8}{'PPV':<8}{'NPV':<8}{'Kappa':<8}{'MCC':<8}")
    print("-" * 96)
    for r in fold_results:
        print(f"{r['fold']:<6}{r['val_auc']:<8.4f}{r['val_aupr']:<8.4f}{r['val_balanced_accuracy']:<8.4f}"
              f"{r['val_sensitivity']:<8.4f}{r['val_specificity']:<8.4f}{r['val_ppv']:<8.4f}"
              f"{r['val_npv']:<8.4f}{r['val_kappa']:<8.4f}{r['val_mcc']:<8.4f}")

    cv_auc_mean = np.mean([r["val_auc"] for r in fold_results])
    cv_auc_std = np.std([r["val_auc"] for r in fold_results])
    cv_aupr_mean = np.mean([r["val_aupr"] for r in fold_results])
    cv_aupr_std = np.std([r["val_aupr"] for r in fold_results])
    cv_acc_mean = np.mean([r["val_accuracy"] for r in fold_results])
    cv_acc_std = np.std([r["val_accuracy"] for r in fold_results])
    cv_bacc_mean = np.mean([r["val_balanced_accuracy"] for r in fold_results])
    cv_bacc_std = np.std([r["val_balanced_accuracy"] for r in fold_results])
    cv_sens_mean = np.mean([r["val_sensitivity"] for r in fold_results])
    cv_sens_std = np.std([r["val_sensitivity"] for r in fold_results])
    cv_spec_mean = np.mean([r["val_specificity"] for r in fold_results])
    cv_spec_std = np.std([r["val_specificity"] for r in fold_results])
    cv_ppv_mean = np.mean([r["val_ppv"] for r in fold_results])
    cv_ppv_std = np.std([r["val_ppv"] for r in fold_results])
    cv_npv_mean = np.mean([r["val_npv"] for r in fold_results])
    cv_npv_std = np.std([r["val_npv"] for r in fold_results])
    cv_f1_mean = np.mean([r["val_f1"] for r in fold_results])
    cv_f1_std = np.std([r["val_f1"] for r in fold_results])
    cv_f1w_mean = np.mean([r["val_f1_weighted"] for r in fold_results])
    cv_f1w_std = np.std([r["val_f1_weighted"] for r in fold_results])
    cv_kappa_mean = np.mean([r["val_kappa"] for r in fold_results])
    cv_kappa_std = np.std([r["val_kappa"] for r in fold_results])
    cv_mcc_mean = np.mean([r["val_mcc"] for r in fold_results])
    cv_mcc_std = np.std([r["val_mcc"] for r in fold_results])

    print("-" * 96)
    print(f"{'Mean':<6}{cv_auc_mean:<8.4f}{cv_aupr_mean:<8.4f}{cv_bacc_mean:<8.4f}"
          f"{cv_sens_mean:<8.4f}{cv_spec_mean:<8.4f}{cv_ppv_mean:<8.4f}"
          f"{cv_npv_mean:<8.4f}{cv_kappa_mean:<8.4f}{cv_mcc_mean:<8.4f}")
    print(f"{'Std':<6}{cv_auc_std:<8.4f}{cv_aupr_std:<8.4f}{cv_bacc_std:<8.4f}"
          f"{cv_sens_std:<8.4f}{cv_spec_std:<8.4f}{cv_ppv_std:<8.4f}"
          f"{cv_npv_std:<8.4f}{cv_kappa_std:<8.4f}{cv_mcc_std:<8.4f}")

    print(f"\n{'='*100}")
    print("SUMMARY")
    print(f"{'='*100}")
    print(f"  AUC:              {cv_auc_mean:.4f} +/- {cv_auc_std:.4f}")
    print(f"  AUPR:             {cv_aupr_mean:.4f} +/- {cv_aupr_std:.4f}")
    print(f"  Accuracy:         {cv_acc_mean:.4f} +/- {cv_acc_std:.4f}")
    print(f"  Balanced Acc:     {cv_bacc_mean:.4f} +/- {cv_bacc_std:.4f}")
    print(f"  Sensitivity:      {cv_sens_mean:.4f} +/- {cv_sens_std:.4f}")
    print(f"  Specificity:      {cv_spec_mean:.4f} +/- {cv_spec_std:.4f}")
    print(f"  PPV (Precision):  {cv_ppv_mean:.4f} +/- {cv_ppv_std:.4f}")
    print(f"  NPV:              {cv_npv_mean:.4f} +/- {cv_npv_std:.4f}")
    print(f"  F1-Score:         {cv_f1_mean:.4f} +/- {cv_f1_std:.4f}")
    print(f"  Cohen's Kappa:    {cv_kappa_mean:.4f} +/- {cv_kappa_std:.4f}")
    print(f"  MCC:              {cv_mcc_mean:.4f} +/- {cv_mcc_std:.4f}")
    print(f"{'='*100}")

    # Statistical validation
    stat_tests = compute_statistical_tests(fold_results)
    print(f"\n{'='*100}")
    print("STATISTICAL VALIDATION (One-sample t-test, H0: metric = chance level)")
    print(f"{'='*100}")
    print(f"  {'Metric':<20} {'Mean+/-SD':<18} {'t-stat':>8} {'p-value':>10} {'Sig.':>6}")
    print(f"  {'-'*62}")
    metric_display = {
        "val_auc": ("AUC", cv_auc_mean, cv_auc_std),
        "val_aupr": ("AUPR", cv_aupr_mean, cv_aupr_std),
        "val_balanced_accuracy": ("Balanced Acc", cv_bacc_mean, cv_bacc_std),
        "val_sensitivity": ("Sensitivity", cv_sens_mean, cv_sens_std),
        "val_specificity": ("Specificity", cv_spec_mean, cv_spec_std),
        "val_f1": ("F1-Score", cv_f1_mean, cv_f1_std),
        "val_kappa": ("Cohen's Kappa", cv_kappa_mean, cv_kappa_std),
        "val_mcc": ("MCC", cv_mcc_mean, cv_mcc_std),
    }
    for metric_key, test_result in stat_tests.items():
        name, mean, std = metric_display[metric_key]
        p = test_result["p_value"]
        if p < 0.001:
            sig = "***"
        elif p < 0.01:
            sig = "**"
        elif p < 0.05:
            sig = "*"
        else:
            sig = "n.s."
        print(f"  {name:<20} {mean:.4f}+/-{std:.4f}    {test_result['t_statistic']:>8.2f} {p:>10.4f} {sig:>6}")
    print(f"  {'-'*62}")
    print(f"  Significance: *** p<0.001, ** p<0.01, * p<0.05, n.s. not significant")
    print(f"  Note: df=4 (5-fold CV), one-sided test (H1: metric > chance)")
    print(f"{'='*100}")

    target_achieved = cv_auc_mean >= 0.8

    # Save results
    results = {
        "config": vars(args),
        "cv_results": fold_results,
        "cv_auc_mean": float(cv_auc_mean),
        "cv_auc_std": float(cv_auc_std),
        "cv_aupr_mean": float(cv_aupr_mean),
        "cv_aupr_std": float(cv_aupr_std),
        "cv_accuracy_mean": float(cv_acc_mean),
        "cv_accuracy_std": float(cv_acc_std),
        "cv_balanced_accuracy_mean": float(cv_bacc_mean),
        "cv_balanced_accuracy_std": float(cv_bacc_std),
        "cv_sensitivity_mean": float(cv_sens_mean),
        "cv_sensitivity_std": float(cv_sens_std),
        "cv_specificity_mean": float(cv_spec_mean),
        "cv_specificity_std": float(cv_spec_std),
        "cv_ppv_mean": float(cv_ppv_mean),
        "cv_ppv_std": float(cv_ppv_std),
        "cv_npv_mean": float(cv_npv_mean),
        "cv_npv_std": float(cv_npv_std),
        "cv_f1_mean": float(cv_f1_mean),
        "cv_f1_std": float(cv_f1_std),
        "cv_f1_weighted_mean": float(cv_f1w_mean),
        "cv_f1_weighted_std": float(cv_f1w_std),
        "cv_kappa_mean": float(cv_kappa_mean),
        "cv_kappa_std": float(cv_kappa_std),
        "cv_mcc_mean": float(cv_mcc_mean),
        "cv_mcc_std": float(cv_mcc_std),
        "statistical_tests": stat_tests,
        "target_achieved": target_achieved,
        "timestamp": datetime.now().isoformat()
    }

    with open(os.path.join(args.save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {args.save_dir}")

    return results


if __name__ == "__main__":
    main()
