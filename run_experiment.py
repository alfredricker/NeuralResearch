#!/usr/bin/env python3
"""
Cortical Column vs. Modern Hopfield vs. CNN  —  MNIST classification benchmark.

All three models are sized to roughly the same parameter budget (~50 K) so
the comparison reflects architectural differences, not raw capacity.

Usage:
    python run_experiment.py                     # full run (15 epochs)
    python run_experiment.py --epochs 3 --quick  # quick sanity check
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from src.cnn import SmallCNN
from src.cortical_column import CorticalColumn
from src.data import get_dataloaders
from src.hopfield import ModernHopfieldNetwork
from src.train import RunMetrics, print_comparison, train_model


def build_cortical_column(n_classes: int = 10) -> CorticalColumn:
    """~48 K parameters."""
    return CorticalColumn(
        patch_size=7,
        n_feed_in=64,
        n_m=160,
        grid_periods_x=(5, 7),
        grid_periods_y=(5, 7),
        n_context=16,
        n_classes=n_classes,
        lambda_decay=0.15,
        theta_w=1.0,
        kappa=0.05,
        n_saccades_train=12,
        n_saccades_eval=16,
    )


def build_hopfield(n_classes: int = 10) -> ModernHopfieldNetwork:
    """~50 K parameters."""
    return ModernHopfieldNetwork(
        input_dim=784,
        hidden_dim=48,
        n_patterns=128,
        n_classes=n_classes,
        beta=8.0,
        n_steps=3,
    )


def build_cnn(n_classes: int = 10) -> SmallCNN:
    """~47 K parameters."""
    return SmallCNN(
        in_channels=1,
        conv1_out=10,
        conv2_out=20,
        fc1_out=128,
        n_classes=n_classes,
    )


def main():
    parser = argparse.ArgumentParser(description="Cortical Column benchmark")
    parser.add_argument("--data-dir", type=str, default="MNIST", help="path to MNIST parquet dir")
    parser.add_argument("--epochs", type=int, default=15, help="training epochs per model")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--quick", action="store_true", help="fast sanity check (fewer workers)")
    parser.add_argument("--save-results", type=str, default="results.json", help="save metrics to JSON")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["cortical", "hopfield", "cnn"],
        choices=["cortical", "hopfield", "cnn"],
        help="which models to train",
    )
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Data
    num_workers = 0 if args.quick else 2
    train_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=num_workers,
    )
    print(f"Train: {len(train_loader.dataset):,} samples, Test: {len(test_loader.dataset):,} samples")

    # Build and report parameter counts
    builders = {
        "cortical": ("CorticalColumn", build_cortical_column),
        "hopfield": ("ModernHopfield", build_hopfield),
        "cnn": ("SmallCNN", build_cnn),
    }

    print(f"\n{'─' * 40}")
    print("  Model parameter counts:")
    for key in args.models:
        name, builder = builders[key]
        m = builder()
        print(f"    {name:<22} {m.count_parameters():>8,}")
    print(f"{'─' * 40}")

    # Train each model
    all_results: list[RunMetrics] = []

    for key in args.models:
        name, builder = builders[key]
        model = builder()

        # Cortical column benefits from a slightly higher LR and more grad-clip headroom
        lr = args.lr * 1.5 if key == "cortical" else args.lr
        clip = 2.0 if key == "cortical" else 1.0

        metrics = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            model_name=name,
            n_epochs=args.epochs,
            lr=lr,
            weight_decay=1e-4,
            grad_clip=clip,
            scheduler_type="cosine",
        )
        all_results.append(metrics)

    # Comparison
    print_comparison(all_results)

    # Save results
    if args.save_results:
        out = {}
        for r in all_results:
            out[r.model_name] = {
                "n_params": r.n_params,
                "best_test_acc": r.best_test_acc,
                "final_test_acc": r.test_accs[-1],
                "train_accs": r.train_accs,
                "test_accs": r.test_accs,
                "train_losses": r.train_losses,
                "test_losses": r.test_losses,
                "epoch_times": r.epoch_times,
            }
        save_path = Path(args.save_results)
        save_path.write_text(json.dumps(out, indent=2))
        print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
