"""
Training and evaluation utilities shared by all three models.
"""

import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class RunMetrics:
    """Stores per-epoch metrics for a single model run."""
    model_name: str
    n_params: int
    train_losses: list[float] = field(default_factory=list)
    train_accs: list[float] = field(default_factory=list)
    test_losses: list[float] = field(default_factory=list)
    test_accs: list[float] = field(default_factory=list)
    epoch_times: list[float] = field(default_factory=list)

    @property
    def best_test_acc(self) -> float:
        return max(self.test_accs) if self.test_accs else 0.0

    def summary(self) -> str:
        lines = [
            f"\n{'=' * 55}",
            f"  {self.model_name}",
            f"  Parameters : {self.n_params:,}",
            f"  Best test  : {self.best_test_acc:.2%}",
            f"  Final test : {self.test_accs[-1]:.2%}" if self.test_accs else "",
            f"  Total time : {sum(self.epoch_times):.1f}s",
            f"{'=' * 55}",
        ]
        return "\n".join(lines)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
) -> tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = nn.functional.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = nn.functional.cross_entropy(logits, labels)

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    model_name: str,
    n_epochs: int = 15,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    scheduler_type: str = "cosine",
) -> RunMetrics:
    """Full training loop with logging."""
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metrics = RunMetrics(model_name=model_name, n_params=n_params)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, n_epochs // 3), gamma=0.3)
    else:
        scheduler = None

    print(f"\n{'─' * 55}")
    print(f"  Training: {model_name}  ({n_params:,} parameters)")
    print(f"{'─' * 55}")

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, grad_clip)
        test_loss, test_acc = evaluate(model, test_loader, device)

        elapsed = time.time() - t0

        metrics.train_losses.append(train_loss)
        metrics.train_accs.append(train_acc)
        metrics.test_losses.append(test_loss)
        metrics.test_accs.append(test_acc)
        metrics.epoch_times.append(elapsed)

        if scheduler is not None:
            scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch:3d}/{n_epochs}  "
            f"train {train_loss:.4f} / {train_acc:.2%}  "
            f"test {test_loss:.4f} / {test_acc:.2%}  "
            f"lr={lr_now:.1e}  ({elapsed:.1f}s)"
        )

    print(metrics.summary())
    return metrics


def print_comparison(results: list[RunMetrics]):
    """Print a side-by-side comparison table."""
    print(f"\n{'━' * 62}")
    print(f"  {'Model':<25} {'Params':>10} {'Best Test':>10} {'Final Test':>10}")
    print(f"{'━' * 62}")
    for r in results:
        print(
            f"  {r.model_name:<25} {r.n_params:>10,} "
            f"{r.best_test_acc:>10.2%} {r.test_accs[-1]:>10.2%}"
        )
    print(f"{'━' * 62}")
