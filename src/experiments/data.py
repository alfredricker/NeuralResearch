"""
MNIST data loading from HuggingFace-format parquet files.
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class MNISTParquetDataset(Dataset):
    """MNIST dataset loaded from a parquet file with 'image' (PNG bytes) and 'label' columns."""

    def __init__(self, parquet_path: str, normalize: bool = True):
        df = pd.read_parquet(parquet_path)
        self.labels = torch.tensor(df["label"].values, dtype=torch.long)

        # Decode PNG bytes → float32 tensors
        images = []
        for img_dict in df["image"]:
            png_bytes = img_dict["bytes"] if isinstance(img_dict, dict) else img_dict
            pil = Image.open(io.BytesIO(png_bytes)).convert("L")
            arr = np.array(pil, dtype=np.float32) / 255.0
            images.append(arr)

        self.images = torch.tensor(np.stack(images)).unsqueeze(1)  # (N, 1, 28, 28)

        if normalize:
            # Standard MNIST normalization
            self.images = (self.images - 0.1307) / 0.3081

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def get_dataloaders(
    data_dir: str = "MNIST",
    batch_size: int = 128,
    num_workers: int = 2,
):
    """Create train and test dataloaders from parquet files."""
    data_dir = Path(data_dir)

    train_ds = MNISTParquetDataset(data_dir / "train.parquet")
    test_ds = MNISTParquetDataset(data_dir / "test.parquet")

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    return train_loader, test_loader



def make_synthetic_data(n_train: int = 5000, n_test: int = 1000, n_classes: int = 10,
                        img_size: int = 28, seed: int = 42):
    """
    Generate synthetic digit-like patterns for testing.
    Each class has a distinct random template; instances are noisy versions.
    """
    rng = np.random.RandomState(seed)

    # Create class templates: sparse random patterns
    templates = []
    for c in range(n_classes):
        t = np.zeros((img_size, img_size))
        # Each class gets ~40 random active pixels in a class-specific region
        row_offset = (c // 5) * 10
        col_offset = (c % 5) * 5
        for _ in range(40):
            r = rng.randint(max(0, row_offset), min(img_size, row_offset + 14))
            cc = rng.randint(max(0, col_offset), min(img_size, col_offset + 12))
            t[r, cc] = 1.0
        # Add some structure with small gaussian blobs
        for _ in range(3):
            cr = rng.randint(row_offset, min(img_size - 1, row_offset + 13))
            ccc = rng.randint(col_offset, min(img_size - 1, col_offset + 11))
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    rr, rc = cr + dr, ccc + dc
                    if 0 <= rr < img_size and 0 <= rc < img_size:
                        t[rr, rc] += 0.5 * np.exp(-0.5 * (dr**2 + dc**2))
        templates.append(t.flatten())

    templates = np.array(templates)
    # Normalize templates
    for i in range(n_classes):
        mx = templates[i].max()
        if mx > 0:
            templates[i] /= mx

    def generate_set(n):
        images = []
        labels = []
        for _ in range(n):
            c = rng.randint(0, n_classes)
            noise = rng.randn(img_size * img_size) * 0.3
            img = templates[c] + noise
            img = np.clip(img, 0.0, 1.0)
            images.append(img)
            labels.append(c)
        return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int64)

    train_imgs, train_labels = generate_set(n_train)
    test_imgs, test_labels = generate_set(n_test)
    return train_imgs, train_labels, test_imgs, test_labels


# ─── MNIST parquet loading (for use with real data) ─────────────────────────

def load_mnist_parquet(train_path: str, test_path: str):
    """
    Load MNIST from HuggingFace-format parquet files.
    Requires pyarrow or fastparquet as pandas backend.
    Returns (train_images, train_labels, test_images, test_labels)
    where images are (N, 784) float32 and labels are (N,) int64.
    """
    import io
    from PIL import Image
    import pandas as pd

    def load_split(path):
        df = pd.read_parquet(path)
        labels = df["label"].values.astype(np.int64)
        images = []
        for img_dict in df["image"]:
            png_bytes = img_dict["bytes"] if isinstance(img_dict, dict) else img_dict
            pil = Image.open(io.BytesIO(png_bytes)).convert("L")
            arr = np.array(pil, dtype=np.float32) / 255.0
            images.append(arr.flatten())
        return np.array(images), labels

    train_imgs, train_labels = load_split(train_path)
    test_imgs, test_labels = load_split(test_path)
    return train_imgs, train_labels, test_imgs, test_labels



if __name__ == "__main__":
    train_imgs, train_labels, test_imgs, test_labels = load_mnist_parquet(
        "MNIST/train.parquet", "MNIST/test.parquet"
    )
    print(f"Loaded real MNIST: train={len(train_labels)}, test={len(test_labels)}")
    print(f"Image shape: {train_imgs[0].shape}, classes: {np.unique(train_labels)}")
    print(train_imgs[0][:200])