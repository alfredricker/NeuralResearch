from pathlib import Path

parent_dir = Path(__file__).parent.parent
import pandas as pd
import numpy as np
from typing import Tuple
from PIL import Image
import io

def decode_png_bytes(png_bytes: bytes) -> np.ndarray:
    pil = Image.open(io.BytesIO(png_bytes)).convert("L")
    arr = np.array(pil, dtype=np.float32) / 255.0
    return arr

def load_mnist_dataset() -> Tuple[np.ndarray, np.ndarray]:
    data_path = Path(parent_dir, "data", "MNIST", "train.parquet")
    df = pd.read_parquet(data_path)
    print(df.head(10))
    print(df.columns)
    images = []
    for img_dict in df["image"]:
        png_bytes = img_dict["bytes"] if isinstance(img_dict, dict) else img_dict
        images.append(decode_png_bytes(png_bytes))
    return np.array(images), df["label"].values

if __name__ == "__main__":
    images, labels = load_mnist_dataset()
    print(images[0].shape)
    print(labels[0])