"""
data_utils.py — SVHN dataset loading, filtering (digits 1 & 7), and DataLoader creation.
"""

import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

import config


def _set_seeds(seed: int = config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_split(split: str, data_dir) -> datasets.SVHN:
    """Download (if needed) and return an SVHN split dataset (no transform; normalization
    is applied vectorised inside _filter_and_flatten)."""
    return datasets.SVHN(root=str(data_dir), split=split, download=True, transform=None)


def _filter_and_flatten(dataset: datasets.SVHN, max_samples: int = None):
    """
    Keep only digits 1 and 7, remap labels, flatten and normalise images.

    Vectorised: accesses dataset.data [N,3,32,32] uint8 directly instead of
    iterating sample-by-sample through __getitem__ (which was the original bottleneck).

    Returns
    -------
    x : float32 tensor  [N, 3072]
    y : float32 tensor  [N]       values in {0.0, 1.0}
    """
    keep    = set(config.KEEP_DIGITS)
    labels  = np.array(dataset.labels)
    indices = np.where(np.isin(labels, list(keep)))[0]

    if max_samples is not None and max_samples < len(indices):
        rng     = np.random.default_rng(config.SEED)
        indices = rng.choice(indices, size=max_samples, replace=False)

    # Vectorised extraction and normalisation (equivalent to ToTensor + Normalize)
    imgs = torch.from_numpy(dataset.data[indices]).float() / 255.0   # [N, 3, 32, 32]
    mean = torch.tensor(config.SVHN_MEAN).view(1, 3, 1, 1)
    std  = torch.tensor(config.SVHN_STD).view(1, 3, 1, 1)
    x    = ((imgs - mean) / std).reshape(len(indices), -1)            # [N, 3072]

    y = torch.tensor(
        [float(config.LABEL_MAP[int(l)]) for l in labels[indices]],
        dtype=torch.float32,
    )
    return x, y


def get_tensors(
    data_dir=None,
    smoke_test: bool = False,
    max_train: int = None,
    max_test:  int = None,
):
    """
    Return raw tensors (x_train, y_train, x_test, y_test) without DataLoaders.
    Useful for trace extraction.
    """
    _set_seeds()

    data_dir = data_dir or config.DATA_DIR
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_ds = _load_split("train", data_dir)
    test_ds  = _load_split("test",  data_dir)

    if smoke_test:
        max_train = max_train or 500
        max_test  = max_test  or 200

    x_tr, y_tr = _filter_and_flatten(train_ds, max_samples=max_train)
    x_te, y_te = _filter_and_flatten(test_ds,  max_samples=max_test)

    print(f"[data_utils] train samples : {len(x_tr)}")
    print(f"[data_utils] test  samples : {len(x_te)}")
    print(f"[data_utils] class balance  train 0/1: "
          f"{int((y_tr == 0).sum())}/{int((y_tr == 1).sum())}")
    print(f"[data_utils] class balance  test  0/1: "
          f"{int((y_te == 0).sum())}/{int((y_te == 1).sum())}")

    return x_tr, y_tr, x_te, y_te


def get_loaders(
    batch_size: int = config.BATCH_SIZE,
    data_dir=None,
    smoke_test: bool = False,
    max_train: int = None,
    max_test:  int = None,
):
    """
    Return (train_loader, test_loader) for SVHN binary classification (1 vs 7).

    In smoke-test mode a tiny subset is used so the whole pipeline can be
    verified quickly without downloading large amounts of data.
    """
    x_tr, y_tr, x_te, y_te = get_tensors(
        data_dir=data_dir,
        smoke_test=smoke_test,
        max_train=max_train,
        max_test=max_test,
    )

    train_loader = DataLoader(
        TensorDataset(x_tr, y_tr),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    test_loader = DataLoader(
        TensorDataset(x_te, y_te),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = get_loaders(smoke_test=True)
    xb, yb = next(iter(train_loader))
    print(f"batch x: {xb.shape}, y: {yb.shape}")
    print(f"x min/max: {xb.min():.3f} / {xb.max():.3f}")
    print(f"y unique: {yb.unique()}")
