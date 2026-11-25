"""Data loading for image forgery segmentation.

Based on ImgDataset from the original notebook (legacy/assets.py, Hyeonwook Kim), with fixes:
- each forged image is paired with its mask BY FILE NAME (the original paired them by
  os.listdir order, which is arbitrary and can mismatch images and masks)
- train/val/test splits contain both authentic and forged images
- images that share a file name (e.g. an authentic image and its forged version) stay in
  the same split, so no image leaks between training and testing
- masks are resized with nearest-neighbour, never blurred
- no augmentation on validation/test data
- optional Error Level Analysis (ELA) channels
"""
import io
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms.v2 as T
from PIL import Image, ImageChops, ImageOps
from torch.utils.data import Dataset

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _images(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXTS)


def make_splits(auth_dir, forged_dir, mask_dir, val_frac=0.1, test_frac=0.2, seed=42):
    """Returns {"train": [...], "val": [...], "test": [...]}, each a list of
    (image_path, mask_path) with mask_path=None for authentic images."""
    masks = {p.stem: p for p in Path(mask_dir).iterdir() if p.suffix == ".npy"}

    groups = {}            # file stem -> list of (image, mask)
    missing = 0
    for p in _images(auth_dir):
        groups.setdefault(p.stem, []).append((p, None))
    for p in _images(forged_dir):
        m = masks.get(p.stem)
        if m is None:
            missing += 1
            continue
        groups.setdefault(p.stem, []).append((p, m))
    if missing:
        print(f"Warning: {missing} forged images had no mask with the same name and were skipped")

    stems = sorted(groups)
    random.Random(seed).shuffle(stems)
    n_test = round(len(stems) * test_frac)
    n_val = round(len(stems) * val_frac)
    parts = {"test": stems[:n_test],
             "val": stems[n_test:n_test + n_val],
             "train": stems[n_test + n_val:]}
    splits = {k: [item for s in v for item in groups[s]] for k, v in parts.items()}

    for k, items in splits.items():
        n_forged = sum(m is not None for _, m in items)
        print(f"{k:5s}: {len(items):5d} images  ({len(items) - n_forged} authentic, {n_forged} forged)")
    return splits


def balance(items, seed=42):
    """Oversample the smaller class (authentic or forged) so training sees both equally."""
    auth = [it for it in items if it[1] is None]
    forged = [it for it in items if it[1] is not None]
    rng = random.Random(seed)
    small, big = (auth, forged) if len(auth) < len(forged) else (forged, auth)
    if small:
        small = small + [rng.choice(small) for _ in range(len(big) - len(small))]
    return small + big


def ela_image(img, quality=90, scale=15.0):
    """Error Level Analysis: re-save the image as JPEG and amplify the difference.
    Pasted or edited regions often re-compress differently from the rest of the image."""
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    resaved = Image.open(buf).convert("RGB")
    diff = np.asarray(ImageChops.difference(img, resaved), dtype=np.float32) * scale
    return Image.fromarray(np.clip(diff, 0, 255).astype(np.uint8))


class ForgeryDataset(Dataset):
    """Returns (image [C,S,S], mask [1,S,S] in {0,1}, is_forged [scalar]).
    C = 3 (RGB), or 6 (RGB + ELA) when use_ela=True."""

    def __init__(self, items, size=448, augment=False, use_ela=False):
        self.items = items
        self.size = size
        self.augment = augment
        self.use_ela = use_ela
        self.colorjitter = T.ColorJitter(0.2, 0.2, 0.2, 0.1)
        self.jpeg = T.JPEG(quality=(5, 100))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, mask_path = self.items[idx]
        x = Image.open(img_path).convert("RGB")

        if mask_path is not None:
            m = np.load(mask_path)
            if m.ndim == 3:                       # several forged regions -> one mask
                m = np.any(m, axis=0)
            m = (m > 0).astype(np.uint8) * 255
        else:
            m = np.zeros((x.height, x.width), dtype=np.uint8)
        y = Image.fromarray(m, mode="L")
        if y.size != x.size:
            y = y.resize(x.size, Image.Resampling.NEAREST)

        e = ela_image(x) if self.use_ela else None       # from the original file, before augmentation

        x = TF.to_tensor(ImageOps.pad(x, (self.size, self.size), color=(0, 0, 0)))
        y = ImageOps.pad(y, (self.size, self.size), method=Image.Resampling.NEAREST, color=0)
        y = (TF.to_tensor(y) > 0.5).float()

        if self.augment:                                 # colour/JPEG changes: RGB only
            x = (self.jpeg((x * 255).to(torch.uint8)) / 255).float()
            x = self.colorjitter(x)

        if e is not None:
            e = TF.to_tensor(ImageOps.pad(e, (self.size, self.size), color=(0, 0, 0)))
            x = torch.cat([x, e])

        if self.augment:                                 # flips: image, ELA and mask together
            if torch.rand(1) > 0.5:
                x, y = TF.hflip(x), TF.hflip(y)
            if torch.rand(1) > 0.5:
                x, y = TF.vflip(x), TF.vflip(y)

        return x, y, torch.tensor(float(mask_path is not None))
