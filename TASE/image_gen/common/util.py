# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch
from torchvision import datasets, transforms
from tensorboard import program
import socket
from typing import Literal, Optional
from pathlib import Path


# === Dataset ===
class ImageNetDataset(torch.utils.data.Dataset):
    """
    ImageNet dataset for training and validation.

    Args:
        root (str|Path): Path to the dataset root directory (expects subdirs train/val/test).
        resolution (int): Output square resolution.
        split (str): 'train' | 'val' | 'test'.
        normalization: 'dino' | 'vae' | None
        return_imagenet_filename (bool): If True, __getitem__ returns (img, label, filename).
        random_crop (bool): If True, random crop; else center crop.
    """

    def __init__(
        self,
        root: Path,
        resolution: int,
        split: Literal["train", "val", "test"],
        normalization: Optional[Literal["dino", "vae"]] = "vae",
        return_imagenet_filename: bool = False,
        random_crop: bool = True,
    ):
        root = Path(root)
        images = root / split
        ds = datasets.ImageFolder(images)

        # Map class ids to names
        self.class_ids = {i: c for i, c in enumerate(ds.classes)}
        if (root / "classnames.txt").exists():
            self.class_names_path = root / "classnames.txt"
        elif (root / "words.txt").exists():
            self.class_names_path = root / "words.txt"
        else:
            raise FileNotFoundError("No class names file found.")

        with open(self.class_names_path, encoding="utf-8") as f:
            self.class_names = dict(
                line.strip().split(None, 1) for line in f if line.strip()
            )
        self.class_map = {
            idx: self.class_names.get(id_, "<unknown>")
            for idx, id_ in self.class_ids.items()
        }

        self.resolution = resolution
        self.base_ds = ds
        self.return_filename = bool(return_imagenet_filename)
        self.random_crop = bool(random_crop)
        self.normalization = normalization  # remember choice for set_resolution

        # ---- Build transforms ----
        self.transform = self._make_transform(resolution, normalization, random_crop)

    def _make_transform(self, resolution: int, normalization, random_crop: bool):
        crop = (
            transforms.RandomCrop(resolution)
            if random_crop
            else transforms.CenterCrop(resolution)
        )
        if normalization is None:
            norm = []
        elif normalization == "dino":
            norm = [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        elif normalization == "vae":
            norm = [transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        else:
            raise ValueError(f"Unknown normalization type: {normalization}")

        return transforms.Compose(
            [transforms.Resize(resolution), crop, transforms.ToTensor(), *norm]
        )

    def set_resolution(self, resolution: int, random_crop: Optional[bool] = None):
        """Update output resolution (and optionally crop mode) while preserving normalization."""
        self.resolution = resolution
        if random_crop is not None:
            self.random_crop = bool(random_crop)
        self.transform = self._make_transform(
            self.resolution, self.normalization, self.random_crop
        )

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        # ImageFolder keeps (path, class) pairs in .samples
        path, _ = self.base_ds.samples[idx]
        name = Path(path).stem
        img, lbl = self.base_ds[
            idx
        ]  # applies default loader; no transforms (we apply ours)
        img = self.transform(img)

        if self.return_filename:
            # return the relative path inside split for readability
            return img, lbl, name
        return img, lbl


def launch_tensorboard(logdir, host="localhost", port=6006, max_tries=20):
    """
    Launch TensorBoard, trying ports in [port, port+max_tries).
    If the requested port is busy, it will try the next ones until successful.
    """
    for offset in range(max_tries):
        try_port = port + offset
        tb = program.TensorBoard()
        tb.configure(
            argv=[
                None,
                "--logdir",
                str(logdir),
                "--host",
                host,
                "--port",
                str(try_port),
            ]
        )
        try:
            url = tb.launch()
            if offset > 0:
                print(
                    f"[TensorBoard] Port {port} was busy. Using port {try_port} instead."
                )
            else:
                print(f"[TensorBoard] Running on port {try_port}.")
            return url
        except (OSError, socket.error) as e:
            # Typical error if port already in use
            if offset == max_tries - 1:
                print(
                    f"[TensorBoard] Failed to find a free port in range {port}–{port+max_tries-1}."
                )
                raise e
            continue
