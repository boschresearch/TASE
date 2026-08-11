# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from pathlib import Path
from typing import Literal, Optional, Dict, Any, List

import yaml
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


def _build_norm(normalization: Optional[Literal["none", "vae", "dino"]]):
    if normalization is None or normalization == "none":
        return None
    if normalization == "vae":
        return transforms.Normalize(
            [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        )  # maps [0,1] -> [-1,1]
    if normalization == "dino":
        return transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    raise ValueError(f"Unknown normalization: {normalization}")


class SplatCtrlDataset(Dataset):
    """
    Dataset for the generated SplatCtrl samples (clean image, artifact image, feature map, meta).

    Expects a directory with:
      - images: input_XXXXXX.jpg, novel_XXXXXX.jpg
      - features: feature_XXXXXX.npz (with array under key 'feature_map')
      - meta: meta.yml (YAML with entries 'sample_XXXXXX': { label, paths, perturbations, ... })

    Each __getitem__ returns a dict:
      {
        'clean': FloatTensor [3,H,W]   (normalized as requested),
        'artifact': FloatTensor [3,H,W],
        'feature': FloatTensor [C,Hf,Wf],   # if load_features=True
        'label': int,
        'meta': dict,                   # original YAML entry (JSON-safe)
        'paths': dict,                  # resolved absolute paths
        'id': 'sample_XXXXXX'
      }

    Notes:
      - Images are loaded via PIL, converted to RGB and optionally normalized.
      - Default transform maps to [0,1] float; add normalization if specified.
      - Feature .npz is loaded lazily; set mmap=True to lower RAM spikes. Features are loaded in HWC and returned in CHW
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        meta_file: str = "meta.yml",
        normalization: Optional[Literal["none", "vae", "dino"]] = "vae",
        resize: Optional[int] = None,  # if set, resize images to this square size
        load_features: bool = True,
        mmap_features: bool = True,  # np.load(..., mmap_mode="r")
        dtype: torch.dtype = torch.float32,
        return_paths: bool = True,
    ):
        self.root = Path(root) / split
        self.meta_path = self.root / meta_file
        if not self.meta_path.exists():
            raise FileNotFoundError(f"meta file not found: {self.meta_path}")

        # meta.yml may contain multiple YAML documents appended; load all and merge
        entries: Dict[str, Any] = {}
        with open(self.meta_path, "r") as f:
            for doc in yaml.safe_load_all(f):
                if isinstance(doc, dict):
                    entries.update(doc)

        # load the info json for additional info if present
        info_path = self.root / "config.json"
        if info_path.exists():
            import json

            with open(info_path, "r") as f:
                self.info = json.load(f)
        else:
            self.info = {}

        # Load the classmap if present
        classmap_path = self.root / "classmap.json"
        if classmap_path.exists():
            import json

            with open(classmap_path, "r") as f:
                class_map = json.load(f)
            self.class_map = {
                int(k): v for k, v in class_map.items()
            }  # {1: 'a', 2: 'b'}
        else:
            print(
                f"[warn] No classmap.json found in {self.root}; class_map will be empty."
            )
            self.class_map = {}

        # Flatten into indexable list (stable sort by sample id)
        self.items: List[tuple[str, Dict[str, Any]]] = sorted(
            entries.items(), key=lambda kv: kv[0]
        )

        self.load_features = bool(load_features)
        self.mmap_features = bool(mmap_features)
        self.dtype = dtype
        self.return_paths = bool(return_paths)

        self.norm = None
        self.set_resolution(resolution=resize, normalization=normalization)

    def set_resolution(
        self,
        resolution: Optional[int] = None,
        normalization: Optional[Literal["none", "vae", "dino"]] = None,
    ):
        # Build image transform
        t: List[Any] = []
        native_res = self.info.get("output_res", None)
        if resolution is not None and native_res is not None:
            if resolution > native_res:
                print(
                    f"Warning: requested resolution {resolution} is larger than native {native_res}"
                )
            if resolution == native_res:
                resolution = None  # no resize needed
        if resolution is not None:
            t.append(
                transforms.Resize(
                    (resolution, resolution),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True,
                )
            )
        t += [transforms.ToTensor()]  # [0,1]
        norm = _build_norm(normalization)
        if norm is not None:
            self.norm = norm
        if self.norm is not None:
            t.append(self.norm)
        self.img_tf = transforms.Compose(t)

    def __len__(self) -> int:
        return len(self.items)

    def _resolve(self, p: str | Path) -> Path:
        """Resolve sample path: absolute stays; relative is under self.root."""
        p = Path(p)
        return p if p.is_absolute() else (self.root / p)

    def _load_img(self, path: Path) -> torch.Tensor:
        with Image.open(path) as im:
            im = im.convert("RGB")
            return self.img_tf(im).to(self.dtype)

    def _load_feat(self, path: Path) -> torch.Tensor:
        # npz with key 'feature_map'
        mode = "r" if self.mmap_features else None
        with np.load(path, mmap_mode=mode, allow_pickle=False) as data:
            arr = data["feature_map"]
            tensor = torch.from_numpy(arr).to(self.dtype)
            tensor = tensor.permute(2, 0, 1)  # HWC -> CHW
        return tensor

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_id, meta = self.items[idx]
        # Robustly read paths from meta; fall back keys supported
        in_key = (
            meta.get("input_img_path")
            or meta.get("input_path")
            or meta.get("in_img_path")
        )
        out_key = (
            meta.get("novel_img_path")
            or meta.get("output_img_path")
            or meta.get("out_img_path")
        )
        feat_key = meta.get("feature_map_path") or meta.get("feature_path")

        if in_key is None or out_key is None:
            raise KeyError(f"Missing image paths in meta for {sample_id}")

        p_in = self._resolve(in_key)
        p_out = self._resolve(out_key)
        p_feat = (
            self._resolve(feat_key)
            if (self.load_features and feat_key is not None)
            else None
        )

        clean = self._load_img(p_in)
        artifact = self._load_img(p_out)
        feat = self._load_feat(p_feat) if p_feat is not None else None

        label = int(meta.get("label", -1))

        example = {
            "clean": clean,
            "artifact": artifact,
            "label": label,
            "meta": meta,  # includes perturbations, ctrl_dims, etc.
            "id": sample_id,
        }
        if feat is not None:
            example["feature"] = feat
        if self.return_paths:
            example["paths"] = {
                "clean": str(p_in),
                "artifact": str(p_out),
                "feature": str(p_feat) if p_feat is not None else None,
            }
        return example


# Optional: a simple collate_fn that keeps dicts and supports missing features
def splat_ctrl_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collates a list of items from GeneratedSplatDataset into batch tensors where possible,
    and keeps variable-length/meta fields in a list.
    """
    out: Dict[str, Any] = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]
        if torch.is_tensor(vals[0]):
            try:
                out[k] = torch.stack(vals, dim=0)
            except Exception:
                out[k] = vals
        else:
            out[k] = vals
    return out
