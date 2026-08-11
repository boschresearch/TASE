# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from gsplat V 1.5.3
#   (https://github.com/nerfstudio-project/gsplat/blob/main/examples/utils.py)
# Copyright 2025 Nerfstudio Team, licensed under the Apache-2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

import random

import numpy as np
import torch

try:
    from pytorch3d.ops import knn_points

    _HAS_P3D = True
except Exception:
    from sklearn.neighbors import NearestNeighbors

    _HAS_P3D = False

from torch import Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import colormaps


class CameraOptModule(torch.nn.Module):
    """Camera pose optimization module."""

    def __init__(self, n: int):
        super().__init__()
        # Delta positions (3D) + Delta rotations (6D)
        self.embeds = torch.nn.Embedding(n, 9)
        # Identity rotation in 6D representation
        self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def zero_init(self):
        torch.nn.init.zeros_(self.embeds.weight)

    def random_init(self, std: float):
        torch.nn.init.normal_(self.embeds.weight, std=std)

    def forward(self, camtoworlds: Tensor, embed_ids: Tensor) -> Tensor:
        """Adjust camera pose based on deltas.

        Args:
            camtoworlds: (..., 4, 4)
            embed_ids: (...,)

        Returns:
            updated camtoworlds: (..., 4, 4)
        """
        assert camtoworlds.shape[:-2] == embed_ids.shape
        batch_shape = camtoworlds.shape[:-2]
        pose_deltas = self.embeds(embed_ids)  # (..., 9)
        dx, drot = pose_deltas[..., :3], pose_deltas[..., 3:]
        rot = rotation_6d_to_matrix(
            drot + self.identity.expand(*batch_shape, -1)
        )  # (..., 3, 3)
        transform = torch.eye(4, device=pose_deltas.device).repeat((*batch_shape, 1, 1))
        transform[..., :3, :3] = rot
        transform[..., :3, 3] = dx
        return torch.matmul(camtoworlds, transform)


class AppearanceOptModule(torch.nn.Module):
    """Appearance optimization module."""

    def __init__(
        self,
        n: int,
        feature_dim: int,
        embed_dim: int = 16,
        sh_degree: int = 3,
        mlp_width: int = 64,
        mlp_depth: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.sh_degree = sh_degree
        self.embeds = torch.nn.Embedding(n, embed_dim)
        layers = []
        layers.append(
            torch.nn.Linear(embed_dim + feature_dim + (sh_degree + 1) ** 2, mlp_width)
        )
        layers.append(torch.nn.ReLU(inplace=True))
        for _ in range(mlp_depth - 1):
            layers.append(torch.nn.Linear(mlp_width, mlp_width))
            layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Linear(mlp_width, 3))
        self.color_head = torch.nn.Sequential(*layers)

    def forward(
        self, features: Tensor, embed_ids: Tensor, dirs: Tensor, sh_degree: int
    ) -> Tensor:
        """Adjust appearance based on embeddings.

        Args:
            features: (N, feature_dim)
            embed_ids: (C,)
            dirs: (C, N, 3)

        Returns:
            colors: (C, N, 3)
        """
        from gsplat.cuda._torch_impl import _eval_sh_bases_fast

        C, N = dirs.shape[:2]
        # Camera embeddings
        if embed_ids is None:
            embeds = torch.zeros(C, self.embed_dim, device=features.device)
        else:
            embeds = self.embeds(embed_ids)  # [C, D2]
        embeds = embeds[:, None, :].expand(-1, N, -1)  # [C, N, D2]
        # GS features
        features = features[None, :, :].expand(C, -1, -1)  # [C, N, D1]
        # View directions
        dirs = F.normalize(dirs, dim=-1)  # [C, N, 3]
        num_bases_to_use = (sh_degree + 1) ** 2
        num_bases = (self.sh_degree + 1) ** 2
        sh_bases = torch.zeros(C, N, num_bases, device=features.device)  # [C, N, K]
        sh_bases[:, :, :num_bases_to_use] = _eval_sh_bases_fast(num_bases_to_use, dirs)
        # Get colors
        if self.embed_dim > 0:
            h = torch.cat([embeds, features, sh_bases], dim=-1)  # [C, N, D1 + D2 + K]
        else:
            h = torch.cat([features, sh_bases], dim=-1)
        colors = self.color_head(h)
        return colors


def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1]. Adapted from pytorch3d.
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def knn(x: Tensor, K: int = 4) -> Tensor:
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, indices = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x), torch.from_numpy(indices).to(x)


def compute_knn(xyz: torch.Tensor, K: int):
    """
    Returns:
        d2: (N, K) squared distances
        idx: (N, K) neighbor indices
    """
    if _HAS_P3D:
        # Use PyTorch3D implementation
        d2, idx, _ = knn_points(xyz[None, ...], xyz[None, ...], K=K)
        return d2[0], idx[0].long()
    else:
        # Fallback: scikit-learn implementation
        d, idx = knn(xyz, K=K)  # Euclidean distances
        d2 = d**2  # match knn_points semantics
        return d2, idx.long()


def rgb_to_sh(rgb: Tensor) -> Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def sh_to_rgb(sh: Tensor) -> Tensor:
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ref: https://github.com/hbb1/2d-gaussian-splatting/blob/main/utils/general_utils.py#L163
def colormap(img, cmap="jet"):
    W, H = img.shape[:2]
    dpi = 300
    fig, ax = plt.subplots(1, figsize=(H / dpi, W / dpi), dpi=dpi)
    im = ax.imshow(img, cmap=cmap)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = torch.from_numpy(data).float().permute(2, 0, 1)
    plt.close()
    return img


def apply_float_colormap(img: torch.Tensor, colormap: str = "turbo") -> torch.Tensor:
    """Convert single channel to a color img.

    Args:
        img (torch.Tensor): (..., 1) float32 single channel image.
        colormap (str): Colormap for img.

    Returns:
        (..., 3) colored img with colors in [0, 1].
    """
    img = torch.nan_to_num(img, 0)
    if colormap == "gray":
        return img.repeat(1, 1, 3)
    img_long = (img * 255).long()
    img_long_min = torch.min(img_long)
    img_long_max = torch.max(img_long)
    assert img_long_min >= 0, f"the min value is {img_long_min}"
    assert img_long_max <= 255, f"the max value is {img_long_max}"
    return torch.tensor(
        colormaps[colormap].colors,  # type: ignore
        device=img.device,
    )[img_long[..., 0]]


def apply_depth_colormap(
    depth: torch.Tensor,
    acc: torch.Tensor = None,
    near_plane: float = None,
    far_plane: float = None,
) -> torch.Tensor:
    """Converts a depth image to color for easier analysis.

    Args:
        depth (torch.Tensor): (..., 1) float32 depth.
        acc (torch.Tensor | None): (..., 1) optional accumulation mask.
        near_plane: Closest depth to consider. If None, use min image value.
        far_plane: Furthest depth to consider. If None, use max image value.

    Returns:
        (..., 3) colored depth image with colors in [0, 1].
    """
    near_plane = near_plane or float(torch.min(depth))
    far_plane = far_plane or float(torch.max(depth))
    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0.0, 1.0)
    img = apply_float_colormap(depth, colormap="turbo")
    if acc is not None:
        img = img * acc + (1.0 - acc)
    return img

def rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """
    R: (..., 3, 3) rotation matrix
    returns q: (..., 4) as (w, x, y, z), unit-normalized
    """
    assert R.shape[-2:] == (3, 3)
    # normalize the columns to ensure it's a valid rotation matrix (in case of numerical issues)
    R = R / R.norm(dim=-2, keepdim=True).clamp_min(1e-12)  # (..., 3, 3)

    if R.ndim == 2:
        R = R.unsqueeze(0)
        single = True
    else:
        single = False

    assert R.shape[-2:] == (3, 3)
    device, dtype = R.device, R.dtype
    B = R.shape[0]

    r00 = R[:, 0, 0]; r01 = R[:, 0, 1]; r02 = R[:, 0, 2]
    r10 = R[:, 1, 0]; r11 = R[:, 1, 1]; r12 = R[:, 1, 2]
    r20 = R[:, 2, 0]; r21 = R[:, 2, 1]; r22 = R[:, 2, 2]

    t = r00 + r11 + r22
    q = torch.zeros((B, 4), device=device, dtype=dtype)

    # Case 1: trace > 0
    mask0 = t > 0
    idx0 = mask0.nonzero(as_tuple=False).squeeze(-1)
    if idx0.numel() > 0:
        s = 2.0 * torch.sqrt(t[idx0] + 1.0)
        q[idx0, 0] = 0.25 * s
        q[idx0, 1] = (r21[idx0] - r12[idx0]) / s
        q[idx0, 2] = (r02[idx0] - r20[idx0]) / s
        q[idx0, 3] = (r10[idx0] - r01[idx0]) / s

    # Case 2: trace <= 0 -> pick largest diagonal
    idxN = (~mask0).nonzero(as_tuple=False).squeeze(-1)
    if idxN.numel() > 0:
        r00N, r11N, r22N = r00[idxN], r11[idxN], r22[idxN]
        diag = torch.stack([r00N, r11N, r22N], dim=1)   # (M,3)
        k = diag.argmax(dim=1)                          # (M,)

        # k == 0
        m = k == 0
        idx = idxN[m]
        if idx.numel() > 0:
            s = 2.0 * torch.sqrt(1.0 + r00[idx] - r11[idx] - r22[idx])
            q[idx, 0] = (r21[idx] - r12[idx]) / s
            q[idx, 1] = 0.25 * s
            q[idx, 2] = (r01[idx] + r10[idx]) / s
            q[idx, 3] = (r02[idx] + r20[idx]) / s

        # k == 1
        m = k == 1
        idx = idxN[m]
        if idx.numel() > 0:
            s = 2.0 * torch.sqrt(1.0 - r00[idx] + r11[idx] - r22[idx])
            q[idx, 0] = (r02[idx] - r20[idx]) / s
            q[idx, 1] = (r01[idx] + r10[idx]) / s
            q[idx, 2] = 0.25 * s
            q[idx, 3] = (r12[idx] + r21[idx]) / s

        # k == 2
        m = k == 2
        idx = idxN[m]
        if idx.numel() > 0:
            s = 2.0 * torch.sqrt(1.0 - r00[idx] - r11[idx] + r22[idx])
            q[idx, 0] = (r10[idx] - r01[idx]) / s
            q[idx, 1] = (r02[idx] + r20[idx]) / s
            q[idx, 2] = (r12[idx] + r21[idx]) / s
            q[idx, 3] = 0.25 * s

    # Normalize (avoid divide-by-zero)
    q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-12)

    return q[0] if single else q