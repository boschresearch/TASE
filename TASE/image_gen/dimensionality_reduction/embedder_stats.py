# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import json
from scipy.stats import chi2
import math
from TASE.splatting.util.data_util import resolve_path
#from TASE.image_gen.dimensionality_reduction.embedding_model import EmbeddingModel

# -------------------- Online Gaussian estimator (full or diagonal) --------------------
def _chi2_ppf_approx(conf: float, df: float) -> float:
    """
    Wilson–Hilferty approximation for chi-square quantiles.
    Accurate enough for thresholds in a margin loss.
    """
    # inverse standard normal
    z = math.sqrt(2) * torch.erfinv(torch.tensor(2 * conf - 1.0)).item()
    a = 1 - 2 / (9 * df)
    b = math.sqrt(2 / (9 * df))
    return float(df * (a + b * z) ** 3)


class OnlineGaussian:
    """
    Incremental estimator for mean and covariance with O(d^2) memory (full)
    or O(d) memory (diagonal). Uses Welford/Chan updates for stability.
    """

    def __init__(self, d, diagonal=False, dtype=torch.float32, device="cpu"):
        self.diagonal = diagonal
        self.n = 0
        self.mean = torch.zeros(d, dtype=dtype, device=device)
        if diagonal:
            self.M2 = torch.zeros(
                d, dtype=dtype, device=device
            )  # accumulates sum of squared diffs (diag only)
        else:
            self.M2 = torch.zeros(
                d, d, dtype=dtype, device=device
            )  # accumulates sum of outer products of diffs

    @torch.no_grad()
    def update(self, Z):
        """
        Z: (B, d) 2D tensor of feature vectors in float32 on same device.
        """
        if Z.numel() == 0:
            return

        # batch mean and centered
        n_b = Z.shape[0]
        mean_b = Z.mean(dim=0)  # (d,)
        delta = mean_b - self.mean  # (d,)

        # update count & mean
        n_old = self.n
        self.n += n_b
        self.mean = self.mean + delta * (n_b / max(self.n, 1))

        # centered batch
        Zc = Z - mean_b  # (B, d)

        if self.diagonal:
            # batch "M2" for diagonal only
            M2_b = (Zc.pow(2)).sum(dim=0)  # (d,)
            # between-mean term
            between = (delta.pow(2)) * (n_b * n_old / max(self.n, 1))
            self.M2 = self.M2 + M2_b + between
        else:
            # batch covariance scatter
            # (B, d)^T @ (B, d) = (d, d)
            M2_b = Zc.T @ Zc  # (d, d)
            # between-mean outer term
            between = torch.outer(delta, delta) * (n_b * n_old / max(self.n, 1))
            self.M2 = self.M2 + M2_b + between

    @torch.no_grad()
    def finalize(self, eps=1e-5, shrinkage=0.0):
        """
        Returns:
          mu
          Sigma_inv (diag inverse variances)  OR  None
          chol (Cholesky of full covariance)  OR  None
          diagonal flag
        """
        mu = self.mean.clone()
        if self.n <= 1:
            raise RuntimeError("Not enough samples to estimate covariance.")

        if self.diagonal:
            var = self.M2 / (self.n - 1)  # (d,)
            # shrinkage towards isotropic variance
            if shrinkage > 0.0:
                var = (1 - shrinkage) * var + shrinkage * var.mean()
            var = var + eps
            Sigma_inv = 1.0 / var
            chol = None
            return mu, Sigma_inv, chol, True
        else:
            Sigma = self.M2 / (self.n - 1)  # (d, d)
            if shrinkage > 0.0:
                tr = torch.trace(Sigma) / Sigma.shape[0]
                Sigma = (1 - shrinkage) * Sigma + shrinkage * tr * torch.eye(
                    Sigma.shape[0], device=Sigma.device
                )
            # jitter for numerical stability
            Sigma = Sigma + eps * torch.eye(Sigma.shape[0], device=Sigma.device)
            chol = torch.linalg.cholesky(Sigma)  # lower-triangular L, Sigma = L L^T
            return mu, None, chol, False


# -------------------- Loss using the saved stats --------------------
class InDistMarginLoss(nn.Module):
    """
    Zero loss inside a chi-square confidence ellipsoid; increasing loss outside.
    Supports diagonal or full covariance (via Cholesky factor).
    Can also be initialized with mean + std_dev (vector or scalar).
    """

    def __init__(
        self,
        mu,  # (d,) or scalar
        n_dims=None,  # optional, for scalar mu
        Sigma_inv_diag=None,  # (d,) if diagonal=True  (precision diag)
        chol=None,  # (d,d) Cholesky of covariance Σ (lower)  OR precision_chol if use_precision=True
        std_dev=None,  # (d,) or scalar, implies diagonal=True
        diagonal=False,
        confidence=0.99,  # confidence level for zero-loss region
        tau=None,  # optionally pass explicit threshold in squared Mahalanobis units
        mode="softplus",  # "hinge" or "softplus"
        temperature=1.0,  # softness for softplus margin
        power=1.0,  # exponent on margin (1=linear, 2=quadratic, etc.)
        reduction="mean",
        use_precision=False,  # if True, 'chol' is Cholesky of Σ^{-1} (precision factor L), faster & stable
    ):
        super().__init__()
        self.diagonal = diagonal or (std_dev is not None)
        self.reduction = reduction
        self.mode = mode
        self.temperature = float(temperature)
        self.power = float(power)
        self.use_precision = bool(use_precision)

        mu = torch.as_tensor(mu, dtype=torch.float32).detach()
        self.register_buffer("mu", mu)
        d = mu.numel()

        if n_dims is not None:
            if d != 1:
                raise ValueError("n_dims can only be used with scalar mu.")
            d = int(n_dims)
            self.mu = self.mu.expand(d)

        if self.diagonal:
            if Sigma_inv_diag is not None:
                inv_diag = torch.as_tensor(Sigma_inv_diag, dtype=torch.float32).detach()
            elif std_dev is not None:
                std_dev = torch.as_tensor(std_dev, dtype=torch.float32).detach()
                if std_dev.numel() == 1 and d > 1:
                    std_dev = std_dev.expand(d)
                if std_dev.numel() != d:
                    raise ValueError("mu and std_dev must match shape or be scalars.")
                inv_diag = 1.0 / (std_dev**2)  # precision diag
            else:
                raise ValueError("Diagonal mode requires Sigma_inv_diag or std_dev.")
            self.register_buffer("Sigma_inv_diag", inv_diag)
            self.chol = None
        else:
            if chol is None:
                raise ValueError("Full mode requires chol.")
            chol = torch.as_tensor(chol, dtype=torch.float32).detach()
            self.Sigma_inv_diag = None
            self.register_buffer(
                "chol", chol
            )  # Σ^{1/2} (lower) OR precision factor if use_precision=True

        # Base tau for full d (we will adjust for masks on-the-fly)
        if tau is None:
            tau_val = _chi2_ppf_approx(confidence, df=d)
        else:
            tau_val = float(tau)
        self.register_buffer("tau", torch.tensor(tau_val, dtype=torch.float32))
        self.d = d
        self.confidence = float(confidence)

        """
        print(f"Initialized InDistMarginLoss: mean={self.mu}, d={d}, diagonal={self.diagonal}, use_precision={self.use_precision}, tau={self.tau.item():.3f}")
        if self.diagonal:
            print(f"  Sigma_inv_diag: {self.Sigma_inv_diag}")
        """

    def _mahalanobis2_from_residual(self, r: torch.Tensor) -> torch.Tensor:
        """
        r: residuals (B,d) = z - mu AFTER any feat_mask is applied.
        Returns dm2 per sample.
        """
        if self.diagonal:
            dm2 = (r * self.Sigma_inv_diag) * r
            return dm2.sum(dim=-1)
        else:
            if self.use_precision:
                # chol is precision factor L where Σ^{-1} = L L^T -> ||L r^T||^2
                y = torch.matmul(self.chol, r.T)  # (d,B)
                return (y * y).sum(dim=0)  # (B,)
            else:
                # chol is Cholesky of Σ (lower); solve Σ y = r^T -> y = Σ^{-1} r^T
                y = torch.cholesky_solve(r.T, self.chol, upper=False)  # (d,B)
                return (r.T * y).sum(dim=0)

    def _effective_tau(self, feat_mask: torch.Tensor | None) -> torch.Tensor:
        """
        Adjust chi-square threshold if a feature mask is used.
        For hard masks use DoF = #active dims.
        For soft masks, use DoF ≈ sum(mask^2) as a reasonable proxy.
        """
        if feat_mask is None:
            return self.tau
        fm = torch.as_tensor(feat_mask, dtype=torch.float32, device=self.mu.device)
        if fm.numel() != self.d:
            raise ValueError("feat_mask must have shape (d,)")
        # Effective DoF approximation
        dof = (
            float((fm > 0).sum().item())
            if (fm.dtype == torch.bool)
            else float((fm * fm).sum().item())
        )
        dof = max(dof, 1.0)
        tau_eff = _chi2_ppf_approx(self.confidence, df=dof)
        return torch.tensor(tau_eff, dtype=torch.float32, device=self.mu.device)

    def forward(
        self,
        z,  # (B, d_in) with d_in <= D_full
        sample_mask=None,  # optional (B,) boolean
        feat_mask=None,  # optional (d_in,) boolean/float in [0,1]
        sample_weights=None,  # optional (B,) float
    ):
        z = z.float()
        B, d_in = z.shape
        D_full = self.mu.numel()

        if d_in > D_full:
            raise ValueError(
                f"Input has more dims ({d_in}) than distribution ({D_full})."
            )

        # --- Build full-length residual r_full: pad missing dims with zeros (truncate coupling safely) ---
        # r_full = [ (z - mu) for first d_in dims ; 0 for remaining D_full - d_in dims ]
        if d_in < D_full:
            # compute residual only on available dims
            r_first = z - self.mu[:d_in]  # (B, d_in)
            # pad to full length with zeros (ignored dims)
            r = torch.zeros((B, D_full), dtype=z.dtype, device=z.device)
            r[:, :d_in] = r_first
        else:
            r = z - self.mu  # (B, D_full)

        # --- Feature-dimension mask (length D_full): given mask mapped to first d_in, zeros for the rest ---
        if feat_mask is not None:
            fm_in = torch.as_tensor(feat_mask, dtype=r.dtype, device=r.device)
            if fm_in.dim() != 1 or fm_in.numel() != d_in:
                raise ValueError("feat_mask must be (d_in,).")
            if d_in < D_full:
                fm = torch.zeros((D_full,), dtype=r.dtype, device=r.device)
                fm[:d_in] = fm_in
            else:
                fm = fm_in
        else:
            # default: keep first d_in, drop the truncated tail
            fm = torch.zeros((D_full,), dtype=r.dtype, device=r.device)
            fm[:d_in] = 1.0

        # apply mask (ignore/attenuate specific dims; truncated tail is 0 already)
        r = r * fm  # (B, D_full)

        # --- Core loss terms ---
        dm2 = self._mahalanobis2_from_residual(
            r
        )  # expects (B, D_full); zero-padded tail contributes nothing
        tau_eff = self._effective_tau(fm)  # make tau depend on active dims via fm

        margin = dm2 - tau_eff

        if self.mode == "hinge":
            loss_vec = torch.relu(margin)
        elif self.mode == "softplus":
            loss_vec = F.softplus(margin / self.temperature) * self.temperature
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        if self.power != 1.0:
            loss_vec = loss_vec.pow(self.power)

        # --- Sample mask / weights ---
        if sample_mask is not None:
            sm = torch.as_tensor(
                sample_mask, dtype=loss_vec.dtype, device=loss_vec.device
            )
            if sm.shape != loss_vec.shape:
                raise ValueError("sample_mask must be (B,).")
            loss_vec = loss_vec * sm

        if sample_weights is not None:
            w = torch.as_tensor(
                sample_weights, dtype=loss_vec.dtype, device=loss_vec.device
            )
            if w.shape != loss_vec.shape:
                raise ValueError("sample_weights must be (B,).")
            if sample_mask is not None:
                w = w * sm
        else:
            w = None

        # --- Reduction ---
        if self.reduction == "mean":
            if w is not None:
                denom = w.sum().clamp_min(1e-8)
                return ((w * loss_vec).sum() / denom), loss_vec
            else:
                if sample_mask is not None:
                    denom = sm.sum().clamp_min(1.0)
                    return (loss_vec.sum() / denom), loss_vec
                return loss_vec.mean(), loss_vec
        elif self.reduction == "sum":
            return (w * loss_vec).sum() if w is not None else loss_vec.sum()
        else:
            return loss_vec

    @classmethod
    def from_mean_std(cls, mu, std_dev, **kwargs):
        return cls(mu=mu, std_dev=std_dev, diagonal=True, **kwargs)


# -------------------- Utility: safe feature extraction --------------------
@torch.no_grad()
def extract_embeddings(model, images: torch.Tensor, use_pca: bool = False) -> torch.Tensor:
    out = model.encode(images, use_pca=use_pca)  # (B, d) or (B, P, d)
    B, H, W, C = out.shape
    out = out.reshape(B * H * W, C)
    return out


import torch
from pathlib import Path

# from yourpkg.losses import GaussianInDistLoss, InDistMarginLoss  # import your classes


# --- helper: chi-square quantile using SciPy ---
def _chi2_quantile(p: float, df: int) -> torch.Tensor:
    val = chi2.ppf(p, df)  # returns a Python float or numpy scalar
    return torch.tensor(val, dtype=torch.float32)


# --- load stats once (tensors only) ---
def load_gaussian_stats(stats_path: str | Path, device: str = "cuda"):
    stats_path = resolve_path(stats_path)
    if not Path(stats_path).is_file:
        return None
    ckpt = torch.load(str(stats_path), map_location=device)
    mu = ckpt["mu"].to(device).float()
    diagonal = bool(ckpt["diagonal"])
    Sigma_inv_diag = ckpt.get("Sigma_inv_diag")
    chol = ckpt.get("chol")
    if Sigma_inv_diag is not None:
        Sigma_inv_diag = Sigma_inv_diag.to(device).float()
    if chol is not None:
        chol = chol.to(device).float()
    return mu, diagonal, Sigma_inv_diag, chol


# --- make a torch distribution from stats ---
def make_gaussian_distribution(stats_path: str | Path, device: str = "cuda"):
    mu, diagonal, Sigma_inv_diag, chol = load_gaussian_stats(stats_path, device)
    if diagonal:
        var = 1.0 / Sigma_inv_diag
        std = var.sqrt()
        base = torch.distributions.Normal(loc=mu, scale=std)
        return torch.distributions.Independent(base, 1)
    else:
        Sigma = chol @ chol.T
        return torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=Sigma)


# --- draw B samples from the saved distribution ---
def sample_from_stats(
    dist: str | Path | torch.distributions.MultivariateNormal,
    B: int,
    device: str = "cuda",
):
    if isinstance(dist, (str, Path)):
        dist = make_gaussian_distribution(dist, device)
    return dist.sample((B,))  # shape: (B, d)


# --- build a ready-to-use loss from stats ---
def build_loss_from_stats(
    stats_path: str | Path,
    device: str = "cuda",
    *,
    kind: str = "margin",  # "margin" -> InDistMarginLoss, "gaussian" -> GaussianInDistLoss
    confidence: float = 0.95,  # used for "margin" if tau not provided
    tau: float | None = None,
    mode: str = "softplus",
    temperature: float = 0.5,
    power: float = 1.0,
    reduction: str = "mean",
):
    mu, diagonal, Sigma_inv_diag, chol = load_gaussian_stats(stats_path, device)

    if kind == "gaussian":
        return (
            InDistMarginLoss(
                mu=mu,
                Sigma_inv_diag=Sigma_inv_diag,
                chol=chol,
                diagonal=diagonal,
                reduction=reduction,
            )
            .to(device)
            .eval()
        )

    if kind == "margin":
        if tau is None:
            tau = float(_chi2_quantile(confidence, df=mu.numel()))
        return (
            InDistMarginLoss(
                mu=mu,
                Sigma_inv_diag=Sigma_inv_diag,
                chol=chol,
                diagonal=diagonal,
                tau=tau,
                mode=mode,
                temperature=temperature,
                power=power,
                reduction=reduction,
            )
            .to(device)
            .eval()
        )

    raise ValueError(f"Unknown kind='{kind}'")


# -------------------- Main: incremental pass over ImageNet --------------------
def compute_stats_incremental(
    imagenet_path: Path | str,
    mrl_model_path: Path | str,
    pca_path: Path | str | None = None,
    num_samples: int = 50000,
    use_diagonal: bool = True,
    shrinkage: float = 0.1,
    use_pca: bool = False,
):
    from TASE.image_gen.dimensionality_reduction.embedding_model import EmbeddingModel
    from TASE.image_gen.common.util import ImageNetDataset

    print("Starting incremental Gaussian stats computation...")
    # define some parameters
    imagenet_path = Path(imagenet_path)
    dataset = ImageNetDataset(root=imagenet_path, resolution=512, split="train")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (
        torch.bfloat16
        if device == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )

    # Use bfloat16 for the model, but ALWAYS accumulate stats in float32
    stats_dtype = torch.float32

    mrl_model_path = Path(mrl_model_path)
    stats_path = Path(mrl_model_path).parent.parent / "stats" if not use_pca else Path(pca_path).parent / "stats"
    stats_path.mkdir(parents=True, exist_ok=True)

    eps = 1e-5

    # load the DINO model and the dimensionality reduction model
    ctrl_encoder = EmbeddingModel.load_ckpt(mrl_model_path, pca_path=pca_path if use_pca else None)
    ctrl_encoder.eval()
    for p in ctrl_encoder.parameters():
        p.requires_grad_(False)
    ctrl_encoder.to(device).to(dtype)
    torch.set_grad_enabled(False)

    # Dry-run to get embedding dimensionality d
    print("Inferring embedding dimensionality...")
    first_img, *_ = next(
        iter(dataloader)
    )  # assuming dataset returns (image, label, ...) or (image, ...)
    if isinstance(first_img, (list, tuple)):
        first_img = first_img[0]
    first_img = first_img.to(device, dtype=dtype, non_blocking=True)
    with torch.amp.autocast(
        device_type="cuda",
        enabled=(device == "cuda"),
        dtype=torch.bfloat16 if dtype == torch.bfloat16 else torch.float16,
    ):
        test_emb = extract_embeddings(ctrl_encoder, first_img)
    d = test_emb.shape[-1]
    print(f"Embedding dimension: {d}")

    online = OnlineGaussian(
        d=d, diagonal=use_diagonal, dtype=stats_dtype, device=device
    )

    processed = 0
    step = 0
    print("Starting incremental pass...")
    for batch in dataloader:
        # support datasets that return (img, label) or (img,)
        if isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch
        images = images.to(device, dtype=dtype, non_blocking=True)
        B = images.shape[0]

        # If we would overshoot num_samples, subsample the last chunk
        remaining = max(num_samples - processed, 0)
        if remaining <= 0:
            break
        if B > remaining:
            images = images[:remaining]

        # Mixed precision forward; cast embeddings to float32 for stats
        with torch.amp.autocast(
            device_type="cuda",
            enabled=(device == "cuda"),
            dtype=torch.bfloat16 if dtype == torch.bfloat16 else torch.float16,
        ):
            Z = extract_embeddings(ctrl_encoder, images, use_pca=use_pca)  # (B, d) or (B*P, d)
        Z = Z.to(stats_dtype)

        online.update(Z)
        processed += B
        step += 1

        if step % 50 == 0 or processed >= num_samples:
            print(f"[{processed}/{num_samples}] samples processed")

        if processed >= num_samples:
            break

    print("Finalizing stats...")
    mu, Sigma_inv_diag, chol, diagonal = online.finalize(eps=eps, shrinkage=shrinkage)

    # Save stats to disk
    out = {
        "mu": mu.cpu(),
        "diagonal": diagonal,
        "eps": eps,
        "shrinkage": shrinkage,
        "num_samples": processed,
        "dtype": str(stats_dtype),
    }
    if diagonal:
        out["Sigma_inv_diag"] = Sigma_inv_diag.cpu()
    else:
        out["chol"] = chol.cpu()

    stats_file = (
        stats_path
        / f"{mrl_model_path.stem}_stats_{'diag' if diagonal else 'full'}_N{processed}.pt"
    ) 
    
    torch.save(out, stats_file)
    print(f"Saved stats to: {stats_file}")

    # Prepare JSON-serializable dict
    json_out = {
        "mu": mu.cpu().tolist(),
        "diagonal": diagonal,
        "eps": eps,
        "shrinkage": shrinkage,
        "num_samples": processed,
        "dtype": str(stats_dtype),
    }
    if diagonal:
        json_out["Sigma_inv_diag"] = Sigma_inv_diag.cpu().tolist()
    else:
        json_out["chol"] = chol.cpu().tolist()

    # Write JSON
    json_file = stats_file.with_suffix(".json")
    with open(json_file, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"Saved stats JSON to: {json_file}")


if __name__ == "__main__":
    import os

    imgnet_root = os.getenv("IMGNET_ROOT")
    checkpoint_file = "/fs/scratch/rb_bd_dlp_rng-dl01_cr_AIQ_employees/fat2hi/Workspace/Repos/TASE/OUTPUT/MRL/dinov3/2025-12-03_10:58:39/ckpt/epoch_1.pth"
    pca_path = "/fs/scratch/rb_bd_dlp_rng-dl01_cr_AIQ_employees/fat2hi/Workspace/Repos/TASE/OUTPUT/PCA/dinov3/torch_ipca.pkl"

    compute_stats_incremental(
        imagenet_path=imgnet_root, mrl_model_path=(checkpoint_file), pca_path=pca_path, use_pca=True
    )