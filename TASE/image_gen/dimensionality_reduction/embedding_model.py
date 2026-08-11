# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential
from dinov3.layers import SelfAttentionBlock, RopePositionEmbedding
from dinov3.loss.gram_loss import GramLoss
from TASE.image_gen.dimensionality_reduction.embedder_stats import InDistMarginLoss
from torchvision import transforms
from pathlib import Path
import pickle
import json
from functools import lru_cache
from typing import List, Tuple, Optional, Dict, Any, Union
from torch_incremental_pca import IncrementalPCA
import random
import inspect
from TASE.splatting.util.data_util import gaussian_decay_weights

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dinov3.layers import (
    SelfAttentionBlock,
    RopePositionEmbedding,
    Mlp,
)  # use Mlp (or SwiGLUFFN) as FFN


class AERoPE(nn.Module):
    """
    Autoencoder that reduces channel depth of patch features using a DINOv3
    SelfAttentionBlock with axial RoPE in both encoder and decoder.

    Expects inputs as feature grids: (B, Hf, Wf, C).
    """

    def __init__(
        self,
        embed_dim: int,  # input feature dim C
        transformer_dim: int,  # internal transformer width
        internal_dim: int,  # bottleneck dim
        num_heads: int,
        rope_kwargs: dict | None = None,  # forwarded to RopePositionEmbedding
    ):
        super().__init__()
        rope_kwargs = rope_kwargs or {}

        # Linear adapters
        self.enc_in = nn.Linear(embed_dim, transformer_dim)
        self.enc_out = nn.Linear(
            transformer_dim, internal_dim
        )  # Activation is applied outside
        self.enc_norm = nn.LayerNorm(internal_dim)

        self.dec_in = nn.Linear(internal_dim, transformer_dim)
        self.dec_out = nn.Linear(transformer_dim, embed_dim)

        # RoPE (shared object is fine since it just builds sin/cos per (H,W))
        self.rope = RopePositionEmbedding(
            embed_dim=transformer_dim,
            num_heads=num_heads,
            **rope_kwargs,
        )

        # Separate SA blocks for encoder and decoder
        self.enc_blk = SelfAttentionBlock(
            dim=transformer_dim,
            num_heads=num_heads,
            ffn_ratio=4.0,
            qkv_bias=True,
            proj_bias=True,
            ffn_bias=True,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            act_layer=nn.GELU,
            ffn_layer=Mlp,  # or SwiGLUFFN / swiglu variants
            init_values=None,
            mask_k_bias=False,
        )
        self.dec_blk = SelfAttentionBlock(
            dim=transformer_dim,
            num_heads=num_heads,
            ffn_ratio=4.0,
            qkv_bias=True,
            proj_bias=True,
            ffn_bias=True,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            act_layer=nn.GELU,
            ffn_layer=Mlp,
            init_values=None,
            mask_k_bias=False,
        )

    @staticmethod
    def _flatten_hw(z_bhwc: torch.Tensor):
        # (B,Hf,Wf,C) -> (B,N,C), keep Hf,Wf
        B, Hf, Wf, C = z_bhwc.shape
        return z_bhwc.view(B, Hf * Wf, C), Hf, Wf

    @staticmethod
    def _unflatten_hw(z_bnc: torch.Tensor, Hf: int, Wf: int):
        # (B,N,C) -> (B,Hf,Wf,C)
        B, N, C = z_bnc.shape
        assert N == Hf * Wf, f"N={N} != Hf*Wf={Hf*Wf}"
        return z_bnc.view(B, Hf, Wf, C)

    def _run_block(
        self, x_bnc: torch.Tensor, Hf: int, Wf: int, block: SelfAttentionBlock
    ) -> torch.Tensor:
        """
        DINOv3 SelfAttentionBlock API takes lists: x_list, rope_list.
        We pass a single-view list and unwrap the result.
        """
        rope_sincos = self.rope(H=Hf, W=Wf)  # build axial RoPE (sin/cos) for this grid
        x_list = block([x_bnc], [rope_sincos])  # -> list with one tensor
        return x_list[0]

    def encode_tokens(self, feats_bhwc: torch.Tensor) -> torch.Tensor:
        x, Hf, Wf = self._flatten_hw(feats_bhwc)  # (B,N,Cin)
        x = self.enc_in(x)  # (B,N,transformer_dim)
        x = self._run_block(x, Hf, Wf, self.enc_blk)  # axial RoPE attention
        x = self.enc_out(x)  # (B,N,internal_dim)
        x = self.enc_norm(x)
        return self._unflatten_hw(x, Hf, Wf)

    def decode_tokens(self, z_bhwc: torch.Tensor) -> torch.Tensor:
        x, Hf, Wf = self._flatten_hw(z_bhwc)  # (B,N,internal_dim)
        x = self.dec_in(x)  # (B,N,transformer_dim)
        x = self._run_block(x, Hf, Wf, self.dec_blk)  # axial RoPE attention
        x = self.dec_out(x)  # (B,N,embed_dim)
        return self._unflatten_hw(x, Hf, Wf)


class EmbeddingModel(nn.Module):
    def __init__(
        self,
        backbone_model="dinov3",
        num_heads: int = 12,
        internal_dim: int = 128,
        embed_dim: int = 768,
        patch_size: int = 8,
        transformer_dim: int = None,
        build_encoder_decoder: bool = True,
        mrl_dims=None,
        include_cls: bool = False,
        identity_mode: bool = False,
        pca_path: Path = None,
        gram_lambda: float = 1.0,
        cos_lambda: float = 0.8,
        l2_lambda: float = 0.2,
        equivariance_lambda: float = 1.0,
        spacial_bias_lambda: float = 0.1,
        spacial_corr_lambda: float = 1.0,
        moments_lambda: float = 0.1,
        dist_lambda: float = 0.1,
        token_dropout_prob: float = 0.2,
        target_mu: float = 0.0,
        target_std: float = 1.0,  # Natural images: ~0.5 (range -1, 1), as this is applied pre tanh, pre L2 norm, I want it a bit higher
    ):
        """
        I might have to create a proper way to save and load the config values.
        Not sure if saving and loading works properly yet.
        NOTE: Expects unnormalized images as inputs as the normalization is applied according to the backbone model.
        """
        super().__init__()

        self.include_cls = include_cls
        self.identity_mode = identity_mode
        self.feature_hack = False

        self.gram_lambda = gram_lambda
        self.spacial_bias_lambda = spacial_bias_lambda
        self.dist_lambda = dist_lambda
        self.moments_lambda = moments_lambda
        self.equivariance_lambda = equivariance_lambda
        self.spacial_corr_lambda = spacial_corr_lambda
        self.cos_lambda = cos_lambda
        self.l2_lambda = l2_lambda

        self.token_dropout_prob = token_dropout_prob
        self.target_mu = target_mu
        self.target_std = target_std

        self.gram_loss = GramLoss()
        self.dist_loss = InDistMarginLoss.from_mean_std(
            mu=self.target_mu,
            n_dims=internal_dim,
            std_dev=self.target_std,
            confidence=0.67,
            power=2.0,
        )

        if mrl_dims is None:
            mrl_dims = []
            d = 2
            while d <= internal_dim:
                mrl_dims.append(d)
                d *= 2
        for dim in mrl_dims:
            if dim > internal_dim:
                raise ValueError(
                    f"MRL dimension {dim} cannot be greater than internal dimension {internal_dim}."
                )
                # ensure the full embed_dim is included
        if mrl_dims[-1] != internal_dim:
            mrl_dims.append(internal_dim)
        self.mrl_dims = mrl_dims

        self.backbone_model_name = "None"
        if isinstance(backbone_model, str):
            self.backbone_model_name = backbone_model
            if backbone_model == "dinov2":
                from dinov2.hub.backbones import dinov2_vitb14_reg

                backbone_model = dinov2_vitb14_reg(pretrained=True)
            elif backbone_model == "dinov3":
                from transformers import AutoModel

                backbone_model = AutoModel.from_pretrained(
                    "facebook/dinov3-vitb16-pretrain-lvd1689m"
                )
            elif backbone_model == "vae":
                from diffusers import AutoencoderKL

                backbone_model = AutoencoderKL.from_pretrained(
                    "black-forest-labs/FLUX.1-dev", subfolder="vae"
                )
            else:
                raise ValueError(
                    f"Unknown backbone model string: {self.backbone_model_name}"
                )
            print(
                f"Loaded backbone model {self.backbone_model_name} from model specifier."
            )

        self.backbone_model = backbone_model
        # freeze backbone
        for p in self.backbone_model.parameters():
            p.requires_grad = False

        try:
            if hasattr(backbone_model, "patch_size"):
                patch_size = backbone_model.patch_size
            elif hasattr(backbone_model.config, "patch_size"):
                patch_size = backbone_model.config.patch_size
            else:
                raise AttributeError
        except AttributeError:
            print("No patch size found in backbone model. Using default of 8.")
            patch_size = 8
        try:
            if hasattr(backbone_model, "num_features"):
                embed_dim = backbone_model.num_features
            elif hasattr(backbone_model.config, "hidden_size"):
                embed_dim = backbone_model.config.hidden_size
            else:
                raise AttributeError
        except AttributeError:
            print("No embed dim found in backbone model. Using default of 16.")
            embed_dim = 16

        if transformer_dim is None:
            transformer_dim = embed_dim * 2
        self.transformer_dim = transformer_dim
        self.num_heads = num_heads
        self.internal_dim = internal_dim
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # --- Identity mode: skip autoencoder, use raw backbone features ---
        if self.identity_mode:
            build_encoder_decoder = False
            self.internal_dim = embed_dim
            internal_dim = embed_dim
            # Recalculate MRL dims for the raw embedding dimension
            mrl_dims = []
            d = 2
            while d <= internal_dim:
                mrl_dims.append(d)
                d *= 2
            if mrl_dims[-1] != internal_dim:
                mrl_dims.append(internal_dim)
            self.mrl_dims = mrl_dims

        self.eqv_decay_weights = gaussian_decay_weights(self.internal_dim)

        print(f"Using patch size {self.patch_size}, embed dim {self.embed_dim}")

        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        self.pca = None
        if pca_path is not None:
            # load PCA model
            # This is used for inference only, not training.
            self.pca = pickle.load(open(pca_path, "rb"))
        if build_encoder_decoder:
            # build the transformer encoder and decoder
            self.autoencoder = AERoPE(
                embed_dim=embed_dim,
                transformer_dim=transformer_dim,
                internal_dim=internal_dim,
                num_heads=num_heads,
            )

    def pca_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply PCA transformation to the input tensor.
        Args:
            pca: PCA model.
            x: Tensor of shape (B, H, W, C).
        Returns:
            Tensor of shape (B, H, W, n_components).
        """
        pca = self.pca
        dtype = x.dtype
        if x.ndim == 4:
            B, H, W, C = x.shape
            new_shape = (B, H, W, -1)
        elif x.ndim == 3:
            B, N, C = x.shape
            new_shape = (B, N, -1)
        flat = x.reshape(-1, C).double()
        if isinstance(pca, dict):
            red = flat - pca["mean"]
            red = torch.mm(red, pca["components"].T)
        else:
            red = pca.transform(flat)
        return red.view(new_shape).to(dtype=dtype)

    def base_encode(self, x: torch.Tensor) -> torch.Tensor:
        B, img_C, img_H, img_W = x.shape
        # check for patch size compatibility
        if x.shape[-1] % self.patch_size != 0:
            # reshape to size divisible by patch size
            new_H = (img_H // self.patch_size) * self.patch_size
            new_W = (img_W // self.patch_size) * self.patch_size
            x = F.interpolate(
                x, size=(new_H, new_W), mode="bilinear", align_corners=False
            )
        else:
            new_H, new_W = img_H, img_W

        if hasattr(self.backbone_model, "forward_features"):
            # for DINO
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
            x = normalize(x)
            with torch.no_grad():
                features = self.backbone_model.forward_features(x)
                base = features["x_norm_patchtokens"]  # [B, N, E]
                if self.include_cls:
                    cls_token = features["x_norm_clstoken"]  # [B, 1, E]
                    base = torch.cat(
                        (cls_token.unsqueeze(1), base), dim=1
                    )  # [B, N+1, E]
        elif hasattr(self.backbone_model, "encode"):
            # for VAE
            normalize = transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            )
            x = normalize(x)
            with torch.no_grad():
                feats = self.backbone_model.encode(x).latent_dist.sample()
                feats = (
                    feats - self.backbone_model.config.shift_factor
                ) * self.backbone_model.config.scaling_factor
                feats = feats.permute(0, 2, 3, 1)  # (B, H, W, E)
                B, H, W, C = feats.shape
                base = feats.reshape(B, H * W, C)
        else:
            # for DINOv3
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
            x = normalize(x)
            with torch.no_grad():
                outputs = self.backbone_model(x)
            base = outputs.last_hidden_state  # (B, N, 1 CLS + D + 4 REG)
            if not self.include_cls:
                # remove CLS token and 4 reg tokens
                base = base[:, 5:, :]

        H, W = int(new_H / self.patch_size), int(new_W / self.patch_size)
        base = base.reshape(B, H, W, -1)

        return base

    def encode(
        self,
        x: torch.Tensor,
        return_dict: bool = False,
        use_pca: bool = False,
        normalize: bool = False,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        if self.feature_hack:
            # apply reflect padding to simulate larger images for feature extraction
            num_pad_patches = 2
            pad_size = (
                self.patch_size * num_pad_patches
            )  # pad by 2 patches on each side
            x = F.pad(x, (pad_size, pad_size, pad_size, pad_size), mode="reflect")
        B, img_C, img_H, img_W = x.shape
        H, W = int(img_H / self.patch_size), int(img_W / self.patch_size)
        base = self.base_encode(x)  # [B, H, W, C]
        if use_pca:
            if self.pca is None:
                raise ValueError("PCA requested but PCA model not loaded.")
            # apply PCA transformation if available
            base_flat = base.reshape(B, H * W, -1)  # [B, N, C]
            ordered = self.pca_transform(base_flat)
            ordered = ordered.reshape(B, H, W, -1)
        elif self.autoencoder is not None:
            ordered = self.autoencoder.encode_tokens(base)  # [B, H, W, internal_dim]
        elif self.identity_mode:
            ordered = base  # raw backbone features as-is
        else:
            raise RuntimeError(
                "No encoding path available: autoencoder is None, identity_mode is False, and PCA not requested."
            )

        if self.feature_hack:
            # remove padding
            ordered = ordered[
                :, num_pad_patches:-num_pad_patches, num_pad_patches:-num_pad_patches, :
            ]
            base = base[
                :, num_pad_patches:-num_pad_patches, num_pad_patches:-num_pad_patches, :
            ]

        ordered_tanh = torch.nn.functional.tanh(ordered)  # ensure bounded activations
        if normalize:
            ordered_tanh = F.normalize(ordered_tanh, dim=-1, eps=1e-6)

        if return_dict:
            # return a dict with the base and ordered embeddings
            return {
                "base_embedding": base,
                "ordered_embedding": ordered_tanh,
                "ordered_raw": ordered,
                "recon_embedding": None,
            }

        return ordered_tanh  # [B, H, W, internal_dim]

    def decode(self, embedding: torch.Tensor, reshape: bool = False) -> torch.Tensor:
        # expects [B, H, W, C]
        if self.autoencoder is None:
            raise RuntimeError("decode() is not available in identity mode (no autoencoder).")
        recon = self.autoencoder.decode_tokens(embedding)  # [B, H, W, E]
        return recon

    @staticmethod
    def get_tuncated(x: torch.Tensor, k: int):
        if k >= x.shape[-1]:
            return x
        mask = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)  # [E]
        mask[k:] = 0
        mask = mask.view(1, 1, 1, -1)  # [1,1,E]
        masked = x * mask
        return masked

    def forward(self, x: torch.Tensor, random_spatial_mask: bool = False) -> dict:
        result = self.encode(x, return_dict=True)
        ordered = result["ordered_embedding"]  # [seq, batch, E]
        # optionally apply a random spatial mask
        if random_spatial_mask:
            B, H, W, C = ordered.shape
            mask = (
                torch.rand(1, H, W, 1, device=ordered.device) > self.token_dropout_prob
            ).float()
            ordered = ordered * mask
            result["mask"] = mask
        # produce reconstructions at each MRL level
        decoded = []
        for mrl in self.mrl_dims:
            masked = EmbeddingModel.get_tuncated(ordered, mrl)
            recon = self.decode(masked)
            decoded.append(recon)

        result["recon_embedding"] = decoded
        return result

    def compute_recon_loss(
        self,
        base: torch.Tensor,
        recon: torch.Tensor,
        mask: torch.Tensor = None,
        normalize: bool = False,
    ) -> torch.Tensor:
        if normalize:
            base = F.normalize(base, dim=-1, eps=1e-6)
            recon = F.normalize(recon, dim=-1, eps=1e-6)
        cos = 1 - F.cosine_similarity(base, recon, dim=-1)
        l2 = (base - recon).pow(2).mean(dim=-1)
        if mask is not None:
            mask = mask.squeeze(-1) if mask.dim() == 4 else mask  # [1,H,W]
            cos = cos * mask
            l2 = l2 * mask
        cos = cos.mean()
        l2 = l2.mean()
        return self.cos_lambda * cos + self.l2_lambda * l2

    def gaussian_moments_loss(
        self,
        z_bhwc: torch.Tensor,  # (B,H,W,C)
        mask: torch.Tensor = None,  # (1,H,W,1) or (B,H,W,1); 1=keep, 0=ignore
        use_lower_bound_var: bool = True,
        per_image: bool = False,
        cov_lambda: float = 1.0,  # strength of off-diagonal covariance penalty
    ) -> torch.Tensor:
        """
        Encourages per-channel mean/variance to match targets and suppresses cross-channel covariance.
        If per_image=True, moments are computed per sample; otherwise over the whole batch.
        """
        B, H, W, C = z_bhwc.shape
        device = z_bhwc.device
        dtype = z_bhwc.dtype

        # -- mask to weights (B,H,W,1) --
        if mask is None:
            w = torch.ones((B, H, W, 1), device=device, dtype=dtype)
        else:
            if mask.shape[0] == 1:  # broadcast (1,H,W,1) -> (B,H,W,1)
                w = mask.expand(B, H, W, 1).to(device=device, dtype=dtype)
            else:
                w = mask.to(device=device, dtype=dtype)

        # -- axes over which we aggregate tokens --
        if per_image:
            reduce_axes = (1, 2)  # per image (B,C)
            N_tokens = w.sum(dim=reduce_axes, keepdim=False).clamp_min(1e-6)  # (B,1)
        else:
            reduce_axes = (0, 1, 2)  # global (C,)
            N_tokens = w.sum(dim=reduce_axes, keepdim=False).clamp_min(1e-6)  # (1,)

        # -- means: (B,1,1,C) or (1,1,1,C) --
        # sum over tokens, divide by valid count
        sum_wz = (z_bhwc * w).sum(dim=reduce_axes, keepdim=True)  # (*,1,1,C)
        sum_w = w.sum(dim=reduce_axes, keepdim=True).clamp_min(1e-6)  # (*,1,1,1)
        mu = sum_wz / sum_w  # (*,1,1,C)

        # -- variances per channel: same broadcasting --
        zc = z_bhwc - mu
        var = ((zc * zc) * w).sum(dim=reduce_axes, keepdim=True) / sum_w  # (*,1,1,C)

        target_mu = torch.as_tensor(self.target_mu, dtype=dtype, device=device).view(
            1, 1, 1, -1
        )
        target_std = torch.as_tensor(self.target_std, dtype=dtype, device=device).view(
            1, 1, 1, -1
        )
        target_var = target_std * target_std

        # -- losses on mean, variance --
        mean_loss = (mu - target_mu).pow(2).mean()
        if use_lower_bound_var:
            var_loss = (
                F.relu(target_var - var).pow(2).mean()
            )  # penalize only if too small
        else:
            var_loss = (target_var - var).pow(2).mean()

        total = mean_loss + var_loss

        # ===== covariance penalty (off-diagonals) =====
        # Goal: Cov ≈ diag(var); i.e., off-diagonal entries ~ 0.
        # Compute weighted covariance over tokens, per image or globally.

        # weight each token by sqrt(w) to realize weighted least squares
        if cov_lambda > 0:
            wsqrt = w.sqrt()

            if per_image:
                # reshape per image: (B, N, C)
                X = (zc * wsqrt).reshape(B, H * W, C)
                # effective counts per image (scalar per batch element)
                Neff = (
                    wsqrt.reshape(B, H * W, 1).pow(2).sum(dim=1).clamp_min(1e-6)
                )  # (B,1)
                # covariance: (B,C,C)
                Cov = torch.matmul(X.transpose(1, 2), X) / Neff.unsqueeze(
                    -1
                )  # divide by Neff
                # remove diagonals
                offdiag_mask = ~torch.eye(C, dtype=torch.bool, device=device).unsqueeze(
                    0
                )  # (1,C,C)
                offdiag = Cov[offdiag_mask]  # (B*C*(C-1))
                cov_loss = (offdiag.pow(2)).mean()
            else:
                # global aggregation: merge B,H,W
                X = (zc * wsqrt).reshape(-1, C)  # (BN, C)
                Neff = wsqrt.reshape(-1, 1).pow(2).sum().clamp_min(1e-6)  # scalar
                Cov = (X.t() @ X) / Neff  # (C,C)
                offdiag_mask = ~torch.eye(C, dtype=torch.bool, device=device)
                cov_loss = Cov[offdiag_mask].pow(2).mean()

                # (Optional) normalize by channel scale so magnitudes are comparable:
                # You can divide cov_loss by (var.mean().detach() + eps) to be scale-invariant.
                # eps = 1e-6
                # cov_loss = cov_loss / (var.mean().detach() + eps)

            total = total + cov_lambda * cov_loss

        return total

    def in_dist_loss(
        self, z_bhwc: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        z_bhwc: (B,H,W,C) patch/token embeddings
        Returns: loss encouraging embedding distribution to be in-distribution.
        """
        B, H, W, C = z_bhwc.shape
        z_flat = z_bhwc.reshape(-1, C)  # (B*H*W, C)
        mask = (
            mask.expand(B, H, W, 1).reshape(-1) if mask is not None else None
        )  # (B*H*W,)
        loss, loss_vec = self.dist_loss(z_flat, sample_mask=mask)
        loss_vec = loss_vec.view(B, H, W) if loss_vec is not None else None
        return loss, loss_vec

    @staticmethod
    def spacial_bias_loss(
        z_bhwc: torch.Tensor,
        mask: torch.Tensor = None,
        normalize: bool = True,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        z_bhwc: (B,H,W,C) patch/token embeddings
        Returns: loss encouraging no absolute positional bias.
        NOTE: If masks are applied during the embedding, the same mask has to be applied across the batch!
        """
        # per-batch per-channel z-score across tokens
        if normalize:
            z = z_bhwc
            mu = z.mean(dim=(0, 1, 2), keepdim=True)
            std = z.std(dim=(0, 1, 2), keepdim=True, unbiased=False).clamp_min(eps)
            z_bhwc = (z - mu) / std
        # Spatial bias suppression: per-(y,x) mean across batch should be ~0
        pos_mean = z_bhwc.mean(dim=0)  # (H,W,C)  mean over B
        loss_pos = pos_mean**2
        if mask is not None:
            loss_pos = loss_pos * mask
        loss_pos = loss_pos.mean()
        return loss_pos

    @staticmethod
    def spatial_corr_loss(
        z_bhwc: torch.Tensor, freqs=(1, 2, 4), eps: float = 1e-6
    ) -> torch.Tensor:
        """
        Penalize linear dependence of features on absolute position.
        z_bhwc: (B,H,W,C). Returns scalar in [0,1].
        """
        B, H, W, C = z_bhwc.shape
        device = z_bhwc.device

        # --- Build bases ---
        xs = torch.linspace(-1, 1, W, device=device)
        ys = torch.linspace(-1, 1, H, device=device)
        Y, X = torch.meshgrid(ys, xs, indexing="ij")  # (H,W)
        bases = [X, Y]
        for f in freqs:
            bases += [
                torch.sin(f * torch.pi * X),
                torch.cos(f * torch.pi * X),
                torch.sin(f * torch.pi * Y),
                torch.cos(f * torch.pi * Y),
            ]
        Bm = torch.stack(bases, dim=-1).reshape(
            -1, len(bases)
        )  # (N, D), N=H*W, D=bases

        # z-score columns of bases
        Bm = Bm - Bm.mean(dim=0, keepdim=True)
        Bm = Bm / Bm.std(dim=0, keepdim=True).clamp_min(eps)  # (N,D)

        # Flatten features to (B,C,N) and z-score along N
        Z = z_bhwc.reshape(B, -1, C).permute(0, 2, 1).contiguous()  # (B,C,N)
        Z = Z - Z.mean(dim=2, keepdim=True)
        Z = Z / Z.std(dim=2, keepdim=True).clamp_min(eps)

        # Pearson correlations r = (Z @ B) / (N-1) with standardized columns -> just (Z @ B)/N
        # Compute batched: (B,C,N) @ (N,D) -> (B,C,D)
        r = Z @ Bm / Z.shape[2]  # (B,C,D)

        loss = (r**2).mean()  # in [0,1]
        return loss

    def gram_anchoring_loss(
        self,
        z_student_bhwc: torch.Tensor,  # (B, H, W, D_s)
        z_teacher_bhwc: torch.Tensor,  # (B, H, W, D_t)
    ) -> torch.Tensor:
        """
        Args:
            z_student_bhwc: (B,H,W,C_s)
            z_teacher_bhwc: (B,H,W,C_t)
            normalize_tokens: If True, L2-normalize tokens along channel dim before Gram (cosine-like).
            eps: Numerical stability.
        Returns:
            Scalar torch.Tensor
        """
        # For small feature maps, the exact loss is fine.
        # For larger feature maps, it becomes slow due to O(N^2) memory/time.
        B, Hs, Ws, Cs = z_student_bhwc.shape
        Bt, Ht, Wt, Ct = z_teacher_bhwc.shape
        assert B == Bt and Hs == Ht and Ws == Wt, "Batch and spatial dims must match"
        N = Hs * Ws
        zs = z_student_bhwc.reshape(B, N, Cs)
        zt = z_teacher_bhwc.reshape(B, N, Ct)
        return self.gram_loss(zs, zt)

    def compute_losses(self, result: dict) -> list:
        # retrieve embeddings
        base = result["base_embedding"]
        reduced_raw = result["ordered_raw"]  # pre-tanh features
        reduced = result["ordered_embedding"]  # post-tanh features
        recon_list = result["recon_embedding"]  # list of reconstructions at each MRL

        if "mask" in result:
            mask = result["mask"]  # (B, H, W)
        else:
            mask = None

        # Main AE reconstruction loss applied on all MRL levels
        ae_recon_loss = [
            self.compute_recon_loss(base, recon, mask) for recon in recon_list
        ]
        total_loss = sum(ae_recon_loss) / len(ae_recon_loss)
        info = {
            "ae_recon_loss": {
                mrl: l.item() for mrl, l in zip(self.mrl_dims, ae_recon_loss)
            },
        }

        # Gram anchoring loss on a random MRL level (to save computation)
        if self.gram_lambda > 0:
            k = self.mrl_dims[
                torch.randint(0, len(self.mrl_dims), (1,)).item()
            ]  # sample uniformly from mrl_dims
            trunc = reduced[
                ..., :k
            ]  # truely truncate to keep the loss scale consistent
            gram_loss = self.gram_anchoring_loss(trunc, base) * self.gram_lambda / k
            total_loss += gram_loss
            info["gram_loss"] = {k: gram_loss.item()}

        # Gaussian moments loss on pre-tanh features (to encourage image-like distribution)
        if self.moments_lambda > 0:
            moments_loss = (
                self.gaussian_moments_loss(reduced_raw, mask) * self.moments_lambda
            )
            total_loss += moments_loss
            info["moments_loss"] = moments_loss.item()
        if self.dist_lambda > 0:
            dist_loss, dist_loss_map = self.in_dist_loss(reduced_raw, mask)
            dist_loss = dist_loss * self.dist_lambda
            total_loss += dist_loss
            info["dist_loss"] = dist_loss.item()
            info["dist_loss_map"] = dist_loss_map  # (B,H,W) or None

        # Regularizations to suppress absolute positional encoding in the adapted features
        if self.spacial_bias_lambda > 0:
            spacial_bias = (
                EmbeddingModel.spacial_bias_loss(reduced_raw) * self.spacial_bias_lambda
            )
            total_loss += spacial_bias
            info["spacial_bias"] = spacial_bias.item()
        if self.spacial_corr_lambda > 0:
            spacial_corr = (
                EmbeddingModel.spatial_corr_loss(reduced_raw) * self.spacial_corr_lambda
            )
            total_loss += spacial_corr
            info["spacial_corr"] = spacial_corr.item()

        return total_loss, info

    def compute_validation_losses(self, result: dict, full_gram: bool = True) -> dict:
        # retrieve embeddings
        base = result["base_embedding"]
        reduced_raw = result["ordered_raw"]  # pre-tanh features
        reduced = result["ordered_embedding"]  # post-tanh features
        recon_list = result["recon_embedding"]  # list of reconstructions at each MRL

        if "mask" in result:
            print(
                "[WARNING]: Mask found in validation result. Are you sure you want to use masking during validation?"
            )
            mask = result["mask"]  # (B, H, W)
        else:
            mask = None

        # Main AE reconstruction loss applied on all MRL levels
        ae_recon_loss = [
            self.compute_recon_loss(base, recon, mask) for recon in recon_list
        ]
        info = {
            "ae_recon_loss": {
                mrl: l.item() for mrl, l in zip(self.mrl_dims, ae_recon_loss)
            },
        }

        # Gram anchoring loss on a random MRL level (to save computation)
        if full_gram:
            ks = self.mrl_dims
        else:
            k = self.mrl_dims[
                torch.randint(0, len(self.mrl_dims), (1,)).item()
            ]  # sample uniformly from mrl_dims
            ks = [k]
        for k in ks:
            trunc = reduced[
                ..., :k
            ]  # truely truncate to keep the loss scale consistent
            gram_loss = self.gram_anchoring_loss(trunc, base)
            if "gram_loss" not in info:
                info["gram_loss"] = {}
            info["gram_loss"].update({k: gram_loss.item()})

        return info

    @torch.no_grad()
    def _sizes_in_patches(self, H: int, W: int, min_scale=0.5, max_scale=1.0):
        """Enumerate admissible crop sizes in patch units within the scale range."""
        ps = self.patch_size
        Hp, Wp = H // ps, W // ps

        # Convert scale bounds (on pixels) to patch counts
        min_h = max(1, (int(H * min_scale) // ps))
        max_h = max(1, (int(H * max_scale) // ps))
        min_w = max(1, (int(W * min_scale) // ps))
        max_w = max(1, (int(W * max_scale) // ps))

        # Clamp to feasible ranges
        max_h = min(max_h, Hp)
        max_w = min(max_w, Wp)

        sizes = []
        for Kh in range(min_h, max_h + 1):
            for Kw in range(min_w, max_w + 1):
                if Kh <= Hp and Kw <= Wp:
                    sizes.append((Kh, Kw))
        # Fallback to at least 1×1 if scales were tiny
        if not sizes:
            sizes = [(1, 1)]
        return sizes, Hp, Wp

    # ---------- helpers ----------
    def _ensure_leastcov_state(
        self, H: int, W: int, ps: int, device: torch.device | None
    ):
        device = device or torch.device("cpu")
        Hp, Wp = H // ps, W // ps
        if not hasattr(self, "_leastcov_state") or self._leastcov_state is None:
            assert (
                H % ps == 0 and W % ps == 0
            ), "H and W must be multiples of patch_size"
            self._leastcov_state = {
                "H": H,
                "W": W,
                "ps": ps,
                "Hp": Hp,
                "Wp": Wp,
                "device": device,
                "counts": torch.zeros((Hp, Wp), dtype=torch.int64, device=device),
                "calls": 0,
            }
        else:
            st = self._leastcov_state
            if (st["H"], st["W"], st["ps"]) != (H, W, ps):
                raise ValueError(
                    f"Least-covered sampler initialized for "
                    f"(H,W,ps)=({st['H']},{st['W']},{st['ps']}); "
                    f"got ({H},{W},{ps}). Keep image size & patch_size constant."
                )

    def _maybe_renormalize_counts(self):
        """Keep counters small to avoid overflow while preserving ordering."""
        st = self._leastcov_state
        c = st["counts"]
        st["calls"] += 1
        # Periodic baseline removal: subtract current min so the least-covered tokens are at 0
        if st["calls"] % 10_000 == 0:
            m = int(c.min().item())
            if m > 0:
                c.sub_(m)
        # Hard cap: if numbers ever get huge, do a safe right-shift
        if int(c.max().item()) > (1 << 60):
            # Shift divides all counts by 2 (preserves ordering)
            c >>= 1

    # -------------------------- public API: least-covered sampler ----------------------
    @torch.no_grad()
    def sample_crop_least_covered(
        self,
        H: int,
        W: int,
        min_scale: float = 0.5,
        max_scale: float = 1.0,
        fixed_scale: float | None = None,  # if not None, overrides min/max
        preserve_ar: bool = True,  # if True: AR preserved; else: AR free
        device: torch.device | None = None,
    ):
        """
        Stateful sampler:
          1) keeps per-token coverage counts,
          2) picks a least-covered token,
          3) samples an in-bounds crop that includes it,
          4) updates counts for all tokens inside the crop.
        Returns (top, left, ch, cw) in *pixels*.
        """
        patch_size = self.patch_size
        self._ensure_leastcov_state(H, W, patch_size, device)
        st = self._leastcov_state
        ps, Hp, Wp = st["ps"], st["Hp"], st["Wp"]
        counts = st["counts"]

        # ---- choose crop size (tokens) ----
        if preserve_ar:
            s = (
                float(max(0.0, min(1.0, fixed_scale)))
                if fixed_scale is not None
                else random.uniform(min_scale, max_scale)
            )
            Kh = max(1, min(Hp, int(round(Hp * s))))
            Kw = max(1, min(Wp, int(round(Wp * s))))
        else:
            if fixed_scale is not None:
                sh = sw = float(max(0.0, min(1.0, fixed_scale)))
            else:
                sh, sw = random.uniform(min_scale, max_scale), random.uniform(
                    min_scale, max_scale
                )
            Kh = max(1, min(Hp, int(round(Hp * sh))))
            Kw = max(1, min(Wp, int(round(Wp * sw))))

        # Degenerate full-frame case
        if (Hp - Kh + 1) == 1 and (Wp - Kw + 1) == 1:
            # Update all tokens and return
            counts.add_(1)
            self._maybe_renormalize_counts()
            return 0, 0, Kh * ps, Kw * ps

        # ---- pick a least-covered token (uniform among ties) ----
        minval = counts.min()
        idxs = (counts == minval).nonzero(as_tuple=False)  # (M, 2)
        pr, pc = idxs[random.randrange(idxs.shape[0])].tolist()  # target token

        # ---- valid placement range to include (pr,pc) ----
        top_lo = max(0, pr - (Kh - 1))
        top_hi = min(pr, Hp - Kh)
        left_lo = max(0, pc - (Kw - 1))
        left_hi = min(pc, Wp - Kw)

        # sample top-left uniformly within valid range
        top_p = random.randint(top_lo, top_hi)
        left_p = random.randint(left_lo, left_hi)

        # ---- update coverage counts for all tokens inside the crop ----
        counts[top_p : top_p + Kh, left_p : left_p + Kw] += 1
        self._maybe_renormalize_counts()

        # ---- to pixels ----
        return top_p * ps, left_p * ps, Kh * ps, Kw * ps

    @torch.no_grad()
    @staticmethod
    def _sample_patch_aligned_crop_equal_area(
        self,
        H: int,
        W: int,
        min_scale: float = 0.5,
        max_scale: float = 1.0,
        fixed_scale: float | None = None,  # if not None, overrides min/max
        preserve_ar: bool = True,  # if True: AR preserved; else: AR free
    ):
        """
        Equal-area coverage (uniform over tokens).
        - If `fixed_scale` in [0,1] is provided: crop size fixed, only position random.
        - If `preserve_ar=True`: AR preserved (one scalar scale).
        - If `preserve_ar=False`: H and W scales chosen independently.
        Returns (top, left, ch, cw) in *pixels*.
        """
        ps = self.patch_size
        Hp, Wp = H // ps, W // ps
        assert Hp > 0 and Wp > 0

        # ----- 1) Choose crop size -----
        if preserve_ar:
            if fixed_scale is not None:
                s = float(max(0.0, min(1.0, fixed_scale)))
            else:
                s = random.uniform(min_scale, max_scale)
            Kh = max(1, min(Hp, int(round(Hp * s))))
            Kw = max(1, min(Wp, int(round(Wp * s))))
        else:
            if fixed_scale is not None:
                # Use same fixed scale for both axes, but allow AR change via rounding
                sh = sw = float(max(0.0, min(1.0, fixed_scale)))
            else:
                sh = random.uniform(min_scale, max_scale)
                sw = random.uniform(min_scale, max_scale)
            Kh = max(1, min(Hp, int(round(Hp * sh))))
            Kw = max(1, min(Wp, int(round(Wp * sw))))

        # ----- 2) Pick a target token uniformly -----
        pr = random.randrange(Hp)
        pc = random.randrange(Wp)

        # ----- 3) Valid placement ranges so crop includes that token -----
        top_lo = max(0, pr - (Kh - 1))
        top_hi = min(pr, Hp - Kh)
        left_lo = max(0, pc - (Kw - 1))
        left_hi = min(pc, Wp - Kw)

        top_p = random.randint(top_lo, top_hi)
        left_p = random.randint(left_lo, left_hi)

        # ----- 4) Convert to pixels -----
        top = top_p * ps
        left = left_p * ps
        ch = Kh * ps
        cw = Kw * ps
        return top, left, ch, cw

    # ---------- Helpers ----------

    @staticmethod
    def _pair_indices_for_triplets(B: int) -> List[Tuple[int, int]]:
        """Return floor(B/2) disjoint random pairs of indices from [0..B-1]."""
        perm = torch.randperm(B).tolist()
        return [(perm[i], perm[i + 1]) for i in range(0, len(perm) - 1, 2)]

    @torch.no_grad()
    @staticmethod
    def _build_collage_triplets(
        x: torch.Tensor,
        patch_size: int,
        modes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Build collage triplets from a batch x.
        Returns dict with:
        - pairs: List[(i,j)]
        - modes: List[str] in {"hcat","vcat","paste"}
        - paste_specs: List[dict|None]
        - imgs_a: (K,3,H,W)
        - imgs_b: (K,3,H,W)
        - imgs_collage: List of collages stacked to (K,3,Hc,Wc) if uniform size, else list
        """
        B, C, H, W = x.shape
        assert B >= 2, "Need at least 2 images for collages"
        ps = patch_size

        pairs = EmbeddingModel._pair_indices_for_triplets(B)
        K = len(pairs)
        imgs_a = torch.stack([x[i] for i, _ in pairs], dim=0)  # (K,3,H,W)
        imgs_b = torch.stack([x[j] for _, j in pairs], dim=0)  # (K,3,H,W)

        if modes is None:
            modes = [random.choice(["hcat", "vcat", "paste"]) for _ in range(K)]

        collages = []
        paste_specs: List[Optional[Dict[str, int]]] = []

        Hp, Wp = H // ps, W // ps
        for k in range(K):
            a = imgs_a[k : k + 1]  # (1,3,H,W)
            b = imgs_b[k : k + 1]
            mode = modes[k]

            if mode == "hcat":
                # horizontal concat with width crop (keeps same H,W as inputs)
                Hp, Wp = H // ps, W // ps
                assert Wp >= 2, "need at least 2 patch columns for hcat"
                split_w = random.randint(1, Wp - 1)  # patches taken from left image
                w_left = split_w * ps
                w_right = (Wp - split_w) * ps
                collage = torch.cat(
                    [a[:, :, :, :w_left], b[:, :, :, -w_right:]],  # left part of A
                    dim=3,
                )  # right part of B
                # spec in PATCH units
                spec = dict(mode="hcat", split_p=split_w)

            elif mode == "vcat":
                # vertical concat with height crop (keeps same H,W as inputs)
                Hp, Wp = H // ps, W // ps
                assert Hp >= 2, "need at least 2 patch rows for vcat"
                split_h = random.randint(1, Hp - 1)  # patches taken from top image
                h_top = split_h * ps
                h_bot = (Hp - split_h) * ps
                collage = torch.cat(
                    [a[:, :, :h_top, :], b[:, :, -h_bot:, :]], dim=2  # top part of A
                )  # bottom part of B
                spec = dict(mode="vcat", split_p=split_h)

            else:  # "paste": patch-aligned paste of a window from b into a
                # sample patch window size and locations
                h_sel = random.randint(1, Hp)
                w_sel = random.randint(1, Wp)
                tb = random.randint(0, Hp - h_sel)  # source top (patch coords)
                lb = random.randint(0, Wp - w_sel)  # source left
                ta = random.randint(0, Hp - h_sel)  # dest top in a
                la = random.randint(0, Wp - w_sel)  # dest left in a

                tb_px, lb_px = tb * ps, lb * ps
                ta_px, la_px = ta * ps, la * ps
                hh_px, ww_px = h_sel * ps, w_sel * ps

                collage = a.clone()
                collage[:, :, ta_px : ta_px + hh_px, la_px : la_px + ww_px] = b[
                    :, :, tb_px : tb_px + hh_px, lb_px : lb_px + ww_px
                ]

                spec = dict(h_p=h_sel, w_p=w_sel, tb=tb, lb=lb, ta=ta, la=la)

            collages.append(collage.squeeze(0))
            paste_specs.append(spec)

        # If all collages are same size, stack to tensor; else keep list
        uniform_size = all(c.shape == collages[0].shape for c in collages)
        imgs_collage = torch.stack(collages, dim=0) if uniform_size else collages

        return dict(
            pairs=pairs,
            modes=modes,
            paste_specs=paste_specs,
            imgs_a=imgs_a,
            imgs_b=imgs_b,
            imgs_collage=imgs_collage,
        )

    @staticmethod
    def _reconstruct_collage_features(
        z_a: torch.Tensor,  # (K, Ha, Wa, C)
        z_b: torch.Tensor,  # (K, Hb, Wb, C)
        modes: list[str],  # ["hcat"|"vcat"|"paste"]
        paste_specs: list[dict | None],  # for paste + cropped cats
    ) -> torch.Tensor:
        """
        Rebuild collage features from z_a and z_b using the *same* geometry:
        - "hcat": take first split_p columns from A, last (Wa - split_p) from B
        - "vcat": take first split_p rows from A, last (Ha - split_p) from B
        - "paste": patch-aligned paste from z_b into z_a
        All split_p are **patch units**, i.e., feature cells (since features are 1 token per patch).
        Returns a stacked tensor (K, Hr, Wr, C). For all three modes here, Hr/Wr == Ha/Wa.
        """
        assert z_a.shape == z_b.shape, "z_a and z_b must share spatial size per pair"
        K, Ha, Wa, C = z_a.shape
        out = []

        for k in range(K):
            mode = modes[k]
            spec = paste_specs[k]

            if mode == "hcat":
                # spec: {"mode":"hcat", "split_p": int}
                split_p = spec.get("split_p", None) if spec else None
                if split_p is None:
                    raise ValueError("hcat requires spec['split_p']")
                assert 1 <= split_p <= Wa - 1, "invalid split_p for hcat"
                z_rec = torch.cat(
                    [
                        z_a[k : k + 1, :, :split_p, :],  # left part from A
                        z_b[k : k + 1, :, split_p:, :],
                    ],  # right part from B
                    dim=2,
                )

            elif mode == "vcat":
                # spec: {"mode":"vcat", "split_p": int}
                split_p = spec.get("split_p", None) if spec else None
                if split_p is None:
                    raise ValueError("vcat requires spec['split_p']")
                assert 1 <= split_p <= Ha - 1, "invalid split_p for vcat"
                z_rec = torch.cat(
                    [
                        z_a[k : k + 1, :split_p, :, :],  # top part from A
                        z_b[k : k + 1, split_p:, :, :],
                    ],  # bottom part from B
                    dim=1,
                )

            else:  # "paste"
                # spec: {"h_p","w_p","tb","lb","ta","la"} in PATCH units
                if spec is None:
                    raise ValueError("paste requires paste_specs")
                h_sel = spec["h_p"]
                w_sel = spec["w_p"]
                tb = spec["tb"]
                lb = spec["lb"]
                ta = spec["ta"]
                la = spec["la"]

                # bounds checks in feature space (patch units == feature cells)
                assert 0 <= ta and ta + h_sel <= Ha
                assert 0 <= la and la + w_sel <= Wa
                assert 0 <= tb and tb + h_sel <= Ha
                assert 0 <= lb and lb + w_sel <= Wa

                z_rec = z_a[k : k + 1].clone()
                z_rec[:, ta : ta + h_sel, la : la + w_sel, :] = z_b[k : k + 1][
                    :, tb : tb + h_sel, lb : lb + w_sel, :
                ]

            out.append(z_rec)

        return torch.cat(out, dim=0)

    # ---------- Main (uses helpers) ----------
    def equivariance_loss_crop_flip(
        self,
        x: torch.Tensor,
        min_scale: float = 0.2,
        max_scale: float = 0.8,
        p_flip: float = 0.5,
        p_collage: float = 0.1,
        weighted: bool = False,
        preserve_ar: bool = False,
        fixed_scale: Optional[float] = None,
        return_collage_info: bool = False,
        reduction: str = "mean",
    ):
        """
        Compute equivariance loss for embeddings under spatial transformations.

        This function enforces consistency of the learned embeddings with respect
        to either (a) random crops + optional horizontal flips, or (b) collage-based
        triplet compositions of two images. Only one path is chosen per call.

        Args:
            x (torch.Tensor):
                Input batch of images of shape (B, 3, H, W).
            min_scale (float, default=0.5):
                Minimum crop scale factor relative to full image size.
            max_scale (float, default=1.0):
                Maximum crop scale factor.
            p_flip (float, default=0.5):
                Probability of applying a horizontal flip to the cropped region.
            p_collage (float, default=0.25):
                Probability of using the collage-triplet path (requires B ≥ 2).
            weighted (bool, default=False):
                Whether to apply per-channel decay weights to embeddings.
            normalize (bool, default=True):
                Whether to L2-normalize embeddings before reconstruction loss.
            preserve_ar (bool, default=True):
                Preserve aspect ratio in random crop sampling.
            fixed_scale (float, optional):
                If provided, overrides min/max_scale with a fixed crop scale.
            return_collage_info (bool, default=False):
                If True, return additional tensors (imgs_a, imgs_b, imgs_collage,
                and corresponding embeddings) in the info dict for debugging or analysis.

        Returns:
            loss (torch.Tensor):
                Scalar tensor, equivariance loss (reconstruction loss between transformed
                embeddings and reference embeddings), scaled by `self.equivariance_lambda`.
            info (Dict[str, Any]):
                Dictionary with metadata about the applied transformation:
                  - If collage path: includes "pairs", "modes", "paste_specs",
                    and optionally raw images + embeddings if
                    `return_collage_info=True`.
                  - If crop+flip path: includes "crop_px" (pixel coordinates),
                    "crop_patch" (patch grid coordinates), and "flip" (bool).

        Notes:
            - Collage path compares embeddings of direct collage images vs.
              reconstructed collages from component embeddings.
            - Crop+flip path compares embeddings of a cropped image vs.
              the corresponding region of the full-image embedding.
        """
        assert x.dim() == 4, "x must be (B,3,H,W)"
        B, _, H, W = x.shape
        ps = self.patch_size

        # -- Collage triplets (optional) --
        info: Dict[str, Any] = {}

        if (random.random() < p_collage) and (B >= 2):
            info["collage"] = True

            col = EmbeddingModel._build_collage_triplets(x, patch_size=ps)
            pairs = col["pairs"]
            modes = col["modes"]
            paste_specs = col["paste_specs"]
            imgs_a = col["imgs_a"]
            imgs_b = col["imgs_b"]
            imgs_collage = col["imgs_collage"]

            # Encode a, b, and direct collage images
            # (Batch once if imgs_collage is a tensor; else loop)
            z_a = self.encode(imgs_a)
            z_b = self.encode(imgs_b)
            if isinstance(imgs_collage, torch.Tensor):
                z_collage_dir = self.encode(imgs_collage)
            else:
                z_collage_dir = torch.stack(
                    [self.encode(ci) for ci in imgs_collage], dim=0
                )

            # Rebuild collage features from z_a/z_b
            z_collage_rec = EmbeddingModel._reconstruct_collage_features(
                z_a, z_b, modes, paste_specs
            )

            if weighted:
                self.eqv_decay_weights = self.eqv_decay_weights.to(z_collage_dir.device)
                wv = self.eqv_decay_weights.view(1, 1, 1, -1)
                z_collage_dir = z_collage_dir * wv
                z_collage_rec = z_collage_rec * wv

            # cosine sim only to prevent feature shrinkage
            loss = 1.0 - F.cosine_similarity(z_collage_dir, z_collage_rec, dim=-1)
            if reduction == "mean":
                loss = loss.mean()

            # Attach light info; add heavy tensors only if requested
            info.update(
                {
                    "pairs": pairs,
                    "modes": modes,
                    "paste_specs": paste_specs,
                }
            )
            if return_collage_info:
                info.update(
                    {
                        "imgs_a": imgs_a,
                        "imgs_b": imgs_b,
                        "imgs_collage": imgs_collage,
                        "z_a": z_a,
                        "z_b": z_b,
                        "z_collage_dir": z_collage_dir,
                        "z_collage_rec": z_collage_rec,
                    }
                )
        else:
            # -- Crop+flip path (unchanged) --
            # top, left, ch, cw = EmbeddingModel._sample_patch_aligned_crop_equal_area(ps,H, W, min_scale, max_scale, fixed_scale=fixed_scale, preserve_ar=preserve_ar)
            top, left, ch, cw = self.sample_crop_least_covered(
                H,
                W,
                min_scale,
                max_scale,
                fixed_scale=fixed_scale,
                preserve_ar=preserve_ar,
            )
            do_flip = random.random() < p_flip

            x_crop = x[:, :, top : top + ch, left : left + cw]
            if do_flip:
                x_crop = torch.flip(x_crop, dims=[3])

            z_full = self.encode(x)
            z_timg = self.encode(x_crop)

            top_p, left_p = top // ps, left // ps
            h_p, w_p = ch // ps, cw // ps
            z_ref = z_full[:, top_p : top_p + h_p, left_p : left_p + w_p, :]
            if do_flip:
                z_ref = torch.flip(z_ref, dims=[2])

            if weighted:
                self.eqv_decay_weights = self.eqv_decay_weights.to(z_timg.device)
                wv = self.eqv_decay_weights.view(1, 1, 1, -1)
                z_timg = z_timg * wv
                z_ref = z_ref * wv

            # cosine sim only to prevent feature shrinkage
            loss = 1.0 - F.cosine_similarity(z_timg, z_ref, dim=-1)
            if reduction == "mean":
                loss = loss.mean()

            info.update(
                {
                    "crop_px": (top, left, ch, cw),
                    "crop_patch": (top_p, left_p, h_p, w_p),
                    "flip": do_flip,
                }
            )

        return loss, info

    def save_ckpt(self, path: Union[str, Path], save_backbone_weights: bool = False):
        model_state_dict = self.state_dict()
        # drop the loss params
        model_state_dict = {
            k: v for k, v in model_state_dict.items() if "loss" not in k
        }
        # drop the backbone weights if specified
        if not save_backbone_weights:
            model_state_dict = {
                k: v for k, v in model_state_dict.items() if "backbone_model" not in k
            }
        state = {
            "backbone_model": self.backbone_model_name,
            "mrl_dims": self.mrl_dims,
            "internal_dim": self.internal_dim,
            "patch_size": self.patch_size,
            "transformer_dim": self.transformer_dim,
            "num_heads": self.num_heads,
            "embed_dim": self.embed_dim,
            "include_cls": self.include_cls,
            "identity_mode": self.identity_mode,
            "gram_lambda": self.gram_lambda,
            "cos_lambda": self.cos_lambda,
            "l2_lambda": self.l2_lambda,
            "equivariance_lambda": self.equivariance_lambda,
            "spacial_bias_lambda": self.spacial_bias_lambda,
            "spacial_corr_lambda": self.spacial_corr_lambda,
            "moments_lambda": self.moments_lambda,
            "dist_lambda": self.dist_lambda,
            "token_dropout_prob": self.token_dropout_prob,
        }
        path = str(path)
        json_path = path + ".json"
        with open(json_path, "w") as f:
            json.dump(state, f, indent=4)
        state["model_state_dict"] = model_state_dict
        torch.save(state, path + ".pth")

    @classmethod
    def load_ckpt(
        cls,
        path: Union[str, Path],
        pca_path: Optional[Union[str, Path]] = None,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
        feature_hack: bool = True,
    ):
        state = torch.load(path, map_location=device)

        # Extract init signature defaults
        sig = inspect.signature(cls.__init__)
        defaults = {
            k: v.default
            for k, v in sig.parameters.items()
            if v.default is not inspect._empty and k != "self"
        }

        # Start from defaults, override with checkpoint values
        init_args = {
            **defaults,
            **{k: v for k, v in state.items() if k in sig.parameters},
        }

        if pca_path is not None:
            init_args["pca_path"] = pca_path  # allow override or pass-through of PCA path

        model = cls(**init_args).to(device)
        model.load_state_dict(state["model_state_dict"], strict=False)
        model.to(dtype)

        if feature_hack:
            model.feature_hack = True

        return model

    @classmethod
    def from_backbone(
        cls,
        backbone: str = "dinov3",
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    ) -> "EmbeddingModel":
        """Create an EmbeddingModel in identity mode (raw backbone features, no adapter)."""
        model = cls(backbone_model=backbone, build_encoder_decoder=False, identity_mode=True)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model.to(device).to(dtype)


def test_gram_loss():
    B, W, H, C = 8, 512, 512, 3

    mrl = [2, 4, 8, 16, 32, 64, 128]
    avgs = {}
    for k in mrl:
        avgs[k] = 0.0

    total_time_model = 0.0
    total_time_gram = 0.0
    from datetime import datetime

    start_t = datetime.now()
    for i in range(100):
        img = torch.randn(B, C, H, W).cuda()
        start = datetime.now()
        with torch.no_grad():
            result = model.forward(img)
        end = datetime.now()
        total_time_model += (end - start).total_seconds()
        zs = result["ordered_embedding"]  # (B, H, W, Cs)
        zt = result["base_embedding"]  # (B, H, W, Ct
        start = datetime.now()
        for k in mrl:
            trunc = zs[..., :k]
            loss = model.gram_anchoring_loss(trunc, zt) * model.gram_lambda
            avgs[k] += loss.item()
        end = datetime.now()
        total_time_gram += (end - start).total_seconds()
    end_t = datetime.now()
    total_time = (end_t - start_t).total_seconds()
    print(
        f"Time for 100 iters: {total_time:.2f} sec ({100.0 / total_time:.2f}) iters/sec"
    )
    print(
        f"  Model forward time: {total_time_model:.2f} sec ({100.0 / total_time_model:.2f} iters/sec)"
    )
    print(
        f"  Gram loss time:     {total_time_gram:.2f} sec ({100.0 / total_time_gram:.2f} iters/sec)"
    )
    print(f"Compute for forward pass = {total_time_model / total_time * 100.0:.2f} %")
    print(f"Compute for gram loss = {total_time_gram / total_time * 100.0:.2f} %")

    print("Averages over 100 iters:")
    for k in mrl:
        avgs[k] /= 100.0
        print(f"k={k}: {avgs[k]:.6f}")


def test_sample_crop(sample_fn):
    H, W, ps = 512, 512, 16  # 8x8 tokens
    Hp, Wp = H // ps, W // ps

    # run many draws
    counts = torch.zeros((Hp, Wp), dtype=torch.int32)
    N = 10_000
    for _ in range(N):
        top, left, ch, cw = sample_fn(
            H, W, min_scale=0.2, max_scale=0.8, preserve_ar=False
        )
        top_p, left_p = top // ps, left // ps
        # mark all tokens covered by the crop
        counts[top_p : top_p + ch // ps, left_p : left_p + ch // ps] += 1

    # normalize to probabilities
    probs = counts.float() / counts.sum()

    print("Coverage probabilities over tokens:")
    print(probs)

    # check corners vs center
    corner = probs[0, 0].item()
    center = probs[Hp // 2, Wp // 2].item()

    print(f"corner prob {corner:.5f}, center prob {center:.5f}")
    # heuristic: corners should not be << center
    ratio = corner / center
    print(f"corner/center ratio = {ratio:.3f}")
    if ratio < 0.2:
        print("corners are badly underrepresented!")
    else:
        print("corner OK")


if __name__ == "__main__":
    model = EmbeddingModel()
    test_sample_crop(model.sample_crop_least_covered)
    test_sample_crop(model._sample_patch_aligned_crop_equal_area)
