# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from diffusers V 0.35.2
#   (https://github.com/huggingface/diffusers)
# Copyright 2025 The HuggingFace Inc. team. All rights reserved., licensed
# under the Apache-2.0 license, cf. 3rd-party-licenses.txt file in the root
# directory of this source tree.

import math
import torch
import torch.nn.functional as F
from torchvision import transforms
from diffusers.models import FluxControlNetModel
from diffusers import (
    FluxPipeline,
    FlowMatchEulerDiscreteScheduler,
    FluxControlNetPipeline,
)
from typing import Optional, Union, Sequence, List, Literal, Tuple, Dict
from dataclasses import dataclass, field
from torch.utils.data import DataLoader, DistributedSampler
import debugpy
import datetime
from tqdm import tqdm
from accelerate import Accelerator
from torch import nn
from torch.optim import Optimizer
from pathlib import Path
from logging import Logger
from collections import deque

from TASE.image_gen.dimensionality_reduction.embedding_model import EmbeddingModel
from TASE.image_gen.common.util import *
from TASE.image_gen.controlnet.splat_ctrl_dataset import (
    SplatCtrlDataset,
    splat_ctrl_collate,
)
from TASE.splatting.common.rendering import norm_channel_wise, norm_across_channels
from TASE.splatting.common.generative_helper import (
    generate_conditional_image,
    mask_suffix_channels,
)

import os
import datetime
from dataclasses import asdict as dc_asdict
from pathlib import Path
from typing import List, Tuple, Optional, Literal
import torch
import yaml


# === Config ===
def _default_results_root() -> Path:
    result_root = Path("./OUTPUT")
    if os.environ.get("TASE_ROOT") is not None:
        result_root = Path(os.environ["TASE_ROOT"]) / "OUTPUT" / "ControlNet"
    return result_root


def _expand_env_path(p: Optional[Path]) -> Optional[Path]:
    if p is None:
        return None
    return Path(os.path.expandvars(str(p))).expanduser()


@dataclass
class Config:
    # -------- I/O --------
    dataset_root: Path = "$IMGNET_ROOT"
    result_dir: Path = field(default_factory=_default_results_root)
    log_dir: Optional[Path] = None
    tb_dir: Optional[Path] = None
    ckpt_dir: Optional[Path] = None
    run_id: str = field(
        default_factory=lambda: datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    # -------- Training --------
    stages: List[Tuple[int, int, int]] = field(
        default_factory=lambda: [(512, 1, 1_500_000)]
    )
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 1e-3
    lr_scheduler: Literal["constant", "cosine"] = "cosine"
    lr_warmup_steps: int = 500
    lr_num_cycles: int = 1
    lr_power: float = 1.0
    max_grad_norm: float = 1.0

    # -------- Precision --------
    dtype: torch.dtype = torch.bfloat16

    # -------- Control signal --------
    ctrl_backbone: Literal["dino", "dinov3", "vae", "gsam2", "none"] = "dino"
    max_control_channels: Optional[int] = 64
    guidance_dim_dropout_start_iter: Optional[int] = 100_000
    dim_reduction: Optional[Literal["pca", "mrl", "none"]] = "mrl"
    pca_path: Optional[Path] = f"$TASE_ROOT/OUTPUT/PCA/dinov3/torch_ipca.pkl"
    use_pca:bool = False
    mrl_embedding_model_path: Optional[Path] = None
    raw_dino: bool = False
    feature_hacks: bool = False
    gsam2_root: Optional[Path] = None
    colormap_path: Optional[Path] = None
    splat_ctrl_finetuning: bool = False
    splat_ctrl_finetune_transformer: bool = True
    splat_ctrl_lora_training: bool = False
    splat_ctrl_transformer_lr: float = 1e-5
    splat_ctrl_base_noise_level = 0.2

    # -------- Data / misc --------
    use_tiny_imagenet: bool = False
    num_workers: int = 4
    rand_seed: int = 42
    launch_tensorboard: bool = True
    tb_host: str = "0.0.0.0"
    tb_port: int = 6006
    debug: bool = field(default_factory=lambda: debugpy.is_client_connected())

    # -------- Guidance / logging knobs --------
    n_running_average: int = 1000
    dynamic_q: Optional[float] = None
    dynamic_q_lambda: Optional[float] = None
    text_guidance_scale: float = 3.5
    val_ctrl_guidance_scale: float = 0.65
    text_guidance_dropout: float = 0.5
    val_loss_frequency: int = 50
    val_image_frequency: int = 1000
    val_image_truncation: int = 8
    checkpoint_frequency: int = 5000
    print_frequency: int = 100
    _bucket_size: int = 8
    _local_batch: int = 1
    _global_batch: int = 1

    # -------- Checkpointing / bookkeeping --------
    resume_from_ckpt: Optional[Path] = None
    resume_transformer_from_ckpt: Optional[Path] = None
    resume_global_step: int = 0
    total_training_steps: int = -1

    # ---- lifecycle helpers ----
    def finalize(self) -> None:
        # Expand env vars in critical paths
        self.dataset_root = _expand_env_path(self.dataset_root)
        self.result_dir = _expand_env_path(self.result_dir) or Path("./OUTPUT/CtrlNet")

        if self.splat_ctrl_finetuning:
            self.result_dir = self.result_dir / "splat_finetune"
            self.guidance_dim_dropout_start_iter = (
                0  # always enable dim dropout for splat finetuning
            )
            if not self.splat_ctrl_lora_training:
                # reduce the learning rates for full finetuning
                self.learning_rate = self.learning_rate / 10.0
                self.splat_ctrl_transformer_lr = self.learning_rate / 10.0
            else:
                # keep the learning rate for LoRA finetuning
                self.splat_ctrl_transformer_lr = self.learning_rate

        # --- Raw DINO mode: override control signal settings ---
        if self.raw_dino:
            self.dim_reduction = "none"
            self.ctrl_backbone = "dinov3"
            if self.max_control_channels == 64:  # only override default
                self.max_control_channels = 768
            self.guidance_dim_dropout_start_iter = None  # no MRL truncation dropout

        # Compute total steps (if not explicitly set)
        if self.total_training_steps is None or self.total_training_steps <= 0:
            self.total_training_steps = sum(steps for _, _, steps in self.stages)

        # Build derived directories from run_id
        self.result_dir = self.result_dir / self.run_id
        self.ckpt_dir = self.result_dir / "ckpts"
        self.log_dir = self.result_dir / "logs"
        self.tb_dir = self.result_dir / "tb"

        # create a list of MRL truncation levels to sample from
        if self.dim_reduction == "mrl":
            self.mrl_levels = []
            level = 2
            while level <= self.max_control_channels:
                self.mrl_levels.append(level)
                level *= 2

    def asdict(self, *, for_tracker: bool = False) -> dict:
        d = dc_asdict(self)

        # Normalize paths to strings
        for k in (
            "dataset_root",
            "result_dir",
            "log_dir",
            "tb_dir",
            "ckpt_dir",
            "pca_path",
            "mrl_embedding_model_path",
            "gsam2_root",
            "colormap_path",
            "resume_from_ckpt",
            "resume_transformer_from_ckpt",
        ):
            if d.get(k) is not None:
                d[k] = str(d[k])

        # Make tracker-friendly values
        d["dtype"] = str(self.dtype).replace("torch.", "")  # e.g., "bfloat16"
        d["stages"] = ";".join(
            f"{r},{b},{s}" for r, b, s in self.stages
        )  # "512,2,1281167;..."

        if not for_tracker:
            return d

        # Accelerate/TensorBoard accept only basic primitives
        out = {}
        for k, v in d.items():
            if isinstance(v, (int, float, str, bool)):
                out[k] = v
            elif v is None:
                out[k] = "None"
            else:
                out[k] = str(v)
        return out

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load config values from a YAML file into a Config instance."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # Convert path-like strings to Path objects where relevant
        path_keys = {
            "dataset_root",
            "result_dir",
            "log_dir",
            "tb_dir",
            "ckpt_dir",
            "pca_path",
            "mrl_embedding_model_path",
            "gsam2_root",
            "colormap_path",
            "resume_from_ckpt",
        }
        for k in path_keys:
            if k in data and data[k] is not None:
                data[k] = Path(data[k])

        # Convert dtype string back to torch dtype if provided
        if "dtype" in data and isinstance(data["dtype"], str):
            data["dtype"] = getattr(torch, data["dtype"])

        # Convert stages if given as string
        if "stages" in data and isinstance(data["stages"], str):
            data["stages"] = [
                tuple(map(int, part.split(","))) for part in data["stages"].split(";")
            ]

        return cls(**data)


class RunningAverage:
    def __init__(self, window_size: int):
        self.buf = deque(maxlen=window_size)

    def update(self, loss: float):
        self.buf.append(loss)

    @property
    def avg(self) -> float:
        return sum(self.buf) / len(self.buf) if self.buf else 0.0


# === Helpers ===
def get_sigmas(idx, noise_scheduler, device, dtype):
    sigma = noise_scheduler.sigmas[idx].flatten().to(device=device, dtype=dtype)
    while sigma.ndim < 3:
        sigma = sigma.unsqueeze(-1)
    return sigma


def encode_prompt(prompts, pipe):
    return pipe.encode_prompt(prompts, prompt_2=prompts)


def get_timestep_idx(
    noise_frac: float | torch.Tensor, noise_scheduler: FlowMatchEulerDiscreteScheduler
):
    """
    Map noise fraction f∈[0,1] (0=clean, 1=max noise) to the scheduler index.
    If f=0, return the last index (max noise); if f=1, return the first index (min noise).
    """
    if isinstance(noise_frac, float):
        noise_frac = torch.tensor([noise_frac]).unsqueeze(0)
    f = noise_frac.clamp(0.0, 1.0)
    n = len(noise_scheduler.timesteps) - 1
    # assume sigmas are descending (σ_max -> σ_min), f=1 -> idx=0; f=0 -> idx=n
    idx = ((1.0 - f) * n).long().clamp(0, n).cpu()  # insdices should always be on CPU
    return idx


def prepare_latents(
    config: Config,
    input_imgs,
    target_imgs,
    vae,
    noise_scheduler,
    device,
    dtype,
    noise_frac=None,
    generator=None,
    normalize=True,
):
    # helper
    def encode(imgs, vae, device, dtype, normalize):
        # imgs: [B, 3, H, W] in [0,1]
        if normalize:
            # normalize for vae
            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            imgs = normalize(imgs)
        imgs = imgs.to(device=device, dtype=dtype)
        lat = vae.encode(imgs).latent_dist.sample()
        lat = (lat - vae.config.shift_factor) * vae.config.scaling_factor
        lat = lat.to(device=device, dtype=dtype)
        return lat

    if config.splat_ctrl_finetuning:
        # encode both input and target images, then add additional noise, returned noise is the difference between noisy latent and target latent
        assert (
            input_imgs is not None
        ), "input_imgs must be provided for splat finetuning"
        lat = encode(target_imgs, vae, device, dtype, normalize)
        lat_in = encode(input_imgs, vae, device, dtype, normalize)
        assert (
            lat_in.shape == lat.shape
        ), f"Input and target latents must have the same shape, got {lat_in.shape} and {lat.shape}"
        B, C, H, W = lat.shape
        noise = torch.randn(lat.shape, device=device, dtype=dtype, generator=generator)
        noise = FluxControlNetPipeline._pack_latents(noise, B, C, H, W)
        packed = FluxControlNetPipeline._pack_latents(lat, B, C, H, W)
        packed_in = FluxControlNetPipeline._pack_latents(lat_in, B, C, H, W)
        if noise_frac is None:
            # for training: randomly sample noise index
            noise_frac = torch.rand((1,), device=device)
        # high frac -> low idx -> high sigma, high noise
        noise_idx = get_timestep_idx(noise_frac, noise_scheduler)
        noise_idx = noise_idx.expand(B)
        # handle the assumed base noise fraction for splat finetuning
        diffusion_frac = (
            noise_frac * (1.0 - config.splat_ctrl_base_noise_level)
            + config.splat_ctrl_base_noise_level
        )
        diffusion_idx = get_timestep_idx(diffusion_frac, noise_scheduler)
        diffusion_idx = diffusion_idx.expand(B)
        # create the noisy input latent
        sig = get_sigmas(noise_idx, noise_scheduler, device, dtype)
        noisy = (1 - sig) * packed_in + sig * noise
        # The new target noise is the difference between noisy and target latent
        noise = noisy - packed
        t = noise_scheduler.timesteps[diffusion_idx].to(device=device, dtype=dtype)
        sig = get_sigmas(diffusion_idx, noise_scheduler, device, dtype)
    else:
        # only encode the target image which is assumed to be identical to the input image
        lat = encode(target_imgs, vae, device, dtype, normalize)
        B, C, H, W = lat.shape
        noise = torch.randn(lat.shape, device=device, dtype=dtype, generator=generator)
        noise = FluxControlNetPipeline._pack_latents(noise, B, C, H, W)
        packed = FluxControlNetPipeline._pack_latents(lat, B, C, H, W)
        if noise_frac is None:
            # for training: randomly sample noise index
            idx = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (B,),
                device=noise_scheduler.timesteps.device,
            )
        else:
            # high frac -> low idx -> high sigma, high noise
            idx = get_timestep_idx(noise_frac, noise_scheduler)
            idx = idx.expand(B)
        t = noise_scheduler.timesteps[idx].to(device=device, dtype=dtype)
        # noise = torch.randn(packed.shape, device=device, dtype=dtype, generator=generator)
        sig = get_sigmas(idx, noise_scheduler, device, dtype)
        noisy = (1 - sig) * packed + sig * noise
    return lat, packed, noisy, noise, t, sig


def build_control_map(imgs, lbls, embedder, config: Config) -> torch.Tensor:
    """
    Build control map from images using DINO or VAE.
    Args:
        imgs: Tensor of shape (B, 3, H, W).
        dino: DINO model.
        vae: VAE model.
        pca: PCA model.
    Returns:
        ctrl: Tensor of shape (B, C, H, W).
    """
    device, dtype = imgs.device, imgs.dtype
    # The model handles reshaping and normalization
    if isinstance(embedder, EmbeddingModel):
        feats = embedder.encode(imgs, use_pca=config.use_pca)  # [B, H, W, C]
        ctrl = (
            feats[..., : config.max_control_channels]
            if config.max_control_channels is not None
            else feats
        )
        if config.feature_hacks:
            # crop out the outer 3% of the feature map to remove border artifacts
            H, W = ctrl.shape[1], ctrl.shape[2]
            # crop_h = int(H * 0.03); crop_w = int(W * 0.03)
            crop_h = 1
            crop_w = 1
            ctrl = ctrl[:, crop_h : H - crop_h, crop_w : W - crop_w, :]
            # apply the equivalent crop to the input image as well
            ps = embedder.patch_size
            img_h, img_w = imgs.shape[2], imgs.shape[3]
            crop_img_w = int(crop_w * ps)
            crop_img_h = int(crop_h * ps)
            imgs = imgs[
                :, :, crop_img_h : img_h - crop_img_h, crop_img_w : img_w - crop_img_w
            ]
    else:
        # semseg model takes images in [0,1] and labels
        feats = embedder.encode((imgs, lbls))  # [B, C, H, W]
    ctrl = ctrl.permute(0, 3, 1, 2)  # (B, C, H, W)
    ctrl.to(device=device, dtype=dtype)
    return ctrl, imgs


def sample_truncation(config: Config) -> torch.Tensor:
    # select the mrl level
    mrl_index = torch.randint(0, len(config.mrl_levels), (1,)).item()
    mrl_level = config.mrl_levels[mrl_index]
    # add an offset between the selected level and the next one
    if mrl_index + 1 < len(config.mrl_levels):
        next_level = config.mrl_levels[mrl_index + 1]
        offset = torch.randint(0, next_level - mrl_level, (1,)).item()
        mrl_level = mrl_level + offset
    return config.max_control_channels - mrl_level


def forward_step(
    accelerator: Accelerator,
    config: Config,
    target_imgs,
    input_imgs,
    ctrl,
    labels,
    pipe,
    vae,
    transformer,
    controlnet,
    noise_scheduler,
    ctrl_embedder: EmbeddingModel,
    device,
    dtype,
    class_map,
    guidance_scale,
    ctrl_mask: Optional[Union[int, torch.Tensor, Sequence[int]]] = None,
    drop_txt: Optional[float] = None,
    reduction: str = "mean",
):
    # Prepare the model inputs
    with torch.no_grad():
        # Prepare the model inputs
        target_imgs = target_imgs.to(device, dtype=dtype)
        B = target_imgs.shape[0]
        # Text
        labels = [
            f"A photo of a {class_map[l.item()]}" if isinstance(l, torch.Tensor) else l
            for l in labels
        ]
        if drop_txt:
            rand_f = torch.rand((B,), device=device)
            prompts = ["" if rand_f[i] < drop_txt else l for i, l in enumerate(labels)]
            guidance = torch.full((B,), guidance_scale, device=device, dtype=dtype)
            guidance = torch.where(
                rand_f < drop_txt, torch.zeros_like(guidance), guidance
            )
        else:
            prompts = labels
            guidance = torch.full((B,), guidance_scale, device=device, dtype=dtype)
        prompt_emb, pooled, txt_ids = encode_prompt(prompts, pipe)
        # ControlNet conditioning
        # NOTE: The image may be cropped compared to the original input image if the feature hacks are enabled!
        # Build the control map before preparing the latents, so we can crop the input image as well
        if ctrl is None:
            ctrl, target_imgs = build_control_map(
                target_imgs, labels, ctrl_embedder, config
            )
        # Latents
        lat, packed, noisy, noise, timesteps, sigma = prepare_latents(
            config, input_imgs, target_imgs, vae, noise_scheduler, device, dtype
        )
        if ctrl.shape == target_imgs.shape:
            # Image aligned control map -> pass through VAE
            ctrl = ctrl.to(device=device, dtype=dtype)
            ctrl = vae.encode(ctrl).latent_dist.sample()
            ctrl = (ctrl - vae.config.shift_factor) * vae.config.scaling_factor
        _, _, H_lat, W_lat = lat.shape
        if ctrl.shape[2] != H_lat or ctrl.shape[3] != W_lat:
            ctrl = F.interpolate(
                ctrl, size=(H_lat, W_lat), mode="bilinear", align_corners=False
            )
        if ctrl_mask:
            ctrl = mask_suffix_channels(ctrl, num_channels=ctrl_mask)
        image_ids = FluxControlNetPipeline._prepare_latent_image_ids(
            noisy.shape[0], lat.shape[2] // 2, lat.shape[3] // 2, device, dtype
        )

        ctrl = FluxControlNetPipeline._pack_latents(ctrl, *ctrl.shape)
        ctrl = ctrl.to(device=device, dtype=dtype)

    # The actual forward pass
    with accelerator.autocast():
        samples, single = controlnet(
            hidden_states=noisy,
            timestep=timesteps / 1000,
            encoder_hidden_states=prompt_emb,
            pooled_projections=pooled,
            txt_ids=txt_ids,
            img_ids=image_ids,
            guidance=guidance,
            return_dict=False,
            controlnet_cond=ctrl,
        )
        pred = transformer(
            hidden_states=noisy,
            timestep=timesteps / 1000,
            encoder_hidden_states=prompt_emb,
            pooled_projections=pooled,
            txt_ids=txt_ids,
            img_ids=image_ids,
            guidance=guidance,
            return_dict=False,
            controlnet_block_samples=(
                [s.to(dtype=dtype) for s in samples] if samples else None
            ),
            controlnet_single_block_samples=(
                [s.to(dtype=dtype) for s in single] if single else None
            ),
        )[0]
    loss = F.mse_loss(pred, (noise - packed), reduction=reduction)
    return loss


def train_loop(
    config: Config,
    accelerator: Accelerator,
    logger: Logger,
    # data
    train_dataset: Union[ImageNetDataset, SplatCtrlDataset],
    val_dataset: Union[ImageNetDataset, SplatCtrlDataset],
    class_map: Dict[int, str],
    # components
    pipe: FluxPipeline,
    vae: nn.Module,
    transformer: nn.Module,
    controlnet: FluxControlNetModel,
    optimizer: Optimizer,
    lr_scheduler,
    transformer_optimizer: Optional[Optimizer],
    transformer_lr_scheduler,
    noise_scheduler: FlowMatchEulerDiscreteScheduler,
    ctrl_embedder: EmbeddingModel,
    # other
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """
    Single training loop that supports both ImageNet and SplatCtrlDataset.
    1. For ImageNetDataset, ctrl_embedder is used to build control maps from target images.
    2. For SplatCtrlDataset, ctrl_embedder is not used, and the dataset provides control maps directly.
    """
    # compute total global images
    total_images = sum(tot for _, _, tot in config.stages)
    total_steps = math.ceil(
        total_images / (config.stages[0][1] * accelerator.num_processes)
    )
    pbar = tqdm(total=total_images, desc="Training", disable=None)

    # tensorboard writer (only on main)
    tb_writer = (
        accelerator.get_tracker("tensorboard").writer
        if accelerator.is_main_process
        else None
    )

    logger.info(f"Starting training on {accelerator.num_processes} devices")
    global_step = config.resume_global_step
    best_val_loss = float("inf")

    # Fast-forward LR scheduler(s) to match resumed step
    if global_step > 0:
        # Compute global batch from the first stage (used for step counting)
        first_batch = config.stages[0][1] * accelerator.num_processes
        resume_iterations = global_step // first_batch
        logger.info(
            f"Resuming from global_step={global_step} "
            f"(fast-forwarding LR scheduler by {resume_iterations} steps)"
        )
        for _ in range(resume_iterations):
            lr_scheduler.step()
        if transformer_lr_scheduler is not None:
            for _ in range(resume_iterations):
                transformer_lr_scheduler.step()
        pbar.update(global_step)

    for stage_idx, (res, local_batch, stage_images) in enumerate(
        config.stages, start=1
    ):
        # update batch sizes
        config._local_batch = local_batch
        config._global_batch = local_batch * accelerator.num_processes
        logger.info(
            f"Stage {stage_idx}: {res}×{res}, global_batch={config._global_batch}, images={stage_images}"
        )

        # build loaders
        seed = config.rand_seed + accelerator.process_index
        # training loader
        train_dataset.set_resolution(res)
        train_loader = DataLoader(
            train_dataset,
            batch_size=local_batch,
            shuffle=True,
            collate_fn=(
                splat_ctrl_collate
                if isinstance(train_dataset, SplatCtrlDataset)
                else None
            ),
            generator=torch.Generator(device="cpu").manual_seed(seed),
            num_workers=config.num_workers,
            pin_memory=True,
        )
        train_loader = accelerator.prepare(train_loader)

        # validation loader
        val_dataset.set_resolution(res)
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=True,
            seed=config.rand_seed,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=local_batch,
            collate_fn=(
                splat_ctrl_collate
                if isinstance(val_dataset, SplatCtrlDataset)
                else None
            ),
            sampler=val_sampler,
            num_workers=config.num_workers,
            pin_memory=True,
        )
        val_loader = accelerator.prepare(val_loader)
        val_iter = iter(val_loader)

        # running averages and action counters
        train_avg = RunningAverage(config.n_running_average)
        val_avg = RunningAverage(config.n_running_average)
        action = dict(val_loss=0, val_img=0, print=0, ckpt=config.checkpoint_frequency)

        # optional static validation sample for images
        if config.val_image_frequency and accelerator.is_main_process:
            # grab one sample from validation
            # NOTE: This is an unbatched, single sample!
            if isinstance(train_dataset, SplatCtrlDataset):
                batch = val_dataset[1]
                target_img0 = batch["clean"].to(device)
                input_img0 = batch["artifact"].to(device)
                ctrl0 = batch["feature"].to(device)
                lbl0 = batch["label"]
                lbl0 = [f"A photo of a {class_map[lbl0]}"]
                print(
                    f"Validation sample: {lbl0} shape: {target_img0.shape}, Control shape : {ctrl0.shape}"
                )
                ctrl0 = ctrl0.to(device).to(dtype)
            else:
                batch = val_dataset[10_042]
                target_img0, lbl0 = batch
                ctrl0, target_img0 = build_control_map(
                    target_img0.unsqueeze(0).to(device).to(dtype),
                    lbl0,
                    ctrl_embedder,
                    config,
                )
                target_img0 = target_img0.squeeze(0)
                input_img0 = target_img0.clone().to(device)
                ctrl0 = ctrl0.squeeze(0).to(device).to(dtype)
                lbl0 = [f"A photo of a {class_map[lbl0]}"]

            target_img0 = target_img0.to(device).to(dtype)
            # Add to tensorboard
            if tb_writer is not None:
                tb_writer.add_image(
                    f"val/img0_{lbl0}", target_img0 / 2.0 + 0.5, global_step=global_step
                )
                if input_img0 is not None:
                    tb_writer.add_image(
                        f"val/img0_input_{lbl0}",
                        input_img0 / 2.0 + 0.5,
                        global_step=global_step,
                    )
                if ctrl0 is not None:
                    ctrl_img = norm_channel_wise(ctrl0[:3].permute(1, 2, 0)).permute(
                        2, 0, 1
                    )  # [C, H, W]
                    tb_writer.add_image(
                        f"val/ctrl0_{lbl0}", ctrl_img, global_step=global_step
                    )

        # how many steps in this stage
        steps = math.ceil(stage_images / config._global_batch)
        step = 0

        # Skip already-completed steps when resuming
        if config.resume_global_step > 0:
            steps_to_skip = config.resume_global_step // config._global_batch
            # Account for images consumed by prior stages
            prior_images = sum(s for _, _, s in config.stages[:stage_idx - 1])
            steps_to_skip -= prior_images // config._global_batch
            if steps_to_skip >= steps:
                # This entire stage was already completed
                logger.info(
                    f"Stage {stage_idx}: skipping entirely (already completed)"
                )
                continue
            elif steps_to_skip > 0:
                step = steps_to_skip
                logger.info(
                    f"Stage {stage_idx}: skipping {steps_to_skip} steps "
                    f"(resuming at step {step}/{steps})"
                )

        while step < steps:
            for batch in train_loader:
                if step >= steps:
                    break

                # unpack batch
                if isinstance(train_dataset, SplatCtrlDataset):
                    target_imgs = batch["clean"].to(device).to(dtype)
                    input_imgs = batch["artifact"].to(device).to(dtype)
                    ctrls = batch["feature"].to(device).to(dtype)
                    lbls = batch["label"]
                    lbls = [f"A photo of a {class_map[l]}" for l in lbls]
                else:
                    target_imgs, lbls = batch
                    input_imgs = None
                    lbls = [f"A photo of a {class_map[l.item()]}" for l in lbls]
                    target_imgs = target_imgs.to(device).to(dtype)
                    ctrls = None

                # optional control-channel dropout
                dim_dropout_start = getattr(
                    config, "guidance_dim_dropout_start_iter", 0
                )
                if (
                    config.max_control_channels
                    and dim_dropout_start is not None
                    and global_step > dim_dropout_start
                    and hasattr(config, "mrl_levels")
                ):
                    truncate = sample_truncation(config)
                else:
                    truncate = None

                # forward/backward
                loss = forward_step(
                    accelerator,
                    config,
                    target_imgs,
                    input_imgs,
                    ctrls,
                    lbls,
                    pipe,
                    vae,
                    transformer,
                    controlnet,
                    noise_scheduler,
                    ctrl_embedder,
                    device,
                    dtype,
                    class_map,
                    config.text_guidance_scale,
                    ctrl_mask=truncate,
                    drop_txt=config.text_guidance_dropout,
                )
                accelerator.backward(loss)

                # gradient clipping & step
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        controlnet.parameters(), config.max_grad_norm
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                if config.splat_ctrl_finetuning and transformer_optimizer is not None:
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            transformer.parameters(), config.max_grad_norm
                        )
                    transformer_optimizer.step()
                    transformer_lr_scheduler.step()
                    transformer_optimizer.zero_grad()

                # log train loss
                if tb_writer is not None:
                    train_avg.update(loss.item())
                    accelerator.log({"train/loss": loss.item()}, step=global_step)
                    dim = config.max_control_channels - (
                        truncate if truncate is not None else 0
                    )
                    dim = dim // 4 * 4
                    if truncate is not None:
                        tb_writer.add_scalars(
                            f"train", {f"dim_{dim:04d}": loss.item()}, global_step
                        )

                # periodic validation loss
                if (
                    config.val_loss_frequency
                    and step * config._global_batch > action["val_loss"]
                ):
                    action["val_loss"] += config.val_loss_frequency
                    controlnet.eval()
                    with torch.no_grad():
                        try:
                            val_batch = next(val_iter)
                        except StopIteration:
                            val_iter = iter(val_loader)
                            val_batch = next(val_iter)
                        # unpack validation batch
                        if isinstance(val_dataset, SplatCtrlDataset):
                            v_target_imgs = val_batch["clean"].to(device).to(dtype)
                            v_input_imgs = val_batch["artifact"].to(device).to(dtype)
                            v_ctrls = val_batch["feature"].to(device).to(dtype)
                            v_lbls = val_batch["label"]
                            v_lbls = [f"A photo of a {class_map[l]}" for l in v_lbls]
                        else:
                            v_target_imgs, v_lbls = val_batch
                            v_lbls = [
                                f"A photo of a {class_map[l.item()]}" for l in v_lbls
                            ]
                            v_target_imgs = v_target_imgs.to(device).to(dtype)
                            v_input_imgs = None
                            v_ctrls = None
                        val_loss = forward_step(
                            accelerator,
                            config,
                            v_target_imgs,
                            v_input_imgs,
                            v_ctrls,
                            v_lbls,
                            pipe,
                            vae,
                            transformer,
                            controlnet,
                            noise_scheduler,
                            ctrl_embedder,
                            device,
                            dtype,
                            class_map,
                            config.text_guidance_scale,
                            ctrl_mask=truncate,
                        )
                    all_losses = accelerator.gather(val_loss.unsqueeze(0))
                    avg_loss = all_losses.mean().item()
                    if tb_writer is not None:
                        dim = config.max_control_channels - (
                            truncate if truncate is not None else 0
                        )
                        dim = dim // 4 * 4
                        tb_writer.add_scalar("val/loss", avg_loss, global_step)
                        if truncate is not None:
                            tb_writer.add_scalars(
                                f"val", {f"dim_{dim:04d}": avg_loss}, global_step
                            )
                    val_avg.update(avg_loss)
                    controlnet.train()

                # periodic validation image
                if (
                    config.val_image_frequency
                    and step * config._global_batch > action["val_img"]
                    and tb_writer is not None
                ):
                    action["val_img"] += config.val_image_frequency
                    ctrl_trunc = (
                        mask_suffix_channels(
                            ctrl0,
                            num_channels=config.max_control_channels
                            - config.val_image_truncation,
                        )
                        if truncate
                        else ctrl0
                    )
                    controlnet.eval()
                    if config.splat_ctrl_finetuning:
                        img_guidance = input_img0
                    else:
                        img_guidance = None
                    with torch.no_grad():
                        # reuse static img0, ctrl0 from above
                        pred = generate_conditional_image(
                            pipe,
                            transformer,
                            controlnet,
                            FlowMatchEulerDiscreteScheduler.from_config(
                                noise_scheduler.config
                            ),
                            prompts=lbl0,
                            ctrl=ctrl_trunc,
                            height=res,
                            width=res,
                            num_inference_steps=28,
                            rgb_guide_image=img_guidance,
                            assumed_noise_level=(
                                config.splat_ctrl_base_noise_level
                                if config.splat_ctrl_finetuning
                                else 0.0
                            ),
                            noise_frac=0.9,
                            text_guidance_scale=config.text_guidance_scale,
                            ctrl_guidance_scale=config.val_ctrl_guidance_scale,
                            device=device,
                            dtype=dtype,
                            seed=config.rand_seed,
                        )
                    try:
                        caption = (
                            f"val/pred_{lbl0}_{config.val_image_truncation}"
                            if truncate
                            else f"val/pred_{lbl0}_full"
                        )
                        tb_writer.add_image(caption, pred[0], global_step)
                    except Exception as e:
                        print(
                            f"[warn] Could not add validation image to tensorboard: {e}. Disableing image validation."
                        )
                        config.val_image_frequency = None
                    controlnet.train()

                # checkpointing
                if (
                    config.checkpoint_frequency
                    and step * config._global_batch > action["ckpt"]
                    and accelerator.is_main_process
                ):
                    action["ckpt"] += config.checkpoint_frequency
                    # save best or latest
                    tag = "best" if val_avg.avg < best_val_loss else "latest"
                    best_val_loss = min(val_avg.avg, best_val_loss)
                    save_models(
                        accelerator,
                        config,
                        logger,
                        controlnet,
                        transformer,
                        desc=f"{tag}_stage_{stage_idx}_res_{res}",
                    )

                # print progress
                if (
                    config.print_frequency
                    and step * config._global_batch > action["print"]
                ):
                    action["print"] += config.print_frequency
                    logger.info(
                        f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}  "
                        f"Stage {stage_idx} step {step}/{steps}, "
                        f"train_loss={train_avg.avg:.4f}, val_loss={val_avg.avg:.4f}"
                    )

                # step counters
                if accelerator.is_main_process:
                    global_step += config._global_batch
                step += 1
                pbar.update(config._global_batch)

        # final checkpoint at end of stage
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            save_models(
                accelerator,
                config,
                logger,
                controlnet,
                transformer,
                desc=f"final_stage_{stage_idx}_res_{res}",
            )

    accelerator.end_training()
    pbar.close()
    logger.info("Training complete.")


def save_models(
    accelerator: Accelerator,
    config: Config,
    logger: Logger,
    controlnet: FluxControlNetModel,
    transformer: nn.Module,
    desc: str,
):
    # final checkpoint at end of stage
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = Path(config.ckpt_dir) / f"controlnet_{desc}"
        unwrapped = accelerator.unwrap_model(controlnet)
        if config.splat_ctrl_lora_training:
            unwrapped.save_lora_adapter(final_dir, adapter_name="splat_ctrl_lora")
            logger.info(f"Saved controlnet LoRA adapter to {final_dir}")
        else:
            unwrapped.save_pretrained(
                final_dir, max_shard_size="16GB", save_function=accelerator.save
            )
            logger.info(f"Saved controlnet checkpoint to {final_dir}")
        if config.splat_ctrl_finetuning and config.splat_ctrl_finetune_transformer:
            # also save the transformer
            transformer_dir = Path(config.ckpt_dir) / f"transformer_{desc}"
            unwrapped_t = accelerator.unwrap_model(transformer)
            if config.splat_ctrl_lora_training:
                unwrapped_t.save_lora_adapter(
                    transformer_dir, adapter_name="splat_ctrl_lora"
                )
                logger.info(f"Saved transformer LoRA adapter to {transformer_dir}")
            else:
                unwrapped_t.save_pretrained(
                    transformer_dir,
                    max_shard_size="32GB",
                    save_function=accelerator.save,
                )
                logger.info(f"Saved transformer checkpoint to {transformer_dir}")


if __name__ == "__main__":
    # sample 1000 random truncation levels
    config = Config()
    config.finalize()
    levels = [sample_truncation(config) for _ in range(1000)]
    # print statistics
    import numpy as np

    arr = np.array(levels)
    print(
        f"Levels: min {arr.min()}, max {arr.max()}, mean {arr.mean():.1f}, median {np.median(arr)}, std {arr.std():.1f}"
    )
    import matplotlib.pyplot as plt

    plt.hist(arr, bins=64)
    plt.title("Sampled truncation levels")
    plt.xlabel("Truncation level")
    plt.ylabel("Count")
    plt.savefig("truncation_levels.png")
