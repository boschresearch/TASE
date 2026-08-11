# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import threading
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch import nn
from accelerate import Accelerator
from accelerate.logging import get_logger
from transformers import Adafactor
from diffusers import FluxControlPipeline, FlowMatchEulerDiscreteScheduler
from diffusers.models import FluxControlNetModel, AutoencoderKL, FluxTransformer2DModel
from diffusers.models.controlnet import zero_module
from diffusers.optimization import get_scheduler
from diffusers.models.model_loading_utils import load_state_dict
import yaml
import os

# --- Project imports (local) ---
from TASE.image_gen.controlnet.train_util import train_loop, Config
from TASE.image_gen.common.util import launch_tensorboard
from TASE.image_gen.dimensionality_reduction.embedding_model import EmbeddingModel
from TASE.image_gen.common.util import ImageNetDataset

# from TASE.image_gen.controlnet.semseg.grounded_sam2_helper import SemSeg_Wrapper
# from TASE.image_gen.controlnet.semseg.segmentation_datasets import LVISDataset


# ============================
# Helpers
# ============================
def parse_stages(stage_args: List[str]) -> List[Tuple[int, int, int]]:
    """Parse stages passed as repeated --stages R,B,S or a single 'R,B,S'."""
    stages: List[Tuple[int, int, int]] = []
    for s in stage_args:
        try:
            r, b, n = (int(x) for x in s.split(","))
            stages.append((r, b, n))
        except Exception as e:
            raise argparse.ArgumentTypeError(
                f"Invalid --stages entry '{s}'. Expected 'res,batch,steps'"
            ) from e
    return stages


def str2dtype(x: str) -> torch.dtype:
    x = x.lower()
    if x in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if x in {"fp16", "float16", "half"}:
        return torch.float16
    if x in {"fp32", "float32"}:
        return torch.float32
    raise argparse.ArgumentTypeError("dtype must be one of: bf16, fp16, fp32")


def setup_logging(log_dir: Path, is_main: bool) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(sys.stdout)]
    if is_main:
        handlers.append(logging.FileHandler(log_dir / "trainer.log", mode="a"))
    logging.basicConfig(level=logging.INFO, handlers=handlers)
    logger = get_logger("trainer")
    return logger


def save_config(cfg: Config, path: Path, is_main: bool, logger: logging.Logger) -> None:
    if not is_main:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving configuration to {path}")
    with open(path, "w") as f:
        yaml.dump(cfg.asdict(), f, default_flow_style=False, sort_keys=False)


def init_accelerator(cfg: Config) -> Accelerator:
    if cfg.dtype == torch.bfloat16:
        mp = "bf16"
    elif cfg.dtype == torch.float16:
        mp = "fp16"
    else:
        mp = "no"
    accelerator = Accelerator(
        log_with="tensorboard",
        project_dir=cfg.tb_dir,
        mixed_precision=mp,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
    )
    return accelerator


def load_flux_pipeline(
    dtype: torch.dtype, device, logger: logging.Logger
) -> Tuple[FluxControlPipeline, AutoencoderKL, FluxTransformer2DModel]:
    logger.info("Loading FLUX pipeline from black-forest-labs/FLUX.1-dev")
    pipe = FluxControlPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", torch_dtype=dtype
    )
    vae, transformer = pipe.vae, pipe.transformer
    for m in (vae, transformer, pipe.text_encoder, pipe.text_encoder_2):
        m.requires_grad_(False)
        m.eval()
    pipe.to(device).to(dtype)
    return pipe, vae, transformer


def load_transformer_checkpoint(
    cfg: Config, transformer: FluxTransformer2DModel, logger: logging.Logger
) -> FluxTransformer2DModel:
    if cfg.resume_transformer_from_ckpt:
        logger.info(
            f"Loading transformer from checkpoint {cfg.resume_transformer_from_ckpt}"
        )
        state = load_state_dict(str(cfg.resume_transformer_from_ckpt))
        transformer.load_state_dict(state, strict=False)
        logger.info("Loaded transformer checkpoint")
    else:
        logger.info("No transformer checkpoint provided; using pretrained weights")
    return transformer


def build_ctrl_encoder(cfg: Config, vae, device, logger: logging.Logger):
    ctrl_encoder = None
    if cfg.raw_dino:
        logger.info("Using raw DINOv3 features (identity mode, no adapter)")
        ctrl_encoder = EmbeddingModel.from_backbone("dinov3", device=device, dtype=cfg.dtype)
    elif cfg.ctrl_backbone in {"dino", "dinov3", "vae"}:
        if cfg.mrl_embedding_model_path is None:
            raise ValueError(
                "--mrl-embedding-model-path is required for ctrl-backbone 'dino' or 'vae'"
            )
        ctrl_encoder = EmbeddingModel.load_ckpt(cfg.mrl_embedding_model_path, pca_path = cfg.pca_path, device=device)
    elif cfg.ctrl_backbone == "none":
        logger.info(
            "No control backbone specified; expecting control signal from dataset (SplatCtrl)"
        )
        ctrl_encoder = None
    else:
        raise ValueError(f"Unknown control backbone: {cfg.ctrl_backbone}")

    if isinstance(ctrl_encoder, torch.nn.Module):
        ctrl_encoder.eval()
        for p in ctrl_encoder.parameters():
            p.requires_grad_(False)
        ctrl_encoder.to(device)
        if cfg.dtype in (torch.bfloat16, torch.float16):
            ctrl_encoder.to(cfg.dtype)

    return ctrl_encoder


def build_controlnet(
    cfg: Config, transformer, logger: logging.Logger
) -> FluxControlNetModel:
    controlnet = FluxControlNetModel.from_transformer(transformer)
    if cfg.ctrl_backbone in {"dino", "dinov3", "vae"} or cfg.splat_ctrl_finetuning:
        if cfg.max_control_channels is None:
            raise ValueError(
                "--max-control-channels must be set for ctrl-backbone 'dino'/'vae'"
            )
        controlnet.controlnet_x_embedder = zero_module(
            nn.Linear(cfg.max_control_channels * 4, controlnet.inner_dim)
        )
        controlnet.config.x_embed_in = cfg.max_control_channels * 4
        controlnet.config.x_embed_out = controlnet.inner_dim
    if cfg.resume_from_ckpt:
        logger.info(f"Loading controlnet from checkpoint {cfg.resume_from_ckpt}")
        state = load_state_dict(str(cfg.resume_from_ckpt))
        controlnet.load_state_dict(state, strict=False)
    else:
        logger.info("Created new controlnet")
    controlnet.to(torch.float32)
    return controlnet


def build_optimizer_and_schedulers(cfg: Config, params, accelerator: Accelerator):
    optimizer = Adafactor(
        params,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        scale_parameter=False,
        relative_step=False,
    )
    lr_scheduler = get_scheduler(
        cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=cfg.total_training_steps * accelerator.num_processes,
        num_cycles=cfg.lr_num_cycles,
        power=cfg.lr_power,
    )
    noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
    return optimizer, lr_scheduler, noise_scheduler


# ============================
# Main
# ============================
def env_path(val: str) -> Path:
    return Path(os.path.expandvars(val)).expanduser()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train ControlNet with Flux + optional control backbones"
    )

    SUP = argparse.SUPPRESS

    # I/O
    parser.add_argument("--dataset-root", type=Path, default=SUP)
    parser.add_argument("--result-dir", type=Path, default=SUP)
    parser.add_argument(
        "--feature-hacks",
        action="store_true",
        default=SUP,
        help="Enable feature hacks to remove positional artifacts",
    )

    # Stages
    parser.add_argument(
        "--stages", type=str, nargs="+", default=SUP, help="res,batch,steps"
    )
    parser.add_argument("--grad-accum", type=int, default=SUP)

    # Optim
    parser.add_argument("--learning-rate", type=float, default=SUP)
    parser.add_argument("--weight-decay", type=float, default=SUP)
    parser.add_argument("--lr-scheduler", type=str, default=SUP)
    parser.add_argument("--lr-warmup-steps", type=int, default=SUP)
    parser.add_argument("--lr-num-cycles", type=int, default=SUP)
    parser.add_argument("--lr-power", type=float, default=SUP)

    # Precision
    parser.add_argument("--dtype", type=str2dtype, default=SUP)

    # Control backbone
    parser.add_argument(
        "--ctrl-backbone",
        type=str,
        default=SUP,
        choices=["dino", "dinov3", "vae", "none"],
    )
    parser.add_argument("--max-control-channels", type=int, default=SUP)
    parser.add_argument("--guidance-dim-dropout-start-iter", type=int, default=SUP)
    parser.add_argument("--dim-reduction", type=str, default=SUP)
    parser.add_argument("--pca-path", type=Path, default=SUP)
    parser.add_argument("--use-pca", action="store_true", default=SUP)
    parser.add_argument("--mrl-embedding-model-path", type=Path, default=SUP)
    parser.add_argument(
        "--raw-dino",
        action="store_true",
        default=SUP,
        help="Use raw DINOv3 features without adapter (identity mode). No --mrl-embedding-model-path needed.",
    )
    parser.add_argument("--gsam2-root", type=Path, default=SUP)
    parser.add_argument("--colormap-path", type=Path, default=SUP)
    parser.add_argument(
        "--splat-ctrl-finetuning",
        action="store_true",
        default=SUP,
        help="Use SplatCtrlDataset for finetuning on SplatCtrl data",
    )
    parser.add_argument(
        "--splat-ctrl-lora-training",
        action="store_true",
        default=SUP,
        help="Use LoRA for finetuning on SplatCtrl data",
    )

    # Checkpoint
    parser.add_argument("--resume-from-ckpt", type=Path, default=SUP)
    parser.add_argument("--resume-transformer-from-ckpt", type=Path, default=SUP)
    parser.add_argument(
        "--resume-global-step",
        type=int,
        default=SUP,
        help="Global step (images seen) to resume from. Fast-forwards LR scheduler.",
    )

    # Misc
    parser.add_argument("--seed", type=int, default=SUP)
    parser.add_argument(
        "--tb",
        action="store_true",
        default=SUP,
        help="Launch TensorBoard (main process only)",
    )
    parser.add_argument("--tb-host", type=str, default=SUP)
    parser.add_argument("--tb-port", type=int, default=SUP)

    args = parser.parse_args(argv)

    # Build kwargs only from provided CLI args
    cfg_kwargs = {}

    def set_if_present(ns, key, cfg_key=None, transform=lambda x: x):
        if hasattr(ns, key):
            cfg_kwargs[cfg_key or key] = transform(getattr(ns, key))

    # Map CLI -> Config fields (only when present)
    set_if_present(args, "dataset_root")
    set_if_present(args, "result_dir")
    set_if_present(args, "feature_hacks", "feature_hacks", bool)

    # Stages need parsing
    if hasattr(args, "stages"):
        cfg_kwargs["stages"] = parse_stages(args.stages)

    set_if_present(args, "grad_accum", "gradient_accumulation_steps")
    set_if_present(args, "learning_rate")
    set_if_present(args, "weight_decay")
    set_if_present(args, "lr_scheduler")
    set_if_present(args, "lr_warmup_steps")
    set_if_present(args, "lr_num_cycles")
    set_if_present(args, "lr_power")
    set_if_present(args, "dtype")
    set_if_present(args, "ctrl_backbone")
    set_if_present(args, "max_control_channels")
    set_if_present(args, "guidance_dim_dropout_start_iter")
    set_if_present(args, "dim_reduction")
    set_if_present(args, "pca_path")
    set_if_present(args, "use_pca", "use_pca", bool)
    set_if_present(args, "mrl_embedding_model_path")
    set_if_present(args, "raw_dino", "raw_dino", bool)
    set_if_present(args, "gsam2_root")
    set_if_present(args, "colormap_path")
    set_if_present(args, "splat_ctrl_finetuning", "splat_ctrl_finetuning", bool)
    set_if_present(args, "splat_ctrl_lora_training", "splat_ctrl_lora_training", bool)
    set_if_present(args, "resume_from_ckpt")
    set_if_present(args, "resume_transformer_from_ckpt")
    set_if_present(args, "resume_global_step")
    set_if_present(args, "seed", "rand_seed")
    set_if_present(args, "tb", "launch_tensorboard", bool)
    set_if_present(args, "tb_host")
    set_if_present(args, "tb_port")

    cfg = Config(**cfg_kwargs)
    cfg.finalize()

    if cfg.splat_ctrl_finetuning:
        print(
            f"SplatCtrl finetuning enabled. Learning rates adjusted: {cfg.learning_rate}, {cfg.splat_ctrl_transformer_lr}"
        )

    # Accelerator before any logging files created
    accelerator = init_accelerator(cfg)

    # Clean previous run dir only on main process
    if (cfg.result_dir / cfg.run_id).exists() and accelerator.is_main_process:
        accelerator.print(
            f"Removing previous run directory: {(cfg.result_dir / cfg.run_id)}"
        )
        shutil.rmtree(cfg.result_dir / cfg.run_id)

    # Prepare directories on main process
    if accelerator.is_main_process:
        cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
        cfg.tb_dir.mkdir(parents=True, exist_ok=True)
        cfg.log_dir.mkdir(parents=True, exist_ok=True)

    # Logging
    logger = setup_logging(cfg.log_dir, accelerator.is_main_process)
    logger.info(f"Run ID: {cfg.run_id}")

    # Launch TensorBoard only on main
    tb_thread = None
    if cfg.launch_tensorboard and accelerator.is_main_process:
        try:
            tb_thread = threading.Thread(
                target=launch_tensorboard,
                args=(cfg.tb_dir,),
                kwargs={"host": cfg.tb_host, "port": cfg.tb_port},
                daemon=True,
            )
            tb_thread.start()
        except Exception as e:
            logger.error(f"Error launching tensorboard server: {e}")

    # Reduce noise from other loggers
    for logger_name in ["tensorboard", "werkzeug", "prometheus_client", "root"]:
        lg = logging.getLogger(logger_name)
        if lg:
            lg.setLevel(logging.WARNING)
            lg.propagate = False

    # Trackers (after logger setup)
    tracker_cfg = cfg.asdict()
    accelerator.init_trackers(cfg.run_id, config=tracker_cfg)
    device = accelerator.device

    # Save config (main only)
    save_config(cfg, cfg.result_dir / "config.yml", accelerator.is_main_process, logger)

    # Pipeline
    pipe, vae, transformer = load_flux_pipeline(cfg.dtype, device, logger)
    transformer = load_transformer_checkpoint(cfg, transformer, logger)

    if cfg.splat_ctrl_lora_training:
        try:
            from peft import LoraConfig
            from peft.utils.peft_types import TaskType
        except ImportError as e:
            raise ImportError(
                "Please install the 'peft' library to use LoRA training: pip install peft"
            ) from e
        # Prepare LoRA config
        TARGETS_FLUX = [
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.to_out.0",
            "ff.net.0.proj",
            "ff.net.2",
            "attn.to_qkv",  # include to_qkv for fused variants
        ]
        config = LoraConfig(
            r=16,
            lora_alpha=16,
            task_type=TaskType.FEATURE_EXTRACTION,
            target_modules=TARGETS_FLUX,
        )

    # unfreeze the transformer if finetuning on SplatCtrl
    if cfg.splat_ctrl_finetuning and cfg.splat_ctrl_finetune_transformer:
        if cfg.splat_ctrl_lora_training:
            transformer.add_adapter(
                adapter_config=config, adapter_name="splat_ctrl_lora"
            )
            transformer.enable_lora()
            logger.info("Enabled LoRA adapters for FLUX transformer")
        else:
            transformer.requires_grad_(True)
            transformer.to(cfg.dtype)
            logger.info("Unfroze FLUX transformer for finetuning on SplatCtrl")

    # ControlNet
    controlnet = build_controlnet(cfg, transformer, logger)
    if not cfg.splat_ctrl_lora_training:
        controlnet.requires_grad_(True)
    else:
        controlnet.add_adapter(adapter_config=config, adapter_name="splat_ctrl_lora")
        controlnet.enable_lora()
        logger.info("Enabled LoRA adapters for ControlNet")

    # Check if the require_grad is set correctly
    if cfg.splat_ctrl_lora_training:
        for n, p in controlnet.named_parameters():
            if (p.requires_grad and "lora" not in n) or (
                "lora" in n and not p.requires_grad
            ):
                print(
                    f"Param: {n} requires_grad={p.requires_grad} but should be the opposite"
                )
        for n, p in controlnet.named_parameters():
            if (p.requires_grad and "lora" not in n) or (
                "lora" in n and not p.requires_grad
            ):
                print(
                    f"Param: {n} requires_grad={p.requires_grad} but should be the opposite"
                )

    # Control encoder
    ctrl_encoder = build_ctrl_encoder(cfg, vae, device, logger)

    # Dataset
    if cfg.splat_ctrl_finetuning:
        from TASE.image_gen.controlnet.splat_ctrl_dataset import SplatCtrlDataset

        dataset = SplatCtrlDataset(
            root=cfg.dataset_root,
            normalization="vae",
            resize=cfg.stages[0][0],
            split="train",
        )
        val_dataset = SplatCtrlDataset(
            root=cfg.dataset_root,
            normalization="vae",
            resize=cfg.stages[0][0],
            split="val",
        )
        classmap = dataset.class_map
    else:
        res = cfg.stages[0][0]
        if cfg.feature_hacks:
            ps = ctrl_encoder.patch_size
            res = res + int(res // ps * 0.03) * ps * 2
        dataset = ImageNetDataset(cfg.dataset_root, res, split="train")
        val_dataset = ImageNetDataset(cfg.dataset_root, res, split="val")
        classmap = dataset.class_map
    logger.info(f"Loaded dataset from {cfg.dataset_root}")

    def _trainable_params(module):
        # Only parameters marked trainable (e.g., LoRA adapters) are returned.
        return [p for p in module.parameters() if p.requires_grad]

    # ControlNet params:
    # - Full finetune: you likely set requires_grad_(True) earlier on controlnet
    # - LoRA finetune: only LoRA layers should have requires_grad=True
    controlnet_trainable = _trainable_params(controlnet)

    if len(controlnet_trainable) == 0:
        raise ValueError(
            "No trainable ControlNet parameters. Did you forget to set requires_grad_(True) or enable LoRA?"
        )
    else:
        optimizer, lr_scheduler, noise_scheduler = build_optimizer_and_schedulers(
            cfg, controlnet_trainable, accelerator
        )

    # Transformer (optional)
    if cfg.splat_ctrl_finetuning and cfg.splat_ctrl_finetune_transformer:
        transformer_trainable = _trainable_params(transformer)

        if len(transformer_trainable) == 0:
            raise ValueError(
                "No trainable transformer parameters. Did you forget to set requires_grad_(True) or enable LoRA?"
            )
        else:
            transformer_optimizer = Adafactor(
                transformer_trainable,
                lr=cfg.splat_ctrl_transformer_lr,
                weight_decay=0.0,
                scale_parameter=False,
                relative_step=False,
            )
            transformer_lr_scheduler = get_scheduler(
                cfg.lr_scheduler,
                optimizer=transformer_optimizer,
                num_warmup_steps=cfg.lr_warmup_steps * accelerator.num_processes,
                num_training_steps=cfg.total_training_steps * accelerator.num_processes,
                num_cycles=cfg.lr_num_cycles,
                power=cfg.lr_power,
            )
    else:
        transformer_optimizer = None
        transformer_lr_scheduler = None

    # If we are LoRA finetuning, save the full models once at the start of training
    if cfg.splat_ctrl_lora_training:
        if accelerator.is_main_process:
            final_dir = Path(cfg.ckpt_dir) / f"base_controlnet"
            controlnet.save_pretrained(
                final_dir, max_shard_size="16GB", save_function=accelerator.save
            )
            logger.info(f"Saved base controlnet checkpoint to {final_dir}")
            if cfg.splat_ctrl_finetuning and cfg.splat_ctrl_finetune_transformer:
                # also save the transformer
                transformer_dir = Path(cfg.ckpt_dir) / "base_transformer"
                transformer.save_pretrained(
                    transformer_dir,
                    max_shard_size="32GB",
                    save_function=accelerator.save,
                )
                logger.info(f"Saved base transformer checkpoint to {transformer_dir}")

    # According to HF example: trainable ControlNet in fp32; frozen modules can be bf16/fp16
    controlnet.to(device).to(torch.float32)

    # Prepare with accelerate
    pipe, controlnet, optimizer, lr_scheduler = accelerator.prepare(
        pipe, controlnet, optimizer, lr_scheduler
    )

    # Train
    train_loop(
        cfg,
        accelerator,
        logger,
        dataset,
        val_dataset,
        classmap,
        pipe,
        vae,
        transformer,
        controlnet,
        optimizer,
        lr_scheduler,
        transformer_optimizer,
        transformer_lr_scheduler,
        noise_scheduler,
        ctrl_encoder,
        device,
        cfg.dtype,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
