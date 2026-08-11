# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import os
import sys
import shutil
import logging
import argparse
from datetime import datetime
from pathlib import Path
from collections import deque
import threading

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import debugpy

from TASE.image_gen.dimensionality_reduction.embedding_model import EmbeddingModel
from TASE.image_gen.common.util import ImageNetDataset, launch_tensorboard


def parse_args():
    p = argparse.ArgumentParser(
        description="Train MRL embeddings with CLI-configurable params"
    )

    # Paths & backbone
    p.add_argument(
        "--ckpt", type=str, default=None, help="Path to checkpoint to resume from"
    )
    p.add_argument(
        "--imagenet_root",
        type=Path,
        default="$IMGNET_ROOT",
        help="Path to ILSVRC2012 images root",
    )
    p.add_argument(
        "--backbone", type=str, default="dinov3", choices=["dinov3", "dino", "vae"]
    )

    # Training setup
    p.add_argument("--res", type=int, default=512)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_dims", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Num dataloader workers (defaults 0 if debug, else 4)",
    )

    # Logging / checkpoints
    p.add_argument("--val_freq", type=int, default=100)
    p.add_argument("--log_freq", type=int, default=1000)
    p.add_argument("--ckpt_freq", type=int, default=10000)
    p.add_argument("--img_freq", type=int, default=10000)
    p.add_argument(
        "--results_root",
        type=Path,
        default=None,
    )

    # Loss weights
    p.add_argument("--gram_lambda", type=float, default=0.0)
    p.add_argument("--equivariance_lambda", type=float, default=0.0)
    p.add_argument("--spacial_bias_lambda", type=float, default=0.0)
    p.add_argument("--spacial_corr_lambda", type=float, default=0.0)
    p.add_argument("--moments_lambda", type=float, default=0.0)
    p.add_argument("--dist_lambda", type=float, default=0.0)

    # Equivariance warmup (if not set, auto: 50k when training from scratch, else 0 when resuming)
    p.add_argument(
        "--eqv_warmup",
        type=int,
        default=None,
        help="Steps to warm up before applying equivariance loss (default: 50k if training from scratch, else 0)",
    )

    # TensorBoard server
    p.add_argument("--tb_host", type=str, default="0.0.0.0")
    p.add_argument("--tb_port", type=int, default=6008)

    return p.parse_args()


class RunningAverage:
    def __init__(self, window_size: int):
        self.buf = deque(maxlen=window_size)

    def update(self, loss: float):
        self.buf.append(float(loss))

    @property
    def avg(self) -> float:
        return sum(self.buf) / len(self.buf) if self.buf else 0.0


def log_losses(
    tb_writer,
    prefix: str,
    step: int,
    info: dict = None,
    maps: bool = False,
    RES: int = 512,
):
    if info is None:
        return
    for key, value in info.items():
        if isinstance(value, dict):
            new_dict = {f"{k:04d}": v for k, v in value.items()}
            tb_writer.add_scalars(f"{prefix}/{key}", new_dict, step)
        elif isinstance(value, torch.Tensor):
            if maps and value.ndim == 3:
                # treat as image map
                value = value[0, :, :]
                value = value - value.min()
                denom = max(value.max(), torch.tensor(1e-6, device=value.device))
                value = value / denom
                tb_writer.add_images(
                    f"{prefix}/{key}", value.unsqueeze(0).unsqueeze(0), step
                )
        else:
            tb_writer.add_scalar(f"{prefix}/{key}", value, step)


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Debug mode detection affects workers and run id
    debug = debugpy.is_client_connected()
    if debug:
        datetime_str = "debug"
        workers = 0 if args.workers is None else args.workers
        img_freq = 1000  # faster feedback when debugging
    else:
        datetime_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        workers = 4 if args.workers is None else args.workers
        img_freq = args.img_freq

    # EQV warmup default logic
    if args.eqv_warmup is None:
        eqv_warmup = 0 if args.ckpt else 50_000
    else:
        eqv_warmup = args.eqv_warmup

    if args.results_root is None:
        result_root = Path("./OUTPUT")
        if os.environ.get("TASE_ROOT") is not None:
            result_root = Path(os.environ["TASE_ROOT"]) / "OUTPUT"
    else:
        result_root = args.results_root

    # Paths
    run_id = datetime_str
    result_root = result_root / "models" / "MRL" / args.backbone / run_id
    log_dir = result_root / "logs"
    ckpt_dir = result_root / "ckpt"
    tb_dir = result_root / "tb"

    # Prepare result directories
    if result_root.exists():
        print("Removing previous results (logs & tensorboard)...")
        for d in (log_dir, tb_dir):
            if d.exists():
                print(f"Removing {d}")
                shutil.rmtree(d)
    for d in (log_dir, ckpt_dir, tb_dir):
        os.makedirs(d, exist_ok=True)

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "trainer.log", mode="a"),
        ],
    )
    logger = logging.getLogger()

    # TensorBoard server
    try:
        tb_server = threading.Thread(
            target=launch_tensorboard,
            args=(tb_dir,),
            kwargs={"host": args.tb_host, "port": args.tb_port},
            daemon=True,
        )
        tb_server.start()
    except Exception as e:
        logger.error(f"Error launching tensorboard server: {e}")
        tb_server = None

    for logger_name in ["tensorboard", "werkzeug", "prometheus_client"]:
        lg = logging.getLogger(logger_name)
        if lg:
            lg.setLevel(logging.WARNING)
            lg.propagate = False

    tb_writer = SummaryWriter(tb_dir)

    # Device info print
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_properties(0).name
    else:
        gpu_name = "CPU"
    print(f"Using device: {gpu_name} (debug: {debug})")

    # Build model
    if args.ckpt:
        print(f"Loading checkpoint from {args.ckpt}")
        model = EmbeddingModel.load_ckpt(args.ckpt, device)
        # You can still override loss weights after load if desired:
        model.gram_lambda = args.gram_lambda
        model.equivariance_lambda = args.equivariance_lambda
        model.spacial_bias_lambda = args.spacial_bias_lambda
        model.spacial_corr_lambda = args.spacial_corr_lambda
        model.moments_lambda = args.moments_lambda
        model.dist_lambda = args.dist_lambda
    else:
        print("Training from scratch")
        model = EmbeddingModel(
            backbone_model=args.backbone,
            internal_dim=args.max_dims,
            gram_lambda=args.gram_lambda,
            equivariance_lambda=args.equivariance_lambda,
            spacial_bias_lambda=args.spacial_bias_lambda,
            spacial_corr_lambda=args.spacial_corr_lambda,
            moments_lambda=args.moments_lambda,
            dist_lambda=args.dist_lambda,
        ).to(device)

    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Datasets / loaders
    train_ds = ImageNetDataset(args.imagenet_root, args.res, split="train")
    val_ds = ImageNetDataset(args.imagenet_root, args.res, split="val")

    max_iter = len(train_ds) * args.epochs
    logger.info(
        f"Training with resolution {args.res}×{args.res}, epochs={args.epochs} ({max_iter}), batch_size={args.batch}"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
    )
    val_iter = iter(val_loader)

    train_avg = RunningAverage(1000)
    val_avg = RunningAverage(1000)
    best_val_loss = float("inf")

    next_val = 0
    next_log = 0
    next_ckpt = args.ckpt_freq
    next_img = 0

    step = 0
    pbar = tqdm(total=max_iter, desc=f"Train {args.res}", unit="img", disable=None)

    def add_images_for_val(result, images, step):
        with torch.no_grad():
            feat = result["ordered_embedding"]  # (B, N, C)
            B, img_H, img_W = images.shape[0], images.shape[2], images.shape[3]
            H, W = int(img_H / model.patch_size), int(img_W / model.patch_size)
            feat = feat.reshape(B, H, W, -1)
            feat = feat[0, :, :, :3]  # first 3 dims, first image
            feat = feat.permute(2, 0, 1)  # (3, H, W)
            feat = feat - feat.min()
            feat = feat / max(feat.max(), torch.tensor(1e-6, device=feat.device))
            feat = torch.clamp(feat, 0, 1)
            feat = F.interpolate(
                feat.unsqueeze(0), size=(args.res, args.res), mode="nearest"
            )
            tb_writer.add_images("val_img/mrl_img", feat, step)
            tb_writer.add_images("val_img/orig_img", (images[:1]).clamp(0, 1), step)

    logger.info(f"Starting training run {run_id}")
    model.train()
    for epoch in range(args.epochs):
        logger.info(f"Starting epoch {epoch+1} / {args.epochs}")
        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)
            optimizer.zero_grad()

            result = model(images, random_spatial_mask=True)
            loss, info = model.compute_losses(result)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += args.batch

            log_losses(tb_writer, prefix="train", step=step, info=info)

            pbar.set_postfix(loss=f"{loss.item():.4f}")
            pbar.update(images.size(0))

            # Equivariance reg (after warmup)
            if model.equivariance_lambda > 0 and step > eqv_warmup:
                eqv_loss, _ = model.equivariance_loss_crop_flip(images)
                eqv_loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                tb_writer.add_scalar("train/equivariance_loss", eqv_loss.item(), step)
                loss = loss + eqv_loss * model.equivariance_lambda

            train_avg.update(loss)

            # Validation
            if step > next_val:
                model.eval()
                try:
                    images_val, _ = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    images_val, _ = next(val_iter)

                with torch.no_grad():
                    images_val = images_val.to(device)
                    result_val = model(images_val)
                    vloss, vinfo = model.compute_losses(result_val)

                log_losses(
                    tb_writer,
                    prefix="val",
                    step=step,
                    info=vinfo,
                    maps=(step > next_img),
                    RES=args.res,
                )

                with torch.no_grad():
                    eqv_loss_val, _ = model.equivariance_loss_crop_flip(images_val)
                tb_writer.add_scalar("val/equivariance_loss", eqv_loss_val.item(), step)

                vloss = vloss + eqv_loss_val
                val_avg.update(vloss)
                next_val += args.val_freq

                if step > next_img:
                    add_images_for_val(result_val, images_val, step)
                    next_img += img_freq

                model.train()

            # Logging
            if step > next_log:
                logger.info(
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
                    f"Step {step} / {max_iter}, Train loss: {train_avg.avg:.4f}, Val loss: {val_avg.avg:.4f}"
                )
                next_log += args.log_freq

            # Checkpointing
            if step > next_ckpt:
                if best_val_loss > val_avg.avg:
                    ckpt_file = ckpt_dir / "best"
                    best_val_loss = val_avg.avg
                else:
                    ckpt_file = ckpt_dir / "latest"
                model.save_ckpt(ckpt_file, save_backbone_weights=False)
                logger.info(
                    f"Saved checkpoint: {ckpt_file}, step {step}, val loss {val_avg.avg:.4f}"
                )
                next_ckpt += args.ckpt_freq

            #### DEBUG ####
            break
            #### DEBUG ####
        pbar.close()

        # Save epoch checkpoint
        ckpt_file = os.path.join(ckpt_dir, f"epoch_{epoch+1}")
        model.save_ckpt(ckpt_file, save_backbone_weights=False)
        logger.info(
            f"Saved checkpoint: {ckpt_file}, step {step}, val loss {val_avg.avg:.4f}"
        )

    logger.info("Training complete.")
    tb_writer.close()

    logger.info("Computing feature stats...")
    from TASE.image_gen.dimensionality_reduction.embedder_stats import (
        compute_stats_incremental,
    )

    compute_stats_incremental(
        imagenet_path=args.imagenet_root, mrl_model_path=(ckpt_file + ".pth")
    )


if __name__ == "__main__":
    main()

"""
cd TASE/image_gen/dimensionality_reduction
python mrl_trainer.py \
  --imagenet_root "$IMGNET_ROOT" \
    --backbone dinov3 --res 512 --batch 64 --epochs 1 --max_dims 64 \
      --equivariance_lambda 1000.0 
"""
