#!/usr/bin/env python
# coding=utf-8
# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from TASE.image_gen.common.util import ImageNetDataset

from TASE.image_gen.dimensionality_reduction.embedding_model import EmbeddingModel
from torch_incremental_pca import IncrementalPCA
import pickle

# === Config ===
BACKBONE = "dinov3"                        # set to "dino" or "vae" (Flux vae)
IMAGENET_ROOT      = os.path.expandvars("$IMGNET_ROOT")        # path to ImageNet dataset
NUM_IMAGES         = 50_000             # how many images to sample
PCA_COMPONENTS     = 64
BATCH_SIZE         = 128                # around 50GB GPU memory for the VAE, 13GB for DINO
RESOLUTION         = 512
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_PCA_PATH    = os.path.expandvars(f"$TASE_ROOT/OUTPUT/PCA/{BACKBONE}/torch_ipca.pkl")
DEVICE = "cuda" 
DTYPE = torch.bfloat16

# === Load DINOv2 backbone ===
print(f"Loading backbone model: {BACKBONE}")
embedding_model = EmbeddingModel(backbone_model=BACKBONE, build_encoder_decoder=False)
embedding_model.to(DEVICE)
embedding_model.eval()

# === Prepare dataset & loader ===
split = "train"

dataset = ImageNetDataset(root=IMAGENET_ROOT, resolution=RESOLUTION, split=split)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                     shuffle=True, num_workers=8, pin_memory=True)

# === Incremental PCA setup ===
ipca = IncrementalPCA(n_components=PCA_COMPONENTS)

images_processed = 0
with torch.no_grad():
    pbar = tqdm(loader, total=(NUM_IMAGES // BATCH_SIZE) + 1,
                desc="Fitting IncrementalPCA", unit="batch")
    for images, _ in pbar:
        images = images.to(DEVICE, dtype=DTYPE, non_blocking=True)
        # maybe crop to exact NUM_IMAGES
        if images_processed >= NUM_IMAGES:
            break
        remaining = NUM_IMAGES - images_processed
        if images.shape[0] > remaining:
            images = images[:remaining]

        images = images.to(DEVICE)
        with torch.no_grad():
            feats = embedding_model.base_encode(images)  # [B, num_patches, embed_dim]
        toks = feats.reshape(-1, feats.shape[-1]).float() # [B*num_patches, embed_dim]

        # partial fit
        ipca.partial_fit(toks)

        images_processed += images.shape[0]
        pbar.set_postfix({"imgs": images_processed})

# === Print Stats ===

print(f"Processed {images_processed} images.")
print(f"Incremental PCA mean: {ipca.mean_}")
print(f"Incremental PCA components: {ipca.components_}")
print(f"Incremental PCA explained variance ratio: {ipca.explained_variance_ratio_}")
print(f"Incremental PCA explained variance: {ipca.explained_variance_}")
for i in range(PCA_COMPONENTS):
    print(f"Variance explained by {i} components: {torch.sum(ipca.explained_variance_ratio_[:i]) * 100:.2f}%")

# === Save PCA model ===
if not os.path.exists(os.path.dirname(OUTPUT_PCA_PATH)):
    os.makedirs(os.path.dirname(OUTPUT_PCA_PATH))
pickle.dump(ipca, open(OUTPUT_PCA_PATH, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
torch.save(
    {"mean": ipca.mean_,
     "components": ipca.components_,
     "explained_variance": ipca.explained_variance_,
     "explained_variance_ratio": ipca.explained_variance_ratio_},
    OUTPUT_PCA_PATH.replace(".pkl", ".pt")
)
print(f"Incremental PCA trained on {images_processed} images; saved to:\n  {OUTPUT_PCA_PATH}")
