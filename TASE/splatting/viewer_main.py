# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from gsplat V 1.5.3
#   (https://github.com/nerfstudio-project/gsplat/blob/main/examples/simple_viewer.py)
# Copyright 2025 Nerfstudio Team, licensed under the Apache-2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

import argparse
import math
import time
from typing import Tuple
import viser

# import the modified viewer class
from TASE.splatting.viewer.viewer import Viewer, ColorMapper
from TASE.splatting.util.data_util import load_TASE_samples, load_cameras_json

import torch
import torch.nn.functional as F
import numpy as np


def main(gpu_index: int, args):
    torch.manual_seed(42)
    device = torch.device("cuda", gpu_index)

    server = viser.ViserServer(port=args.port, verbose=False)

    # load the splats
    if args.ckpt is None:
        print("No checkpoint path provided. Trying to load dataset for visualization.")
        if hasattr(args, "dataset") and args.dataset is not None:
            samples_dict = load_TASE_samples(args.dataset, device=device)
            if samples_dict is None:
                print("No samples found in the dataset directory. Exiting.")
                return
        else:
            print("No dataset directory provided. Exiting.")
            return

        viewer = Viewer(
            server=server,
            scene=None,
            cfg=None,  # No config since we are loading samples
            mode="samples",
            initial_pose=args.pose,
            colormapper=ColorMapper(),
            samples=samples_dict,
        )
    else:
        # Construct the viewer without a scene first, then delegate the
        # actual checkpoint/dataset loading to the viewer's own native
        # `load_scene` function (the same one used by the "Load scene" GUI
        # button), instead of duplicating the loading logic here.
        viewer = Viewer(
            server=server,
            scene=None,
            cfg=None,
            mode="rendering",
            initial_pose=args.pose,
            colormapper=None,
            samples=None,
            segmentation_module=args.segmentation,
        )
        viewer.load_scene(
            args.ckpt,
            load_dataset=True,
            device=device,
            renormalize_coordinates=args.normalize,
            dataparser_transform=args.dataparser_transform,
        )
        if viewer.scene is not None:
            print(f"Loaded scene '{viewer.scene.name}'.")

    # Merge in any cameras.json passed on the CLI using the viewer's own
    # `add_samples` function.
    if args.cameras is not None:
        extra_cameras = load_cameras_json(args.cameras)
        viewer.add_samples(extra_cameras)
        print(f"Loaded {len(extra_cameras)} cameras from {args.cameras}.")

    print("Viewer running... Ctrl+C to exit.")
    time.sleep(100000)


if __name__ == "__main__":
    """
    # Use single GPU to view the scene
    CUDA_VISIBLE_DEVICES=0 python simple_viewer.py \
        --ckpt OUTPUT/reconstruction/my_run/bicycle/ckpt/step29999.pt \
        --port 8081
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
    )
    parser.add_argument(
        "--cameras",
        type=str,
        default=None,
        help="Path to a cameras.json file (3DGS/COLMAP format) to overlay in the viewer.",
    )
    parser.add_argument(
        "--samples",
        type=str,
    )
    parser.add_argument(
        "--segmentation",
        action="store_true",
    )
    parser.add_argument(
        "--dataset",
        type=str,
    )
    parser.add_argument(
        "--dataparser-transform",
        type=str,
        help="The nerfstudio-compatible transform file to use to transform the scene back to dataset coordinates.",
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="port for the viewer server"
    )
    parser.add_argument(
        "--pose",
        type=str,
        help="a 3x4 or 4x4 transformation matrix to set the initial camera pose",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="whether to normalize the scene coordinates",
    )
    args = parser.parse_args()

    # For distributed viewing of large scenes
    # cli(main, args, verbose=True)
    main(gpu_index=0, args=args)
