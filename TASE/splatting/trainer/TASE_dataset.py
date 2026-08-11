# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from nerfstudio
#   (https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/data/datasets/base_dataset.py)
# Copyright 2022 the Regents of the University of California, Nerfstudio Team
# and contributors. All rights reserved., licensed under the Apache-2.0
# license, cf. 3rd-party-licenses.txt file in the root directory of this
# source tree.

"""
Dataset including all the modalities supported by TASE: RGB images, depth maps, normal maps, semantic features.
"""

from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch
import json

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset


class TASE_dataset(InputDataset):
    """Dataset that returns images and depths. If no depths are found, then we generate them with Zoe Depth.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs.
    """

    def __init__(
        self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0
    ):
        super().__init__(dataparser_outputs, scale_factor)
        self.scene_metadata = None
        self.data_dir = None

    def get_metadata(self, data: Dict) -> Dict:
        # file lists from dataparser outputs
        idx = data["image_idx"]
        scale = self._dataparser_outputs.dataparser_scale
        sample_meta = self._dataparser_outputs.metadata.get(idx, None)
        if sample_meta is None:
            print(f"No metadata found for image index {idx}.")
            return {}

        if self.scene_metadata is None:
            scene_metadata = self._dataparser_outputs.metadata.get(
                "scene_metadata_filename", None
            )
            self.scene_metadata = (
                json.load(open(scene_metadata, "r")) if scene_metadata else None
            )
        if self.scene_metadata is None:
            raise ValueError("Could not load metadata file.")

        if self.data_dir is None:
            self.data_dir = self._dataparser_outputs.metadata.get("data_dir", None)
        if self.data_dir is None:
            raise ValueError("Could not find data directory in metadata.")

        datamap_filename = sample_meta.get("datamap_filename", None)
        datamap_path = (
            self.data_dir
            / self.scene_metadata["subdirs"]["datamap_dir"]
            / datamap_filename
            if datamap_filename
            else None
        )

        try:
            datamap_dict = np.load(
                Path(datamap_path).with_suffix(".npz"), allow_pickle=True
            )
            datamap_dict = {k: datamap_dict[k] for k in datamap_dict.files}
            for key in datamap_dict:
                if isinstance(datamap_dict[key], np.ndarray):
                    datamap_dict[key] = torch.tensor(
                        datamap_dict[key], dtype=torch.float32
                    )
                    if datamap_dict[key].ndim == 2:
                        datamap_dict[key] = datamap_dict[key].unsqueeze(
                            -1
                        )  # Add channel dimension if missing
                if "mask" in key and datamap_dict[key].max() > 1:
                    datamap_dict[key] = (
                        datamap_dict[key] / datamap_dict[key].max()
                    )  # Normalize mask values to [0, 1]
            if "depth" in datamap_dict:
                datamap_dict["depth"] = (
                    datamap_dict["depth"] * scale
                )  # Apply scale factor to depth

        except Exception as e:
            print(f"Error loading datamaps for image {idx}: {e}.")
            return {}

        return datamap_dict
