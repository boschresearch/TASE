# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch
import torch.nn as nn
import math


class FourierEmbedding(nn.Module):
    def __init__(
        self, input_dim: int, num_frequencies: int, include_input: bool = True
    ):
        """
        Creates a Fourier embedding of the input.

        Parameters:
            input_dim (int): Dimension of the input vector (e.g., 3 for viewing directions).
            num_frequencies (int): Number of frequency bands.
            include_input (bool): Whether to include the original input in the output.
        """
        super(FourierEmbedding, self).__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        # Precompute frequency bands (e.g., [1, 2, 4, ..., 2^(num_frequencies-1)])
        self.register_buffer(
            "freq_bands", 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Fourier embedding.

        Parameters:
            x (torch.Tensor): Tensor of shape (B, input_dim).

        Returns:
            torch.Tensor: Fourier-embedded tensor of shape (B, embed_dim),
                          where embed_dim = input_dim + 2 * input_dim * num_frequencies (if include_input is True).
        """
        out = []
        if self.include_input:
            out.append(x)
        for freq in self.freq_bands:
            out.append(torch.sin(x * freq * math.pi))
            out.append(torch.cos(x * freq * math.pi))
        return torch.cat(out, dim=-1)


class MLP_BackgroundModel(nn.Module):

    def render_bg_colors(
        self, Ks: torch.Tensor, extrinsics: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        B = Ks.shape[0]
        viewing_directions = self.get_viewing_directions_batch(
            Ks, extrinsics, height, width
        )
        viewing_directions = viewing_directions.reshape(-1, 3)
        output_colors = self(viewing_directions)
        output_colors = output_colors.reshape(B, height, width, -1)
        return output_colors

    def get_viewing_directions_batch(
        self, K: torch.Tensor, extrinsic: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        """
        Computes viewing directions for each pixel in batched images given pinhole camera intrinsics
        and camera-to-world extrinsics. The output viewing directions are in world coordinates.

        Parameters:
            K (torch.Tensor): Tensor of shape (B, 3, 3) or (3, 3) containing intrinsic matrices.
                            If a single camera is provided (3, 3), it is unsqueezed.
            extrinsic (torch.Tensor): Tensor of shape (B, 4, 4) or (4, 4) containing camera-to-world matrices.
                                    If a single camera is provided (4, 4), it is unsqueezed.
            height (int): Image height (number of rows).
            width (int): Image width (number of columns).

        Returns:
            torch.Tensor: Tensor of shape (B, height, width, 3) with normalized viewing directions in world coordinates.
        """
        device = K.device

        # Ensure K and extrinsic have a batch dimension.
        if K.dim() == 2:
            K = K.unsqueeze(0)  # (1, 3, 3)
        if extrinsic.dim() == 2:
            extrinsic = extrinsic.unsqueeze(0)  # (1, 4, 4)

        B = K.shape[0]

        # Create a mesh grid of pixel coordinates (H, W).
        grid_v, grid_u = torch.meshgrid(
            torch.arange(height, device=device, dtype=K.dtype),
            torch.arange(width, device=device, dtype=K.dtype),
            indexing="ij",
        )
        ones = torch.ones_like(grid_u)
        # Create homogeneous pixel coordinates: (H, W, 3) with [u, v, 1].
        pixels = torch.stack([grid_u, grid_v, ones], dim=-1)

        # Flatten to shape (H*W, 3) then transpose to (3, H*W).
        pixels_flat = pixels.view(-1, 3).t()  # (3, H*W)

        # Expand pixel coordinates to the batch: (B, 3, H*W)
        pixels_flat = pixels_flat.unsqueeze(0).expand(B, -1, -1)

        # Compute inverse intrinsic matrices for each batch.
        invK = torch.inverse(K)  # (B, 3, 3)

        # Convert pixels to camera space directions: (B, 3, H*W).
        cam_dirs = torch.bmm(invK, pixels_flat)
        # Transpose to (B, H*W, 3).
        cam_dirs = cam_dirs.transpose(1, 2)

        # Extract the rotation part of the extrinsic matrices: (B, 3, 3).
        R = extrinsic[:, :3, :3]
        # Rotate camera space directions into world space.
        # Since our directions are column vectors, we multiply on the right with R^T.
        world_dirs = torch.bmm(cam_dirs, R.transpose(1, 2))

        # Reshape to (B, height, width, 3).
        world_dirs = world_dirs.view(B, height, width, 3)

        # Normalize each viewing direction.
        world_dirs = world_dirs / world_dirs.norm(dim=-1, keepdim=True)

        return world_dirs

    def __init__(
        self,
        input_dim: int = 3,
        num_frequencies: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 3,
        output_dim: int = 3,
    ):
        """
        Simple background model that predicts an RGB color from a Fourier-embedded viewing direction.

        Parameters:
            input_dim (int): Dimension of the viewing direction (default is 3).
            num_frequencies (int): Number of Fourier frequencies to use.
            hidden_dim (int): Number of neurons in hidden layers.
            num_layers (int): Number of hidden layers.
            output_dim (int): Dimension of the output color (default is 3 for RGB).
        """
        super(MLP_BackgroundModel, self).__init__()
        self.output_dim = output_dim
        self.embedding = FourierEmbedding(
            input_dim, num_frequencies, include_input=True
        )
        # Calculate the embedding dimension.
        embed_dim = input_dim + 2 * input_dim * num_frequencies

        layers = []
        in_dim = embed_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim
        # Final layer outputs 3 channels for RGB.
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, viewing_direction: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that computes the background color given a viewing direction.

        Parameters:
            viewing_direction (torch.Tensor): Tensor of shape (B, 3) representing viewing directions.

        Returns:
            torch.Tensor: Tensor of shape (B, 3) with predicted RGB colors in [0, 1].
        """
        # Apply Fourier embedding.
        embedded = self.embedding(viewing_direction)
        # Pass through the MLP.
        color = self.mlp(embedded)
        # Ensure the output color is in [0, 1].
        if self.output_dim == 3:
            torch.sigmoid(color)
        return color


if __name__ == "__main__":
    # Example usage:
    model = MLP_BackgroundModel(
        input_dim=3, num_frequencies=6, hidden_dim=128, num_layers=3
    )

    # Create a batch of random viewing directions (e.g., 4 directions).
    viewing_directions = torch.randn(4, 3)
    viewing_directions = viewing_directions / viewing_directions.norm(
        dim=-1, keepdim=True
    )

    # Get the predicted background colors.
    output_colors = model(viewing_directions)
    print("Predicted background colors:")
    print(output_colors)
