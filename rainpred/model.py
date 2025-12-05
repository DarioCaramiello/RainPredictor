import math
from typing import Tuple

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

from .config import PRED_LENGTH, PATCH_HEIGHT, PATCH_WIDTH


class UNet_Encoder(nn.Module):
    """Simple U-Net encoder for single-frame 2D feature extraction."""
    def __init__(self, input_channels: int):
        super().__init__()
        # First convolutional block: keep spatial size, go to 64 channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, 1, 1, bias=False),  # 3x3 conv, stride 1, padding 1
            nn.BatchNorm2d(64),                                   # normalize each of the 64 channels
            nn.ReLU(True),                                        # non-linear activation
        )
        # Second conv block, keep 64 channels and spatial size
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        # Max pooling to downsample spatially by a factor of 2
        self.pool1 = nn.MaxPool2d(2, 2)
        # Third conv block: increase channels to 128
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        # Fourth conv block: keep 128 channels
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a single frame.

        Args:
            x: input tensor of shape (B, C_in, H, W).

        Returns:
            A tuple (x_down, skip1) where:
              - x_down is the downsampled feature map (B, 128, H/2, W/2)
              - skip1 is the high-resolution feature map for skip connection (B, 64, H, W)
        """
        x = self.conv1(x)   # (B, 64, H, W)
        x = self.conv2(x)   # (B, 64, H, W)
        skip1 = x           # save high-res features for the decoder
        x = self.pool1(x)   # (B, 64, H/2, W/2)
        x = self.conv3(x)   # (B, 128, H/2, W/2)
        x = self.conv4(x)   # (B, 128, H/2, W/2)
        return x, skip1


class UNet_Decoder(nn.Module):
    """Simple U-Net decoder that merges encoder and transformer features.

    This module takes:
      - a low-resolution feature map from the temporal transformer
      - the encoder's low-res features (same resolution)
      - a high-resolution skip connection
    and reconstructs an output with `output_channels` channels (e.g. 1 for radar field).
    """
    def __init__(self, output_channels: int):
        super().__init__()
        # First decoder block: merge transformer (128ch) and encoder low-res (128ch) => 256ch -> 128ch
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        # Second decoder block: refine 128ch features
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        # Upsampling layer: go back to encoder's high-res spatial size
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 1),  # reduce channels to 64 after upsampling
        )
        # Third decoder block: merge upsampled features (64ch) + high-res skip (64ch) => 128ch -> 64ch
        self.conv7 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        # Fourth decoder block: refine 64ch features
        self.conv8 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        # Final 1x1 conv to get the desired number of output channels (e.g. 1)
        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, skip0: torch.Tensor, skip1: torch.Tensor):
        """Decode feature maps into the final prediction.

        Args:
            x: transformer features at low resolution (B, 128, H/2, W/2)
            skip0: encoder low-res features (B, 128, H/2, W/2)
            skip1: encoder high-res features (B, 64, H, W)

        Returns:
            A tuple (out, out_noact) where:
              - out is the final activated prediction (B, C_out, H, W)
              - out_noact is the same before the final activation (useful for metrics)
        """
        # Concatenate low-res encoder and transformer features along channel dimension
        x = torch.cat([skip0, x], dim=1)  # (B, 256, H/2, W/2)
        x = self.conv5(x)                 # (B, 128, H/2, W/2)
        x = self.conv6(x)                 # (B, 128, H/2, W/2)
        # Upsample back to original spatial resolution
        x = self.up1(x)                   # (B, 64, H, W)
        # Concatenate with high-res skip connection
        x = torch.cat([skip1, x], dim=1)  # (B, 128, H, W)
        x = self.conv7(x)                 # (B, 64, H, W)
        x = self.conv8(x)                 # (B, 64, H, W)
        # Map 64 feature channels to the requested output channels (e.g. 1)
        x = self.final_conv(x)            # (B, C_out, H, W)
        # Save pre-activation output
        x_last = x
        # Apply bounded activation; change if you prefer a different output range
        x = torch.tanh(x)
        return x, x_last


def generate_positional_encoding(seq_len: int, d_model: int, device: torch.device) -> torch.Tensor:
    """Create standard Transformer sinusoidal positional encodings.

    Args:
        seq_len: number of time/patch tokens.
        d_model: embedding dimension.
        device: torch device where the tensor will be allocated.

    Returns:
        Positional encoding tensor of shape (1, seq_len, d_model).
    """
    pe = torch.zeros(seq_len, d_model, device=device)  # (seq_len, d_model)
    # Positions 0..seq_len-1
    position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
    # Divisor term for sine/cosine frequencies
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model)
    )
    # Apply sine to even indices in the embedding
    pe[:, 0::2] = torch.sin(position * div_term)
    # Apply cosine to odd indices in the embedding
    pe[:, 1::2] = torch.cos(position * div_term)
    # Add batch dimension
    return pe.unsqueeze(0)  # (1, seq_len, d_model)


class TemporalTransformerBlock(nn.Module):
    """Temporal Transformer over spatio-temporal patches of encoder features.

    It treats each (time, patch) as a token, encodes the full sequence,
    then reshapes a subset of tokens into future frames.
    """
    def __init__(
        self,
        channels: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        pred_length: int,
        patch_height: int = PATCH_HEIGHT,
        patch_width: int = PATCH_WIDTH,
    ):
        super().__init__()
        # How many future frames we want to predict
        self.pred_length = pred_length
        # Patch size used for tokenization
        self.patch_height = patch_height
        self.patch_width = patch_width
        # Dimensionality of a single patch (flattened)
        patch_dim = channels * patch_height * patch_width

        # Map 5D tensor (B, T, C, H, W) to sequence of patch tokens
        self.to_patch_embedding = nn.Sequential(
            # Split H,W into non-overlapping patches of size (patch_height, patch_width)
            Rearrange(
                "b t c (h p1) (w p2) -> b (t h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),        # normalize patch features
            nn.Linear(patch_dim, d_model),  # project to transformer dimension
            nn.LayerNorm(d_model),          # stabilize training
        )

        # Standard Transformer encoder over the patch tokens
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=num_encoder_layers,
        )

        # Map transformer outputs back to patch feature dimension
        self.to_feature_map = nn.Sequential(
            nn.Linear(d_model, patch_dim),
            nn.LayerNorm(patch_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the temporal transformer.

        Args:
            x: encoder features of shape (B, T_in, C, H, W).

        Returns:
            Tensor of predicted features with shape
            (B, pred_length, C, H, W), where C == `channels` used at init.
        """
        B, Tin, C, H, W = x.shape
        # Number of patches along height/width
        ph = H // self.patch_height
        pw = W // self.patch_width

        # Convert to patch tokens: (B, Tin * ph * pw, patch_dim)
        tokens = self.to_patch_embedding(x)
        B2, T, D = tokens.shape  # T = Tin * ph * pw

        # Create positional encodings and add them
        pe = generate_positional_encoding(T, D, tokens.device)  # (1, T, D)
        mem = self.encoder(tokens + pe)                         # (B, T, D)

        # How many tokens we need for the future frames
        tokens_per_frame = ph * pw
        needed = self.pred_length * tokens_per_frame
        # Sanity check: we cannot ask for more future tokens than we have in memory
        assert needed <= T, f"pred_length * ph*pw too large: {needed} > {T}"

        # Take the last `needed` tokens as the representation for the future frames
        mem = mem[:, -needed:, :]  # (B, needed, D)

        # Map back from d_model to patch_dim
        out = self.to_feature_map(mem)  # (B, needed, patch_dim)

        # Reshape tokens back to (B, T_pred, C, H, W)
        out = rearrange(
            out,
            "b (t h w) (p1 p2 c) -> b t c (h p1) (w p2)",
            t=self.pred_length,
            h=ph,
            w=pw,
            p1=self.patch_height,
            p2=self.patch_width,
        )
        return out


class RainPredRNN(nn.Module):
    """Spatio-temporal radar nowcasting model using UNet + temporal Transformer."""
    def __init__(
        self,
        input_dim: int = 1,
        num_hidden: int = 256,
        max_hidden_channels: int = 128,
        patch_height: int = PATCH_HEIGHT,
        patch_width: int = PATCH_WIDTH,
        pred_length: int = PRED_LENGTH,
    ):
        super().__init__()
        # U-Net encoder operating on each input frame independently
        self.encoder = UNet_Encoder(input_dim)
        # U-Net decoder that reconstructs frames from transformer features
        # output_channels == input_dim (e.g. 1 radar channel)
        self.decoder = UNet_Decoder(input_dim)
        # Number of frames to predict
        self.pred_length = pred_length
        # Temporal transformer over encoder features
        self.transformer_block = TemporalTransformerBlock(
            channels=max_hidden_channels,   # must match encoder output channels (128 here)
            d_model=num_hidden,             # transformer embedding dim
            nhead=8,                        # number of attention heads
            num_encoder_layers=3,           # number of transformer layers
            pred_length=pred_length,
            patch_height=patch_height,
            patch_width=patch_width,
        )

    def forward(self, input_sequence: torch.Tensor, pred_length: int):
        """Forward pass of the full model.

        Args:
            input_sequence: input radar sequence of shape (B, T_in, C, H, W).
            pred_length: number of future frames to generate (can override default).

        Returns:
            A tuple (preds, preds_noact) where:
              - preds has shape (B, T_pred, C_out, H, W), with activation applied
              - preds_noact has the same shape, but before the final activation
        """
        B, Tin, C, H, W = input_sequence.size()

        # Collect per-frame encoder outputs
        enc_feats = []    # low-res encoder features for each input frame
        skip1_list = []   # high-res skip features for each input frame
        for t in range(Tin):
            # Encode each frame independently: (B, C, H, W) -> (B, 128, H/2, W/2), (B, 64, H, W)
            x, sk1 = self.encoder(input_sequence[:, t])
            enc_feats.append(x)
            skip1_list.append(sk1)

        # Stack along the temporal dimension: (B, Tin, 128, H/2, W/2)
        enc_feats = torch.stack(enc_feats, dim=1)
        # Stack high-res skips: (B, Tin, 64, H, W)
        skip1 = torch.stack(skip1_list, dim=1)

        # Apply temporal transformer to the low-res encoder features
        pred_feats = self.transformer_block(enc_feats)  # (B, T_pred, 128, H/2, W/2)

        preds = []        # list of activated predictions per frame
        preds_noact = []  # list of raw predictions per frame
        for t in range(pred_length):
            # Decode one predicted feature map and corresponding skip connection
            y, y_no = self.decoder(
                pred_feats[:, t],   # transformer features at time t
                enc_feats[:, t],    # encoder low-res features at time t
                skip1[:, t],        # encoder high-res features at time t
            )
            preds.append(y)
            preds_noact.append(y_no)

        # Stack predictions back into a single tensor: (B, T_pred, C_out, H, W)
        return torch.stack(preds, dim=1), torch.stack(preds_noact, dim=1)
