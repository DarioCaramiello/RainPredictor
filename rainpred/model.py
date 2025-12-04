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
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.conv2(x)
        skip1 = x
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x, skip1

class UNet_Decoder(nn.Module):
    """Simple U-Net decoder that merges encoder and transformer features."""
    def __init__(self, output_channels: int):
        super().__init__()
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, 2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, skip0: torch.Tensor, skip1: torch.Tensor):
        x = torch.cat([skip0, x], dim=1)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.up1(x)
        x = torch.cat([skip1, x], dim=1)
        x = self.conv7(x)
        x = self.conv8(x)
        x_last = x
        x = torch.tanh(x)
        return x, x_last

def generate_positional_encoding(seq_len: int, d_model: int, device: torch.device) -> torch.Tensor:
    """Create standard Transformer sinusoidal positional encodings."""
    pe = torch.zeros(seq_len, d_model, device=device)
    position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)

class TemporalTransformerBlock(nn.Module):
    """Temporal Transformer over spatio-temporal patches of encoder features."""
    def __init__(self, channels: int, d_model: int, nhead: int, num_encoder_layers: int,
                 pred_length: int, patch_height: int = PATCH_HEIGHT, patch_width: int = PATCH_WIDTH):
        super().__init__()
        self.pred_length = pred_length
        self.patch_height = patch_height
        self.patch_width = patch_width
        patch_dim = channels * patch_height * patch_width
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b t c (h p1) (w p2) -> b (t h w) (p1 p2 c)", p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=num_encoder_layers,
        )
        self.to_feature_map = nn.Sequential(
            nn.Linear(d_model, patch_dim),
            nn.LayerNorm(patch_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, Tin, C, H, W = x.shape
        ph = H // self.patch_height
        pw = W // self.patch_width
        tokens = self.to_patch_embedding(x)
        B2, T, D = tokens.shape
        pe = generate_positional_encoding(T, D, tokens.device)
        mem = self.encoder(tokens + pe)
        tokens_per_frame = ph * pw
        needed = self.pred_length * tokens_per_frame
        assert needed <= T, f"pred_length * ph*pw too large: {needed} > {T}"
        mem = mem[:, -needed:, :]
        out = self.to_feature_map(mem)
        out = rearrange(
            out,
            "b (t h w) (p1 p2 c) -> b t c (h p1) (w p2)",
            t=self.pred_length, h=ph, w=pw, p1=self.patch_height, p2=self.patch_width,
        )
        return out

class RainPredRNN(nn.Module):
    """Spatio-temporal radar nowcasting model using UNet + temporal Transformer."""
    def __init__(self, input_dim: int = 1, num_hidden: int = 256,
                 max_hidden_channels: int = 128, patch_height: int = PATCH_HEIGHT,
                 patch_width: int = PATCH_WIDTH, pred_length: int = PRED_LENGTH):
        super().__init__()
        self.encoder = UNet_Encoder(input_dim)
        self.decoder = UNet_Decoder(input_dim)
        self.pred_length = pred_length
        self.transformer_block = TemporalTransformerBlock(
            channels=max_hidden_channels,
            d_model=num_hidden,
            nhead=8,
            num_encoder_layers=3,
            pred_length=pred_length,
            patch_height=patch_height,
            patch_width=patch_width,
        )

    def forward(self, input_sequence: torch.Tensor, pred_length: int):
        B, Tin, C, H, W = input_sequence.size()
        enc_feats = []
        skip1_list = []
        for t in range(Tin):
            x, sk1 = self.encoder(input_sequence[:, t])
            enc_feats.append(x)
            skip1_list.append(sk1)
        enc_feats = torch.stack(enc_feats, dim=1)
        skip1 = torch.stack(skip1_list, dim=1)
        pred_feats = self.transformer_block(enc_feats)
        preds = []
        preds_noact = []
        for t in range(pred_length):
            y, y_no = self.decoder(pred_feats[:, t], enc_feats[:, t], skip1[:, t])
            preds.append(y)
            preds_noact.append(y_no)
        return torch.stack(preds, dim=1), torch.stack(preds_noact, dim=1)
