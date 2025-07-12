import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)].unsqueeze(0).to(x.device)
        return x

class DiffusionBridge(nn.Module):
    def __init__(self, d_model, noise_scale=0.1):
        super().__init__()
        self.noise_scale = noise_scale
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, x):
        noise = torch.randn_like(x) * self.noise_scale
        x_noisy = x + noise
        x_corrected = self.proj(x_noisy)
        return x_corrected

class TransformerDiffusionModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, output_seq_len, dropout, input_seq_len):
        super().__init__()
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.diffusion_bridge = DiffusionBridge(d_model)
        self.output_layer = nn.Linear(input_seq_len * d_model, output_seq_len)

    def forward(self, src, src_mask=None):
        x = self.input_embedding(src)
        x = self.positional_encoding(x)
        output = self.transformer_encoder(x, src_mask)
        output = self.norm(output)
        output = self.diffusion_bridge(output)
        predictions = self.output_layer(output.reshape(output.size(0), -1))
        return predictions
