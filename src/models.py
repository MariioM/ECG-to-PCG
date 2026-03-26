import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import Config
import math


#====================== VAE =======================
class Encoder1D(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, z_dim=32):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(base_channels),
            nn.SiLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(base_channels, base_channels*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(base_channels*2),
            nn.SiLU()
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(base_channels*2, base_channels*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(base_channels*4),
            nn.SiLU()
        )
        self.block4 = nn.Sequential(
            nn.Conv1d(base_channels*4, base_channels*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(base_channels*8),
            nn.SiLU()
        )
        self.block5 = nn.Sequential(
            nn.Conv1d(base_channels*8, base_channels*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(base_channels*8),
            nn.SiLU()
        )
        self.final_conv = nn.Conv1d(base_channels*8, z_dim*2, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        z = self.final_conv(x)
        mu, logvar = torch.chunk(z, 2, dim=1)
        return mu, logvar

class Decoder1D(nn.Module):
    def __init__(self, out_channels=1, base_channels=32, z_dim=32):
        super().__init__()
        self.initial_conv = nn.Conv1d(z_dim, base_channels*8, kernel_size=3, padding=1)
        self.block1 = nn.Sequential(
            nn.ConvTranspose1d(base_channels*8, base_channels*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(base_channels*8),
            nn.SiLU()
        )
        self.block2 = nn.Sequential(
            nn.ConvTranspose1d(base_channels*8, base_channels*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(base_channels*4),
            nn.SiLU()
        )
        self.block3 = nn.Sequential(
            nn.ConvTranspose1d(base_channels*4, base_channels*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(base_channels*2),
            nn.SiLU()
        )
        self.block4 = nn.Sequential(
            nn.ConvTranspose1d(base_channels*2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(base_channels),
            nn.SiLU()
        )
        self.block5 = nn.Sequential(
            nn.ConvTranspose1d(base_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(base_channels),
            nn.SiLU()
        )
        self.final_conv = nn.Conv1d(base_channels, out_channels, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, z):
        x = self.initial_conv(z)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.final_conv(x)
        return self.tanh(x)

class PCGVAE(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, z_dim=16):
        super().__init__()
        self.encoder = Encoder1D(in_channels, base_channels, z_dim)
        self.decoder = Decoder1D(in_channels, base_channels, z_dim)
        self.z_dim = z_dim

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def encode(self, x):
        mu, _ = self.encoder(x)
        return mu

    def decode(self, z, target_len=None):
        recon = self.decoder(z)
        if target_len is not None and recon.shape[-1] != target_len:
            recon = F.interpolate(recon, size=target_len, mode='linear', align_corners=False)
        return recon

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        if recon.shape[-1] != x.shape[-1]:
            recon = F.interpolate(recon, size=x.shape[-1], mode='linear', align_corners=False)
        return recon, mu, logvar
    

#===================== TRANSFORMER RECTIFIED FLOW =====================#

class TBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 6, bias=True)
        )
    
    def forward(self, x, c, return_attn=False):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        x_norm1 = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, attn_weights = self.attn(x_norm1, x_norm1, x_norm1, need_weights=True)
        x = x + gate_msa.unsqueeze(1) * attn_out

        x_norm2 = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm2)

        if return_attn: return x, attn_weights
        return x

class FlowTransformer(nn.Module):
    def __init__(self, in_channels=16, ecg_channels=16, hidden_size=256, depth=8, num_heads=4):
        super().__init__()
        
        self.audio_proj = nn.Conv1d(in_channels, hidden_size, kernel_size=1)
        self.ecg_proj = nn.Conv1d(ecg_channels, hidden_size, kernel_size=1)
        
        self.time_embedder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, 312, hidden_size))
        self.blocks = nn.ModuleList([TBlock(hidden_size, num_heads) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 2, bias=True)
        )
        self.final_linear = nn.Linear(hidden_size, 16) 
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.pos_embedding, std=0.02)
        nn.init.constant_(self.final_linear.weight, 0)
        nn.init.constant_(self.final_linear.bias, 0)
        
    def timestep_embedding(self, t, dim=256):
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(start=0, end=half_dim, dtype=torch.float32) / half_dim
        ).to(t.device)
        args = t[:, None] * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding
        
    def forward(self, x, t, c_ecg, return_attn=False):
        t_emb = self.timestep_embedding(t, self.audio_proj.out_channels)
        t_cond = self.time_embedder(t_emb)

        h_audio = self.audio_proj(x).transpose(1, 2)
        h_ecg = self.ecg_proj(c_ecg).transpose(1, 2)
        h = h_audio + h_ecg
        
        h = h + self.pos_embedding

        for block in self.blocks:
            h = block(h, t_cond)
            
        shift, scale = self.adaLN_modulation(t_cond).chunk(2, dim=1)
        h = self.final_norm(h) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        out = self.final_linear(h).transpose(1, 2)
        return out