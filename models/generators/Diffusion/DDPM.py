import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── UTILS ───────────────────────────────────────────────────────────────────

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

# ─── BLOCKS ──────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch),
        )
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.drop  = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip  = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return h + self.skip(x)

class SelfAttention(nn.Module):
    def __init__(self, ch, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.attn = nn.MultiheadAttention(ch, num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W).transpose(1, 2)
        h, _ = self.attn(h, h, h)
        return x + h.transpose(1, 2).view(B, C, H, W)

class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
    def forward(self, x): return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)
    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))

# ─── ARCHITECTURE ────────────────────────────────────────────────────────────

class UNet(nn.Module):
    def __init__(self, in_ch=3, ch1=64, ch2=128, ch3=192, ch4=256, time_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.init_conv = nn.Conv2d(in_ch, ch1, 3, padding=1)

        self.down1a = ResBlock(ch1, ch1, time_dim)
        self.down1b = ResBlock(ch1, ch1, time_dim)
        self.pool1  = Downsample(ch1)
        self.down2a = ResBlock(ch1, ch2, time_dim)
        self.down2b = ResBlock(ch2, ch2, time_dim)
        self.pool2  = Downsample(ch2)
        self.down3a = ResBlock(ch2, ch3, time_dim)
        self.down3b = ResBlock(ch3, ch3, time_dim)
        self.pool3  = Downsample(ch3)
        self.down4a = ResBlock(ch3, ch4, time_dim)
        self.attn4  = SelfAttention(ch4)
        self.down4b = ResBlock(ch4, ch4, time_dim)
        self.pool4  = Downsample(ch4)

        self.mid1 = ResBlock(ch4, ch4, time_dim)
        self.mid_attn = SelfAttention(ch4)
        self.mid2 = ResBlock(ch4, ch4, time_dim)

        self.up4   = Upsample(ch4)
        self.up4a  = ResBlock(ch4 + ch4, ch4, time_dim)
        self.uattn4 = SelfAttention(ch4)
        self.up4b  = ResBlock(ch4, ch3, time_dim)
        self.up3   = Upsample(ch3)
        self.up3a  = ResBlock(ch3 + ch3, ch3, time_dim)
        self.up3b  = ResBlock(ch3, ch2, time_dim)
        self.up2   = Upsample(ch2)
        self.up2a  = ResBlock(ch2 + ch2, ch2, time_dim)
        self.up2b  = ResBlock(ch2, ch1, time_dim)
        self.up1   = Upsample(ch1)
        self.up1a  = ResBlock(ch1 + ch1, ch1, time_dim)
        self.up1b  = ResBlock(ch1, ch1, time_dim)

        self.out_norm = nn.GroupNorm(8, ch1)
        self.out_conv = nn.Conv2d(ch1, in_ch, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x = self.init_conv(x)
        d1 = self.down1b(self.down1a(x, t_emb), t_emb)
        d2 = self.down2b(self.down2a(self.pool1(d1), t_emb), t_emb)
        d3 = self.down3b(self.down3a(self.pool2(d2), t_emb), t_emb)
        d4 = self.down4b(self.attn4(self.down4a(self.pool3(d3), t_emb)), t_emb)
        h = self.mid2(self.mid_attn(self.mid1(self.pool4(d4), t_emb)), t_emb)
        h = self.up4b(self.uattn4(self.up4a(torch.cat([self.up4(h), d4], 1), t_emb)), t_emb)
        h = self.up3b(self.up3a(torch.cat([self.up3(h), d3], 1), t_emb), t_emb)
        h = self.up2b(self.up2a(torch.cat([self.up2(h), d2], 1), t_emb), t_emb)
        h = self.up1b(self.up1a(torch.cat([self.up1(h), d1], 1), t_emb), t_emb)
        return self.out_conv(F.silu(self.out_norm(h)))