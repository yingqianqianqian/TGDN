import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
from config import config


class MultiScaleTransformer(nn.Module):
    def __init__(self, in_channels=3, dim=config.TRANSFORMER_DIM, depth=config.TRANSFORMER_DEPTH,
                 heads=config.TRANSFORMER_HEADS):
        super().__init__()
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, dim // 4, 3, stride=2, padding=1),
            nn.GroupNorm(8, dim // 4),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, 1)
        )

        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=dim * 4,
                batch_first=True,
                norm_first=True
            ) for _ in range(depth)
        ])

        num_scales = len(config.NOISE_LEVELS)
        self.scale_fusion = nn.Linear(dim * num_scales, config.TRANSFORMER_DIM)
        self.norm = nn.LayerNorm(config.TRANSFORMER_DIM)

    def forward(self, noisy_imgs):
        features = []
        for img in noisy_imgs:
            x = self.patch_embed(img)
            x = rearrange(x, 'b c h w -> b (h w) c')
            for layer in self.transformer_layers:
                x = layer(x)
            features.append(torch.mean(x, dim=1))

        fused_features = torch.cat(features, dim=1)
        return self.norm(self.scale_fusion(fused_features))


class AttentionGuidanceBlock(nn.Module):
    def __init__(self, dim, out_dim=None):
        super().__init__()
        out_dim = out_dim or dim
        self.norm1 = nn.GroupNorm(8, dim)
        self.conv1 = nn.Conv2d(dim, out_dim, 3, padding=1)

        # 条件投影层
        self.cond_proj = nn.Sequential(
            nn.Linear(config.COND_DIM, out_dim * 2),
            nn.GELU()
        )

        self.norm2 = nn.GroupNorm(8, out_dim)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, padding=1)
        self.attn = nn.MultiheadAttention(out_dim, 4, batch_first=True)

        self.residual = nn.Conv2d(dim, out_dim, 1) if dim != out_dim else nn.Identity()

    def forward(self, x, cond):
        # 确保条件向量形状正确
        if cond.dim() != 2:
            cond = cond.view(cond.size(0), -1)
        if cond.size(1) != config.COND_DIM:
            cond = cond[:, :config.COND_DIM]

        residual = self.residual(x)

        x = self.norm1(x)
        x = F.gelu(x)
        x = self.conv1(x)

        # 应用条件投影
        cond_proj = self.cond_proj(cond)
        scale, shift = cond_proj.chunk(2, dim=-1)

        # 确保scale和shift有正确的维度
        scale = scale.view(scale.size(0), scale.size(1), 1, 1)
        shift = shift.view(shift.size(0), shift.size(1), 1, 1)

        # 应用条件调制
        x = x * (1 + scale) + shift

        batch, c, h, w = x.shape
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        attn_out = rearrange(attn_out, 'b (h w) c -> b c h w', h=h, w=w)

        x = self.norm2(attn_out)
        x = F.gelu(x)
        x = self.conv2(x)

        return x + residual


class TGDN_UNet(nn.Module):
    def __init__(self, in_channels=3, base_dim=config.BASE_DIM):
        super().__init__()
        self.init_conv = nn.Conv2d(in_channels, base_dim, 3, padding=1)

        # 下采样路径
        self.down_pool1 = nn.AvgPool2d(2)
        self.down_block1 = AttentionGuidanceBlock(base_dim, base_dim * 2)

        self.down_pool2 = nn.AvgPool2d(2)
        self.down_block2 = AttentionGuidanceBlock(base_dim * 2, base_dim * 4)

        self.down_pool3 = nn.AvgPool2d(2)
        self.down_block3 = AttentionGuidanceBlock(base_dim * 4, base_dim * 8)

        # 中间层
        self.mid_block1 = AttentionGuidanceBlock(base_dim * 8)
        self.mid_block2 = AttentionGuidanceBlock(base_dim * 8)

        # 上采样路径
        self.up_block1 = AttentionGuidanceBlock(base_dim * 8, base_dim * 4)  # 输出256通道
        self.up_sample1 = nn.Upsample(scale_factor=2, mode='nearest')

        self.up_block2 = AttentionGuidanceBlock(base_dim * 4, base_dim * 2)  # 输出128通道
        self.up_sample2 = nn.Upsample(scale_factor=2, mode='nearest')

        self.up_block3 = AttentionGuidanceBlock(base_dim * 2, base_dim)  # 输出64通道
        self.up_sample3 = nn.Upsample(scale_factor=2, mode='nearest')

        # 输出层
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, base_dim),
            nn.GELU(),
            nn.Conv2d(base_dim, in_channels, 3, padding=1)
        )

        # 时间嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(config.TIME_EMBED_DIM, config.TIME_EMBED_DIM * 2),
            nn.GELU(),
            nn.Linear(config.TIME_EMBED_DIM * 2, config.TIME_EMBED_DIM)
        )

        # 跳跃连接调整层 - 关键修复
        self.skip_conv3 = nn.Conv2d(base_dim * 8, base_dim * 4, 1)  # 512 -> 256
        self.skip_conv2 = nn.Conv2d(base_dim * 4, base_dim * 2, 1)  # 256 -> 128
        self.skip_conv1 = nn.Conv2d(base_dim * 2, base_dim, 1)  # 128 -> 64

    def forward(self, x, t, cond_vector):
        # 确保条件向量形状正确
        cond_vector = cond_vector.view(cond_vector.size(0), -1)
        cond_vector = cond_vector[:, :config.TRANSFORMER_DIM]

        # 时间嵌入
        t_emb = self._timestep_embedding(t, config.TIME_EMBED_DIM)
        t_emb = self.time_embed(t_emb)

        # 连接条件向量和时间嵌入
        cond = torch.cat([cond_vector, t_emb], dim=1)

        # 确保总维度正确
        if cond.size(1) > config.COND_DIM:
            cond = cond[:, :config.COND_DIM]
        elif cond.size(1) < config.COND_DIM:
            padding = torch.zeros(cond.size(0), config.COND_DIM - cond.size(1),
                                  device=cond.device)
            cond = torch.cat([cond, padding], dim=1)

        # 初始卷积
        x0 = self.init_conv(x)

        # 下采样路径
        x0_pooled = self.down_pool1(x0)
        x1 = self.down_block1(x0_pooled, cond)  # 128通道

        x1_pooled = self.down_pool2(x1)
        x2 = self.down_block2(x1_pooled, cond)  # 256通道

        x2_pooled = self.down_pool3(x2)
        x3 = self.down_block3(x2_pooled, cond)  # 512通道

        # 中间处理
        x = self.mid_block1(x3, cond)  # 输入512，输出512
        x = self.mid_block2(x, cond)  # 输入512，输出512

        # 上采样路径 + 跳跃连接 - 关键修复
        # 调整x3的通道数从512到256
        x3_adjusted = self.skip_conv3(x3)
        x = self.up_block1(x, cond) + x3_adjusted  # 256 + 256
        x = self.up_sample1(x)

        # 调整x2的通道数从256到128
        x2_adjusted = self.skip_conv2(x2)
        x = self.up_block2(x, cond) + x2_adjusted  # 128 + 128
        x = self.up_sample2(x)

        # 调整x1的通道数从128到64
        x1_adjusted = self.skip_conv1(x1)
        x = self.up_block3(x, cond) + x1_adjusted  # 64 + 64
        x = self.up_sample3(x)

        # 最后的跳跃连接 (x0是64通道)
        x = x + x0

        return self.out_conv(x)

    def _timestep_embedding(self, t, dim):
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)


class TGDN(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_module = MultiScaleTransformer()
        self.diffusion_unet = TGDN_UNet()
        self.noise_levels = config.NOISE_LEVELS

    def forward(self, x, t):
        noisy_imgs = []
        for noise_level in self.noise_levels:
            noise = torch.randn_like(x) * noise_level
            noisy_imgs.append(x + noise)

        cond_vector = self.transformer_module(noisy_imgs)
        return self.diffusion_unet(x, t, cond_vector)