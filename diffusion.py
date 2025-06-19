import torch
import torch.nn as nn
import numpy as np
from config import config
import torchvision
import torch.nn.functional as F

class Diffusion:
    def __init__(self, timesteps=config.TIMESTEPS, schedule=config.BETA_SCHEDULE):
        self.timesteps = timesteps

        if schedule == "linear":
            self.betas = torch.linspace(1e-4, 0.02, timesteps)
        elif schedule == "cosine":
            # 余弦调度
            s = 0.008
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # 注册缓冲区以便在不同设备间移动
        self.register_buffer('betas', self.betas)
        self.register_buffer('alphas', self.alphas)
        self.register_buffer('alphas_cumprod', self.alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', self.sqrt_alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', self.sqrt_one_minus_alphas_cumprod)

    def register_buffer(self, name, tensor):
        """用于在不同设备间移动张量"""
        setattr(self, name, tensor.to(config.DEVICE))

    def sample(self, model, num_samples, img_size=32, device=None):
        """从纯噪声生成样本"""
        device = device or config.DEVICE
        model.eval()

        # 从纯噪声开始
        x = torch.randn(num_samples, 3, img_size, img_size, device=device)

        # 逐步去噪
        for t in range(self.timesteps - 1, -1, -1):
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            with torch.no_grad():
                predicted_noise = model(x, t_batch)

            # 计算系数
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t]
            sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t]

            # 更新图像
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = (1 / torch.sqrt(alpha)) * (
                    x - ((1 - alpha) / sqrt_one_minus_alpha_cumprod) * predicted_noise
            ) + torch.sqrt(self.betas[t]) * noise

        # 将图像值限制在[-1, 1]范围内
        x = torch.clamp(x, -1.0, 1.0)
        return x

    def generate_image_grid(self, samples, nrow=8, normalize=True):
        """将样本转换为网格图像"""
        if normalize:
            # 反归一化到 [0, 1] 范围
            samples = samples * 0.5 + 0.5

        # 创建网格
        grid = torchvision.utils.make_grid(samples, nrow=nrow)
        return grid

    def forward_diffusion(self, x0, t):
        """添加噪声到输入图像"""
        noise = torch.randn_like(x0)
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        noisy_x = sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha * noise
        return noisy_x, noise

    def loss_fn(self, model, x0, t):
        """计算扩散损失"""
        noisy_x, noise = self.forward_diffusion(x0, t)
        predicted_noise = model(noisy_x, t)
        return F.mse_loss(predicted_noise, noise)