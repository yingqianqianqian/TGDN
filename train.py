import torch
import torch.optim as optim
import torch.nn as nn
import os
import time
import logging
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision
from PIL import Image
import numpy as np

# 导入自定义模块
from model import TGDN
from diffusion import Diffusion
from dataset import get_cifar10_loaders
from config import config

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建保存目录
os.makedirs(config.SAVE_DIR, exist_ok=True)
os.makedirs(config.LOG_DIR, exist_ok=True)


def setup_device():
    """设置训练设备"""
    device = torch.device(config.DEVICE)
    logger.info(f"Using device: {device}")

    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()

    return device


def generate_and_save_samples(model, diffusion, epoch, num_samples=16, save_dir="./samples"):
    """生成并保存样本图片"""
    os.makedirs(save_dir, exist_ok=True)

    # 生成样本
    samples = diffusion.sample(model, num_samples, img_size=32)

    # 转换为网格图像
    grid = diffusion.generate_image_grid(samples, nrow=int(num_samples ** 0.5))

    # 转换为PIL图像
    grid = grid.cpu().permute(1, 2, 0).numpy()
    grid = (grid * 255).astype(np.uint8)
    img = Image.fromarray(grid)

    # 保存图片
    img_path = os.path.join(save_dir, f"samples_epoch_{epoch:03d}.png")
    img.save(img_path)
    logger.info(f"Saved samples to {img_path}")

    # 同时保存到TensorBoard
    writer.add_image("Generated_Samples", torch.tensor(grid).permute(2, 0, 1), epoch)

    return img


def train():
    """主训练函数"""
    # 初始化
    device = setup_device()
    global writer
    writer = SummaryWriter(log_dir=config.LOG_DIR)

    # 加载数据
    train_loader, _ = get_cifar10_loaders()
    logger.info("Dataset loaded successfully")

    # 初始化模型和扩散过程
    model = TGDN().to(device)
    diffusion = Diffusion(timesteps=config.TIMESTEPS, schedule=config.BETA_SCHEDULE)

    # 优化器和学习率调度器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.EPOCHS * len(train_loader)
    )

    # 训练循环
    best_loss = float('inf')
    global_step = 0

    for epoch in range(config.EPOCHS):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.EPOCHS}") as pbar:
            for batch_idx, (images, _) in enumerate(pbar):
                images = images.to(device)

                # 采样时间步
                t = torch.randint(0, config.TIMESTEPS, (images.size(0),), device=device).long()

                # 前向传播和损失计算
                optimizer.zero_grad()
                loss = diffusion.loss_fn(model, images, t)

                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                # 记录损失
                epoch_loss += loss.item()
                global_step += 1
                writer.add_scalar('Train/Loss', loss.item(), global_step)
                writer.add_scalar('Train/LR', scheduler.get_last_lr()[0], global_step)

                pbar.set_postfix(loss=loss.item())

        # 计算平均损失
        epoch_loss /= len(train_loader)
        epoch_time = time.time() - start_time

        logger.info(f"Epoch {epoch + 1}/{config.EPOCHS} | Loss: {epoch_loss:.4f} | Time: {epoch_time:.2f}s")
        writer.add_scalar('Epoch/Loss', epoch_loss, epoch + 1)

        # 保存最佳模型（如果启用）
        if config.SAVE_BEST_MODEL and epoch_loss < best_loss:
            best_loss = epoch_loss
            save_path = os.path.join(config.SAVE_DIR, f"best_model_epoch_{epoch + 1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, save_path)
            logger.info(f"Saved best model to {save_path}")

        # 定期保存检查点
        if (epoch + 1) % config.SAVE_CHECKPOINT_FREQ == 0:
            save_path = os.path.join(config.SAVE_DIR, f"checkpoint_epoch_{epoch + 1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'scheduler_state_dict': scheduler.state_dict(),
            }, save_path)
            logger.info(f"Saved checkpoint to {save_path}")

        # 定期生成样本
        if (epoch + 1) % config.GENERATE_SAMPLES_FREQ == 0:
            try:
                generate_and_save_samples(
                    model, diffusion, epoch + 1,
                    num_samples=16,
                    save_dir=os.path.join(config.SAVE_DIR, "samples")
                )
            except Exception as e:
                logger.error(f"Error generating samples: {str(e)}")

    # 生成最终样本
    try:
        generate_and_save_samples(
            model, diffusion, config.EPOCHS,
            num_samples=16,
            save_dir=os.path.join(config.SAVE_DIR, "samples")
        )
    except Exception as e:
        logger.error(f"Error generating final samples: {str(e)}")

    # 保存最终模型
    final_save_path = os.path.join(config.SAVE_DIR, "final_model.pth")
    torch.save(model.state_dict(), final_save_path)
    logger.info(f"Training completed. Final model saved to {final_save_path}")

    writer.close()


if __name__ == "__main__":
    train()