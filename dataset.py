import os
import tarfile
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import config


def ensure_cifar10_ready():
    """确保CIFAR-10数据集已准备就绪"""
    # 路径设置
    tar_path = os.path.join(config.DATASET_PATH, config.COMPRESSED_FILE)
    extracted_path = os.path.join(config.DATASET_PATH, config.EXTRACTED_DIR)

    # 检查是否需要下载
    if not os.path.exists(tar_path) and not os.path.exists(extracted_path):
        print("Downloading CIFAR-10 dataset...")
        datasets.CIFAR10(root=config.DATASET_PATH, train=True, download=True)
        return

    # 检查是否需要解压
    if os.path.exists(tar_path) and not os.path.exists(extracted_path):
        print(f"Extracting {tar_path}...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=config.DATASET_PATH)
        print("Extraction complete!")


def get_cifar10_loaders():
    """获取CIFAR10数据加载器"""
    # 确保数据集已准备就绪
    ensure_cifar10_ready()

    # 定义数据预处理
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(config.NORMALIZE_MEAN, config.NORMALIZE_STD)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config.NORMALIZE_MEAN, config.NORMALIZE_STD)
    ])

    # 加载数据集
    train_set = datasets.CIFAR10(
        root=config.DATASET_PATH,
        train=True,
        download=False,  # 禁用自动下载
        transform=transform_train
    )

    test_set = datasets.CIFAR10(
        root=config.DATASET_PATH,
        train=False,
        download=False,
        transform=transform_test
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_set,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    return train_loader, test_loader