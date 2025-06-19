import torch


class Config:
    # 训练参数

    DATASET_PATH = "./data/cifar-10-python"  # 数据集根目录
    COMPRESSED_FILE = "cifar-10-python.tar.gz"  # 压缩文件名
    EXTRACTED_DIR = "cifar-10-batches-py"  # 解压后目录

    TIME_EMBED_DIM = 128
    COND_DIM = 256  # TRANSFORMER_DIM + TIME_EMBED_DIM

    EPOCHS = 200
    BATCH_SIZE = 128
    LR = 1e-4
    WEIGHT_DECAY = 1e-4

    # 数据集
    DATASET = "CIFAR10"
    IMG_SIZE = 32
    NORMALIZE_MEAN = (0.5, 0.5, 0.5)
    NORMALIZE_STD = (0.5, 0.5, 0.5)

    # 扩散过程
    TIMESTEPS = 1000
    BETA_SCHEDULE = "cosine"

    # 模型架构
    BASE_DIM = 64
    TRANSFORMER_DIM = 128
    TRANSFORMER_DEPTH = 4
    TRANSFORMER_HEADS = 4
    NOISE_LEVELS = [0.1, 0.3, 0.6, 0.9]  # 多尺度噪声级别

    # 设备设置
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # 保存路径
    SAVE_DIR = "./saved_models"
    LOG_DIR = "./logs"
    SAVE_CHECKPOINT_FREQ = 10  # 每10个epoch保存一次检查点
    GENERATE_SAMPLES_FREQ = 10  # 每10个epoch生成一次样本
    SAVE_BEST_MODEL = True  # 是否保存最佳模型

    # 分布式设置 (单卡训练，但保留配置)
    MASTER_ADDR = "10.152.38.203"
    MASTER_PORT = "10043"
    WORLD_SIZE = 1  # 单卡训练


config = Config()