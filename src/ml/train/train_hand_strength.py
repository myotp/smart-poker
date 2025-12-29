"""
手牌强度预测模型训练脚本
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.ml.data import PreflopDataset
from src.ml.models import HandStrengthMLP


@dataclass
class TrainConfig:
    """训练配置"""
    # 数据
    cache_path: str = "data/preflop_cache.npz"
    n_opponents: int = 20
    mc_samples: int = 2000
    train_ratio: float = 0.8
    batch_size: int = 64

    # 模型
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.2

    # 优化
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 100
    early_stop_patience: int = 10

    # 其他
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    seed: int = 42
    save_dir: str = "checkpoints"


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str
) -> float:
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)

    return total_loss / len(loader.dataset)  # type: ignore


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str
) -> tuple[float, float]:
    """评估模型"""
    model.eval()
    total_loss = 0.0
    all_preds: list[float] = []
    all_labels: list[float] = []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred, y)

            total_loss += loss.item() * len(y)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)  # type: ignore
    mae = float(np.mean(np.abs(np.array(all_preds) - np.array(all_labels))))

    return avg_loss, mae


def train(config: TrainConfig | None = None) -> HandStrengthMLP:
    """
    训练手牌强度预测模型

    Args:
        config: 训练配置，默认使用 TrainConfig()

    Returns:
        训练好的模型
    """
    if config is None:
        config = TrainConfig()

    # 设置随机种子
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    print(f"设备: {config.device}")
    print(f"配置: {config}")

    # 加载数据
    print("\n加载数据...")
    dataset = PreflopDataset(
        cache_path=config.cache_path,
        n_opponents=config.n_opponents,
        mc_samples=config.mc_samples,
        seed=config.seed
    )

    # 划分训练集和验证集
    train_size = int(len(dataset) * config.train_ratio)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed)
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")

    # 创建模型
    model = HandStrengthMLP(
        hidden_dims=config.hidden_dims,
        dropout=config.dropout
    ).to(config.device)

    print(f"\n模型结构:\n{model}")

    # 优化器和损失函数
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    criterion = nn.MSELoss()

    # 训练循环
    best_val_loss = float("inf")
    patience_counter = 0
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n开始训练...")
    for epoch in range(config.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config.device)
        val_loss, val_mae = evaluate(model, val_loader, criterion, config.device)

        print(
            f"Epoch {epoch + 1:3d}/{config.epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val MAE: {val_mae:.4f} ({val_mae * 100:.2f}%)"
        )

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_dir / "hand_strength_best.pt")
            print(f"  -> 保存最佳模型 (Val Loss: {val_loss:.6f})")
        else:
            patience_counter += 1

        # 早停
        if patience_counter >= config.early_stop_patience:
            print(f"\n早停: {config.early_stop_patience} epochs 没有改善")
            break

    # 加载最佳模型
    model.load_state_dict(torch.load(save_dir / "hand_strength_best.pt"))

    # 最终评估
    final_loss, final_mae = evaluate(model, val_loader, criterion, config.device)
    print(f"\n最终评估 - Loss: {final_loss:.6f}, MAE: {final_mae:.4f} ({final_mae * 100:.2f}%)")

    return model


if __name__ == "__main__":
    train()
