"""
手牌强度预测模型
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.core import Card


class HandStrengthMLP(nn.Module):
    """
    手牌强度预测 MLP

    输入：2张牌的 one-hot 编码 (104维)
    输出：胜率 (0-1)
    """

    def __init__(
        self,
        input_dim: int = 104,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2
    ):
        """
        初始化模型

        Args:
            input_dim: 输入维度，默认 104 (2×52 one-hot)
            hidden_dims: 隐藏层维度列表，默认 [256, 128, 64]
            dropout: Dropout 概率
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        layers: list[nn.Module] = []
        prev_dim = input_dim

        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim

        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 (batch, 104)

        Returns:
            预测胜率 (batch,)
        """
        return self.net(x).squeeze(-1)


class HandStrengthPredictor:
    """
    手牌强度预测器（推理用）

    封装模型加载和预测逻辑，提供简洁的接口。
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        model: HandStrengthMLP | None = None,
        device: str | None = None
    ):
        """
        初始化预测器

        Args:
            model_path: 模型权重文件路径
            model: 直接传入模型实例（与 model_path 二选一）
            device: 推理设备，默认自动选择
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = HandStrengthMLP()
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            raise ValueError("必须提供 model_path 或 model")

        self.model = self.model.to(device)
        self.model.eval()

    def _encode_hand(self, hand: list[Card]) -> torch.Tensor:
        """将手牌编码为张量"""
        x = np.concatenate([c.to_vector("onehot") for c in hand])
        return torch.from_numpy(x).float()

    def predict(self, hand: list[Card]) -> float:
        """
        预测单手牌的强度

        Args:
            hand: 2张牌的列表

        Returns:
            预测胜率 (0-1)
        """
        if len(hand) != 2:
            raise ValueError(f"手牌必须为2张，当前: {len(hand)}")

        x = self._encode_hand(hand).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.model(x).item()

    def predict_batch(self, hands: list[list[Card]]) -> list[float]:
        """
        批量预测手牌强度

        Args:
            hands: 多手牌的列表

        Returns:
            预测胜率列表
        """
        X = torch.stack([self._encode_hand(hand) for hand in hands])
        X = X.to(self.device)
        with torch.no_grad():
            return self.model(X).cpu().tolist()

    def rank_hands(self, hands: list[list[Card]]) -> list[tuple[list[Card], float]]:
        """
        对多手牌进行排名

        Args:
            hands: 多手牌的列表

        Returns:
            按强度降序排列的 (手牌, 胜率) 列表
        """
        strengths = self.predict_batch(hands)
        ranked = sorted(zip(hands, strengths), key=lambda x: x[1], reverse=True)
        return ranked
