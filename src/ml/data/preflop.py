"""
Preflop 手牌强度数据集生成
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.core import Card, Deck
from src.equity import EquityCalculator


def generate_preflop_data(
    n_opponents: int = 20,
    mc_samples: int = 2000,
    seed: int = 42,
    verbose: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    生成 preflop 手牌强度数据

    枚举所有 C(52,2) = 1326 种起手牌，对抗多个随机对手取平均胜率。

    Args:
        n_opponents: 每手牌对抗的随机对手数量
        mc_samples: 每次对抗的蒙特卡洛采样数
        seed: 随机种子
        verbose: 是否打印进度

    Returns:
        (X, y): X 为手牌编码 (1326, 104)，y 为胜率 (1326,)
    """
    rng = np.random.default_rng(seed)

    # 生成所有 52 张牌
    all_cards = [Card(rank, suit) for rank in range(2, 15) for suit in range(4)]

    # 枚举所有 C(52,2) 种起手牌
    all_hands = list(combinations(all_cards, 2))
    n_hands = len(all_hands)

    X = np.zeros((n_hands, 104), dtype=np.float32)
    y = np.zeros(n_hands, dtype=np.float32)

    for i, hand in enumerate(all_hands):
        hand = list(hand)

        # 编码手牌
        X[i] = np.concatenate([c.to_vector("onehot") for c in hand])

        # 剩余牌堆
        remaining = [c for c in all_cards if c not in hand]

        # 对抗多个随机对手
        win_rates = []
        for _ in range(n_opponents):
            # 随机选择对手手牌
            opp_indices = rng.choice(len(remaining), size=2, replace=False)
            opponent = [remaining[opp_indices[0]], remaining[opp_indices[1]]]

            # 计算胜率
            r1, _ = EquityCalculator.preflop(
                hand, opponent, method="monte_carlo", samples=mc_samples, rng=rng
            )
            win_rates.append(r1.win)

        # 取平均胜率
        y[i] = np.mean(win_rates)

        if verbose and (i + 1) % 100 == 0:
            print(f"进度: {i + 1}/{n_hands} ({(i + 1) / n_hands * 100:.1f}%)")

    return X, y


def save_preflop_data(
    path: str | Path,
    n_opponents: int = 20,
    mc_samples: int = 2000,
    seed: int = 42
) -> None:
    """生成并保存 preflop 数据"""
    X, y = generate_preflop_data(n_opponents, mc_samples, seed)
    np.savez(path, X=X, y=y)
    print(f"数据已保存到: {path}")


def load_preflop_data(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """加载已保存的 preflop 数据"""
    data = np.load(path)
    return data["X"], data["y"]


class PreflopDataset(Dataset):
    """
    Preflop 手牌强度数据集

    支持从缓存文件加载或实时生成数据。
    """

    def __init__(
        self,
        cache_path: str | Path | None = None,
        n_opponents: int = 20,
        mc_samples: int = 2000,
        seed: int = 42,
        regenerate: bool = False
    ):
        """
        初始化数据集

        Args:
            cache_path: 缓存文件路径，如果存在则从缓存加载
            n_opponents: 每手牌对抗的随机对手数量
            mc_samples: 每次对抗的蒙特卡洛采样数
            seed: 随机种子
            regenerate: 是否强制重新生成（忽略缓存）
        """
        if cache_path is not None:
            cache_path = Path(cache_path)

        # 尝试从缓存加载
        if cache_path is not None and cache_path.exists() and not regenerate:
            print(f"从缓存加载数据: {cache_path}")
            self.X, self.y = load_preflop_data(cache_path)
        else:
            # 生成数据
            print("生成 preflop 数据...")
            self.X, self.y = generate_preflop_data(
                n_opponents=n_opponents,
                mc_samples=mc_samples,
                seed=seed
            )
            # 保存到缓存
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                save_preflop_data(cache_path, n_opponents, mc_samples, seed)

        # 转换为 tensor
        self.X = torch.from_numpy(self.X)
        self.y = torch.from_numpy(self.y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

    @staticmethod
    def get_hand_from_index(idx: int) -> list[Card]:
        """根据数据集索引获取对应的手牌"""
        all_cards = [Card(rank, suit) for rank in range(2, 15) for suit in range(4)]
        all_hands = list(combinations(all_cards, 2))
        return list(all_hands[idx])
