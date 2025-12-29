"""
扑克牌 Card 类定义
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np


@dataclass(frozen=True, order=True)
class Card:
    """
    表示一张扑克牌

    Attributes:
        rank: 点数 2-14 (2-10, J=11, Q=12, K=13, A=14)
        suit: 花色 0-3 (0=♠黑桃, 1=♥红桃, 2=♦方块, 3=♣梅花)
    """
    rank: int
    suit: int

    # 类常量
    RANK_SYMBOLS: ClassVar[dict[int, str]] = {
        2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
        10: 'T', 11: 'J', 12: 'Q', 13: 'K', 14: 'A'
    }
    RANK_FROM_SYMBOL: ClassVar[dict[str, int]] = {
        '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
        'T': 10, 't': 10, '10': 10,
        'J': 11, 'j': 11,
        'Q': 12, 'q': 12,
        'K': 13, 'k': 13,
        'A': 14, 'a': 14
    }
    SUIT_SYMBOLS: ClassVar[dict[int, str]] = {
        0: '♠', 1: '♥', 2: '♦', 3: '♣'
    }
    SUIT_FROM_SYMBOL: ClassVar[dict[str, int]] = {
        's': 0, 'S': 0, '♠': 0,  # 黑桃 Spade
        'h': 1, 'H': 1, '♥': 1,  # 红桃 Heart
        'd': 2, 'D': 2, '♦': 2,  # 方块 Diamond
        'c': 3, 'C': 3, '♣': 3   # 梅花 Club
    }

    def __post_init__(self) -> None:
        """验证牌的有效性"""
        if not (2 <= self.rank <= 14):
            raise ValueError(f"rank 必须在 2-14 之间，当前值: {self.rank}")
        if not (0 <= self.suit <= 3):
            raise ValueError(f"suit 必须在 0-3 之间，当前值: {self.suit}")

    @classmethod
    def from_str(cls, s: str) -> "Card":
        """
        从字符串创建牌

        格式: {rank}{suit}
        rank: 2-9, T(10), J, Q, K, A
        suit: s(黑桃), h(红桃), d(方块), c(梅花)

        示例:
            Card.from_str("As")  -> 黑桃A
            Card.from_str("Th")  -> 红桃10
            Card.from_str("2c")  -> 梅花2
        """
        s = s.strip()
        if len(s) < 2:
            raise ValueError(f"无效的牌字符串: {s}")

        # 解析点数（可能是1-2个字符）
        if s[:2] == "10":
            rank_str = "10"
            suit_str = s[2:]
        else:
            rank_str = s[0]
            suit_str = s[1:]

        if rank_str not in cls.RANK_FROM_SYMBOL:
            raise ValueError(f"无效的点数: {rank_str}")
        if len(suit_str) != 1 or suit_str not in cls.SUIT_FROM_SYMBOL:
            raise ValueError(f"无效的花色: {suit_str}")

        return cls(
            rank=cls.RANK_FROM_SYMBOL[rank_str],
            suit=cls.SUIT_FROM_SYMBOL[suit_str]
        )

    def __str__(self) -> str:
        """返回可读的字符串表示，如 A♠"""
        return f"{self.RANK_SYMBOLS[self.rank]}{self.SUIT_SYMBOLS[self.suit]}"

    def __repr__(self) -> str:
        """返回简洁的代码表示"""
        suit_char = ['s', 'h', 'd', 'c'][self.suit]
        return f"Card.from_str('{self.RANK_SYMBOLS[self.rank]}{suit_char}')"

    def to_short_str(self) -> str:
        """返回简短字符串，如 As, Th, 2c"""
        suit_char = ['s', 'h', 'd', 'c'][self.suit]
        return f"{self.RANK_SYMBOLS[self.rank]}{suit_char}"

    def to_index(self) -> int:
        """
        转换为 0-51 的唯一索引
        用于查表法优化和向量化
        """
        return (self.rank - 2) * 4 + self.suit

    @classmethod
    def from_index(cls, index: int) -> "Card":
        """从 0-51 的索引创建牌"""
        if not (0 <= index <= 51):
            raise ValueError(f"索引必须在 0-51 之间，当前值: {index}")
        rank = index // 4 + 2
        suit = index % 4
        return cls(rank, suit)

    def to_vector(self, encoding: str = "onehot") -> np.ndarray:
        """
        转换为向量表示，供神经网络使用

        Args:
            encoding: 编码方式
                - "onehot": 52维 one-hot 向量
                - "split": 17维向量 (13维点数 one-hot + 4维花色 one-hot)
                - "simple": 2维向量 [rank/14, suit/3]

        Returns:
            numpy 数组
        """
        if encoding == "onehot":
            vec = np.zeros(52, dtype=np.float32)
            vec[self.to_index()] = 1.0
            return vec
        elif encoding == "split":
            vec = np.zeros(17, dtype=np.float32)
            vec[self.rank - 2] = 1.0  # 点数 one-hot (13维)
            vec[13 + self.suit] = 1.0  # 花色 one-hot (4维)
            return vec
        elif encoding == "simple":
            return np.array([self.rank / 14.0, self.suit / 3.0], dtype=np.float32)
        else:
            raise ValueError(f"未知的编码方式: {encoding}")


class Deck:
    """一副标准的52张扑克牌"""

    def __init__(self) -> None:
        """创建一副新牌"""
        self.cards: list[Card] = [
            Card(rank, suit)
            for rank in range(2, 15)
            for suit in range(4)
        ]
        self._index = 0

    def shuffle(self, rng: np.random.Generator | None = None) -> "Deck":
        """洗牌"""
        if rng is None:
            rng = np.random.default_rng()
        rng.shuffle(self.cards)  # type: ignore
        self._index = 0
        return self

    def deal(self, n: int = 1) -> list[Card]:
        """发牌"""
        if self._index + n > len(self.cards):
            raise ValueError(f"剩余牌不足: 需要 {n} 张，剩余 {len(self.cards) - self._index} 张")
        dealt = self.cards[self._index:self._index + n]
        self._index += n
        return dealt

    def remaining(self) -> int:
        """剩余牌数"""
        return len(self.cards) - self._index

    def reset(self) -> "Deck":
        """重置发牌位置"""
        self._index = 0
        return self
