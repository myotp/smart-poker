"""
德州扑克牌型枚举和评估结果定义
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .card import Card


class HandRank(IntEnum):
    """
    德州扑克牌型枚举（从低到高）

    数值越大，牌型越强
    """
    HIGH_CARD = 0        # 高牌
    ONE_PAIR = 1         # 一对
    TWO_PAIR = 2         # 两对
    THREE_OF_A_KIND = 3  # 三条
    STRAIGHT = 4         # 顺子
    FLUSH = 5            # 同花
    FULL_HOUSE = 6       # 葫芦（三带二）
    FOUR_OF_A_KIND = 7   # 四条（金刚）
    STRAIGHT_FLUSH = 8   # 同花顺
    ROYAL_FLUSH = 9      # 皇家同花顺

    def __str__(self) -> str:
        """返回中文名称"""
        names = {
            HandRank.HIGH_CARD: "高牌",
            HandRank.ONE_PAIR: "一对",
            HandRank.TWO_PAIR: "两对",
            HandRank.THREE_OF_A_KIND: "三条",
            HandRank.STRAIGHT: "顺子",
            HandRank.FLUSH: "同花",
            HandRank.FULL_HOUSE: "葫芦",
            HandRank.FOUR_OF_A_KIND: "四条",
            HandRank.STRAIGHT_FLUSH: "同花顺",
            HandRank.ROYAL_FLUSH: "皇家同花顺",
        }
        return names[self]

    @property
    def english_name(self) -> str:
        """返回英文名称"""
        return self.name.replace("_", " ").title()


@dataclass
class HandResult:
    """
    牌型评估结果

    Attributes:
        rank: 牌型类型
        best_five: 构成该牌型的最佳5张牌（按重要性排序）
        kickers: 用于同牌型比较的踢脚牌点数元组
        score: 综合评分，可直接用于比较大小
    """
    rank: HandRank
    best_five: list["Card"]
    kickers: tuple[int, ...]
    score: int = 0

    def __post_init__(self) -> None:
        """计算综合评分"""
        if self.score == 0:
            # 分数格式：rank * 15^5 + k0 * 15^4 + k1 * 15^3 + k2 * 15^2 + k3 * 15 + k4
            # 固定5个 kicker 位置，不足的用0填充
            # 使用15作为基数（大于最大点数14）确保不会溢出
            padded_kickers = (self.kickers + (0, 0, 0, 0, 0))[:5]
            self.score = self.rank.value
            for k in padded_kickers:
                self.score = self.score * 15 + k

    def __lt__(self, other: "HandResult") -> bool:
        return self.score < other.score

    def __le__(self, other: "HandResult") -> bool:
        return self.score <= other.score

    def __gt__(self, other: "HandResult") -> bool:
        return self.score > other.score

    def __ge__(self, other: "HandResult") -> bool:
        return self.score >= other.score

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HandResult):
            return NotImplemented
        return self.score == other.score

    def __str__(self) -> str:
        cards_str = " ".join(str(c) for c in self.best_five)
        return f"{self.rank} [{cards_str}]"

    def describe(self) -> str:
        """返回详细描述"""
        cards_str = ", ".join(str(c) for c in self.best_five)
        return f"牌型: {self.rank}\n最佳组合: {cards_str}\nKickers: {self.kickers}"
