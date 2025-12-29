"""
Equity 计算器主类
"""

from typing import Literal

import numpy as np

from src.core import Card

from .enumerate import enumerate_equity
from .monte_carlo import monte_carlo_equity
from .result import EquityResult


class EquityCalculator:
    """
    德州扑克 Equity（胜率）计算器

    支持两种计算方法：
    - enumerate: 完全枚举，精确但 preflop 较慢
    - monte_carlo: 蒙特卡洛模拟，快速但有误差
    """

    @staticmethod
    def preflop(
        hand1: list[Card],
        hand2: list[Card],
        method: Literal["enumerate", "monte_carlo"] = "enumerate",
        samples: int = 10000,
        rng: np.random.Generator | None = None
    ) -> tuple[EquityResult, EquityResult]:
        """
        计算 preflop（无公共牌）对抗胜率

        Args:
            hand1: 玩家1的手牌（2张）
            hand2: 玩家2的手牌（2张）
            method: 计算方法，"enumerate" 或 "monte_carlo"
            samples: 蒙特卡洛采样次数（仅 monte_carlo 时有效）
            rng: 随机数生成器（仅 monte_carlo 时有效）

        Returns:
            (玩家1的结果, 玩家2的结果)

        Examples:
            >>> from src.core import Card
            >>> hand1 = [Card.from_str("Ah"), Card.from_str("Kh")]
            >>> hand2 = [Card.from_str("Qc"), Card.from_str("Qd")]
            >>> r1, r2 = EquityCalculator.preflop(hand1, hand2)
            >>> print(f"AKs vs QQ: {r1.win_percent} vs {r2.win_percent}")
        """
        return EquityCalculator.with_board(
            hand1, hand2, board=[], method=method, samples=samples, rng=rng
        )

    @staticmethod
    def with_board(
        hand1: list[Card],
        hand2: list[Card],
        board: list[Card],
        method: Literal["enumerate", "monte_carlo"] = "enumerate",
        samples: int = 10000,
        rng: np.random.Generator | None = None
    ) -> tuple[EquityResult, EquityResult]:
        """
        计算有公共牌时的对抗胜率

        Args:
            hand1: 玩家1的手牌（2张）
            hand2: 玩家2的手牌（2张）
            board: 公共牌（0-5张）
            method: 计算方法，"enumerate" 或 "monte_carlo"
            samples: 蒙特卡洛采样次数（仅 monte_carlo 时有效）
            rng: 随机数生成器（仅 monte_carlo 时有效）

        Returns:
            (玩家1的结果, 玩家2的结果)

        Examples:
            >>> from src.core import Card
            >>> hand1 = [Card.from_str("Ah"), Card.from_str("Kh")]
            >>> hand2 = [Card.from_str("Qc"), Card.from_str("Qd")]
            >>> board = [Card.from_str("Kc"), Card.from_str("7s"), Card.from_str("2d")]
            >>> r1, r2 = EquityCalculator.with_board(hand1, hand2, board)
            >>> print(f"AKs vs QQ on K72: {r1.win_percent} vs {r2.win_percent}")
        """
        if method == "enumerate":
            return enumerate_equity(hand1, hand2, board)
        elif method == "monte_carlo":
            return monte_carlo_equity(hand1, hand2, board, samples=samples, rng=rng)
        else:
            raise ValueError(f"未知的计算方法: {method}，支持 'enumerate' 或 'monte_carlo'")

    @staticmethod
    def flop(
        hand1: list[Card],
        hand2: list[Card],
        flop: list[Card],
        method: Literal["enumerate", "monte_carlo"] = "enumerate",
        samples: int = 10000
    ) -> tuple[EquityResult, EquityResult]:
        """
        计算 flop（3张公共牌）时的对抗胜率

        Args:
            hand1: 玩家1的手牌（2张）
            hand2: 玩家2的手牌（2张）
            flop: flop 公共牌（3张）
            method: 计算方法
            samples: 蒙特卡洛采样次数

        Returns:
            (玩家1的结果, 玩家2的结果)
        """
        if len(flop) != 3:
            raise ValueError(f"Flop 必须为3张牌，当前: {len(flop)}")
        return EquityCalculator.with_board(hand1, hand2, flop, method=method, samples=samples)

    @staticmethod
    def turn(
        hand1: list[Card],
        hand2: list[Card],
        board: list[Card],
        method: Literal["enumerate", "monte_carlo"] = "enumerate",
        samples: int = 10000
    ) -> tuple[EquityResult, EquityResult]:
        """
        计算 turn（4张公共牌）时的对抗胜率

        Args:
            hand1: 玩家1的手牌（2张）
            hand2: 玩家2的手牌（2张）
            board: 公共牌（4张，flop + turn）
            method: 计算方法
            samples: 蒙特卡洛采样次数

        Returns:
            (玩家1的结果, 玩家2的结果)
        """
        if len(board) != 4:
            raise ValueError(f"Turn 必须为4张公共牌，当前: {len(board)}")
        return EquityCalculator.with_board(hand1, hand2, board, method=method, samples=samples)

    @staticmethod
    def river(
        hand1: list[Card],
        hand2: list[Card],
        board: list[Card]
    ) -> tuple[EquityResult, EquityResult]:
        """
        计算 river（5张公共牌）时的对抗结果

        Args:
            hand1: 玩家1的手牌（2张）
            hand2: 玩家2的手牌（2张）
            board: 公共牌（5张）

        Returns:
            (玩家1的结果, 玩家2的结果) - 结果为确定性的 (1,0,0) 或 (0,1,0) 或 (0,0,1)
        """
        if len(board) != 5:
            raise ValueError(f"River 必须为5张公共牌，当前: {len(board)}")
        return EquityCalculator.with_board(hand1, hand2, board)
