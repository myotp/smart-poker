"""
蒙特卡洛模拟法计算 Equity
"""

import numpy as np

from src.core import Card, HandEvaluator

from .enumerate import get_remaining_deck
from .result import EquityResult


def monte_carlo_equity(
    hand1: list[Card],
    hand2: list[Card],
    board: list[Card] | None = None,
    samples: int = 10000,
    rng: np.random.Generator | None = None
) -> tuple[EquityResult, EquityResult]:
    """
    使用蒙特卡洛模拟法计算两个玩家的 equity

    Args:
        hand1: 玩家1的手牌（2张）
        hand2: 玩家2的手牌（2张）
        board: 已有的公共牌（0-5张），默认为空
        samples: 模拟次数，默认10000
        rng: 随机数生成器，可用于复现结果

    Returns:
        (玩家1的结果, 玩家2的结果)

    Raises:
        ValueError: 手牌数量不正确或公共牌超过5张
    """
    if len(hand1) != 2:
        raise ValueError(f"玩家1手牌必须为2张，当前: {len(hand1)}")
    if len(hand2) != 2:
        raise ValueError(f"玩家2手牌必须为2张，当前: {len(hand2)}")

    if board is None:
        board = []

    if len(board) > 5:
        raise ValueError(f"公共牌不能超过5张，当前: {len(board)}")

    # 检查是否有重复的牌
    all_known = hand1 + hand2 + board
    if len(set(all_known)) != len(all_known):
        raise ValueError("存在重复的牌")

    # 需要补充的公共牌数量
    cards_needed = 5 - len(board)

    if cards_needed == 0:
        # 已有5张公共牌，直接比较
        return _compare_hands_direct(hand1, hand2, board)

    # 初始化随机数生成器
    if rng is None:
        rng = np.random.default_rng()

    # 获取剩余牌堆
    deck = get_remaining_deck(all_known)

    # 蒙特卡洛模拟
    win1 = win2 = tie = 0
    for _ in range(samples):
        # 随机抽取需要的公共牌
        indices = rng.choice(len(deck), size=cards_needed, replace=False)
        additional = [deck[i] for i in indices]
        full_board = board + additional

        # 评估双方牌型
        result1 = HandEvaluator.evaluate(hand1 + full_board)
        result2 = HandEvaluator.evaluate(hand2 + full_board)

        # 比较结果
        if result1 > result2:
            win1 += 1
        elif result1 < result2:
            win2 += 1
        else:
            tie += 1

    total = win1 + win2 + tie
    return (
        EquityResult(win1 / total, win2 / total, tie / total, total),
        EquityResult(win2 / total, win1 / total, tie / total, total)
    )


def _compare_hands_direct(
    hand1: list[Card],
    hand2: list[Card],
    board: list[Card]
) -> tuple[EquityResult, EquityResult]:
    """
    直接比较两手牌（公共牌已满5张）

    Returns:
        (玩家1的结果, 玩家2的结果)
    """
    result1 = HandEvaluator.evaluate(hand1 + board)
    result2 = HandEvaluator.evaluate(hand2 + board)

    if result1 > result2:
        return (
            EquityResult(1.0, 0.0, 0.0, 1),
            EquityResult(0.0, 1.0, 0.0, 1)
        )
    elif result1 < result2:
        return (
            EquityResult(0.0, 1.0, 0.0, 1),
            EquityResult(1.0, 0.0, 0.0, 1)
        )
    else:
        return (
            EquityResult(0.0, 0.0, 1.0, 1),
            EquityResult(0.0, 0.0, 1.0, 1)
        )
