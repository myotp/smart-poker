"""Equity 计算器单元测试"""

import numpy as np
import pytest

from src.core import Card
from src.equity import EquityCalculator, EquityResult


def cards_from_str(s: str) -> list[Card]:
    """辅助函数：从空格分隔的字符串创建牌列表"""
    return [Card.from_str(c) for c in s.split()]


class TestEquityResult:
    """EquityResult 测试"""

    def test_create_result(self):
        """测试创建结果"""
        result = EquityResult(0.6, 0.3, 0.1, 1000)
        assert result.win == 0.6
        assert result.lose == 0.3
        assert result.tie == 0.1
        assert result.sample_count == 1000

    def test_invalid_probability_sum(self):
        """测试概率之和不为1时报错"""
        with pytest.raises(ValueError):
            EquityResult(0.5, 0.3, 0.1, 1000)

    def test_percent_format(self):
        """测试百分比格式"""
        result = EquityResult(0.6, 0.3, 0.1, 1000)
        assert result.win_percent == "60.00%"
        assert result.lose_percent == "30.00%"
        assert result.tie_percent == "10.00%"

    def test_str_representation(self):
        """测试字符串表示"""
        result = EquityResult(0.6, 0.3, 0.1, 1000)
        assert "60.00%" in str(result)
        assert "30.00%" in str(result)


class TestRiverEquity:
    """River（5张公共牌）测试 - 结果确定性"""

    def test_player1_wins(self):
        """玩家1获胜"""
        hand1 = cards_from_str("Ah Kh")
        hand2 = cards_from_str("Qc Qd")
        board = cards_from_str("Kc Ks 7s 2d 3h")  # 玩家1有三条K

        r1, r2 = EquityCalculator.river(hand1, hand2, board)

        assert r1.win == 1.0
        assert r1.lose == 0.0
        assert r2.win == 0.0
        assert r2.lose == 1.0

    def test_player2_wins(self):
        """玩家2获胜"""
        hand1 = cards_from_str("Ah Kh")
        hand2 = cards_from_str("Qc Qd")
        board = cards_from_str("Qs 7s 2d 3h 9c")  # 玩家2有三条Q

        r1, r2 = EquityCalculator.river(hand1, hand2, board)

        assert r1.win == 0.0
        assert r1.lose == 1.0
        assert r2.win == 1.0
        assert r2.lose == 0.0

    def test_tie(self):
        """平局"""
        hand1 = cards_from_str("2h 3h")
        hand2 = cards_from_str("2c 3c")
        board = cards_from_str("As Ks Qs Js Ts")  # 公共牌皇家同花顺

        r1, r2 = EquityCalculator.river(hand1, hand2, board)

        assert r1.tie == 1.0
        assert r2.tie == 1.0


class TestFlopEquity:
    """Flop（3张公共牌）测试"""

    def test_enumerate_flop(self):
        """完全枚举 flop"""
        hand1 = cards_from_str("Ah Kh")
        hand2 = cards_from_str("Qc Qd")
        flop = cards_from_str("Kc 7s 2d")  # 玩家1中了顶对

        r1, r2 = EquityCalculator.flop(hand1, hand2, flop, method="enumerate")

        # 玩家1应该领先
        assert r1.win > r2.win
        # 枚举 C(45,2) = 990 种组合
        assert r1.sample_count == 990

    def test_monte_carlo_flop(self):
        """蒙特卡洛 flop"""
        hand1 = cards_from_str("Ah Kh")
        hand2 = cards_from_str("Qc Qd")
        flop = cards_from_str("Kc 7s 2d")

        rng = np.random.default_rng(42)
        r1, r2 = EquityCalculator.flop(hand1, hand2, flop, method="monte_carlo", samples=5000)

        # 玩家1应该领先
        assert r1.win > r2.win
        assert r1.sample_count == 5000


class TestTurnEquity:
    """Turn（4张公共牌）测试"""

    def test_enumerate_turn(self):
        """完全枚举 turn"""
        hand1 = cards_from_str("Ah Kh")
        hand2 = cards_from_str("Qc Qd")
        board = cards_from_str("Kc 7s 2d 9h")

        r1, r2 = EquityCalculator.turn(hand1, hand2, board, method="enumerate")

        # 玩家1领先（顶对 vs 中对）
        assert r1.win > r2.win
        # 枚举 C(44,1) = 44 种组合
        assert r1.sample_count == 44


class TestPreflopEquity:
    """Preflop 测试"""

    def test_monte_carlo_preflop(self):
        """蒙特卡洛 preflop（快速测试）"""
        hand1 = cards_from_str("Ah As")  # AA
        hand2 = cards_from_str("Kh Ks")  # KK

        rng = np.random.default_rng(42)
        r1, r2 = EquityCalculator.preflop(
            hand1, hand2, method="monte_carlo", samples=5000, rng=rng
        )

        # AA vs KK，AA 应该有约 80% 胜率
        assert r1.win > 0.75
        assert r1.win < 0.90
        assert r1.sample_count == 5000

    def test_pocket_pair_vs_overcards(self):
        """对子 vs 高牌（经典对抗）"""
        hand1 = cards_from_str("Qc Qd")  # QQ
        hand2 = cards_from_str("Ah Kh")  # AKs

        rng = np.random.default_rng(123)
        r1, r2 = EquityCalculator.preflop(
            hand1, hand2, method="monte_carlo", samples=10000, rng=rng
        )

        # QQ vs AKs 大约是 54% vs 46%（俗称 coin flip）
        assert 0.50 < r1.win < 0.60


class TestEdgeCases:
    """边界情况测试"""

    def test_invalid_hand_size(self):
        """手牌数量错误"""
        with pytest.raises(ValueError):
            EquityCalculator.preflop(
                cards_from_str("Ah"),  # 只有1张
                cards_from_str("Qc Qd")
            )

        with pytest.raises(ValueError):
            EquityCalculator.preflop(
                cards_from_str("Ah Kh Qh"),  # 3张
                cards_from_str("Qc Qd")
            )

    def test_invalid_board_size(self):
        """公共牌数量错误"""
        with pytest.raises(ValueError):
            EquityCalculator.flop(
                cards_from_str("Ah Kh"),
                cards_from_str("Qc Qd"),
                cards_from_str("Kc 7s")  # 只有2张
            )

        with pytest.raises(ValueError):
            EquityCalculator.river(
                cards_from_str("Ah Kh"),
                cards_from_str("Qc Qd"),
                cards_from_str("Kc 7s 2d 9h")  # 只有4张
            )

    def test_duplicate_cards(self):
        """重复的牌"""
        with pytest.raises(ValueError):
            EquityCalculator.preflop(
                cards_from_str("Ah Kh"),
                cards_from_str("Ah Qd")  # Ah 重复
            )

    def test_invalid_method(self):
        """无效的计算方法"""
        with pytest.raises(ValueError):
            EquityCalculator.preflop(
                cards_from_str("Ah Kh"),
                cards_from_str("Qc Qd"),
                method="invalid"  # type: ignore
            )


class TestMonteCarloConsistency:
    """蒙特卡洛一致性测试"""

    def test_reproducibility_with_seed(self):
        """使用相同种子应该得到相同结果"""
        hand1 = cards_from_str("Ah Kh")
        hand2 = cards_from_str("Qc Qd")

        rng1 = np.random.default_rng(42)
        r1_a, _ = EquityCalculator.preflop(
            hand1, hand2, method="monte_carlo", samples=1000, rng=rng1
        )

        rng2 = np.random.default_rng(42)
        r1_b, _ = EquityCalculator.preflop(
            hand1, hand2, method="monte_carlo", samples=1000, rng=rng2
        )

        assert r1_a.win == r1_b.win
        assert r1_a.lose == r1_b.lose
        assert r1_a.tie == r1_b.tie


class TestSymmetry:
    """对称性测试"""

    def test_result_symmetry(self):
        """玩家1和玩家2的结果应该对称"""
        hand1 = cards_from_str("Ah Kh")
        hand2 = cards_from_str("Qc Qd")
        board = cards_from_str("Kc 7s 2d")

        r1, r2 = EquityCalculator.flop(hand1, hand2, board)

        # r1.win == r2.lose, r1.lose == r2.win, r1.tie == r2.tie
        assert abs(r1.win - r2.lose) < 1e-9
        assert abs(r1.lose - r2.win) < 1e-9
        assert abs(r1.tie - r2.tie) < 1e-9
