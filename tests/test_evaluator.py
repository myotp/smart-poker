"""HandEvaluator 单元测试"""

import pytest

from src.core import Card, HandEvaluator, HandRank


def cards_from_str(s: str) -> list[Card]:
    """辅助函数：从空格分隔的字符串创建牌列表"""
    return [Card.from_str(c) for c in s.split()]


class TestHandRankDetection:
    """牌型识别测试"""

    def test_royal_flush(self):
        """皇家同花顺"""
        cards = cards_from_str("As Ks Qs Js Ts")
        result = HandEvaluator.evaluate(cards)
        assert result.rank == HandRank.ROYAL_FLUSH

    def test_straight_flush(self):
        """同花顺"""
        cards = cards_from_str("9h 8h 7h 6h 5h")
        result = HandEvaluator.evaluate(cards)
        assert result.rank == HandRank.STRAIGHT_FLUSH
        assert result.kickers[0] == 9  # 最高牌是9

    def test_straight_flush_wheel(self):
        """同花顺 A2345（钢轮）"""
        cards = cards_from_str("Ah 2h 3h 4h 5h")
        result = HandEvaluator.evaluate(cards)
        assert result.rank == HandRank.STRAIGHT_FLUSH
        assert result.kickers[0] == 5  # 最高牌是5

    def test_four_of_a_kind(self):
        """四条"""
        cards = cards_from_str("Ah As Ad Ac Kh")
        result = HandEvaluator.evaluate(cards)
        assert result.rank == HandRank.FOUR_OF_A_KIND
        assert result.kickers[0] == 14  # 四条A

    def test_full_house(self):
        """葫芦"""
        cards = cards_from_str("Ah As Ad Kh Ks")
        result = HandEvaluator.evaluate(cards)
        assert result.rank == HandRank.FULL_HOUSE
        assert result.kickers == (14, 13)  # 三条A带一对K

    def test_flush(self):
        """同花"""
        cards = cards_from_str("Ah Kh 9h 5h 2h")
        result = HandEvaluator.evaluate(cards)
        assert result.rank == HandRank.FLUSH

    def test_straight(self):
        """顺子"""
        cards = cards_from_str("9h 8s 7d 6c 5h")
        result = HandEvaluator.evaluate(cards)
        assert result.rank == HandRank.STRAIGHT
        assert result.kickers[0] == 9

    def test_straight_wheel(self):
        """顺子 A2345（轮子）"""
        cards = cards_from_str("Ah 2s 3d 4c 5h")
        result = HandEvaluator.evaluate(cards)
        assert result.rank == HandRank.STRAIGHT
        assert result.kickers[0] == 5  # 最高牌是5，不是A

    def test_straight_broadway(self):
        """顺子 AKQJT（百老汇）"""
        cards = cards_from_str("Ah Ks Qd Jc Th")
        result = HandEvaluator.evaluate(cards)
        assert result.rank == HandRank.STRAIGHT
        assert result.kickers[0] == 14

    def test_three_of_a_kind(self):
        """三条"""
        cards = cards_from_str("Ah As Ad Kh Qc")
        result = HandEvaluator.evaluate(cards)
        assert result.rank == HandRank.THREE_OF_A_KIND
        assert result.kickers[0] == 14

    def test_two_pair(self):
        """两对"""
        cards = cards_from_str("Ah As Kh Ks Qc")
        result = HandEvaluator.evaluate(cards)
        assert result.rank == HandRank.TWO_PAIR
        assert result.kickers[:2] == (14, 13)  # AA 和 KK

    def test_one_pair(self):
        """一对"""
        cards = cards_from_str("Ah As Kh Qc Jd")
        result = HandEvaluator.evaluate(cards)
        assert result.rank == HandRank.ONE_PAIR
        assert result.kickers[0] == 14  # 一对A

    def test_high_card(self):
        """高牌"""
        cards = cards_from_str("Ah Ks Qd Jc 9h")
        result = HandEvaluator.evaluate(cards)
        assert result.rank == HandRank.HIGH_CARD
        assert result.kickers[0] == 14


class TestSevenCardEvaluation:
    """7张牌评估测试（德州扑克标准场景）"""

    def test_find_royal_flush_in_seven(self):
        """从7张牌中找出皇家同花顺"""
        cards = cards_from_str("As Ks Qs Js Ts 3h 7d")
        result = HandEvaluator.evaluate(cards)
        assert result.rank == HandRank.ROYAL_FLUSH

    def test_find_flush_in_seven(self):
        """从7张牌中找出同花"""
        cards = cards_from_str("Ah Kh 9h 5h 2h Qs Jd")
        result = HandEvaluator.evaluate(cards)
        assert result.rank == HandRank.FLUSH

    def test_find_straight_in_seven(self):
        """从7张牌中找出顺子"""
        cards = cards_from_str("9h 8s 7d 6c 5h Ah Kc")
        result = HandEvaluator.evaluate(cards)
        assert result.rank == HandRank.STRAIGHT

    def test_find_full_house_in_seven(self):
        """从7张牌中找出葫芦"""
        cards = cards_from_str("Ah As Ad Kh Ks 2c 3d")
        result = HandEvaluator.evaluate(cards)
        assert result.rank == HandRank.FULL_HOUSE

    def test_best_full_house_selection(self):
        """选择最佳葫芦"""
        # 有 AAA 和 KKK，应该选 AAA KK
        cards = cards_from_str("Ah As Ad Kh Ks Kd 2c")
        result = HandEvaluator.evaluate(cards)
        assert result.rank == HandRank.FULL_HOUSE
        assert result.kickers == (14, 13)  # AAA + KK


class TestHandComparison:
    """牌型比较测试"""

    def test_rank_comparison(self):
        """不同牌型比较"""
        royal = HandEvaluator.evaluate(cards_from_str("As Ks Qs Js Ts"))
        flush = HandEvaluator.evaluate(cards_from_str("Ah Kh 9h 5h 2h"))
        pair = HandEvaluator.evaluate(cards_from_str("Ah As Kh Qc Jd"))

        assert royal > flush
        assert flush > pair
        assert pair < royal

    def test_same_rank_different_kickers(self):
        """同牌型不同踢脚牌"""
        pair_a = HandEvaluator.evaluate(cards_from_str("Ah As Kh Qc Jd"))
        pair_k = HandEvaluator.evaluate(cards_from_str("Kh Ks Ah Qc Jd"))

        assert pair_a > pair_k  # 一对A > 一对K

    def test_same_pair_different_kicker(self):
        """同样一对，不同踢脚牌"""
        pair_a_k = HandEvaluator.evaluate(cards_from_str("Ah As Kh Qc Jd"))
        pair_a_q = HandEvaluator.evaluate(cards_from_str("Ah As Qh Jc Td"))

        assert pair_a_k > pair_a_q  # 踢脚牌 K > Q

    def test_tie(self):
        """平局"""
        hand1 = HandEvaluator.evaluate(cards_from_str("Ah Ks Qd Jc 9h"))
        hand2 = HandEvaluator.evaluate(cards_from_str("As Kh Qc Jd 9s"))

        assert hand1 == hand2

    def test_compare_method(self):
        """compare 方法测试"""
        high = HandEvaluator.evaluate(cards_from_str("As Ks Qs Js Ts"))
        low = HandEvaluator.evaluate(cards_from_str("2h 3s 4d 5c 7h"))

        assert HandEvaluator.compare(high, low) == 1
        assert HandEvaluator.compare(low, high) == -1
        assert HandEvaluator.compare(high, high) == 0


class TestBatchEvaluation:
    """批量评估测试"""

    def test_evaluate_batch(self):
        """批量评估"""
        hands = [
            cards_from_str("As Ks Qs Js Ts"),
            cards_from_str("Ah Kh 9h 5h 2h"),
            cards_from_str("Ah As Kh Qc Jd"),
        ]
        results = HandEvaluator.evaluate_batch(hands)

        assert len(results) == 3
        assert results[0].rank == HandRank.ROYAL_FLUSH
        assert results[1].rank == HandRank.FLUSH
        assert results[2].rank == HandRank.ONE_PAIR

    def test_get_winners(self):
        """找赢家"""
        hands = [
            cards_from_str("As Ks Qs Js Ts"),  # 皇家同花顺
            cards_from_str("Ah Kh 9h 5h 2h"),  # 同花
            cards_from_str("Ah As Kh Qc Jd"),  # 一对
        ]
        winners, best = HandEvaluator.get_winners(hands)

        assert winners == [0]
        assert best.rank == HandRank.ROYAL_FLUSH

    def test_get_winners_tie(self):
        """平局找赢家"""
        hands = [
            cards_from_str("Ah Ks Qd Jc 9h"),  # 高牌
            cards_from_str("As Kh Qc Jd 9s"),  # 高牌（相同）
            cards_from_str("2h 3s 4d 5c 7h"),  # 高牌（较小）
        ]
        winners, _ = HandEvaluator.get_winners(hands)

        assert winners == [0, 1]  # 前两个平局获胜


class TestEdgeCases:
    """边界情况测试"""

    def test_two_cards(self):
        """仅2张牌"""
        cards = cards_from_str("Ah As")
        result = HandEvaluator.evaluate(cards)
        assert result.rank == HandRank.ONE_PAIR

    def test_three_cards(self):
        """3张牌"""
        cards = cards_from_str("Ah As Ad")
        result = HandEvaluator.evaluate(cards)
        assert result.rank == HandRank.THREE_OF_A_KIND

    def test_invalid_card_count(self):
        """无效牌数"""
        with pytest.raises(ValueError):
            HandEvaluator.evaluate([Card.from_str("Ah")])

        with pytest.raises(ValueError):
            HandEvaluator.evaluate(cards_from_str("Ah As Ad Ac Kh Ks Kd Kc"))
