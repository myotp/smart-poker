"""
德州扑克牌型评估器
"""

from collections import Counter
from itertools import combinations

from .card import Card
from .hand_rank import HandRank, HandResult


class HandEvaluator:
    """
    德州扑克牌型评估器

    支持评估 2-7 张牌，自动找出最佳的 5 张牌组合
    """

    @staticmethod
    def evaluate(cards: list[Card]) -> HandResult:
        """
        评估一组牌，返回最佳牌型

        Args:
            cards: 2-7 张牌的列表

        Returns:
            HandResult 包含牌型、最佳5张牌、踢脚牌

        Raises:
            ValueError: 牌数不在 2-7 范围内
        """
        if len(cards) < 2 or len(cards) > 7:
            raise ValueError(f"牌数必须在 2-7 之间，当前: {len(cards)}")

        # 如果不足5张，直接评估现有的牌
        if len(cards) <= 5:
            return HandEvaluator._evaluate_five_or_less(cards)

        # 超过5张，枚举所有 C(n,5) 组合，找最佳
        best_result: HandResult | None = None
        for combo in combinations(cards, 5):
            result = HandEvaluator._evaluate_exactly_five(list(combo))
            if best_result is None or result > best_result:
                best_result = result

        assert best_result is not None
        return best_result

    @staticmethod
    def _evaluate_five_or_less(cards: list[Card]) -> HandResult:
        """评估5张或更少的牌"""
        if len(cards) == 5:
            return HandEvaluator._evaluate_exactly_five(cards)

        # 少于5张牌，只能判断部分牌型
        ranks = sorted([c.rank for c in cards], reverse=True)
        rank_counts = Counter(ranks)
        counts = sorted(rank_counts.values(), reverse=True)

        # 按出现次数和点数排序的牌
        sorted_cards = sorted(cards, key=lambda c: (-rank_counts[c.rank], -c.rank))

        if counts[0] >= 4:
            # 四条
            quad_rank = [r for r, c in rank_counts.items() if c >= 4][0]
            kickers = (quad_rank,) + tuple(r for r in ranks if r != quad_rank)[:1]
            return HandResult(HandRank.FOUR_OF_A_KIND, sorted_cards, kickers)

        if counts[0] >= 3:
            if len(counts) > 1 and counts[1] >= 2:
                # 葫芦
                trip_rank = [r for r, c in rank_counts.items() if c >= 3][0]
                pair_rank = [r for r, c in rank_counts.items() if c >= 2 and r != trip_rank][0]
                return HandResult(HandRank.FULL_HOUSE, sorted_cards, (trip_rank, pair_rank))
            else:
                # 三条
                trip_rank = [r for r, c in rank_counts.items() if c >= 3][0]
                kickers = (trip_rank,) + tuple(r for r in ranks if r != trip_rank)[:2]
                return HandResult(HandRank.THREE_OF_A_KIND, sorted_cards, kickers)

        if counts[0] >= 2:
            pairs = [r for r, c in rank_counts.items() if c >= 2]
            if len(pairs) >= 2:
                # 两对
                pairs = sorted(pairs, reverse=True)[:2]
                kickers = tuple(pairs) + tuple(r for r in ranks if r not in pairs)[:1]
                return HandResult(HandRank.TWO_PAIR, sorted_cards, kickers)
            else:
                # 一对
                pair_rank = pairs[0]
                kickers = (pair_rank,) + tuple(r for r in ranks if r != pair_rank)[:3]
                return HandResult(HandRank.ONE_PAIR, sorted_cards, kickers)

        # 高牌
        return HandResult(HandRank.HIGH_CARD, sorted_cards, tuple(ranks[:5]))

    @staticmethod
    def _evaluate_exactly_five(cards: list[Card]) -> HandResult:
        """评估恰好5张牌"""
        ranks = sorted([c.rank for c in cards], reverse=True)
        suits = [c.suit for c in cards]

        rank_counts = Counter(ranks)
        counts = sorted(rank_counts.values(), reverse=True)

        # 检查同花
        is_flush = len(set(suits)) == 1

        # 检查顺子
        straight_high = HandEvaluator._get_straight_high(ranks)
        is_straight = straight_high is not None

        # 按出现次数和点数排序
        sorted_cards = sorted(cards, key=lambda c: (-rank_counts[c.rank], -c.rank))

        # 同花顺 / 皇家同花顺
        if is_flush and is_straight:
            if straight_high == 14:
                # 皇家同花顺：A K Q J 10 同花
                return HandResult(HandRank.ROYAL_FLUSH, sorted_cards, (14,))
            else:
                # 同花顺
                return HandResult(HandRank.STRAIGHT_FLUSH, sorted_cards, (straight_high,))

        # 四条
        if counts[0] == 4:
            quad_rank = [r for r, c in rank_counts.items() if c == 4][0]
            kicker = [r for r in ranks if r != quad_rank][0]
            return HandResult(HandRank.FOUR_OF_A_KIND, sorted_cards, (quad_rank, kicker))

        # 葫芦
        if counts[0] == 3 and counts[1] == 2:
            trip_rank = [r for r, c in rank_counts.items() if c == 3][0]
            pair_rank = [r for r, c in rank_counts.items() if c == 2][0]
            return HandResult(HandRank.FULL_HOUSE, sorted_cards, (trip_rank, pair_rank))

        # 同花
        if is_flush:
            return HandResult(HandRank.FLUSH, sorted_cards, tuple(ranks))

        # 顺子
        if is_straight:
            # 特殊处理 A2345，此时排序应该是 5 4 3 2 A
            if straight_high == 5:
                sorted_cards = sorted(cards, key=lambda c: (-(c.rank if c.rank <= 5 else 1)))
            return HandResult(HandRank.STRAIGHT, sorted_cards, (straight_high,))

        # 三条
        if counts[0] == 3:
            trip_rank = [r for r, c in rank_counts.items() if c == 3][0]
            kickers = tuple(r for r in ranks if r != trip_rank)
            return HandResult(HandRank.THREE_OF_A_KIND, sorted_cards, (trip_rank,) + kickers)

        # 两对
        if counts[0] == 2 and counts[1] == 2:
            pairs = sorted([r for r, c in rank_counts.items() if c == 2], reverse=True)
            kicker = [r for r in ranks if r not in pairs][0]
            return HandResult(HandRank.TWO_PAIR, sorted_cards, tuple(pairs) + (kicker,))

        # 一对
        if counts[0] == 2:
            pair_rank = [r for r, c in rank_counts.items() if c == 2][0]
            kickers = tuple(r for r in ranks if r != pair_rank)
            return HandResult(HandRank.ONE_PAIR, sorted_cards, (pair_rank,) + kickers)

        # 高牌
        return HandResult(HandRank.HIGH_CARD, sorted_cards, tuple(ranks))

    @staticmethod
    def _get_straight_high(ranks: list[int]) -> int | None:
        """
        检查是否为顺子，返回顺子的最高牌点数

        Args:
            ranks: 已排序（降序）的点数列表

        Returns:
            顺子的最高牌点数，如果不是顺子则返回 None
        """
        unique_ranks = sorted(set(ranks), reverse=True)
        if len(unique_ranks) < 5:
            return None

        # 检查连续5张
        if unique_ranks[0] - unique_ranks[4] == 4 and len(unique_ranks) >= 5:
            return unique_ranks[0]

        # 特殊情况：A2345（轮子）
        # A 可以当作 1 来组成最小顺子
        if set(unique_ranks[:5]) == {14, 5, 4, 3, 2}:
            return 5  # A2345 的最高牌是 5

        return None

    @staticmethod
    def compare(result1: HandResult, result2: HandResult) -> int:
        """
        比较两个牌型结果

        Args:
            result1: 第一个牌型
            result2: 第二个牌型

        Returns:
            -1: result1 < result2
             0: result1 == result2
             1: result1 > result2
        """
        if result1.score < result2.score:
            return -1
        elif result1.score > result2.score:
            return 1
        else:
            return 0

    @staticmethod
    def evaluate_batch(hands: list[list[Card]]) -> list[HandResult]:
        """
        批量评估多手牌

        Args:
            hands: 多组牌的列表

        Returns:
            对应的评估结果列表
        """
        return [HandEvaluator.evaluate(hand) for hand in hands]

    @staticmethod
    def get_winners(
        hands: list[list[Card]],
        player_ids: list[int] | None = None
    ) -> tuple[list[int], HandResult]:
        """
        找出赢家

        Args:
            hands: 多个玩家的手牌
            player_ids: 玩家ID列表（可选，默认为索引）

        Returns:
            (赢家ID列表, 最佳牌型) - 可能有多个赢家（平局）
        """
        if player_ids is None:
            player_ids = list(range(len(hands)))

        results = HandEvaluator.evaluate_batch(hands)
        best_result = max(results)
        winners = [
            player_ids[i]
            for i, result in enumerate(results)
            if result == best_result
        ]

        return winners, best_result
