"""Card 类单元测试"""

import numpy as np
import pytest

from src.core import Card, Deck


class TestCard:
    """Card 基本功能测试"""

    def test_create_card(self):
        """测试直接创建牌"""
        card = Card(rank=14, suit=0)
        assert card.rank == 14
        assert card.suit == 0

    def test_from_str(self):
        """测试从字符串创建牌"""
        # 黑桃A
        card = Card.from_str("As")
        assert card.rank == 14
        assert card.suit == 0

        # 红桃10
        card = Card.from_str("Th")
        assert card.rank == 10
        assert card.suit == 1

        # 梅花2
        card = Card.from_str("2c")
        assert card.rank == 2
        assert card.suit == 3

        # 方块K
        card = Card.from_str("Kd")
        assert card.rank == 13
        assert card.suit == 2

    def test_from_str_case_insensitive(self):
        """测试字符串大小写不敏感"""
        assert Card.from_str("as") == Card.from_str("As")
        assert Card.from_str("tH") == Card.from_str("Th")

    def test_from_str_with_10(self):
        """测试 10 的两种写法"""
        assert Card.from_str("10s") == Card.from_str("Ts")

    def test_invalid_rank(self):
        """测试无效的点数"""
        with pytest.raises(ValueError):
            Card(rank=1, suit=0)
        with pytest.raises(ValueError):
            Card(rank=15, suit=0)

    def test_invalid_suit(self):
        """测试无效的花色"""
        with pytest.raises(ValueError):
            Card(rank=14, suit=4)
        with pytest.raises(ValueError):
            Card(rank=14, suit=-1)

    def test_invalid_str(self):
        """测试无效的字符串"""
        with pytest.raises(ValueError):
            Card.from_str("X")
        with pytest.raises(ValueError):
            Card.from_str("Ax")

    def test_str_representation(self):
        """测试字符串表示"""
        card = Card.from_str("As")
        assert str(card) == "A♠"

        card = Card.from_str("Th")
        assert str(card) == "T♥"

    def test_to_short_str(self):
        """测试简短字符串"""
        card = Card.from_str("As")
        assert card.to_short_str() == "As"

    def test_to_index(self):
        """测试索引转换"""
        # 2♠ 应该是索引 0
        assert Card(rank=2, suit=0).to_index() == 0
        # 2♥ 应该是索引 1
        assert Card(rank=2, suit=1).to_index() == 1
        # A♣ 应该是索引 51
        assert Card(rank=14, suit=3).to_index() == 51

    def test_from_index(self):
        """测试从索引创建"""
        card = Card.from_index(0)
        assert card.rank == 2 and card.suit == 0

        card = Card.from_index(51)
        assert card.rank == 14 and card.suit == 3

    def test_index_roundtrip(self):
        """测试索引转换的往返一致性"""
        for i in range(52):
            card = Card.from_index(i)
            assert card.to_index() == i

    def test_hashable(self):
        """测试牌可哈希（可放入集合）"""
        cards = {Card.from_str("As"), Card.from_str("Kh"), Card.from_str("As")}
        assert len(cards) == 2

    def test_ordering(self):
        """测试牌的排序"""
        cards = [Card.from_str("2s"), Card.from_str("As"), Card.from_str("Kh")]
        sorted_cards = sorted(cards)
        assert sorted_cards[0].rank == 2
        assert sorted_cards[-1].rank == 14


class TestCardVector:
    """Card 向量化测试"""

    def test_onehot_encoding(self):
        """测试 one-hot 编码"""
        card = Card.from_str("As")
        vec = card.to_vector("onehot")
        assert vec.shape == (52,)
        assert vec.sum() == 1.0
        assert vec[card.to_index()] == 1.0

    def test_split_encoding(self):
        """测试分离编码"""
        card = Card.from_str("As")  # rank=14(A), suit=0(♠)
        vec = card.to_vector("split")
        assert vec.shape == (17,)
        # A 是 rank=14，在 0-12 索引中是 12
        assert vec[12] == 1.0
        # 黑桃是 suit=0，在 13-16 索引中是 13
        assert vec[13] == 1.0

    def test_simple_encoding(self):
        """测试简单编码"""
        card = Card.from_str("As")
        vec = card.to_vector("simple")
        assert vec.shape == (2,)
        assert vec[0] == 14 / 14.0
        assert vec[1] == 0 / 3.0


class TestDeck:
    """Deck 牌组测试"""

    def test_deck_size(self):
        """测试牌组大小"""
        deck = Deck()
        assert len(deck.cards) == 52

    def test_all_unique(self):
        """测试所有牌唯一"""
        deck = Deck()
        unique_cards = set(deck.cards)
        assert len(unique_cards) == 52

    def test_deal(self):
        """测试发牌"""
        deck = Deck()
        cards = deck.deal(5)
        assert len(cards) == 5
        assert deck.remaining() == 47

    def test_deal_too_many(self):
        """测试发牌过多"""
        deck = Deck()
        deck.deal(50)
        with pytest.raises(ValueError):
            deck.deal(5)

    def test_shuffle(self):
        """测试洗牌"""
        deck1 = Deck()
        deck2 = Deck()

        # 使用固定种子洗牌
        rng = np.random.default_rng(42)
        deck1.shuffle(rng)

        # 洗牌后顺序应该不同
        assert deck1.cards != deck2.cards

    def test_reset(self):
        """测试重置"""
        deck = Deck()
        deck.deal(10)
        assert deck.remaining() == 42

        deck.reset()
        assert deck.remaining() == 52
