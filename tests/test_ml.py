"""机器学习模块单元测试"""

import numpy as np
import pytest
import torch

from src.core import Card
from src.ml.models import HandStrengthMLP, HandStrengthPredictor


class TestHandStrengthMLP:
    """HandStrengthMLP 模型测试"""

    def test_model_creation(self):
        """测试模型创建"""
        model = HandStrengthMLP()
        assert model is not None

    def test_model_forward(self):
        """测试前向传播"""
        model = HandStrengthMLP()
        x = torch.randn(16, 104)  # batch_size=16
        output = model(x)

        assert output.shape == (16,)
        assert torch.all(output >= 0) and torch.all(output <= 1)

    def test_model_single_input(self):
        """测试单个输入"""
        model = HandStrengthMLP()
        x = torch.randn(1, 104)
        output = model(x)

        assert output.shape == (1,)

    def test_custom_hidden_dims(self):
        """测试自定义隐藏层"""
        model = HandStrengthMLP(hidden_dims=[128, 64])
        x = torch.randn(8, 104)
        output = model(x)

        assert output.shape == (8,)


class TestHandStrengthPredictor:
    """HandStrengthPredictor 预测器测试"""

    @pytest.fixture
    def predictor(self):
        """创建预测器"""
        model = HandStrengthMLP()
        return HandStrengthPredictor(model=model)

    def test_predict_single(self, predictor):
        """测试单手牌预测"""
        hand = [Card.from_str("Ah"), Card.from_str("As")]
        strength = predictor.predict(hand)

        assert isinstance(strength, float)
        assert 0 <= strength <= 1

    def test_predict_batch(self, predictor):
        """测试批量预测"""
        hands = [
            [Card.from_str("Ah"), Card.from_str("As")],
            [Card.from_str("Kh"), Card.from_str("Ks")],
            [Card.from_str("2c"), Card.from_str("7d")],
        ]
        strengths = predictor.predict_batch(hands)

        assert len(strengths) == 3
        assert all(0 <= s <= 1 for s in strengths)

    def test_rank_hands(self, predictor):
        """测试手牌排名"""
        hands = [
            [Card.from_str("2c"), Card.from_str("7d")],
            [Card.from_str("Ah"), Card.from_str("As")],
            [Card.from_str("Kh"), Card.from_str("Ks")],
        ]
        ranked = predictor.rank_hands(hands)

        assert len(ranked) == 3
        # 验证按强度降序排列
        strengths = [r[1] for r in ranked]
        assert strengths == sorted(strengths, reverse=True)

    def test_invalid_hand_size(self, predictor):
        """测试无效手牌数量"""
        with pytest.raises(ValueError):
            predictor.predict([Card.from_str("Ah")])

        with pytest.raises(ValueError):
            predictor.predict([Card.from_str("Ah"), Card.from_str("As"), Card.from_str("Kh")])


class TestHandEncoding:
    """手牌编码测试"""

    def test_onehot_encoding(self):
        """测试 one-hot 编码"""
        card = Card.from_str("As")
        vec = card.to_vector("onehot")

        assert vec.shape == (52,)
        assert vec.sum() == 1.0

    def test_two_card_encoding(self):
        """测试两张牌编码"""
        hand = [Card.from_str("Ah"), Card.from_str("As")]
        encoded = np.concatenate([c.to_vector("onehot") for c in hand])

        assert encoded.shape == (104,)
        assert encoded.sum() == 2.0


class TestModelDeterminism:
    """模型确定性测试"""

    def test_eval_mode_determinism(self):
        """测试 eval 模式下结果确定性"""
        model = HandStrengthMLP()
        model.eval()

        x = torch.randn(8, 104)

        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)

        assert torch.allclose(out1, out2)
