# 开发指南

## 运行测试

```bash
# 运行所有测试
uv run pytest

# 显示详细输出
uv run pytest -v

# 运行特定测试文件
uv run pytest tests/test_card.py
uv run pytest tests/test_evaluator.py
uv run pytest tests/test_equity.py

# 运行特定测试类或方法
uv run pytest tests/test_evaluator.py::TestHandRankDetection
uv run pytest tests/test_evaluator.py::TestHandRankDetection::test_royal_flush

# 失败时显示详细错误
uv run pytest -v --tb=short
```

## 交互式测试

启动 Python 交互式 shell：

```bash
uv run python
```

### 牌型判断示例

```python
from src.core import Card, HandEvaluator

# 创建牌
cards = [Card.from_str(s) for s in ["As", "Ks", "Qs", "Js", "Ts"]]
result = HandEvaluator.evaluate(cards)
print(result)  # 皇家同花顺

# 7张牌找最佳组合
cards = [Card.from_str(s) for s in ["As", "Ks", "Qs", "Js", "Ts", "3h", "7d"]]
result = HandEvaluator.evaluate(cards)
print(result.describe())
```

### 胜率计算示例

```python
from src.core import Card
from src.equity import EquityCalculator

# AKs vs QQ
hand1 = [Card.from_str('Ah'), Card.from_str('Kh')]
hand2 = [Card.from_str('Qc'), Card.from_str('Qd')]

# 蒙特卡洛（快速）
r1, r2 = EquityCalculator.preflop(hand1, hand2, method='monte_carlo', samples=10000)
print(f'AKs: {r1.win_percent}')
print(f'QQ:  {r2.win_percent}')

# 完全枚举（精确，较慢）
r1, r2 = EquityCalculator.preflop(hand1, hand2, method='enumerate')
print(f'AKs: {r1.win_percent}')
print(f'QQ:  {r2.win_percent}')
print(f'枚举数: {r1.sample_count:,}')
```

### 有公共牌时的计算

```python
from src.core import Card
from src.equity import EquityCalculator

hand1 = [Card.from_str('Ah'), Card.from_str('Kh')]
hand2 = [Card.from_str('Qc'), Card.from_str('Qd')]

# Flop
flop = [Card.from_str('Kc'), Card.from_str('7s'), Card.from_str('2d')]
r1, r2 = EquityCalculator.flop(hand1, hand2, flop)
print(f'AKs: {r1.win_percent}')  # AK 中了顶对，领先

# Turn
turn = flop + [Card.from_str('Qh')]
r1, r2 = EquityCalculator.turn(hand1, hand2, turn)
print(f'AKs: {r1.win_percent}')  # QQ 中了三条，反超
```

## 机器学习模块

### 训练手牌强度预测模型

```bash
# 运行训练脚本
uv run python -m src.ml.train.train_hand_strength
```

训练过程：
1. 生成数据：枚举所有 1326 种起手牌，计算对抗随机对手的胜率
2. 训练 MLP 模型：104 → 256 → 128 → 64 → 1
3. 保存最佳模型到 `checkpoints/hand_strength_best.pt`

自定义训练配置：

```python
from src.ml.train.train_hand_strength import train, TrainConfig

config = TrainConfig(
    epochs=50,
    mc_samples=1000,      # 蒙特卡洛采样数（越大越精确但越慢）
    n_opponents=10,       # 对抗的随机对手数
    batch_size=64,
    learning_rate=1e-3,
)
model = train(config)
```

### 使用训练好的模型

```python
from src.core import Card
from src.ml.models import HandStrengthPredictor

# 加载模型
predictor = HandStrengthPredictor(model_path="checkpoints/hand_strength_best.pt")

# 预测单手牌强度
hand = [Card.from_str("Ah"), Card.from_str("As")]  # AA
strength = predictor.predict(hand)
print(f"AA 强度: {strength:.2%}")  # 约 85%

# 批量预测
hands = [
    [Card.from_str("Ah"), Card.from_str("As")],  # AA
    [Card.from_str("Kh"), Card.from_str("Ks")],  # KK
    [Card.from_str("Ah"), Card.from_str("Kh")],  # AKs
    [Card.from_str("7c"), Card.from_str("2d")],  # 72o
]
strengths = predictor.predict_batch(hands)
for hand, s in zip(hands, strengths):
    print(f"{hand[0]} {hand[1]}: {s:.2%}")

# 对多手牌排名
ranked = predictor.rank_hands(hands)
for hand, s in ranked:
    print(f"{hand[0]} {hand[1]}: {s:.2%}")
```

### 神经网络 vs 蒙特卡洛对比

```python
import time
from src.core import Card
from src.equity import EquityCalculator
from src.ml.models import HandStrengthPredictor

predictor = HandStrengthPredictor(model_path="checkpoints/hand_strength_best.pt")

hand = [Card.from_str("Ah"), Card.from_str("Kh")]
opponent = [Card.from_str("Qc"), Card.from_str("Qd")]

# 神经网络预测（毫秒级）
start = time.time()
strength = predictor.predict(hand)
print(f"NN 预测: {strength:.2%}, 耗时: {(time.time()-start)*1000:.1f}ms")

# 蒙特卡洛模拟（秒级）
start = time.time()
r1, _ = EquityCalculator.preflop(hand, opponent, method="monte_carlo", samples=10000)
print(f"MC 模拟: {r1.win:.2%}, 耗时: {(time.time()-start)*1000:.1f}ms")
```

**注意**：神经网络预测的是对抗"平均随机对手"的强度，蒙特卡洛计算的是对抗特定对手的胜率，两者含义略有不同但趋势一致。
