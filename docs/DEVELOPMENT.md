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
