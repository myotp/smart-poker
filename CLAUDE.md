# Smart Poker 项目指南

## 项目概述

德州扑克机器学习探索项目，目前已实现：
- 牌型判断（10种牌型识别）
- 胜率计算（Equity Calculator）

## 技术栈

- Python 3.13+
- uv 包管理
- pytest 测试框架
- NumPy, PyTorch

## 项目结构

```
src/
├── core/           # 核心模块：Card, Deck, HandRank, HandEvaluator
└── equity/         # 胜率计算：EquityCalculator, 完全枚举, 蒙特卡洛

tests/              # pytest 测试
docs/               # 设计文档
```

## 常用命令

```bash
uv run pytest           # 运行测试
uv run pytest -v        # 详细输出
uv run python           # 交互式 shell
```

## 代码规范

- 注释使用中文
- 类型注解使用 Python 3.10+ 语法（如 `list[Card]` 而非 `List[Card]`）
- 测试文件命名 `test_*.py`

## 核心类速查

```python
from src.core import Card, Deck, HandRank, HandResult, HandEvaluator
from src.equity import EquityCalculator, EquityResult

# Card
Card.from_str("As")          # 黑桃A
card.to_vector("onehot")     # 52维向量

# HandEvaluator
result = HandEvaluator.evaluate(cards)  # 2-7张牌
result.rank                  # HandRank 枚举
result.best_five             # 最佳5张牌

# EquityCalculator
r1, r2 = EquityCalculator.preflop(hand1, hand2, method="monte_carlo")
r1.win_percent               # "54.32%"
```

## 下一步可能的方向

1. 性能优化：查表法、并行计算
2. 手牌范围（Hand Range）支持
3. 多人对抗
4. ML 模型训练
