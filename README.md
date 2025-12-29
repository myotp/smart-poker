# Smart Poker

德州扑克机器学习探索项目，提供牌型判断、胜率计算等基础功能。

## 功能特性

- **牌型判断**：识别皇家同花顺、同花顺、四条、葫芦等 10 种牌型
- **胜率计算**：支持 Preflop/Flop/Turn/River 各阶段的 Equity 计算
- **两种算法**：完全枚举（精确）和蒙特卡洛模拟（快速）
- **ML 友好**：Card 支持向量化，便于神经网络输入

## 安装

需要 Python 3.13+，使用 [uv](https://github.com/astral-sh/uv) 管理依赖：

```bash
# 克隆项目
git clone <repo-url>
cd smart-poker

# 安装依赖
uv sync
```

## 快速开始

### 牌型判断

```python
from src.core import Card, HandEvaluator

# 创建牌
cards = [Card.from_str(s) for s in ["As", "Ks", "Qs", "Js", "Ts"]]

# 评估牌型
result = HandEvaluator.evaluate(cards)
print(result)  # 皇家同花顺 [A♠ K♠ Q♠ J♠ T♠]
```

### 胜率计算

```python
from src.core import Card
from src.equity import EquityCalculator

# AA vs KK
aa = [Card.from_str("Ah"), Card.from_str("As")]
kk = [Card.from_str("Kh"), Card.from_str("Ks")]

# 蒙特卡洛模拟（快速）
r1, r2 = EquityCalculator.preflop(aa, kk, method="monte_carlo", samples=10000)
print(f"AA: {r1.win_percent}")  # ~82%
print(f"KK: {r2.win_percent}")  # ~18%

# 有公共牌时
board = [Card.from_str("Kc"), Card.from_str("7s"), Card.from_str("2d")]
r1, r2 = EquityCalculator.flop(aa, kk, board)
print(f"AA vs KK on K72: {r1.win_percent} vs {r2.win_percent}")
```

## 项目结构

```
smart-poker/
├── src/
│   ├── core/               # 核心模块
│   │   ├── card.py         # Card 和 Deck 类
│   │   ├── hand_rank.py    # HandRank 枚举和 HandResult
│   │   └── evaluator.py    # HandEvaluator 牌型评估器
│   └── equity/             # 胜率计算模块
│       ├── calculator.py   # EquityCalculator 主类
│       ├── enumerate.py    # 完全枚举实现
│       └── monte_carlo.py  # 蒙特卡洛模拟
├── tests/                  # 测试
├── docs/                   # 文档
│   ├── DESIGN.md           # 核心模块设计
│   ├── EQUITY_DESIGN.md    # Equity 模块设计
│   └── PREFLOP_MATCHUPS.md # 经典对抗胜率参考
└── pyproject.toml
```

## 运行测试

```bash
# 运行所有测试
uv run pytest

# 详细输出
uv run pytest -v

# 运行特定模块测试
uv run pytest tests/test_evaluator.py
uv run pytest tests/test_equity.py
```

## Card 表示法

| 格式 | 说明 | 示例 |
|------|------|------|
| 点数 | 2-9, T, J, Q, K, A | `A`=Ace, `T`=10 |
| 花色 | s, h, d, c | s=♠, h=♥, d=♦, c=♣ |

```python
Card.from_str("As")   # 黑桃A
Card.from_str("Th")   # 红桃10
Card.from_str("2c")   # 梅花2
```

## 牌型等级

从高到低：

1. 皇家同花顺 (Royal Flush)
2. 同花顺 (Straight Flush)
3. 四条 (Four of a Kind)
4. 葫芦 (Full House)
5. 同花 (Flush)
6. 顺子 (Straight)
7. 三条 (Three of a Kind)
8. 两对 (Two Pair)
9. 一对 (One Pair)
10. 高牌 (High Card)

## 依赖

- Python >= 3.13
- NumPy >= 2.4.0
- PyTorch >= 2.9.1 (为后续 ML 功能预留)

## License

MIT
