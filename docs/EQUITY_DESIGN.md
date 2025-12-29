# Equity 计算器设计

## 概述

Equity（权益）是扑克术语，表示一手牌在对抗中获胜的概率。本模块实现两个玩家起手牌对抗的胜率计算，支持：

1. **Preflop 对抗**：仅根据双方起手牌，枚举所有可能的公共牌组合计算胜率
2. **Flop/Turn/River 对抗**（扩展）：已有部分公共牌时的胜率计算

## 核心概念

### 术语定义

- **Equity**：获胜概率（0-1 之间的小数，或百分比）
- **Preflop**：公共牌发出前，只有双方手牌
- **Flop**：前三张公共牌
- **Turn**：第四张公共牌
- **River**：第五张公共牌
- **Dead Cards**：已知不在牌堆中的牌（如双方手牌、已翻的公共牌）

### 计算方法

#### 方法一：完全枚举（精确计算）

枚举所有可能的公共牌组合，统计双方胜/负/平局次数：

| 场景 | 需要枚举的公共牌 | 组合数 |
|------|------------------|--------|
| Preflop (0张公共牌) | 5张 | C(48,5) = 1,712,304 |
| Flop (3张公共牌) | 2张 | C(45,2) = 990 |
| Turn (4张公共牌) | 1张 | C(44,1) = 44 |
| River (5张公共牌) | 0张 | 1（直接比较） |

```
Equity(玩家A) = (A获胜次数 + 0.5 × 平局次数) / 总枚举次数
```

#### 方法二：蒙特卡洛模拟（近似计算）

当完全枚举太慢时，随机采样足够多的公共牌组合：

- 优点：速度快，可控制采样次数
- 缺点：结果有误差，需要足够大的样本量
- 建议：10,000+ 次模拟可达到 ±1% 精度

## 目录结构

```
src/equity/
├── __init__.py
├── calculator.py    # EquityCalculator 主类
├── enumerate.py     # 完全枚举实现
└── monte_carlo.py   # 蒙特卡洛模拟实现
```

## 核心接口设计

### EquityCalculator

```python
from dataclasses import dataclass

@dataclass
class EquityResult:
    """胜率计算结果"""
    win: float          # 获胜概率
    lose: float         # 失败概率
    tie: float          # 平局概率
    sample_count: int   # 样本数（枚举或模拟次数）

class EquityCalculator:
    """Equity 计算器"""

    @staticmethod
    def preflop(
        hand1: list[Card],
        hand2: list[Card],
        method: str = "enumerate"  # "enumerate" 或 "monte_carlo"
        samples: int = 10000       # 仅 monte_carlo 时有效
    ) -> tuple[EquityResult, EquityResult]:
        """
        计算 preflop 对抗胜率

        Args:
            hand1: 玩家1的手牌（2张）
            hand2: 玩家2的手牌（2张）
            method: 计算方法
            samples: 蒙特卡洛采样次数

        Returns:
            (玩家1的结果, 玩家2的结果)
        """
        pass

    @staticmethod
    def with_board(
        hand1: list[Card],
        hand2: list[Card],
        board: list[Card],         # 已有的公共牌（0-5张）
        method: str = "enumerate",
        samples: int = 10000
    ) -> tuple[EquityResult, EquityResult]:
        """
        计算有公共牌时的对抗胜率

        Args:
            hand1: 玩家1的手牌（2张）
            hand2: 玩家2的手牌（2张）
            board: 公共牌（0-5张）
            method: 计算方法
            samples: 蒙特卡洛采样次数

        Returns:
            (玩家1的结果, 玩家2的结果)
        """
        pass
```

## 实现思路

### Preflop 完全枚举

```python
def enumerate_preflop(hand1: list[Card], hand2: list[Card]) -> tuple[EquityResult, EquityResult]:
    # 1. 构建剩余牌堆（52 - 4 = 48张）
    dead_cards = set(hand1 + hand2)
    deck = [c for c in all_cards if c not in dead_cards]

    # 2. 枚举所有 C(48,5) 种公共牌组合
    win1 = win2 = tie = 0
    for board in combinations(deck, 5):
        # 3. 评估双方最终牌型
        result1 = HandEvaluator.evaluate(list(hand1) + list(board))
        result2 = HandEvaluator.evaluate(list(hand2) + list(board))

        # 4. 比较结果
        if result1 > result2:
            win1 += 1
        elif result1 < result2:
            win2 += 1
        else:
            tie += 1

    total = win1 + win2 + tie
    return (
        EquityResult(win1/total, win2/total, tie/total, total),
        EquityResult(win2/total, win1/total, tie/total, total)
    )
```

### 蒙特卡洛模拟

```python
def monte_carlo_preflop(
    hand1: list[Card],
    hand2: list[Card],
    samples: int = 10000,
    rng: np.random.Generator | None = None
) -> tuple[EquityResult, EquityResult]:
    if rng is None:
        rng = np.random.default_rng()

    dead_cards = set(hand1 + hand2)
    deck = np.array([c for c in all_cards if c not in dead_cards])

    win1 = win2 = tie = 0
    for _ in range(samples):
        # 随机抽取5张公共牌
        board = rng.choice(deck, size=5, replace=False)

        result1 = HandEvaluator.evaluate(list(hand1) + list(board))
        result2 = HandEvaluator.evaluate(list(hand2) + list(board))

        if result1 > result2:
            win1 += 1
        elif result1 < result2:
            win2 += 1
        else:
            tie += 1

    return (
        EquityResult(win1/samples, win2/samples, tie/samples, samples),
        EquityResult(win2/samples, win1/samples, tie/samples, samples)
    )
```

## 使用示例

```python
from src.core import Card
from src.equity import EquityCalculator

# 创建手牌
hand1 = [Card.from_str("Ah"), Card.from_str("Kh")]  # AK suited
hand2 = [Card.from_str("Qc"), Card.from_str("Qd")]  # QQ

# Preflop 对抗（完全枚举）
result1, result2 = EquityCalculator.preflop(hand1, hand2)
print(f"AKs vs QQ: {result1.win:.1%} vs {result2.win:.1%}")

# 有 Flop 时的对抗
board = [Card.from_str("Kc"), Card.from_str("7s"), Card.from_str("2d")]
result1, result2 = EquityCalculator.with_board(hand1, hand2, board)
print(f"AKs vs QQ (K72 board): {result1.win:.1%} vs {result2.win:.1%}")
```

## 性能考量

| 方法 | Preflop 时间 | Flop 时间 |
|------|--------------|-----------|
| 完全枚举 | ~10-30秒 | <1秒 |
| 蒙特卡洛 10k | ~0.5秒 | ~0.05秒 |
| 蒙特卡洛 100k | ~5秒 | ~0.5秒 |

### 优化方向（后续可选）

1. **并行计算**：使用多线程/多进程加速枚举
2. **查表优化**：预计算常见起手牌组合的 equity
3. **向量化**：用 NumPy 批量评估牌型
4. **Cython/Rust**：关键路径用低级语言重写

## 扩展设计

### 多人对抗

```python
@staticmethod
def multi_way(
    hands: list[list[Card]],  # N 个玩家的手牌
    board: list[Card] = [],
    method: str = "monte_carlo",
    samples: int = 10000
) -> list[EquityResult]:
    """计算多人对抗的 equity"""
    pass
```

### 手牌范围对抗

```python
@staticmethod
def range_vs_range(
    range1: HandRange,  # 玩家1的手牌范围（如 "AA,KK,QQ,AKs"）
    range2: HandRange,  # 玩家2的手牌范围
    board: list[Card] = [],
    samples: int = 10000
) -> tuple[EquityResult, EquityResult]:
    """计算手牌范围之间的对抗 equity"""
    pass
```

## 下一步计划

1. 实现 EquityResult 数据类
2. 实现 Preflop 完全枚举计算
3. 实现蒙特卡洛模拟
4. 实现 with_board 支持任意公共牌数量
5. 添加单元测试
6. 性能基准测试和优化
