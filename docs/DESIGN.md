# 德州扑克牌型判断系统设计

## 概述

本模块实现德州扑克的牌型判断功能，支持从一组牌（2-7张）中识别最佳的5张牌组合及其牌型。

## 核心数据结构

### 1. Card（单张牌）

使用简洁的数据类表示一张牌：

```python
@dataclass(frozen=True)
class Card:
    rank: int   # 点数: 2-14 (14=A, 13=K, 12=Q, 11=J)
    suit: int   # 花色: 0-3 (0=♠, 1=♥, 2=♦, 3=♣)
```

**设计考量**：
- 使用 `frozen=True` 使 Card 不可变且可哈希，便于放入集合
- rank 使用 2-14 而非 1-13，因为 A 在大多数情况下是最大的牌
- suit 使用数字而非字符串，便于快速比较和位运算优化

### 2. HandRank（牌型枚举）

```python
class HandRank(IntEnum):
    HIGH_CARD = 0       # 高牌
    ONE_PAIR = 1        # 一对
    TWO_PAIR = 2        # 两对
    THREE_OF_A_KIND = 3 # 三条
    STRAIGHT = 4        # 顺子
    FLUSH = 5           # 同花
    FULL_HOUSE = 6      # 葫芦
    FOUR_OF_A_KIND = 7  # 四条
    STRAIGHT_FLUSH = 8  # 同花顺
    ROYAL_FLUSH = 9     # 皇家同花顺
```

### 3. HandResult（评估结果）

```python
@dataclass
class HandResult:
    rank: HandRank           # 牌型
    best_five: list[Card]    # 最佳5张牌
    kickers: tuple[int, ...] # 用于同牌型比较的踢脚牌点数
```

## 输入输出设计

### 输入

使用 `list[Card]` 表示一组牌，支持 2-7 张牌：
- 2 张：仅手牌（preflop 分析）
- 5 张：完整的5张牌组合
- 7 张：2张手牌 + 5张公共牌（标准德州扑克）

### 输出

返回 `HandResult`，包含：
- 识别出的牌型
- 构成该牌型的最佳5张牌
- 用于平局比较的 kickers

## 模块结构

```
src/core/
├── __init__.py
├── card.py          # Card 类定义
├── hand_rank.py     # HandRank 枚举
└── evaluator.py     # HandEvaluator 类
```

## HandEvaluator 核心接口

```python
class HandEvaluator:
    @staticmethod
    def evaluate(cards: list[Card]) -> HandResult:
        """评估一组牌，返回最佳牌型"""
        pass

    @staticmethod
    def compare(result1: HandResult, result2: HandResult) -> int:
        """比较两个牌型结果，返回 -1, 0, 1"""
        pass
```

## 牌型判断算法思路

### 基本策略

1. **预处理**：按点数和花色分组统计
2. **从高到低检查**：优先检查高级牌型，命中即返回
3. **组合枚举**：当牌数 > 5 时，枚举所有 C(n,5) 组合

### 判断顺序（从高到低）

1. **同花顺/皇家同花顺**：先检查是否有同花，再检查同花中是否有顺子
2. **四条**：检查是否有4张相同点数
3. **葫芦**：检查是否有3+2组合
4. **同花**：检查是否有5张同花色
5. **顺子**：检查是否有5张连续（注意 A2345 特殊情况）
6. **三条**：检查是否有3张相同点数
7. **两对**：检查是否有2个对子
8. **一对**：检查是否有1个对子
9. **高牌**：取最大的5张

### 辅助函数

```python
def _count_by_rank(cards: list[Card]) -> Counter[int]:
    """按点数统计数量"""

def _group_by_suit(cards: list[Card]) -> dict[int, list[Card]]:
    """按花色分组"""

def _is_straight(ranks: list[int]) -> bool:
    """判断是否为顺子"""

def _get_straight_high(ranks: list[int]) -> int | None:
    """获取顺子的最高牌点数，考虑 A2345"""
```

## 便捷工厂方法

```python
class Card:
    @classmethod
    def from_str(cls, s: str) -> "Card":
        """
        从字符串创建牌，如 "As" (黑桃A), "Th" (红桃10), "2c" (梅花2)
        格式: {rank}{suit}
        rank: 2-9, T, J, Q, K, A
        suit: s(spade), h(heart), d(diamond), c(club)
        """
```

## 使用示例

```python
from core.card import Card
from core.evaluator import HandEvaluator

# 创建手牌
cards = [
    Card.from_str("As"), Card.from_str("Ks"),
    Card.from_str("Qs"), Card.from_str("Js"),
    Card.from_str("Ts"), Card.from_str("3h"),
    Card.from_str("7d")
]

# 评估
result = HandEvaluator.evaluate(cards)
print(result.rank)       # HandRank.ROYAL_FLUSH
print(result.best_five)  # [As, Ks, Qs, Js, Ts]
```

## 性能优化考量（后续可选）

1. **查表法**：预计算所有 C(52,5) = 2,598,960 种5张牌组合的牌型，用于快速查询
2. **位运算**：使用 64 位整数编码牌面，通过位运算快速判断同花、顺子
3. **并行计算**：在大量牌局评估时使用多线程/GPU 加速

## 为 ML 预留的接口

```python
class Card:
    def to_vector(self) -> np.ndarray:
        """转换为 one-hot 或嵌入向量，供神经网络使用"""

class HandEvaluator:
    @staticmethod
    def evaluate_batch(hands: list[list[Card]]) -> list[HandResult]:
        """批量评估，便于训练数据生成"""
```

## 下一步计划

1. 实现 Card 和 HandRank 基础类
2. 实现 HandEvaluator 核心判断逻辑
3. 添加单元测试覆盖各种边界情况
4. 性能基准测试
5. 根据 ML 需求添加向量化接口
