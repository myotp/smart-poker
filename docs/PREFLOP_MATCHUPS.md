# Preflop 经典对抗胜率

## 概述

德州扑克 Preflop（翻牌前）阶段，不同类型的起手牌对抗有其固定的胜率规律。理解这些规律是扑克策略的基础。

## 经典对抗类型

### 1. 超对 vs 低对（Overpair vs Underpair）

高对子 vs 低对子，高对子碾压。

| 对抗 | 胜率 |
|------|------|
| AA vs KK | 82% vs 18% |
| AA vs QQ | 82% vs 18% |
| KK vs QQ | 82% vs 18% |
| KK vs JJ | 82% vs 18% |

**规律**：大对子 vs 小对子 ≈ **82% vs 18%**

### 2. 对子 vs 两高张（Pair vs Two Overcards）

这是最著名的"抛硬币"（Coin Flip）对抗。

| 对抗 | 胜率 | 备注 |
|------|------|------|
| QQ vs AKs | 54% vs 46% | 经典 coin flip |
| QQ vs AKo | 57% vs 43% | 不同花略劣 |
| JJ vs AKs | 54% vs 46% | |
| JJ vs AKo | 57% vs 43% | |
| TT vs AKs | 54% vs 46% | |
| 22 vs AKo | 52% vs 48% | 最小对子仍领先 |

**规律**：
- 对子 vs 两高张同花 ≈ **54% vs 46%**
- 对子 vs 两高张不同花 ≈ **57% vs 43%**

### 3. 高对 vs 包含一张的高牌（Pair vs One Overcard）

| 对抗 | 胜率 |
|------|------|
| AA vs AKs | 87% vs 13% |
| AA vs AKo | 93% vs 7% |
| KK vs AKs | 66% vs 34% |
| KK vs AQs | 71% vs 29% |

**规律**：对子被一张高牌"压制"时，优势缩小但仍领先

### 4. 两高张 vs 两低张（High Cards vs Low Cards）

| 对抗 | 胜率 |
|------|------|
| AKs vs QJs | 63% vs 37% |
| AKo vs QJo | 65% vs 35% |
| AKs vs 76s | 62% vs 38% |

**规律**：高牌 vs 低牌 ≈ **63% vs 37%**

### 5. 同牌不同踢脚（Domination）

当一方的牌被另一方"压制"时，处于极大劣势。

| 对抗 | 胜率 | 说明 |
|------|------|------|
| AK vs AQ | 74% vs 26% | Q 被压制 |
| AK vs AJ | 74% vs 26% | J 被压制 |
| AK vs KQ | 74% vs 26% | K 被压制 |

**规律**：压制局面 ≈ **74% vs 26%**（被压制方只有约 3 个有效 outs）

## 记忆口诀

| 对抗类型 | 胜率比 | 口诀 |
|----------|--------|------|
| 大对 vs 小对 | 82:18 | 八二开 |
| 对子 vs 两高张 | 55:45 | 抛硬币 |
| 高牌 vs 低牌 | 63:37 | 六四开 |
| 压制局面 | 74:26 | 七三开 |

## 影响因素

### 同花（Suited）的价值

同花比不同花大约多 **3-4%** 的胜率：
- AKs vs QQ: 46%
- AKo vs QQ: 43%

### 连张（Connected）的价值

连张牌有更多顺子可能，但影响较小（约 1-2%）。

### 阻断效应（Blockers）

当你持有对手需要的牌时，会降低对手成牌概率：
- 持有 AA 时，对手很难拿到 A
- 持有 KK 且面对 AK 时，对手只有 3 个 K outs

## 实际应用

```python
from src.core import Card
from src.equity import EquityCalculator

# QQ vs AKs
qq = [Card.from_str('Qc'), Card.from_str('Qd')]
aks = [Card.from_str('Ah'), Card.from_str('Kh')]

r1, r2 = EquityCalculator.preflop(qq, aks, method='monte_carlo', samples=50000)
print(f'QQ:  {r1.win_percent}')  # ~54%
print(f'AKs: {r2.win_percent}')  # ~46%
```

## 参考资料

- 以上数据基于蒙特卡洛模拟或完全枚举计算
- 实际胜率可能因具体牌面（如花色阻断）有细微差异
