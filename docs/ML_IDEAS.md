# PyTorch 机器学习方向探索

## 概述

基于现有的牌型判断和胜率计算基础设施，探索德州扑克相关的机器学习应用。

## 方向一：手牌强度预测器（入门）

### 目标

训练神经网络预测起手牌的胜率，替代或加速蒙特卡洛模拟。

### 输入输出

- **输入**：2 张手牌的向量表示
  - 方案 A：2×52 one-hot（104 维）
  - 方案 B：2×17 split encoding（34 维）
  - 方案 C：直接用牌的索引做 embedding

- **输出**：预测胜率（0-1 之间的浮点数）

### 数据生成

利用现有的 `EquityCalculator` 生成训练数据：

```python
from src.core import Card, Deck
from src.equity import EquityCalculator
import numpy as np

def generate_preflop_data(n_samples: int = 10000):
    """生成 preflop 训练数据"""
    X, y = [], []
    deck = Deck()
    rng = np.random.default_rng()

    for _ in range(n_samples):
        deck.shuffle(rng)
        hand1 = deck.deal(2)
        hand2 = deck.deal(2)

        # 用蒙特卡洛计算胜率作为标签
        r1, r2 = EquityCalculator.preflop(
            hand1, hand2, method="monte_carlo", samples=5000
        )

        # 编码手牌
        x = np.concatenate([c.to_vector("onehot") for c in hand1])
        X.append(x)
        y.append(r1.win)

        deck.reset()

    return np.array(X), np.array(y)
```

### 网络结构

```python
import torch
import torch.nn as nn

class HandStrengthNet(nn.Module):
    def __init__(self, input_dim: int = 104):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 输出 0-1 的胜率
        )

    def forward(self, x):
        return self.net(x)
```

### 评估指标

- MAE（平均绝对误差）：预测胜率与实际胜率的差距
- 目标：MAE < 0.02（2% 误差以内）

### 优势

- 推理速度：训练后预测 << 蒙特卡洛模拟
- 批量处理：GPU 可并行处理大量手牌

---

## 方向二：牌面评估网络（中等）

### 目标

给定 2-7 张牌，预测牌型强度分数，可用于快速比较牌力。

### 输入输出

- **输入**：7 张牌的编码（固定长度，不足用 padding）
  - 7×52 one-hot = 364 维
  - 或使用 Transformer 处理变长序列

- **输出**：
  - 牌型分类（10 类）
  - 强度分数（用于同牌型比较）

### 网络结构选项

#### 方案 A：MLP

```python
class BoardEvaluatorMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(364, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.rank_head = nn.Linear(256, 10)   # 牌型分类
        self.score_head = nn.Linear(256, 1)   # 强度分数
```

#### 方案 B：Transformer

```python
class BoardEvaluatorTransformer(nn.Module):
    def __init__(self, d_model: int = 64, nhead: int = 4):
        super().__init__()
        self.card_embedding = nn.Embedding(52, d_model)
        self.pos_embedding = nn.Embedding(7, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Linear(d_model, 10)
```

### 数据生成

```python
def generate_board_data(n_samples: int = 100000):
    """生成牌面评估训练数据"""
    from src.core import Deck, HandEvaluator

    X, y_rank, y_score = [], [], []
    deck = Deck()
    rng = np.random.default_rng()

    for _ in range(n_samples):
        deck.shuffle(rng)
        n_cards = rng.choice([5, 6, 7])  # 随机 5-7 张牌
        cards = deck.deal(n_cards)

        result = HandEvaluator.evaluate(cards)

        # 编码（padding 到 7 张）
        x = [c.to_index() for c in cards] + [-1] * (7 - n_cards)
        X.append(x)
        y_rank.append(result.rank.value)
        y_score.append(result.score)

        deck.reset()

    return np.array(X), np.array(y_rank), np.array(y_score)
```

---

## 方向三：对手手牌范围预测（进阶）

### 目标

根据对手的行动序列，推断其可能持有的手牌范围。

### 输入

- 公共牌编码
- 对手行动序列：(阶段, 行动, 金额)
  - 阶段：preflop/flop/turn/river
  - 行动：fold/check/call/raise/all-in
  - 金额：raise 的大小（相对于底池）

### 输出

169 种起手牌组合的概率分布：
- 13×13 矩阵（对角线为对子，上三角同花，下三角不同花）

### 网络结构

```python
class HandRangePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        # 公共牌编码器
        self.board_encoder = nn.Sequential(
            nn.Linear(5 * 52, 128),
            nn.ReLU(),
        )
        # 行动序列编码器 (LSTM)
        self.action_encoder = nn.LSTM(
            input_size=16,  # 行动编码维度
            hidden_size=64,
            num_layers=2,
            batch_first=True
        )
        # 输出层
        self.output = nn.Sequential(
            nn.Linear(128 + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 169),
            nn.Softmax(dim=-1)
        )
```

### 数据来源

- 在线扑克手牌历史（PokerStars hand history 等）
- 自我对弈生成

---

## 方向四：最优决策网络（挑战）

### 目标

学习近似 GTO（Game Theory Optimal）策略。

### 方法选项

#### 方案 A：监督学习

从 GTO solver（如 PioSOLVER）的输出学习：
- 输入：游戏状态
- 输出：最优行动分布

#### 方案 B：强化学习

使用 self-play 训练：
- 算法：PPO / AlphaZero-style MCTS
- 奖励：筹码变化

#### 方案 C：CFR (Counterfactual Regret Minimization)

深度 CFR 变体：
- Deep CFR
- Single Deep CFR

### 状态表示

```python
@dataclass
class GameState:
    my_hand: list[Card]           # 我的手牌
    board: list[Card]             # 公共牌
    pot: float                    # 底池大小
    my_stack: float               # 我的筹码
    opponent_stack: float         # 对手筹码
    my_position: int              # 位置（0=SB, 1=BB, etc.）
    betting_history: list[Action] # 行动历史
```

### 行动空间

```python
class Action(Enum):
    FOLD = 0
    CHECK = 1
    CALL = 2
    RAISE_HALF_POT = 3
    RAISE_POT = 4
    RAISE_2X_POT = 5
    ALL_IN = 6
```

---

## 推荐路线图

| 阶段 | 方向 | 难度 | 预计工作量 |
|------|------|------|------------|
| 1 | 手牌强度预测器 | ⭐ | 1-2 天 |
| 2 | 牌面评估网络 | ⭐⭐ | 2-3 天 |
| 3 | 对手范围预测 | ⭐⭐⭐ | 1 周 |
| 4 | 最优决策网络 | ⭐⭐⭐⭐ | 2-4 周 |

**建议从方向一开始**，可以快速验证整个训练流程，并且已有完整的数据生成基础设施。

---

## 参考资料

- [Libratus](https://www.science.org/doi/10.1126/science.aao1733) - CMU 的德州扑克 AI
- [Pluribus](https://www.science.org/doi/10.1126/science.aay2400) - 6人德州扑克 AI
- [DeepStack](https://www.science.org/doi/10.1126/science.aam6960) - 深度学习 + CFR
- [OpenSpiel](https://github.com/deepmind/open_spiel) - DeepMind 的游戏 AI 框架
