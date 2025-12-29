# 手牌强度预测器设计

## 概述

训练神经网络预测德州扑克起手牌（2张）对抗随机手牌的胜率，用于快速评估手牌强度。

## 目录结构

```
src/
├── core/           # 已有：牌型判断
├── equity/         # 已有：胜率计算
└── ml/             # 新增：机器学习模块
    ├── __init__.py
    ├── data/           # 数据生成
    │   ├── __init__.py
    │   └── preflop.py  # Preflop 数据集生成
    ├── models/         # 模型定义
    │   ├── __init__.py
    │   └── hand_strength.py
    └── train/          # 训练脚本
        ├── __init__.py
        └── train_hand_strength.py
```

## 问题定义

### 输入

2 张起手牌，编码方式：

| 方案 | 维度 | 说明 |
|------|------|------|
| one-hot | 104 | 2×52 维 one-hot 拼接 |
| split | 34 | 2×17 维 (点数13 + 花色4) |
| index | 2 | 两个 0-51 的索引，用 embedding |

**推荐**：从 one-hot (104维) 开始，简单直接。

### 输出

- 单个浮点数：胜率 (0.0 - 1.0)
- 表示该手牌对抗随机手牌的期望胜率

### 标签生成

使用已有的 `EquityCalculator` 蒙特卡洛模拟生成标签：

```python
# 对抗随机手牌，计算平均胜率
r1, _ = EquityCalculator.preflop(hand, random_hand, method="monte_carlo")
label = r1.win
```

## 数据集设计

### PreflopDataset

```python
class PreflopDataset(torch.utils.data.Dataset):
    """Preflop 手牌强度数据集"""

    def __init__(
        self,
        n_samples: int = 100000,
        n_opponents: int = 10,      # 每手牌对抗的随机对手数
        mc_samples: int = 1000,     # 每次对抗的蒙特卡洛采样数
        encoding: str = "onehot",
        cache_path: str | None = None
    ):
        ...

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
```

### 数据生成策略

1. **枚举所有起手牌组合**：C(52,2) = 1326 种
2. **每种组合对抗多个随机对手**：取平均胜率作为标签
3. **缓存机制**：生成后保存到磁盘，避免重复计算

### 数据增强

德州扑克手牌有对称性：
- `AhKh` 和 `AsKs` 胜率相同（同花）
- `AhKs` 和 `AsKh` 胜率相同（不同花）

可以将 1326 种手牌归类为 169 种类型：
- 13 种对子：AA, KK, ..., 22
- 78 种同花：AKs, AQs, ..., 32s
- 78 种不同花：AKo, AQo, ..., 32o

## 模型设计

### HandStrengthMLP

简单的多层感知机：

```python
class HandStrengthMLP(nn.Module):
    """手牌强度预测 MLP"""

    def __init__(
        self,
        input_dim: int = 104,
        hidden_dims: list[int] = [256, 128, 64],
        dropout: float = 0.2
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
```

### 可选：Embedding 版本

```python
class HandStrengthEmbedding(nn.Module):
    """使用 Embedding 的手牌强度预测器"""

    def __init__(
        self,
        embed_dim: int = 32,
        hidden_dims: list[int] = [128, 64]
    ):
        super().__init__()
        self.card_embed = nn.Embedding(52, embed_dim)

        layers = []
        prev_dim = embed_dim * 2  # 两张牌拼接

        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
            ])
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 2) - 两张牌的索引
        embeds = self.card_embed(x)  # (batch, 2, embed_dim)
        embeds = embeds.view(embeds.size(0), -1)  # (batch, embed_dim * 2)
        return self.net(embeds).squeeze(-1)
```

## 训练配置

```python
@dataclass
class TrainConfig:
    # 数据
    n_samples: int = 100000
    train_ratio: float = 0.8
    batch_size: int = 256

    # 模型
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.2

    # 优化
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 50

    # 其他
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    save_path: str = "checkpoints/hand_strength.pt"
```

## 训练流程

```python
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred, y)

            total_loss += loss.item()
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    mae = np.mean(np.abs(np.array(all_preds) - np.array(all_labels)))
    return total_loss / len(loader), mae
```

## 评估指标

| 指标 | 说明 | 目标 |
|------|------|------|
| MSE Loss | 均方误差 | < 0.01 |
| MAE | 平均绝对误差 | < 0.02 (2%) |
| Max Error | 最大误差 | < 0.05 (5%) |

## 使用接口

训练完成后的推理接口：

```python
class HandStrengthPredictor:
    """手牌强度预测器（推理用）"""

    def __init__(self, model_path: str):
        self.model = HandStrengthMLP()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def predict(self, hand: list[Card]) -> float:
        """预测单手牌的强度"""
        x = torch.tensor([c.to_vector("onehot") for c in hand]).flatten()
        with torch.no_grad():
            return self.model(x.unsqueeze(0)).item()

    def predict_batch(self, hands: list[list[Card]]) -> list[float]:
        """批量预测"""
        X = torch.stack([
            torch.tensor([c.to_vector("onehot") for c in hand]).flatten()
            for hand in hands
        ])
        with torch.no_grad():
            return self.model(X).tolist()
```

## 实现计划

1. 创建 `src/ml/` 目录结构
2. 实现 `PreflopDataset` 数据生成
3. 实现 `HandStrengthMLP` 模型
4. 实现训练脚本
5. 添加测试
6. 训练并评估模型

## 后续优化

1. **169 类型聚合**：利用手牌对称性减少计算
2. **预计算查表**：将 169 种手牌的胜率预计算并缓存
3. **考虑位置**：加入位置信息（按钮位 vs 盲注位）
4. **多人场景**：预测不同人数下的胜率
