"""
Kuhn Poker 游戏实现

Kuhn Poker 是 Harold Kuhn 在 1950 年提出的简化扑克游戏，
是博弈论和扑克 AI 研究的经典入门案例。

游戏规则：
1. 三张牌：A（Ace）> K（King）> Q（Queen）
2. 两个玩家各发一张牌，每人先放 1 筹码底注
3. 先手可以 check 或 bet（下注 1 筹码）
4. 后手根据先手行动响应
5. 大牌获胜
"""

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Literal


class Card(IntEnum):
    """Kuhn Poker 的三张牌"""
    QUEEN = 0   # Q 最小
    KING = 1    # K 中间
    ACE = 2     # A 最大

    def __str__(self) -> str:
        return ["Q", "K", "A"][self.value]


class Action(Enum):
    """玩家行动"""
    CHECK = "check"   # 过牌
    BET = "bet"       # 下注
    FOLD = "fold"     # 弃牌
    CALL = "call"     # 跟注

    def __str__(self) -> str:
        return self.value


@dataclass
class GameState:
    """游戏状态"""
    p1_card: Card           # 玩家1的牌
    p2_card: Card           # 玩家2的牌
    history: tuple[Action, ...]  # 行动历史
    pot: int = 2            # 底池（初始各放1）
    p1_contrib: int = 1     # 玩家1已投入
    p2_contrib: int = 1     # 玩家2已投入

    def is_terminal(self) -> bool:
        """是否为终止状态"""
        if len(self.history) == 0:
            return False

        h = self.history

        # check-check: 摊牌
        if h == (Action.CHECK, Action.CHECK):
            return True
        # check-bet-fold: P1 弃牌
        if h == (Action.CHECK, Action.BET, Action.FOLD):
            return True
        # check-bet-call: 摊牌
        if h == (Action.CHECK, Action.BET, Action.CALL):
            return True
        # bet-fold: P2 弃牌
        if h == (Action.BET, Action.FOLD):
            return True
        # bet-call: 摊牌
        if h == (Action.BET, Action.CALL):
            return True

        return False

    def current_player(self) -> Literal[1, 2]:
        """当前行动的玩家（1 或 2）"""
        if len(self.history) == 0:
            return 1
        if len(self.history) == 1:
            return 2
        if len(self.history) == 2:
            # check-bet 后轮到 P1
            return 1
        raise ValueError("Invalid history")

    def legal_actions(self) -> list[Action]:
        """当前合法行动"""
        if self.is_terminal():
            return []

        h = self.history

        if len(h) == 0:
            # 先手：check 或 bet
            return [Action.CHECK, Action.BET]
        elif len(h) == 1:
            if h[0] == Action.CHECK:
                # 先手 check 后：check 或 bet
                return [Action.CHECK, Action.BET]
            else:  # h[0] == Action.BET
                # 先手 bet 后：fold 或 call
                return [Action.FOLD, Action.CALL]
        elif len(h) == 2:
            # check-bet 后：fold 或 call
            return [Action.FOLD, Action.CALL]

        return []

    def apply_action(self, action: Action) -> "GameState":
        """执行行动，返回新状态"""
        new_history = self.history + (action,)
        new_pot = self.pot
        new_p1 = self.p1_contrib
        new_p2 = self.p2_contrib

        if action == Action.BET:
            if self.current_player() == 1:
                new_p1 += 1
            else:
                new_p2 += 1
            new_pot += 1
        elif action == Action.CALL:
            if self.current_player() == 1:
                new_p1 += 1
            else:
                new_p2 += 1
            new_pot += 1

        return GameState(
            p1_card=self.p1_card,
            p2_card=self.p2_card,
            history=new_history,
            pot=new_pot,
            p1_contrib=new_p1,
            p2_contrib=new_p2
        )

    def payoff(self) -> tuple[int, int]:
        """
        计算收益（终止状态）

        Returns:
            (玩家1收益, 玩家2收益)
        """
        if not self.is_terminal():
            raise ValueError("Not a terminal state")

        h = self.history

        # 有人弃牌
        if Action.FOLD in h:
            if h == (Action.BET, Action.FOLD):
                # P2 弃牌，P1 赢得 P2 的底注
                return (1, -1)
            elif h == (Action.CHECK, Action.BET, Action.FOLD):
                # P1 弃牌，P2 赢得 P1 的底注
                return (-1, 1)

        # 摊牌比大小
        if self.p1_card > self.p2_card:
            # P1 赢
            return (self.p2_contrib, -self.p2_contrib)
        else:
            # P2 赢
            return (-self.p1_contrib, self.p1_contrib)

    def info_set(self, player: Literal[1, 2]) -> str:
        """
        信息集（Information Set）

        玩家只能看到自己的牌和行动历史，看不到对手的牌。
        """
        card = self.p1_card if player == 1 else self.p2_card
        history_str = "".join(a.value[0] for a in self.history)  # c/b/f/c
        return f"{card}{history_str}"


class KuhnPoker:
    """Kuhn Poker 游戏"""

    CARDS = [Card.QUEEN, Card.KING, Card.ACE]

    @staticmethod
    def new_game(p1_card: Card, p2_card: Card) -> GameState:
        """创建新游戏"""
        if p1_card == p2_card:
            raise ValueError("两个玩家不能拿相同的牌")
        return GameState(p1_card=p1_card, p2_card=p2_card, history=())

    @staticmethod
    def all_deals() -> list[tuple[Card, Card]]:
        """所有可能的发牌组合（6种）"""
        deals = []
        for c1 in KuhnPoker.CARDS:
            for c2 in KuhnPoker.CARDS:
                if c1 != c2:
                    deals.append((c1, c2))
        return deals

    @staticmethod
    def play_game(
        p1_card: Card,
        p2_card: Card,
        p1_strategy: dict[str, dict[Action, float]],
        p2_strategy: dict[str, dict[Action, float]]
    ) -> tuple[int, int]:
        """
        用给定策略玩一局游戏

        Args:
            p1_card: 玩家1的牌
            p2_card: 玩家2的牌
            p1_strategy: 玩家1的策略 {信息集: {行动: 概率}}
            p2_strategy: 玩家2的策略

        Returns:
            (玩家1期望收益, 玩家2期望收益)
        """
        import random

        state = KuhnPoker.new_game(p1_card, p2_card)

        while not state.is_terminal():
            player = state.current_player()
            info_set = state.info_set(player)
            strategy = p1_strategy if player == 1 else p2_strategy

            if info_set not in strategy:
                raise ValueError(f"策略中缺少信息集: {info_set}")

            # 按概率选择行动
            probs = strategy[info_set]
            actions = list(probs.keys())
            weights = [probs[a] for a in actions]
            action = random.choices(actions, weights=weights)[0]

            state = state.apply_action(action)

        return state.payoff()

    @staticmethod
    def expected_value(
        p1_strategy: dict[str, dict[Action, float]],
        p2_strategy: dict[str, dict[Action, float]]
    ) -> float:
        """
        计算玩家1的期望收益（遍历所有发牌和行动）

        每种发牌的概率相等（1/6）
        """
        total_ev = 0.0

        for p1_card, p2_card in KuhnPoker.all_deals():
            ev = KuhnPoker._expected_value_deal(
                p1_card, p2_card, p1_strategy, p2_strategy
            )
            total_ev += ev / 6.0  # 6 种发牌等概率

        return total_ev

    @staticmethod
    def _expected_value_deal(
        p1_card: Card,
        p2_card: Card,
        p1_strategy: dict[str, dict[Action, float]],
        p2_strategy: dict[str, dict[Action, float]]
    ) -> float:
        """计算特定发牌下的期望收益"""

        def traverse(state: GameState, prob: float) -> float:
            if state.is_terminal():
                return state.payoff()[0] * prob

            player = state.current_player()
            info_set = state.info_set(player)
            strategy = p1_strategy if player == 1 else p2_strategy
            probs = strategy[info_set]

            ev = 0.0
            for action in state.legal_actions():
                action_prob = probs.get(action, 0.0)
                if action_prob > 0:
                    new_state = state.apply_action(action)
                    ev += traverse(new_state, prob * action_prob)

            return ev

        initial_state = KuhnPoker.new_game(p1_card, p2_card)
        return traverse(initial_state, 1.0)
