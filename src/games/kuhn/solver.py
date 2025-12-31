"""
Kuhn Poker 求解器

使用 CFR (Counterfactual Regret Minimization) 算法求解纳什均衡。
"""

from collections import defaultdict

from .game import KuhnPoker, GameState, Action, Card


class KuhnSolver:
    """
    Kuhn Poker CFR 求解器

    CFR 算法通过迭代最小化"反事实遗憾"来逼近纳什均衡。
    """

    def __init__(self):
        # 累积遗憾：regret_sum[info_set][action]
        self.regret_sum: dict[str, dict[Action, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        # 累积策略：strategy_sum[info_set][action]
        self.strategy_sum: dict[str, dict[Action, float]] = defaultdict(
            lambda: defaultdict(float)
        )

    def get_strategy(self, info_set: str, legal_actions: list[Action]) -> dict[Action, float]:
        """
        根据累积遗憾计算当前策略（Regret Matching）

        正遗憾的行动获得更高概率，负遗憾的行动概率为0
        """
        regrets = self.regret_sum[info_set]

        # 只考虑正遗憾
        positive_regrets = {a: max(0, regrets[a]) for a in legal_actions}
        total = sum(positive_regrets.values())

        if total > 0:
            return {a: positive_regrets[a] / total for a in legal_actions}
        else:
            # 均匀分布
            n = len(legal_actions)
            return {a: 1.0 / n for a in legal_actions}

    def cfr(
        self,
        state: GameState,
        reach_probs: tuple[float, float]
    ) -> tuple[float, float]:
        """
        CFR 递归遍历

        Args:
            state: 当前游戏状态
            reach_probs: (玩家1到达概率, 玩家2到达概率)

        Returns:
            (玩家1的反事实价值, 玩家2的反事实价值)
        """
        if state.is_terminal():
            return state.payoff()

        player = state.current_player()
        info_set = state.info_set(player)
        legal_actions = state.legal_actions()

        # 获取当前策略
        strategy = self.get_strategy(info_set, legal_actions)

        # 计算每个行动的价值
        action_values: dict[Action, tuple[float, float]] = {}
        for action in legal_actions:
            new_state = state.apply_action(action)

            # 更新到达概率
            if player == 1:
                new_reach = (reach_probs[0] * strategy[action], reach_probs[1])
            else:
                new_reach = (reach_probs[0], reach_probs[1] * strategy[action])

            action_values[action] = self.cfr(new_state, new_reach)

        # 计算节点价值（按策略加权）
        node_value = [0.0, 0.0]
        for action in legal_actions:
            for i in range(2):
                node_value[i] += strategy[action] * action_values[action][i]

        # 更新遗憾和策略累积
        opponent = 2 if player == 1 else 1
        opp_reach = reach_probs[opponent - 1]

        for action in legal_actions:
            # 反事实遗憾 = 对手到达概率 × (行动价值 - 节点价值)
            regret = opp_reach * (action_values[action][player - 1] - node_value[player - 1])
            self.regret_sum[info_set][action] += regret

            # 累积策略（用自己的到达概率加权）
            my_reach = reach_probs[player - 1]
            self.strategy_sum[info_set][action] += my_reach * strategy[action]

        return tuple(node_value)  # type: ignore

    def train(self, iterations: int = 10000) -> None:
        """
        训练 CFR

        Args:
            iterations: 迭代次数
        """
        for i in range(iterations):
            # 遍历所有发牌
            for p1_card, p2_card in KuhnPoker.all_deals():
                state = KuhnPoker.new_game(p1_card, p2_card)
                self.cfr(state, (1.0, 1.0))

            if (i + 1) % 1000 == 0:
                ev = self.get_expected_value()
                print(f"迭代 {i + 1}: 玩家1期望收益 = {ev:.6f}")

    def get_average_strategy(self) -> dict[str, dict[Action, float]]:
        """
        获取平均策略（纳什均衡的近似）

        平均策略是所有迭代策略的加权平均，收敛到纳什均衡
        """
        avg_strategy: dict[str, dict[Action, float]] = {}

        for info_set, action_sums in self.strategy_sum.items():
            total = sum(action_sums.values())
            if total > 0:
                avg_strategy[info_set] = {
                    a: s / total for a, s in action_sums.items()
                }
            else:
                # 均匀分布
                n = len(action_sums)
                avg_strategy[info_set] = {a: 1.0 / n for a in action_sums.keys()}

        return avg_strategy

    def get_expected_value(self) -> float:
        """计算当前平均策略下玩家1的期望收益"""
        strategy = self.get_average_strategy()

        # 分离玩家1和玩家2的策略
        p1_strategy = {}
        p2_strategy = {}

        for info_set, probs in strategy.items():
            # 根据信息集判断是哪个玩家
            # P1 的信息集：牌 + 历史（历史长度为 0 或 2）
            # P2 的信息集：牌 + 历史（历史长度为 1）
            history_len = len(info_set) - 1  # 减去牌的字符
            if history_len == 0 or history_len == 2:
                p1_strategy[info_set] = probs
            else:
                p2_strategy[info_set] = probs

        return KuhnPoker.expected_value(p1_strategy, p2_strategy)

    def print_strategy(self) -> None:
        """打印策略"""
        strategy = self.get_average_strategy()

        print("\n=== Kuhn Poker 纳什均衡策略 ===\n")

        # 按玩家分组
        p1_info_sets = []
        p2_info_sets = []

        for info_set in sorted(strategy.keys()):
            history_len = len(info_set) - 1
            if history_len == 0 or history_len == 2:
                p1_info_sets.append(info_set)
            else:
                p2_info_sets.append(info_set)

        print("玩家1（先手）策略：")
        print("-" * 40)
        for info_set in p1_info_sets:
            probs = strategy[info_set]
            card = info_set[0]
            history = info_set[1:] if len(info_set) > 1 else "(开局)"

            probs_str = ", ".join(f"{a.value}: {p:.1%}" for a, p in probs.items())
            print(f"  牌={card}, 历史={history}: {probs_str}")

        print("\n玩家2（后手）策略：")
        print("-" * 40)
        for info_set in p2_info_sets:
            probs = strategy[info_set]
            card = info_set[0]
            history = info_set[1:]

            probs_str = ", ".join(f"{a.value}: {p:.1%}" for a, p in probs.items())
            print(f"  牌={card}, 历史={history}: {probs_str}")

        ev = self.get_expected_value()
        print(f"\n玩家1期望收益: {ev:.6f}")
        print(f"玩家2期望收益: {-ev:.6f}")


def solve_kuhn_poker(iterations: int = 10000) -> KuhnSolver:
    """
    求解 Kuhn Poker

    Args:
        iterations: CFR 迭代次数

    Returns:
        训练好的求解器
    """
    solver = KuhnSolver()
    solver.train(iterations)
    return solver


if __name__ == "__main__":
    print("求解 Kuhn Poker...\n")
    solver = solve_kuhn_poker(10000)
    solver.print_strategy()
