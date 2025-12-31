"""Kuhn Poker 单元测试"""

import pytest

from src.games.kuhn.game import KuhnPoker, GameState, Action, Card
from src.games.kuhn.solver import KuhnSolver, solve_kuhn_poker


class TestCard:
    """牌的测试"""

    def test_card_ordering(self):
        """测试牌的大小顺序"""
        assert Card.ACE > Card.KING > Card.QUEEN

    def test_card_str(self):
        """测试牌的字符串表示"""
        assert str(Card.ACE) == "A"
        assert str(Card.KING) == "K"
        assert str(Card.QUEEN) == "Q"


class TestGameState:
    """游戏状态测试"""

    def test_initial_state(self):
        """测试初始状态"""
        state = KuhnPoker.new_game(Card.ACE, Card.KING)
        assert state.p1_card == Card.ACE
        assert state.p2_card == Card.KING
        assert state.history == ()
        assert state.pot == 2
        assert not state.is_terminal()
        assert state.current_player() == 1

    def test_legal_actions_initial(self):
        """测试初始合法行动"""
        state = KuhnPoker.new_game(Card.ACE, Card.KING)
        actions = state.legal_actions()
        assert Action.CHECK in actions
        assert Action.BET in actions
        assert len(actions) == 2

    def test_check_check(self):
        """测试 check-check 序列"""
        state = KuhnPoker.new_game(Card.ACE, Card.KING)
        state = state.apply_action(Action.CHECK)
        assert state.current_player() == 2

        state = state.apply_action(Action.CHECK)
        assert state.is_terminal()
        # A > K, P1 赢
        assert state.payoff() == (1, -1)

    def test_bet_fold(self):
        """测试 bet-fold 序列"""
        state = KuhnPoker.new_game(Card.QUEEN, Card.ACE)
        state = state.apply_action(Action.BET)
        state = state.apply_action(Action.FOLD)

        assert state.is_terminal()
        # P2 弃牌，P1 赢底注
        assert state.payoff() == (1, -1)

    def test_bet_call(self):
        """测试 bet-call 序列"""
        state = KuhnPoker.new_game(Card.ACE, Card.KING)
        state = state.apply_action(Action.BET)
        state = state.apply_action(Action.CALL)

        assert state.is_terminal()
        # A > K, P1 赢（包括 P2 的 call）
        assert state.payoff() == (2, -2)

    def test_check_bet_fold(self):
        """测试 check-bet-fold 序列"""
        state = KuhnPoker.new_game(Card.QUEEN, Card.ACE)
        state = state.apply_action(Action.CHECK)
        state = state.apply_action(Action.BET)
        state = state.apply_action(Action.FOLD)

        assert state.is_terminal()
        # P1 弃牌，P2 赢
        assert state.payoff() == (-1, 1)

    def test_check_bet_call(self):
        """测试 check-bet-call 序列"""
        state = KuhnPoker.new_game(Card.ACE, Card.KING)
        state = state.apply_action(Action.CHECK)
        state = state.apply_action(Action.BET)
        state = state.apply_action(Action.CALL)

        assert state.is_terminal()
        # A > K, P1 赢
        assert state.payoff() == (2, -2)

    def test_info_set(self):
        """测试信息集"""
        state = KuhnPoker.new_game(Card.ACE, Card.KING)

        # 初始状态
        assert state.info_set(1) == "A"
        assert state.info_set(2) == "K"

        # P1 check 后
        state = state.apply_action(Action.CHECK)
        assert state.info_set(1) == "Ac"
        assert state.info_set(2) == "Kc"


class TestKuhnPoker:
    """Kuhn Poker 游戏测试"""

    def test_all_deals(self):
        """测试所有发牌组合"""
        deals = KuhnPoker.all_deals()
        assert len(deals) == 6

    def test_same_card_error(self):
        """测试相同牌报错"""
        with pytest.raises(ValueError):
            KuhnPoker.new_game(Card.ACE, Card.ACE)


class TestKuhnSolver:
    """CFR 求解器测试"""

    def test_solver_converges(self):
        """测试求解器收敛"""
        solver = KuhnSolver()
        solver.train(iterations=5000)

        # Kuhn Poker 的纳什均衡期望收益约为 -1/18 ≈ -0.0556
        ev = solver.get_expected_value()
        assert -0.1 < ev < 0.0  # P1 略微劣势

    def test_strategy_properties(self):
        """测试策略的性质"""
        solver = solve_kuhn_poker(iterations=5000)
        strategy = solver.get_average_strategy()

        # 检查策略概率和为1
        for info_set, probs in strategy.items():
            total = sum(probs.values())
            assert abs(total - 1.0) < 0.01

    def test_known_equilibrium_properties(self):
        """
        测试已知的纳什均衡性质

        Kuhn Poker 纳什均衡的一些已知性质：
        - 拿 A 时 bet 频率应该较高
        - 拿 Q 时 fold 频率应该较高
        """
        solver = solve_kuhn_poker(iterations=10000)
        strategy = solver.get_average_strategy()

        # 拿 A 开局应该经常 bet（bluff 或 value）
        if "A" in strategy:
            assert strategy["A"].get(Action.BET, 0) > 0.2

        # 拿 Q 面对 bet 应该经常 fold
        if "Qb" in strategy:
            assert strategy["Qb"].get(Action.FOLD, 0) > 0.5


class TestExpectedValue:
    """期望收益计算测试"""

    def test_always_check_strategy(self):
        """测试双方都只 check 的策略"""
        # 构造纯 check 策略
        p1_strategy = {
            "Q": {Action.CHECK: 1.0, Action.BET: 0.0},
            "K": {Action.CHECK: 1.0, Action.BET: 0.0},
            "A": {Action.CHECK: 1.0, Action.BET: 0.0},
            "Qcb": {Action.FOLD: 1.0, Action.CALL: 0.0},
            "Kcb": {Action.FOLD: 1.0, Action.CALL: 0.0},
            "Acb": {Action.FOLD: 1.0, Action.CALL: 0.0},
        }
        p2_strategy = {
            "Qc": {Action.CHECK: 1.0, Action.BET: 0.0},
            "Kc": {Action.CHECK: 1.0, Action.BET: 0.0},
            "Ac": {Action.CHECK: 1.0, Action.BET: 0.0},
            "Qb": {Action.FOLD: 1.0, Action.CALL: 0.0},
            "Kb": {Action.FOLD: 1.0, Action.CALL: 0.0},
            "Ab": {Action.FOLD: 1.0, Action.CALL: 0.0},
        }

        ev = KuhnPoker.expected_value(p1_strategy, p2_strategy)
        # 双方都 check，期望收益为 0（对称）
        assert abs(ev) < 0.01
