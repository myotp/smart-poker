"""
Equity 计算结果数据类
"""

from dataclasses import dataclass


@dataclass
class EquityResult:
    """
    胜率计算结果

    Attributes:
        win: 获胜概率 (0.0 - 1.0)
        lose: 失败概率 (0.0 - 1.0)
        tie: 平局概率 (0.0 - 1.0)
        sample_count: 样本数（枚举或模拟次数）
    """
    win: float
    lose: float
    tie: float
    sample_count: int

    def __post_init__(self) -> None:
        """验证概率之和为1"""
        total = self.win + self.lose + self.tie
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"概率之和必须为1，当前: {total}")

    @property
    def win_percent(self) -> str:
        """返回获胜百分比字符串"""
        return f"{self.win * 100:.2f}%"

    @property
    def lose_percent(self) -> str:
        """返回失败百分比字符串"""
        return f"{self.lose * 100:.2f}%"

    @property
    def tie_percent(self) -> str:
        """返回平局百分比字符串"""
        return f"{self.tie * 100:.2f}%"

    def __str__(self) -> str:
        return f"Win: {self.win_percent}, Lose: {self.lose_percent}, Tie: {self.tie_percent}"

    def __repr__(self) -> str:
        return f"EquityResult(win={self.win:.4f}, lose={self.lose:.4f}, tie={self.tie:.4f}, samples={self.sample_count})"
