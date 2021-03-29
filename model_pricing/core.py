from dataclasses import dataclass
from functools import cache
from typing import List, Tuple
import numpy as np
import scipy.special as sc


@dataclass
class Scenario(object):
    badrate: float
    fee: float

    def __hash__(self):
        return hash((self.badrate, self.fee))


@dataclass
class RocPoint(object):
    false_positive_rate: float
    true_positive_rate: float

    def expected_relative_profit(self, scenario: Scenario):
        """ Expected relative profit under the economic model below"""
        frac_false_positive = self.false_positive_rate * (1 - scenario.badrate)
        frac_false_negative = (1 - self.true_positive_rate) * scenario.badrate
        frac_true_negative = (1 - self.false_positive_rate) * (1 - scenario.badrate)
        return (
            -scenario.fee * frac_false_positive
            # opportunity cost: miss the fee we could have
            - frac_false_negative
            # unfiltered positives (defaults) makes us lose the unitary price
            + scenario.fee * frac_true_negative
            # those corrected predicted as good will pay us a fee
        )


def error_auc(a, b, auc):
    integral_betainc = sc.beta(a, b + 1) / sc.beta(a, b)
    return (auc - integral_betainc) ** 2


@dataclass
class RocCurve(object):
    """ ROC Curve satisfying tpr = sc.betainc(1,b,fpr) for some b"""

    roc_points: List[RocPoint]
    auc: float

    @classmethod
    def from_auc(cls, auc):
        """ Find a ROCCurve with given AUC """

        a = 1
        b = auc / (1 - auc)
        fpr_range = np.linspace(0, 1, 500)
        tpr_values = [sc.betainc(a, b, xi) for xi in fpr_range]
        return cls(
            [RocPoint(fpr, tpr) for fpr, tpr in list(zip(fpr_range, tpr_values))], auc
        )

    @cache
    def optimal_operation_point(self, scenario: Scenario) -> RocPoint:
        """Get the operation point that maximizes the profit"""
        return max(
            self.roc_points, key=lambda rp: rp.expected_relative_profit(scenario)
        )

    @cache
    def optimal_profit(self, scenario: Scenario):
        return self.optimal_operation_point(scenario).expected_relative_profit(scenario)

    def compare_profits(self, other: "RocCurve", scenario: Scenario) -> float:
        """Check how much marginal gain one ROCCurve would bring compared to other.
        The assumption is the economic model of operation"""
        return self.optimal_profit(scenario) - other.optimal_profit(scenario)

    def __hash__(self):
        return hash((((rp.fpr, rp.tpr) for rp in self.roc_points),self.auc))
