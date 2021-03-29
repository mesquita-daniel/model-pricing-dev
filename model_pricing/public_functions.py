from model_pricing.core import *

AUC = float

def _valid_auc(auc):
    assert 0 <= auc <= 1, "AUC can only be between 0 and 1"
    if auc==0:
        return auc+1e-6
    elif auc==1:
        return auc-1e6
    return auc

def critical_auc(scenario: Scenario, base_auc: AUC) -> AUC:
    base_auc =_valid_auc(base_auc)
    base_roc = RocCurve.from_auc(base_auc)
    comp_arr = [
        (RocCurve.from_auc(new_auc).compare_profits(base_roc, scenario), new_auc)
        for new_auc in np.linspace(base_auc, 1 - 1e-6, 200)
    ]
    greater_than_critical = [comp for comp in comp_arr if comp[0] > 1e-6]
    return min(greater_than_critical, key=lambda r: r[1])[1]


def expected_profit(auc: float, scenario: Scenario):
    auc = _valid_auc(auc)
    return RocCurve.from_auc(auc).optimal_profit(scenario)


def expected_profit_increase(base_auc: float, new_model_auc: float, scenario: Scenario):
    base_auc = _valid_auc(base_auc)
    new_model_auc = _valid_auc(new_model_auc)
    return RocCurve.from_auc(new_model_auc).compare_profits(
        RocCurve.from_auc(base_auc), scenario
    )