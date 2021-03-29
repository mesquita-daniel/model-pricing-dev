import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.axes import Axes
from model_pricing import get_critical_auc
from model_pricing.core import *
import matplotlib

matplotlib.rcParams["figure.dpi"] = 500
matplotlib.rcParams["figure.facecolor"] = "white"


def get_xy_bbox(line: Line2D, pos: float):
    data = line.get_data()
    n_points = len(data[0])
    index = int(pos * n_points)
    return data[0][index], data[1][index]


def label_on_line(
    label: str, line: Line2D, ax: Axes, fontweight: str = "ultralight", pos: float = 0.4
):
    x_bb, y_bb = get_xy_bbox(line, pos)
    ax.text(
        x_bb,
        y_bb,
        label,
        fontsize="x-small",
        color=line.get_color(),
        fontweight=fontweight,
        fontstyle="italic",
        bbox={"facecolor": "white", "edgecolor": "white", "pad": 0},
    )


def plot_auc():
    fig, ax = plt.subplots()
    for auc in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        rc = RocCurve.from_auc(auc)
        x_plot = [rp.false_positive_rate for rp in rc.roc_points]
        y_plot = [rp.true_positive_rate for rp in rc.roc_points]
        line = ax.plot(x_plot, y_plot)[0]
        label_on_line(f"{auc:.2f}", line, ax)

    ax.set_title("ROC Curves with different AUC's")
    ax.set_xlim(1e-3, 1 + 1e-3)
    ax.set_ylim(1e-3, 1 + 5e-3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    plt.show()


def plot_profits(ax: Axes, baseauc: float, baseRoc: RocCurve, scenario: Scenario, pos):
    auc_range = np.linspace(baseauc, 1 - 1e-6, 100)
    y_plot = [
        RocCurve.from_auc(auc).compare_profits(baseRoc, scenario) for auc in auc_range
    ]
    line = ax.plot(auc_range, y_plot)[0]
    label_on_line(f"{100*scenario.badrate:.0f}%", line, ax, pos=pos)


def plot_expected_profit_increase_br(
    ax: Axes,
    baseauc: float,
    fee: float,
    bad_rate_range: List[float] = [0.01, 0.05, 0.1, 0.2, 25, 0.35, 0.5, 0.75],
):
    for bad_rate, pos in zip(
        bad_rate_range, np.linspace(0.99, 0.5, len(bad_rate_range))
    ):
        scenario = Scenario(bad_rate, fee)
        plot_profits(ax, baseauc, RocCurve.from_auc(baseauc), scenario, pos)
    ax.set_xlabel("AUC of new model")
    ax.set_ylabel("Expected relative profit increase")
    ax.set_title(
        f"Profit increase from base AUC={baseauc:.2f} and fee={fee:.2f}\n for "
        f"different bad rates"
    )

def plot_critical_auc(base_auc=0.5):
    x_plot = np.linspace(1e-6, 1 - 1e-6, 100)
    fig, ax = plt.subplots()
    for fee, pos in zip([0.05, 0.1, 0.2, 0.3, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5]):
        y_plot = [
                get_critical_auc(Scenario(br, fee), base_auc)
                for br in np.linspace(0, 1, 100)
        ]
        line = ax.plot(x_plot, y_plot)[0]
        label_on_line(f"{fee:.2f}", line, ax, pos=pos)
    ax.set_xlabel("Bad Rate")
    ax.set_ylabel("Critical AUC")
    ax.set_xlim(-1e-3, 1 + 1e-3)
    ax.set_ylim(0.5 - 1e-3, 1 + 1e-3)
    ax.set_title("Critical AUC across bad rate for different fees")
    plt.show()