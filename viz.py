import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import aggregate_results

_STYLE_KWARGS = {
    "font.family": "serif",
    "font.serif": "Times",
    "axes.edgecolor": "0",
    "text.color": "0",
}


def _extract_model_order(label: str) -> float:
    stripped = label.replace("checkpoint_", "").replace("B", "").strip()
    return float(stripped)


def _plot_results(
    df: pd.DataFrame,
    y_col: str,
    y_label: str,
    tick_style: str,
    figsize=(9, 6),
):
    if tick_style == "30BT":
        xticks = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
        xticklabels = ["0"] + [f"{x}B" for x in range(5, 35, 5)]

    elif tick_style == "100BT":
        xticks = [0.0, 10.0, 21.0, 31.0, 40.0, 50.0, 61.0, 71.0, 80.0, 90.0, 100.0]
        xticklabels = ["0"] + [f"{x}B" for x in range(10, 110, 10)]
    else:
        raise ValueError(
            f"Unknown tick_style: {tick_style}. Supported are only 30BT and 100BT"
        )

    fig, ax = plt.subplots(figsize=figsize)

    with sns.axes_style("whitegrid", _STYLE_KWARGS):
        sns.lineplot(
            data=df,
            x="model_order",
            y=y_col,
            hue="corpus",
            style="corpus",
            markers=True,
            markersize=10,
            linewidth=3,
            dashes=False,
            ax=ax,
        )

    ax.set_xlim(left=0)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=14)

    ax.set_xlabel("Training tokens (billions)", fontsize=18)
    ax.set_ylabel(y_label, fontsize=18)
    ax.set_yticklabels(ax.get_yticks(), fontsize=14)

    ax.legend(fontsize=14, frameon=True, loc="best", ncol=1, handlelength=1.5)
    ax.grid(True, which="major", linestyle="--", alpha=0.8)
    fig.tight_layout(pad=0.4)

    return fig, ax


def plot_normalized_results(
    normalized_results: pd.DataFrame, tick_style: str = "100BT"
):
    df = normalized_results.copy()
    df["model_order"] = df["model"].astype(str).map(_extract_model_order)
    df = df.sort_values("model_order")

    return _plot_results(
        df=df,
        y_col="normalized_score",
        y_label="Avg. normalized score",
        tick_style=tick_style,
    )


def plot_results_by_task(
    raw_results: pd.DataFrame, task: str, score_col: str, tick_style: str = "100BT"
):
    aggregated_results = aggregate_results(raw_results)
    df = aggregated_results[aggregated_results["dataset"] == task]
    df["model_order"] = df["model"].astype(str).map(_extract_model_order)
    df = df.sort_values("model_order")

    return _plot_results(
        df=df,
        y_col=score_col,
        y_label="Non-normalized Performance Score",
        tick_style=tick_style,
    )
