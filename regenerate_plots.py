from pathlib import Path
import re
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)

PARITY_LOW = 0.80
PARITY_HIGH = 1.25

# Category colors from plots.ipynb (matplotlib tab10, in sorted-class order).
GROUP_COLORS = list(plt.cm.tab10.colors[:3])
# Biased vs unbiased markers from plots.ipynb.
DISTORTED = "red"
UNDISTORTED = "blue"
# First group is the treated category; the other two are reference groups.
ABLATION_COLORS = ["#FF4C4C", "#3D7EFF", "#3D7EFF"]
# Coral below the parity line, cream at 0.8, blue above.
CLASS_CMAP = LinearSegmentedColormap.from_list(
    "fair_coral_blue",
    ["#F6C4B8", "#FDF1E2", "#FDFBF8", "#E3EDFB", "#9BB6E6"],
)


def save(fig, filename):
    fig.savefig(PLOTS / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {PLOTS / filename}")


def short_group(name):
    match = re.search(r"_(\d+)$", str(name))
    return f"G{match.group(1)}" if match else str(name)


def multi_setting(feature):
    match = re.search(r"BTBP_(ML|MH|LH|MM)_([OS])", str(feature))
    return f"{match.group(1)}-{match.group(2)}" if match else str(feature)


def add_x_parity_regions(ax):
    xmin, xmax = ax.get_xlim()
    if xmin < PARITY_LOW:
        ax.axvspan(xmin, PARITY_LOW, color="#F2D7D5", alpha=0.50, zorder=0)
    if xmax > PARITY_HIGH:
        ax.axvspan(PARITY_HIGH, xmax, color="#F2D7D5", alpha=0.50, zorder=0)


def add_xy_parity_regions(ax):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    if xmin < PARITY_LOW:
        ax.axvspan(xmin, PARITY_LOW, color="#F2D7D5", alpha=0.40, zorder=0)
    if xmax > PARITY_HIGH:
        ax.axvspan(PARITY_HIGH, xmax, color="#F2D7D5", alpha=0.40, zorder=0)
    if ymin < PARITY_LOW:
        ax.axhspan(ymin, PARITY_LOW, color="#F2D7D5", alpha=0.40, zorder=0)
    if ymax > PARITY_HIGH:
        ax.axhspan(PARITY_HIGH, ymax, color="#F2D7D5", alpha=0.40, zorder=0)


# ============================================================
# REGRESSION: PARITY CHECK
# ============================================================

def plot_regression_parity(reg):
    df = reg[reg["simulation_type"] == "low_bias_single_cat_equal_distribution"].copy()
    regimes = ["UTUP", "UTBP", "BTUP", "BTBP"]

    fig, axes = plt.subplots(1, 4, figsize=(14.4, 4.45), sharex=True, sharey=True)

    # Equal limits center the white parity square [0.80, 1.25] on both axes.
    span = (0.55, 1.50)

    for ax, regime in zip(axes, regimes):
        sub = df[df["Protected_Feature"] == regime].sort_values("Protected_Class")

        ax.set_xlim(*span)
        ax.set_ylim(*span)
        ax.set_aspect("equal", adjustable="box")
        add_xy_parity_regions(ax)

        ax.axvline(1.0, color="black", linewidth=0.8, alpha=0.45)
        ax.axhline(1.0, color="black", linewidth=0.8, alpha=0.45)

        for i, (_, row) in enumerate(sub.iterrows()):
            color = GROUP_COLORS[i % len(GROUP_COLORS)]
            target = float(row["FairAI_target"])

            proposed = float(row["FairAI_pred"])
            proposed_plot = min(max(proposed, span[0]), span[1])
            ax.scatter(target, proposed_plot, color=color, marker="o", s=58, zorder=3)

            dalex = float(row["dalex_pred"])
            dalex_plot = min(max(dalex, span[0]), span[1])
            ax.scatter(target, dalex_plot, color=color, marker="^", s=62, zorder=3)

            if proposed > span[1]:
                ax.annotate(f"↑{proposed:.2f}", (target, span[1]), xytext=(0, -3), textcoords="offset points", ha="center", va="top", fontsize=7, color=color)

            if dalex > span[1]:
                ax.annotate(f"↑{dalex:.2f}", (target, span[1]), xytext=(0, -11), textcoords="offset points", ha="center", va="top", fontsize=7, color=color)

        ax.set_title(regime, fontsize=11)
        ax.grid(linestyle=":", alpha=0.25)

    for ax in axes:
        ax.set_xlabel("Target diagnostic")
    axes[0].set_ylabel("Prediction diagnostic")

    legend = [
        Line2D([], [], marker="o", linestyle="none", color="black", label="Proposed"),
        Line2D([], [], marker="^", linestyle="none", color="black", label="fairmodels/DALEX"),
        Line2D([], [], marker="o", linestyle="none", color=GROUP_COLORS[0], label="G1"),
        Line2D([], [], marker="o", linestyle="none", color=GROUP_COLORS[1], label="G2"),
        Line2D([], [], marker="o", linestyle="none", color=GROUP_COLORS[2], label="G3"),
    ]

    fig.legend(handles=legend, loc="lower center", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "target_pred.pdf")


# ============================================================
# REGRESSION: R1-R5
# ============================================================

def prediction_distorted(row):
    return not np.isclose(float(row["target_bias"]), float(row["pred_bias"]))


def prepare_regression_dotplot(df, multi):
    df = df.copy()

    if multi:
        df["setting"] = df["Protected_Feature"].map(multi_setting)
        df["group"] = df["Protected_Class"].map(short_group)

        order = ["ML-O", "ML-S", "MH-O", "MH-S", "LH-O", "LH-S", "MM-O", "MM-S"]
        df["setting"] = pd.Categorical(df["setting"], categories=order, ordered=True)

        df = df.sort_values(["setting", "group"])
        df["label"] = df["setting"].astype(str) + " · " + df["group"]

    else:
        regime_order = ["UTUP", "UTBP", "BTUP", "BTBP"]
        df["Protected_Feature"] = pd.Categorical(df["Protected_Feature"], categories=regime_order, ordered=True)
        df["group"] = df["Protected_Class"].map(short_group)

        df = df.sort_values(["Protected_Feature", "group"])
        df["label"] = df["Protected_Feature"].astype(str) + " · " + df["group"]

    return df


def draw_regression_dotplot(ax, df, *, multi, xlim, show_y):
    y = np.arange(len(df))

    ax.set_xlim(*xlim)
    add_x_parity_regions(ax)
    ax.axvline(1.0, color="black", linewidth=0.8, alpha=0.5)

    for yi, (_, row) in zip(y, df.iterrows()):
        color = DISTORTED if prediction_distorted(row) else UNDISTORTED

        ours = float(row["FairAI_pred"])
        dalex = float(row["dalex_pred"])

        ours_plot = min(ours, xlim[1])
        dalex_plot = min(dalex, xlim[1])

        ax.plot([ours_plot, dalex_plot], [yi, yi], color="#CFCFCF", linewidth=0.8, zorder=1)
        ax.scatter(ours_plot, yi, marker="o", s=48, color=color, zorder=3)
        ax.scatter(dalex_plot, yi, marker="^", s=52, color=color, zorder=3)

        if ours > xlim[1]:
            ax.annotate(f"{ours:.2f}→", (xlim[1], yi), xytext=(-3, 6), textcoords="offset points", ha="right", va="center", fontsize=7, color=color)

        if dalex > xlim[1]:
            ax.annotate(f"{dalex:.2f}→", (xlim[1], yi), xytext=(-3, -6), textcoords="offset points", ha="right", va="center", fontsize=7, color=color)

    if multi:
        for boundary in range(4, len(df), 4):
            ax.axhline(boundary - 0.5, color="#E3E3E3", linewidth=1.0, zorder=0)

    ax.set_ylim(len(df) - 0.35, -0.65)
    if show_y:
        ax.set_yticks(y)
        ax.set_yticklabels(df["label"], fontsize=8)
    else:
        ax.tick_params(axis="y", labelleft=False)

    ax.set_xlabel("Prediction diagnostic")
    ax.grid(axis="x", linestyle=":", alpha=0.28)


def plot_regression_row(reg, configs, filename, *, multi):
    n_panels = len(configs)
    fig_height = 11.2 if multi else 6.2
    fig_width = 5.6 * n_panels + 1.8
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, fig_height), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)

    for index, (ax, (simulation_type, title)) in enumerate(zip(axes, configs)):
        sub = reg[reg["simulation_type"] == simulation_type]
        if multi:
            sub = sub[sub["Protected_Feature"].str.startswith("BTBP_")]
        df = prepare_regression_dotplot(sub, multi)
        draw_regression_dotplot(ax, df, multi=multi, xlim=(0.5, 1.5), show_y=(index == 0))
        ax.set_title(title, fontsize=12)

    legend = [
        Line2D([], [], marker="o", linestyle="none", color="black", label="Proposed"),
        Line2D([], [], marker="^", linestyle="none", color="black", label="fairmodels/DALEX"),
        Line2D([], [], marker="s", linestyle="none", color=DISTORTED, label="Additional prediction distortion"),
        Line2D([], [], marker="s", linestyle="none", color=UNDISTORTED, label="No additional prediction distortion"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, filename)


def plot_regression_results(reg):
    plot_regression_row(
        reg,
        [
            ("low_bias_single_cat_equal_distribution", "Low bias, equal prevalence"),
            ("high_bias_single_cat_equal_distribution", "High bias, equal prevalence"),
            ("high_bias_single_cat_unequal_distribution", "High bias, unequal prevalence"),
        ],
        "reg1-3.pdf",
        multi=False,
    )
    plot_regression_row(
        reg,
        [
            ("high_bias_multiple_cats_unequal_distribution", "Multiple categories"),
            ("high_bias_multiple_cats_unequal_distribution_poor_model", "Multiple categories, poor model"),
        ],
        "reg4-5.pdf",
        multi=True,
    )

    for name in ["reg1.pdf", "reg2.pdf", "reg3.pdf", "reg4.pdf", "reg5.pdf"]:
        leftover = PLOTS / name
        if leftover.exists():
            leftover.unlink()


# ============================================================
# REGRESSION ROBUSTNESS
# ============================================================

def plot_ablation_panel(ax, df, *, show_right_ticks=False, wrap_chars=8, xlim=(0.4, 1.0)):
    """Horizontal bars with the layout from plots.ipynb."""
    classes = list(df["Protected_Class"].unique())
    colors = {
        cat: ABLATION_COLORS[i % len(ABLATION_COLORS)]
        for i, cat in enumerate(classes)
    }

    df_plot = df[::-1].copy()
    levels = df_plot["level"].unique()

    level_spacing = 1.0
    category_spacing = 0.8
    pos_map = {}
    y_positions = []
    level_centers = []
    current_y = 0.0

    for lvl in levels:
        lvl_cats = df_plot.loc[df_plot["level"] == lvl, "Protected_Class"].unique()
        level_start_y = current_y
        for cat in lvl_cats:
            pos_map[(lvl, cat)] = current_y
            y_positions.append(current_y)
            current_y += category_spacing
        level_center = level_start_y + (len(lvl_cats) - 1) * category_spacing / 2.0
        level_centers.append((lvl, level_center))
        if lvl != levels[-1]:
            current_y += level_spacing

    for _, row in df_plot.iterrows():
        lvl = row["level"]
        cat = row["Protected_Class"]
        mean_val = float(row["FairAI_pred_mean"])
        se_val = float(row["FairAI_pred_se"]) * 1.96
        y_pos = pos_map[(lvl, cat)]

        ax.barh(
            y_pos,
            mean_val,
            height=0.6,
            color=colors.get(cat, "gray"),
            alpha=0.95,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.errorbar(mean_val, y_pos, xerr=se_val, fmt="none", color="black", capsize=3, capthick=1)

    if show_right_ticks:
        tick_positions = [pos_map[(row["level"], row["Protected_Class"])] for _, row in df_plot.iterrows()]
        tick_labels = [row["Protected_Class"] for _, row in df_plot.iterrows()]
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels, fontsize=10)
        ax.tick_params(axis="y", labelright=True, labelleft=False, labelsize=10)
        ax.yaxis.set_ticks_position("right")
        ax.yaxis.set_label_position("right")
    else:
        ax.set_yticks([])
        ax.set_yticklabels([])

    for lvl, y_ctr in level_centers:
        raw = str(lvl).replace(",", ",\n")
        txt = textwrap.fill(raw, width=wrap_chars)
        ax.text(
            -0.005,
            y_ctr,
            txt,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=10,
            linespacing=1.1,
            wrap=True,
            clip_on=False,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2),
        )

    ax.set_xlabel("Fairness Score", fontsize=12)
    ax.axvline(x=PARITY_LOW, color="black", linestyle=":", linewidth=2, alpha=0.8)
    ax.set_xlim(*xlim)
    if y_positions:
        ax.set_ylim(min(y_positions) - 0.6, max(y_positions) + 0.6)


def plot_ablations(ab):
    for regime in ["UTUP", "UTBP", "BTUP", "BTBP"]:
        sub = ab[ab["Protected_Feature"] == regime]
        # Drop the (1.0, 1.0) baseline, as in plots.ipynb.
        bias = sub[sub["ablation"] == "vary_bias"].iloc[3:]
        dist = sub[sub["ablation"] == "vary_distribution"]
        model = sub[sub["ablation"] == "vary_model"]

        fig, axes = plt.subplots(1, 3, figsize=(22, 7.2), sharey=False)
        plot_ablation_panel(axes[0], bias, show_right_ticks=False, xlim=(0.4, 1))
        axes[0].set_title("Varying Bias", fontsize=12)
        plot_ablation_panel(axes[1], dist, show_right_ticks=False, xlim=(0.6, 1))
        axes[1].set_title("Varying Distribution", fontsize=12)
        plot_ablation_panel(axes[2], model, show_right_ticks=True, xlim=(0.6, 1))
        axes[2].set_title("Varying Model Quality", fontsize=12)

        fig.suptitle(regime, fontsize=16)
        fig.subplots_adjust(left=0.14, right=0.88, top=0.90, wspace=0.25)
        save(fig, f"ablations_{regime.lower()}.pdf")


# ============================================================
# MULTICLASS
# ============================================================

SIM_LABELS = {
    "high_bias_single_cat_equal_distribution_classification": "Equal prevalence",
    "high_bias_single_cat_unequal_distribution_classification": "Unequal prevalence",
    "high_bias_single_cat_equal_distribution_classification_poor_model": "Equal prevalence, poor model",
}


def classification_panel(cls, regime):
    d = cls[cls["Protected_Feature"] == regime].copy()

    simulation_order = list(SIM_LABELS.keys())
    d["simulation_type"] = pd.Categorical(d["simulation_type"], categories=simulation_order, ordered=True)

    d["group"] = d["Protected_Class"].map(short_group)
    d = d.sort_values(["simulation_type", "group"])

    metrics = ["fairness_class_0", "fairness_class_1", "fairness_class_2", "FairAI_pred"]
    matrix = d[metrics].to_numpy(dtype=float)

    row_labels = [f"{SIM_LABELS[simulation]} · {group}" for simulation, group in zip(d["simulation_type"].astype(str), d["group"])]
    return matrix, row_labels


def draw_classification_panel(ax, matrix, row_labels, norm, *, show_y=True):
    image = ax.imshow(matrix, cmap=CLASS_CMAP, norm=norm, aspect="auto")

    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(["Class 0", "Class 1", "Class 2", "Overall"], fontsize=11)

    ax.set_yticks(np.arange(len(row_labels)))
    if show_y:
        ax.set_yticklabels(row_labels, fontsize=10)
    else:
        ax.set_yticklabels([])

    for boundary in [2.5, 5.5]:
        ax.axhline(boundary, color="white", linewidth=4)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            red, green, blue, _ = CLASS_CMAP(norm(value))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if luminance < 0.45 else "black",
            )

    return image


def plot_classification(cls):
    metrics = ["fairness_class_0", "fairness_class_1", "fairness_class_2", "FairAI_pred"]

    global_min = float(cls[metrics].min().min())
    vmin = max(0.0, np.floor(global_min * 20) / 20)
    if vmin >= PARITY_LOW:
        vmin = PARITY_LOW - 0.05
    norm = TwoSlopeNorm(vmin=vmin, vcenter=PARITY_LOW, vmax=1.0)

    layout = [["UTUP", "UTBP"], ["BTUP", "BTBP"]]
    fig = plt.figure(figsize=(16.2, 10.2))
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=[1, 1, 0.045],
        wspace=0.18,
        hspace=0.32,
        left=0.22,
        right=0.94,
        top=0.94,
        bottom=0.07,
    )
    axes = [
        [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])],
        [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])],
    ]
    color_ax = fig.add_subplot(grid[:, 2])
    image = None

    for row, regimes in enumerate(layout):
        for col, regime in enumerate(regimes):
            matrix, row_labels = classification_panel(cls, regime)
            image = draw_classification_panel(axes[row][col], matrix, row_labels, norm, show_y=(col == 0))
            axes[row][col].set_title(regime, fontsize=15)

    colorbar = fig.colorbar(image, cax=color_ax)
    colorbar.set_label("Fairness score", fontsize=13)
    colorbar.ax.tick_params(labelsize=11)
    save(fig, "class.pdf")

    for name in ["class_utup.png", "class_utbp.png", "class_btup.png", "class_btbp.png"]:
        leftover = PLOTS / name
        if leftover.exists():
            leftover.unlink()


# ============================================================
# MAIN
# ============================================================

def main():
    reg = pd.read_excel(RESULTS / "regression_simulations.xlsx")
    ab = pd.read_excel(RESULTS / "reg_ablation.xlsx")
    cls = pd.read_excel(RESULTS / "classification_simulations.xlsx")

    required_reg = {
        "simulation_type",
        "Protected_Feature",
        "Protected_Class",
        "target_bias",
        "pred_bias",
        "FairAI_target",
        "FairAI_pred",
        "dalex_pred",
    }

    required_ab = {
        "Protected_Feature",
        "Protected_Class",
        "FairAI_pred_mean",
        "FairAI_pred_se",
        "ablation",
        "level",
    }

    required_cls = {
        "simulation_type",
        "Protected_Feature",
        "Protected_Class",
        "FairAI_pred",
        "fairness_class_0",
        "fairness_class_1",
        "fairness_class_2",
    }

    assert required_reg.issubset(reg.columns), f"Regression columns missing: {required_reg - set(reg.columns)}"
    assert required_ab.issubset(ab.columns), f"Ablation columns missing: {required_ab - set(ab.columns)}"
    assert required_cls.issubset(cls.columns), f"Classification columns missing: {required_cls - set(cls.columns)}"

    plot_regression_parity(reg)
    plot_regression_results(reg)
    plot_ablations(ab)
    plot_classification(cls)

    print("\nDone. Regenerated:")
    print("  target_pred.pdf")
    print("  reg1-3.pdf, reg4-5.pdf")
    print("  ablations_utup.pdf - ablations_btbp.pdf")
    print("  class.pdf")


if __name__ == "__main__":
    main()