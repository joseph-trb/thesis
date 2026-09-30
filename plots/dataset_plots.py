import os
import json
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore", category=RuntimeWarning)

BASE = "/work/dldevel/trieba/bachelor_thesis"
EVAL_ROOT = f"{BASE}/results/evaluation"
VERSION = "v4"
PLOT_PATH = f"{EVAL_ROOT}/plots/{VERSION}"

DPI = 200
FORMATS = ("png", "pdf")

MODELS = ["ResNet50", "ConvNeXt"]
MODEL_COLORS = {"ResNet50": "#1f77b4", "ConvNeXt": "#d95f02"}

VARIANTS = [
    "original", "only_bg_b", "only_bg_t", "only_bg_noise",
    "no_fg", "no_fg_noise", "only_fg", "only_fg_noise",
    "mixed_same", "mixed_rand",
]

LABELS = {
    "original": "Original", "only_bg_b": "Only-BG-B", "only_bg_t": "Only-BG-T",
    "only_bg_noise": "Only-BG-Noise", "no_fg": "No-FG", "no_fg_noise": "No-FG-Noise",
    "only_fg": "Only-FG", "only_fg_noise": "Only-FG-Noise",
    "mixed_same": "Mixed-Same", "mixed_rand": "Mixed-Rand",
}

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def load_results():
    data = {}
    for name in os.listdir(EVAL_ROOT):
        path = os.path.join(EVAL_ROOT, name, VERSION, "eval_dataset_results.json")
        if name.isdigit() and os.path.isfile(path):
            with open(path) as f:
                data[int(name)] = json.load(f)
    if not data:
        raise SystemExit(
            f"No eval_dataset_results.json found under {EVAL_ROOT}/<ratio>/{VERSION}/"
        )
    return dict(sorted(data.items()))


DATA = load_results()
RATIOS = list(DATA)
MODELS = [m for m in MODELS if any(m in DATA[r] for r in RATIOS)]
VARIANTS = [
    v for v in VARIANTS
    if any(v in DATA[r].get(m, {}) for r in RATIOS for m in MODELS)
]
CLASSES = sorted({
    c
    for r in RATIOS
    for m in MODELS
    for v in VARIANTS
    for c in DATA[r].get(m, {}).get(v, {}).get("PerClass", {})
})
CHANCE = 100 / max(len(CLASSES), 1)
PAL = dict(zip(RATIOS, plt.cm.viridis(np.linspace(0.05, 0.9, len(RATIOS)))))


def metric(model, variant, key):
    return np.array([
        DATA[r].get(model, {}).get(variant, {}).get(key, np.nan)
        for r in RATIOS
    ], float)


def bg_gap(model, superclass=False):
    key = "background_gap_superclass" if superclass else "background_gap"
    vals = np.array([
        DATA[r].get(model, {}).get(key, np.nan)
        for r in RATIOS
    ], float)
    if np.isnan(vals).all():
        acc = "SuperclassAccuracy" if superclass else "Accuracy"
        vals = metric(model, "mixed_same", acc) - metric(model, "mixed_rand", acc)
    return vals


def rel_bg_gap(model, superclass=False):
    acc = "SuperclassAccuracy" if superclass else "Accuracy"
    same = metric(model, "mixed_same", acc)
    rand = metric(model, "mixed_rand", acc)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = 100 * (same - rand) / same
    return rel


def save(fig, name, formats=FORMATS):
    os.makedirs(PLOT_PATH, exist_ok=True)
    for ext in formats:
        fig.savefig(
            os.path.join(PLOT_PATH, f"{name}.{ext}"),
            dpi=DPI,
            bbox_inches="tight",
        )
    plt.close(fig)
    print(f"  saved {name}")


def sync_ylim(axes, pad=0.05):
    axes = [ax for ax in np.ravel(axes) if ax is not None]
    if not axes:
        return
    lo = min(ax.get_ylim()[0] for ax in axes)
    hi = max(ax.get_ylim()[1] for ax in axes)
    span = hi - lo
    if span <= 0:
        return
    lo, hi = lo - pad * span, hi + pad * span
    for ax in axes:
        ax.set_ylim(lo, hi)


def ratio_ticks(ax):
    ticks = [r for r in RATIOS if r % 25 == 0]
    ax.set_xticks(ticks if ticks else [RATIOS[0], RATIOS[-1]])


def plot_variant_bars():
    fig, axes = plt.subplots(
        1, len(MODELS),
        figsize=(7.2 * len(MODELS), 4.8),
        sharey=True,
        squeeze=False,
    )
    x = np.arange(len(VARIANTS))
    w = 0.8 / len(RATIOS)

    for ax, m in zip(axes[0], MODELS):
        for i, r in enumerate(RATIOS):
            vals = [
                DATA[r].get(m, {}).get(v, {}).get("Accuracy", np.nan)
                for v in VARIANTS
            ]
            ax.bar(
                x + (i - (len(RATIOS) - 1) / 2) * w,
                vals,
                w,
                color=PAL[r],
                label=f"{r}%",
            )
        ax.axhline(CHANCE, ls=":", c="gray", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[v] for v in VARIANTS], rotation=35, ha="right")
        ax.set_title(m)
        ax.grid(axis="x", alpha=0)

    axes[0, 0].set_ylabel("Top-1 accuracy (%)")
    axes[0, -1].legend(title="MAPS share", fontsize=8, ncol=2)
    axes[0, 0].text(
        0.01,
        CHANCE + 1,
        "chance (1/#classes)",
        color="gray",
        fontsize=8,
        transform=axes[0, 0].get_yaxis_transform(),
    )
    fig.tight_layout()
    save(fig, "accuracy_by_variant")


def plot_ratio_lines():
    keys = [
        ("Accuracy", "Top-1 accuracy (%)"),
        ("SuperclassAccuracy", "Superclass accuracy (%)"),
    ]
    fig, axes = plt.subplots(
        2, len(MODELS),
        figsize=(5.5 * len(MODELS), 8),
        sharex=True,
        squeeze=False,
    )
    cmap = plt.get_cmap("tab10")

    for row, (key, ylab) in enumerate(keys):
        for col, m in enumerate(MODELS):
            ax = axes[row, col]
            for i, v in enumerate(VARIANTS):
                ax.plot(
                    RATIOS,
                    metric(m, v, key),
                    marker="o",
                    ms=4,
                    lw=2.6 if v == "original" else 1.4,
                    color="black" if v == "original" else cmap(i % 10),
                    label=LABELS[v],
                )
            ratio_ticks(ax)
            if col == 0:
                ax.set_ylabel(ylab)
            if row == 0:
                ax.set_title(m)
            if row == 1:
                ax.set_xlabel("MAPS share in training data (%)")
        sync_ylim(axes[row, :])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.tight_layout()
    save(fig, "accuracy_vs_ratio")


def plot_foreground_advantage():
    if "only_fg" not in VARIANTS:
        return

    bg_variants = [v for v in ("only_bg_b", "only_bg_t") if v in VARIANTS]
    if not bg_variants:
        return

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for m in MODELS:
        backgrounds = [metric(m, v, "Accuracy") for v in bg_variants]
        foreground_advantage = (
            metric(m, "only_fg", "Accuracy") - np.nanmean(backgrounds, axis=0)
        )
        ax.plot(
            RATIOS,
            foreground_advantage,
            "o-",
            color=MODEL_COLORS.get(m),
            label=m,
        )

    ax.axhline(0, color="k", lw=0.8)
    ratio_ticks(ax)
    ax.set_xlabel("MAPS share in training data (%)")
    ax.set_ylabel("percentage points")
    ax.set_title("Foreground advantage\nOnly-FG − mean(Only-BG-B, Only-BG-T)", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    save(fig, "05a_foreground_advantage")


def plot_relative_background_gap():
    if not {"mixed_same", "mixed_rand"} <= set(VARIANTS):
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    for m in MODELS:
        c = MODEL_COLORS.get(m)
        axes[0].plot(RATIOS, bg_gap(m), "o-", color=c, label=m)

        rel = rel_bg_gap(m)
        axes[1].plot(RATIOS, rel, "o-", color=c, label=f"{m} top-1")
        axes[1].plot(
            RATIOS,
            rel_bg_gap(m, superclass=True),
            "s--",
            color=c,
            label=f"{m} superclass",
        )
        for r, v in zip(RATIOS, rel):
            if r % 25 == 0 and not np.isnan(v):
                axes[1].annotate(
                    f"{v:.0f}%",
                    (r, v),
                    textcoords="offset points",
                    xytext=(0, 6),
                    ha="center",
                    fontsize=7,
                    color=c,
                )

        axes[2].plot(RATIOS, metric(m, "mixed_same", "Accuracy"), "o-", color=c)
        axes[2].plot(RATIOS, metric(m, "mixed_rand", "Accuracy"), "o--", color=c)

    axes[0].set_title("Absolute background gap\nMixed-Same − Mixed-Rand", fontsize=10)
    axes[0].set_ylabel("percentage points")
    axes[1].set_title(
        "Relative background gap\n100 · (Mixed-Same − Mixed-Rand) / Mixed-Same",
        fontsize=10,
    )
    axes[1].set_ylabel("% of Mixed-Same accuracy lost")
    axes[2].set_title("Underlying top-1 accuracies", fontsize=10)
    axes[2].set_ylabel("accuracy (%)")

    handles = [
        Line2D([0], [0], color=MODEL_COLORS.get(m), lw=2, label=m)
        for m in MODELS
    ]
    handles += [
        Line2D([0], [0], color="gray", ls="-", label="Mixed-Same"),
        Line2D([0], [0], color="gray", ls="--", label="Mixed-Rand"),
    ]
    axes[2].legend(handles=handles, fontsize=8)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)

    for ax in axes:
        ratio_ticks(ax)
        ax.set_xlabel("MAPS share in training data (%)")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylim(bottom=0)

    fig.tight_layout()
    save(fig, "relative_background_gap")


def main():
    print(
        f"MAPS ratios found: {RATIOS} | models: {MODELS} | "
        f"{len(CLASSES)} classes"
    )
    print(f"Writing to {PLOT_PATH}")

    plot_variant_bars()
    plot_ratio_lines()
    plot_foreground_advantage()
    plot_relative_background_gap()

    print("Done.")


if __name__ == "__main__":
    main()
