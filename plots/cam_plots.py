import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

VERSION = "v1"
BASE_DIR = "/work/dldevel/trieba/bachelor_thesis"
EVAL_DIR = f"{BASE_DIR}/results/evaluation/hirescam_metrics_{VERSION}"
PLOT_PATH = f"{EVAL_DIR}/plots"

DPI = 200
FORMATS = ("png", "pdf")

MODELS = ["ResNet50", "ConvNeXt"]
MODEL_NAMES_RAW = {"resnet": "ResNet50", "convnext": "ConvNeXt"}
MODEL_COLORS = {"ResNet50": "#1f77b4", "ConvNeXt": "#d95f02"}

VARIANTS = ["original", "mixed_same", "mixed_rand", "only_fg", "only_fg_noise"]
LABELS = {
    "original": "Original",
    "mixed_same": "Mixed-Same",
    "mixed_rand": "Mixed-Rand",
    "only_fg": "Only-FG",
    "only_fg_noise": "Only-FG-Noise",
}

VARIANT_COLORS = {
    "original": "#111111",
    "mixed_same": "#2ca02c",
    "mixed_rand": "#d62728",
    "only_fg": "#9467bd",
    "only_fg_noise": "#8c564b",
}

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


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


def load_cam_data():
    with open(f"{EVAL_DIR}/hirescam_eval_results.json", "r") as f:
        raw_json = json.load(f)

    records = []
    ratios = set()

    for model_key, variants in raw_json.items():
        raw_m, raw_r = model_key.split("_")
        model = MODEL_NAMES_RAW.get(raw_m, raw_m)
        ratio = int(raw_r)
        ratios.add(ratio)

        for vkey, content in variants.items():
            if not vkey.startswith("eval_dataset/"):
                continue

            variant = vkey.replace("eval_dataset/", "")
            class_weighted = content.get("ClassWeighted", {})

            records.append({
                "model": model,
                "ratio": ratio,
                "variant": variant,
                "true_ebpg": class_weighted.get("True_EBPG", np.nan),
                "true_auc": class_weighted.get("True_AUC", np.nan),
            })

    df = pd.DataFrame(records)
    return df, sorted(ratios)


def plot_variant_lines(df, ratios):
    """dataset-variant trends across MAPS ratios."""
    metrics = [
        ("true_ebpg", "Energy-Based Pointing Game (EBPG)", "ebpg_vs_ratio_by_variant"),
        ("true_auc", "Foreground Separation (AUC-ROC)", "auc_vs_ratio_by_variant"),
    ]

    for key, ylabel, fname in metrics:
        fig, axes = plt.subplots(
            1,
            len(MODELS),
            figsize=(6.0 * len(MODELS), 4.8),
            sharey=True,
            squeeze=False,
        )
        axes = axes[0]

        for ax, model in zip(axes, MODELS):
            df_model = df[df["model"] == model]

            for variant in VARIANTS:
                sub = df_model[df_model["variant"] == variant].sort_values("ratio")
                if sub.empty:
                    continue

                ax.plot(
                    sub["ratio"],
                    sub[key],
                    marker="o",
                    ms=5,
                    lw=2.5 if variant == "original" else 1.6,
                    color=VARIANT_COLORS.get(variant, "#555555"),
                    label=LABELS.get(variant, variant),
                )

            ax.set_xticks(ratios)
            ax.set_xlabel("MAPS share in training data (%)")
            ax.set_title(model, fontsize=12)

        axes[0].set_ylabel(ylabel)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(0.99, 0.5),
            frameon=False,
        )
        fig.tight_layout()
        save(fig, fname)


def plot_attribution_shortcut_indicators(df, ratios):
    """Mixed-Same - Mixed-Rand attribution/background gaps."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    for model in MODELS:
        color = MODEL_COLORS.get(model)

        ebpg_same = [
            df[
                (df["model"] == model)
                & (df["ratio"] == ratio)
                & (df["variant"] == "mixed_same")
            ]["true_ebpg"].values[0]
            for ratio in ratios
        ]
        ebpg_rand = [
            df[
                (df["model"] == model)
                & (df["ratio"] == ratio)
                & (df["variant"] == "mixed_rand")
            ]["true_ebpg"].values[0]
            for ratio in ratios
        ]
        gap_ebpg = np.array(ebpg_same) - np.array(ebpg_rand)

        auc_same = [
            df[
                (df["model"] == model)
                & (df["ratio"] == ratio)
                & (df["variant"] == "mixed_same")
            ]["true_auc"].values[0]
            for ratio in ratios
        ]
        auc_rand = [
            df[
                (df["model"] == model)
                & (df["ratio"] == ratio)
                & (df["variant"] == "mixed_rand")
            ]["true_auc"].values[0]
            for ratio in ratios
        ]
        gap_auc = np.array(auc_same) - np.array(auc_rand)

        axes[0].plot(ratios, gap_ebpg, "o-", color=color, lw=1.8, label=model)
        axes[1].plot(ratios, gap_auc, "o-", color=color, lw=1.8, label=model)

    axes[0].axhline(0, color="k", lw=0.8, ls=":")
    axes[0].set_title(
        "Attribution Background Gap (EBPG)\nMixed-Same − Mixed-Rand Focus",
        fontsize=10,
    )
    axes[0].set_ylabel("Δ EBPG")
    axes[0].set_xlabel("MAPS share in training data (%)")
    axes[0].set_xticks(ratios)
    axes[0].legend(fontsize=8)

    axes[1].axhline(0, color="k", lw=0.8, ls=":")
    axes[1].set_title(
        "AUC Separation Background Gap\nMixed-Same − Mixed-Rand AUC",
        fontsize=10,
    )
    axes[1].set_ylabel("Δ AUC")
    axes[1].set_xlabel("MAPS share in training data (%)")
    axes[1].set_xticks(ratios)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    save(fig, "attribution_background_gap")


def main():
    print(f"Loading HiResCAM evaluation results from: {EVAL_DIR}")
    df, ratios = load_cam_data()

    print(f"Ratios: {ratios} | Models: {MODELS}")
    print(f"Writing figures to: {PLOT_PATH}")

    plot_variant_lines(df, ratios)
    plot_attribution_shortcut_indicators(df, ratios)

    print("Selected HiResCAM plots generated successfully.")


if __name__ == "__main__":
    main()
