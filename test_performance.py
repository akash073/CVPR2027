import json
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, balanced_accuracy_score


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path.cwd()
TEST_DIR = ROOT / "test"
MODEL_DIR = ROOT / "saved_models"
OUT_DIR = ROOT / "cvpr_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DELTA = 1.0
DATASET_TO_PLOT = "cifar100"
METRIC = "balanced_accuracy_pct"

# Feature importance settings
COMPUTE_IMPORTANCE = True
IMPORTANCE_REPEATS = 30
IMPORTANCE_SEED = 42

TEACHERS = ["vitb32", "vitl14"]

TEACHER_LABELS = {
    "vitb32": "(a) CLIP ViT-B/32",
    "vitl14": "(b) CLIP ViT-L/14",
}

STUDENTS = [
    "resnet18",
    "mobilenet_v2",
    "efficientnet_b0",
]

STUDENT_LABELS = {
    "resnet18": "ResNet-18",
    "mobilenet_v2": "MobileNetV2",
    "efficientnet_b0": "EfficientNet-B0",
}

# FIX: these two dictionaries were referenced but not defined
STUDENT_COLORS = {
    "resnet18": "#4D83A6",
    "mobilenet_v2": "#E5AB53",
    "efficientnet_b0": "#5E9C84",
}

STUDENT_PATTERNS = {
    "resnet18": "xxx",
    "mobilenet_v2": "////",
    "efficientnet_b0": "----",
}

MODELS = [
    ("Gradient Boosting", "GB", "gradient_boosting.joblib"),
    ("AdaBoost", "Ada", "adaboost.joblib"),
    ("Random Forest", "RF", "random_forest.joblib"),
    ("RBF-SVM", "SVM", "rbf_svm.joblib"),
    ("XGBoost", "XGB", "xgboost.joblib"),
]

CLASS_COUNTS = {
    "fmnist": 10,
    "svhn": 10,
    "cifar10": 10,
    "cifar100": 100,
    "tinyimagenet": 200,
}

NUMERIC = [
    "log2_d",
    "n_classes",
    "log2_mle",
    "log2_pca",
    "log2_erank",
    "log2_width_over_mle",
    "log2_width_over_pca",
    "log2_width_over_erank",
]

CATEGORICAL = ["teacher", "student", "mode"]
FEATURES = NUMERIC + CATEGORICAL

FEATURE_LABELS = {
    "log2_d": r"Width $\log_2 d$",
    "n_classes": "No. of classes",
    "log2_mle": r"$\log_2 d_{\mathrm{MLE}}$",
    "log2_pca": r"$\log_2 d_{\mathrm{PCA}}$",
    "log2_erank": r"$\log_2 d_{\mathrm{eRank}}$",
    "log2_width_over_mle": r"$\log_2(d/d_{\mathrm{MLE}})$",
    "log2_width_over_pca": r"$\log_2(d/d_{\mathrm{PCA}})$",
    "log2_width_over_erank": r"$\log_2(d/d_{\mathrm{eRank}})$",
    "teacher": "Teacher",
    "student": "Student",
    "mode": "Objective",
}


# ============================================================
# 2. CVPR STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.labelweight": "bold",
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.facecolor": "white",
    "hatch.linewidth": 0.5,
})


# ============================================================
# 3. PARSE INPUT FILENAMES
# ============================================================

FILE_RE = re.compile(
    r"^(.*?)_e\d+_pareto_summary(?:\(\d+\))?\.csv$",
    re.IGNORECASE,
)

STUDENT_ALIASES = {
    "resnet_18": "resnet18",
    "mobilenetv2": "mobilenet_v2",
    "efficientnetb0": "efficientnet_b0",
}

DATASET_ALIASES = {
    "fashionmnist": "fmnist",
    "fashion_mnist": "fmnist",
    "tiny_imagenet": "tinyimagenet",
    "tinyimagenet200": "tinyimagenet",
}


def parse_filename(filename):
    match = FILE_RE.fullmatch(filename)
    if match is None:
        return None

    stem = match.group(1).lower()

    student_names = sorted(STUDENTS + list(STUDENT_ALIASES), key=len, reverse=True)
    student = next((s for s in student_names if stem.endswith("_" + s)), None)
    if student is None:
        return None
    stem = stem[:-(len(student) + 1)]

    dataset_names = sorted(list(CLASS_COUNTS) + list(DATASET_ALIASES), key=len, reverse=True)
    dataset = next((d for d in dataset_names if stem.endswith("_" + d)), None)
    if dataset is None:
        return None

    teacher = re.sub(r"[-_/]", "", stem[:-(len(dataset) + 1)])
    if not teacher:
        return None

    return (
        teacher,
        DATASET_ALIASES.get(dataset, dataset),
        STUDENT_ALIASES.get(student, student),
    )


# ============================================================
# 4. EXTRACT GEOMETRY
# ============================================================

def selected_dimension(raw, name):
    column = f"{name}_selected"
    if column not in raw:
        return np.nan

    flags = raw[column].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    values = (pd.to_numeric(raw.loc[flags, "bottleneck_d"], errors="coerce")
                .dropna().unique())

    if len(values) > 1:
        raise ValueError(f"{column} selects multiple dimensions: {values}")

    return float(values[0]) if len(values) else np.nan


# ============================================================
# 5. LOAD CIFAR-100 TEST FILES
# ============================================================

def load_test_data():
    if not TEST_DIR.is_dir():
        raise FileNotFoundError(TEST_DIR)

    records, seen = [], set()

    for path in sorted(TEST_DIR.glob("*.csv")):
        parsed = parse_filename(path.name)
        if parsed is None:
            continue

        teacher, dataset, student = parsed
        if dataset != DATASET_TO_PLOT or teacher not in TEACHERS:
            continue

        key = (teacher, dataset, student)
        if key in seen:
            raise ValueError(f"Duplicate input: {key}")
        seen.add(key)

        raw = pd.read_csv(path)

        required = {"mode", "bottleneck_kind", "bottleneck_d", "test_accuracy_pct"}
        missing = required - set(raw.columns)
        if missing:
            raise ValueError(f"{path.name}: Missing {sorted(missing)}")

        geometry = {name: selected_dimension(raw, name) for name in ("mle", "pca", "erank")}

        linear = raw.loc[
            raw["bottleneck_kind"].astype(str).str.lower().str.strip().eq("linear")
        ].copy()

        for col in ("bottleneck_d", "test_accuracy_pct"):
            linear[col] = pd.to_numeric(linear[col], errors="coerce")

        linear = linear.dropna(subset=["mode", "bottleneck_d", "test_accuracy_pct"])
        linear = linear[linear["bottleneck_d"] > 0]
        if linear.empty:
            raise ValueError(f"No linear rows: {path.name}")

        linear = (linear.groupby(["mode", "bottleneck_d"], as_index=False)
                        ["test_accuracy_pct"].mean())

        best = linear.groupby("mode")["test_accuracy_pct"].transform("max")
        linear["regret_pp"] = best - linear["test_accuracy_pct"]
        linear["feasible"] = (linear["regret_pp"] <= DELTA + 1e-10).astype(int)

        linear["teacher"] = teacher
        linear["dataset"] = dataset
        linear["student"] = student
        linear["n_classes"] = CLASS_COUNTS[dataset]
        linear["log2_d"] = np.log2(linear["bottleneck_d"])

        for name, dim in geometry.items():
            if np.isfinite(dim) and dim > 0:
                linear[f"log2_{name}"] = np.log2(dim)
                linear[f"log2_width_over_{name}"] = np.log2(linear["bottleneck_d"] / dim)
            else:
                linear[f"log2_{name}"] = np.nan
                linear[f"log2_width_over_{name}"] = np.nan

        records.append(linear)
        print(f"Loaded: {path.name}")

    if not records:
        raise FileNotFoundError("No matching CIFAR-100 CSV files found in test/")

    return pd.concat(records, ignore_index=True)


# ============================================================
# 6. LOAD FIVE SAVED CLASSIFIERS
# ============================================================

def load_classifiers():
    classifiers = {}
    for name, _, filename in MODELS:
        path = MODEL_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        classifiers[name] = joblib.load(path)
        print(f"Loaded model: {name}")
    return classifiers


# ============================================================
# 7. EVALUATE CLASSIFIERS
# ============================================================

def evaluate_classifiers(data, classifiers):
    results = []

    for teacher in TEACHERS:
        for student in STUDENTS:
            subset = data[(data["teacher"] == teacher) & (data["student"] == student)]
            if subset.empty:
                warnings.warn(f"Missing evaluation data: {teacher}/{student}")
                continue

            X = subset[FEATURES]
            y = subset["feasible"].to_numpy()

            for name, _, _ in MODELS:
                pred = classifiers[name].predict(X)
                results.append({
                    "teacher": teacher,
                    "student": student,
                    "model": name,
                    "accuracy_pct": 100 * accuracy_score(y, pred),
                    "balanced_accuracy_pct": (100 * balanced_accuracy_score(y, pred)
                                              if len(np.unique(y)) == 2 else np.nan),
                    "n_candidates": len(y),
                    "n_feasible": int(y.sum()),
                    "n_infeasible": int((y == 0).sum()),
                })

    scores = pd.DataFrame(results)
    scores.to_csv(OUT_DIR / "figure3_scores.csv", index=False)
    return scores


# ============================================================
# 7b. FEATURE IMPORTANCE
# ============================================================

def _original_feature(encoded_name):
    """Map an encoded column name (e.g. 'cat__mode_rkd') back to its input feature."""
    raw = encoded_name.split("__", 1)[-1]
    for f in CATEGORICAL:
        if raw == f or raw.startswith(f + "_"):
            return f
    return raw


def compute_feature_importance(data, classifiers):
    """
    1. Permutation importance for all five classifiers: drop in balanced accuracy
       (percentage points) when one ORIGINAL input column is shuffled.
    2. Impurity importance for tree-based classifiers, with one-hot columns
       summed back to their original feature.

    NOTE: on a single held-out dataset, features that are constant there
    (e.g. n_classes, and teacher-geometry features within a teacher) will
    get ~0 permutation importance even if the model relies on them.
    """
    X = data[FEATURES]
    y = data["feasible"].to_numpy()

    if len(np.unique(y)) < 2:
        warnings.warn("Only one feasibility class in the evaluation data; "
                      "balanced-accuracy permutation importance is undefined.")
        return pd.DataFrame(), pd.DataFrame()

    constant = [f for f in FEATURES if X[f].nunique(dropna=False) <= 1]
    if constant:
        print("\nNOTE: constant in the evaluation data, so permutation importance "
              f"will be ~0 regardless of model reliance: {constant}")

    perm_rows, tree_rows = [], []

    for name, short, _ in MODELS:
        model = classifiers[name]
        print(f"Permutation importance: {name} ...")

        r = permutation_importance(
            model, X, y,
            scoring="balanced_accuracy",
            n_repeats=IMPORTANCE_REPEATS,
            random_state=IMPORTANCE_SEED,
            n_jobs=-1,
        )
        for feat, mean, std in zip(FEATURES, r.importances_mean, r.importances_std):
            perm_rows.append({
                "model": name,
                "model_short": short,
                "feature": feat,
                "constant_in_eval": feat in constant,
                "importance_mean_pp": 100 * mean,
                "importance_std_pp": 100 * std,
            })

        # Impurity importance (tree models only)
        steps = getattr(model, "named_steps", {})
        est = steps.get("model")
        pre = steps.get("features")
        if est is not None and pre is not None and hasattr(est, "feature_importances_"):
            try:
                encoded = pre.get_feature_names_out()
            except Exception:
                encoded = [f"f{i}" for i in range(len(est.feature_importances_))]
            for enc_name, imp in zip(encoded, est.feature_importances_):
                tree_rows.append({
                    "model": name,
                    "model_short": short,
                    "feature": _original_feature(enc_name),
                    "importance": float(imp),
                })

    perm = pd.DataFrame(perm_rows)
    tree = pd.DataFrame(tree_rows)
    if not tree.empty:
        tree = tree.groupby(["model", "model_short", "feature"], as_index=False)["importance"].sum()

    perm.to_csv(OUT_DIR / "feature_importance_permutation.csv", index=False)
    tree.to_csv(OUT_DIR / "feature_importance_tree.csv", index=False)

    # Print readable tables (features x models)
    order = [short for _, short, _ in MODELS]
    perm_table = (perm.pivot(index="feature", columns="model_short", values="importance_mean_pp")
                      .reindex(index=FEATURES, columns=order).round(2))
    print("\nPermutation importance (drop in balanced accuracy, pp):")
    print(perm_table.to_string())

    if not tree.empty:
        tree_table = (tree.pivot(index="feature", columns="model_short", values="importance")
                          .reindex(index=FEATURES)
                          .reindex(columns=[c for c in order if c in tree["model_short"].unique()])
                          .round(3))
        print("\nImpurity importance (tree models; one-hot columns summed):")
        print(tree_table.to_string())

    print(f"\nSaved importance CSVs to {OUT_DIR}")
    return perm, tree


def plot_feature_importance(perm):
    """Grouped horizontal bars: features (rows) x classifiers (bars)."""
    if perm.empty:
        return

    order = [short for _, short, _ in MODELS]
    colors = ["#1B3A6B", "#5B8DB8", "#9DC3E6", "#C00000", "#BFBFBF"]
    feats = FEATURES[::-1]                      # top-to-bottom in FEATURES order
    y = np.arange(len(feats))
    h = 0.16

    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    for i, short in enumerate(order):
        sub = perm[perm["model_short"] == short].set_index("feature")
        vals = [sub.loc[f, "importance_mean_pp"] if f in sub.index else np.nan for f in feats]
        errs = [sub.loc[f, "importance_std_pp"] if f in sub.index else 0 for f in feats]
        ax.barh(y + (i - 2) * h, vals, height=h, xerr=errs, color=colors[i],
                edgecolor="black", linewidth=0.3, error_kw={"elinewidth": 0.4, "capsize": 0},
                label=short, zorder=3)

    constant = set(perm.loc[perm["constant_in_eval"], "feature"])
    labels = [FEATURE_LABELS.get(f, f) + (" *" if f in constant else "") for f in feats]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Drop in balanced accuracy (pp)", fontsize=8)
    ax.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.45, 1.09), frameon=False,
              fontsize=6.5, handlelength=1.0, columnspacing=0.8)
    if constant:
        fig.text(0.01, 0.005, "* constant in held-out data (importance not measurable)",
                 fontsize=5.5, style="italic")

    fig.tight_layout(pad=0.3)
    out = OUT_DIR / "feature_importance_cvpr.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
    plt.show()
    plt.close(fig)
    print(f"Saved: {out}")


# ============================================================
# 8. PLOT FIGURE 3
# ============================================================

def plot_figure3(scores):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.35), sharey=True)

    x = np.arange(len(MODELS))
    bar_width = 0.23
    offsets = [-bar_width, 0, bar_width]

    for panel_idx, teacher in enumerate(TEACHERS):
        ax = axes[panel_idx]
        teacher_scores = scores[scores["teacher"] == teacher]

        for student_idx, student in enumerate(STUDENTS):
            values = []
            for name, _, _ in MODELS:
                selected = teacher_scores[(teacher_scores["student"] == student)
                                          & (teacher_scores["model"] == name)]
                values.append(float(selected[METRIC].iloc[0]) if not selected.empty else np.nan)

            positions = x + offsets[student_idx]
            bars = ax.bar(
                positions,
                np.nan_to_num(values, nan=0),
                width=bar_width * 0.93,
                color=STUDENT_COLORS[student],
                hatch=STUDENT_PATTERNS[student],
                edgecolor="black",
                linewidth=0.35,
                label=STUDENT_LABELS[student],
                zorder=3,
            )

            # Percentage values above bars
            for bar, value in zip(bars, values):
                if not np.isfinite(value):
                    bar.set_visible(False)
                    continue
                ax.annotate(
                    f"{value:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=6, fontweight="bold", rotation=90,
                    clip_on=False,
                )

        ax.set_xticks(x)
        ax.set_xticklabels([short for _, short, _ in MODELS], fontsize=11, fontweight="bold")
        ax.set_xlim(-0.55, len(MODELS) - 0.45)
        ax.set_ylim(0, 112)
        ax.set_yticks([0, 20, 40, 60, 80, 100])

        # ViT teacher names BELOW the graphs
        ax.set_xlabel(TEACHER_LABELS[teacher], fontsize=10, fontweight="bold", labelpad=8)

        ax.grid(axis="y", linestyle="--", linewidth=0.45, alpha=0.25)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0, pad=3)
        ax.tick_params(axis="y", labelsize=10, length=2.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(
        "Balanced accuracy" if METRIC == "balanced_accuracy_pct" else "Feasibility accuracy (%)",
        fontsize=10, fontweight="bold",
    )

    # Student legend at TOP
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=3, frameon=False, fontsize=6.5, handlelength=1.3, columnspacing=1.8)

    fig.subplots_adjust(left=0.085, right=0.99, top=0.85, bottom=0.27, wspace=0.14)

    base = OUT_DIR / "figure3_cvpr"
    fig.savefig(f"{base}.pdf", dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.show()
    plt.close(fig)
    print(f"\nSaved:\n{base}.pdf")


# ============================================================
# 9. MAIN
# ============================================================

def main():
    print("=" * 60)
    print("CVPR FIGURE 3 + FEATURE IMPORTANCE")
    print("=" * 60)

    metadata_path = MODEL_DIR / "training_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        trained_delta = float(metadata.get("delta_pp", DELTA))
        if not np.isclose(trained_delta, DELTA):
            raise ValueError("DELTA differs from saved model training.")

        trained_features = metadata.get("features")
        if trained_features and trained_features != FEATURES:
            raise ValueError("Feature schema differs from saved models.")

        trained_datasets = metadata.get("datasets", [])
        if DATASET_TO_PLOT in trained_datasets:
            warnings.warn(f"The saved models were trained WITH {DATASET_TO_PLOT}. "
                          "Scores and importances are NOT held-out (LODO) results.")

    warnings.warn("For independent LODO evaluation, the saved models "
                  "must have been trained without CIFAR-100.")

    data = load_test_data()
    classifiers = load_classifiers()

    scores = evaluate_classifiers(data, classifiers)
    print("\nScores:")
    print(scores.to_string(index=False))

    plot_figure3(scores)

    if COMPUTE_IMPORTANCE:
        perm, _ = compute_feature_importance(data, classifiers)
        plot_feature_importance(perm)

    print("\nCompleted.")


if __name__ == "__main__":
    main()