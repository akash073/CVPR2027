import inspect
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    AdaBoostClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC






DATA_DIR = Path.cwd()
OUT_DIR = DATA_DIR / "saved_models"

DELTA = 1.0
SEED = 42

CLASSES = {
    "fmnist": 10,
    "svhn": 10,
    "cifar10": 10,
    "cifar100": 100,
    "tinyimagenet": 200,
}

DATASET_ALIASES = {
    "fashionmnist": "fmnist",
    "fashion_mnist": "fmnist",
    "tiny_imagenet": "tinyimagenet",
    "tinyimagenet200": "tinyimagenet",
}

STUDENT_ALIASES = {
    "mobilenetv2": "mobilenet_v2",
    "efficientnetb0": "efficientnet_b0",
    "resnet_18": "resnet18",
}

STUDENTS = (
    "resnet18",
    "resnet_18",
    "mobilenet_v2",
    "mobilenetv2",
    "efficientnet_b0",
    "efficientnetb0",
)

FILE_RE = re.compile(
    r"^(?P<prefix>.+)_e(?P<epochs>\d+)"
    r"_pareto_summary(?:\(\d+\))?\.csv$",
    re.IGNORECASE,
)

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

CATEGORICAL = [
    "teacher",
    "student",
    "mode",
]

FEATURES = NUMERIC + CATEGORICAL

MODEL_NAMES = {
    "Gradient Boosting": "gradient_boosting",
    "AdaBoost": "adaboost",
    "Random Forest": "random_forest",
    "RBF-SVM": "rbf_svm",
    "XGBoost": "xgboost",
}






def parse_filename(filename):

    match = FILE_RE.fullmatch(filename)

    if match is None:
        return None

    prefix = match.group("prefix").lower()

    student = next(
        (
            s for s in sorted(STUDENTS, key=len, reverse=True)
            if prefix.endswith("_" + s)
        ),
        None,
    )

    if student is None:
        return None

    base = prefix[:-(len(student) + 1)]

    aliases = {
        **{d: d for d in CLASSES},
        **DATASET_ALIASES,
    }

    dataset = next(
        (
            d for d in sorted(aliases, key=len, reverse=True)
            if base.endswith("_" + d)
        ),
        None,
    )

    if dataset is None:
        return None

    teacher = base[:-(len(dataset) + 1)]

    if not teacher:
        return None

    return (
        teacher,
        aliases[dataset],
        STUDENT_ALIASES.get(student, student),
        int(match.group("epochs")),
    )






def selected_dimension(raw, name):

    column = f"{name}_selected"

    if column not in raw.columns:
        return np.nan

    flags = (
        raw[column]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    dimensions = (
        pd.to_numeric(
            raw.loc[flags, "bottleneck_d"],
            errors="coerce",
        )
        .dropna()
        .unique()
    )

    if len(dimensions) > 1:
        raise ValueError(
            f"{column} selects multiple dimensions: {dimensions}"
        )

    return float(dimensions[0]) if len(dimensions) else np.nan






def load_data(folder, delta):

    records = []
    audits = []
    skipped = []

    for path in sorted(folder.glob("*.csv")):

        parsed = parse_filename(path.name)

        if parsed is None:
            skipped.append(path.name)
            continue

        teacher, dataset, student, epochs = parsed

        raw = pd.read_csv(path)

        required = {
            "mode",
            "bottleneck_kind",
            "bottleneck_d",
            "test_accuracy_pct",
        }

        missing = required - set(raw.columns)

        if missing:
            raise ValueError(
                f"{path.name}: Missing columns {sorted(missing)}"
            )

        geometry = {
            key: selected_dimension(raw, key)
            for key in ("mle", "pca", "erank")
        }

        
        linear = raw.loc[
            raw["bottleneck_kind"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("linear")
        ].copy()

        linear["bottleneck_d"] = pd.to_numeric(
            linear["bottleneck_d"],
            errors="coerce",
        )

        linear["test_accuracy_pct"] = pd.to_numeric(
            linear["test_accuracy_pct"],
            errors="coerce",
        )

        linear = linear.dropna(
            subset=[
                "mode",
                "bottleneck_d",
                "test_accuracy_pct",
            ]
        )

        linear = linear[
            linear["bottleneck_d"] > 0
        ]

        if linear.empty:
            raise ValueError(
                f"{path.name}: No valid LINEAR bottlenecks"
            )

        
        linear = (
            linear.groupby(
                ["mode", "bottleneck_d"],
                as_index=False,
            )["test_accuracy_pct"]
            .mean()
        )

        
        linear["best_test_accuracy_pct"] = (
            linear.groupby("mode")["test_accuracy_pct"]
            .transform("max")
        )

        
        linear["regret_pp"] = (
            linear["best_test_accuracy_pct"]
            - linear["test_accuracy_pct"]
        )

        
        linear["feasible"] = (
            linear["regret_pp"] <= delta + 1e-10
        ).astype(int)

        linear["teacher"] = teacher
        linear["dataset"] = dataset
        linear["student"] = student
        linear["epochs"] = epochs
        linear["source_file"] = path.name

        linear["n_classes"] = CLASSES[dataset]

        linear["log2_d"] = np.log2(
            linear["bottleneck_d"]
        )

        for key, dimension in geometry.items():

            linear[f"{key}_dim"] = dimension

            if pd.notna(dimension) and dimension > 0:

                linear[f"log2_{key}"] = np.log2(
                    dimension
                )

                linear[f"log2_width_over_{key}"] = np.log2(
                    linear["bottleneck_d"] / dimension
                )

            else:

                linear[f"log2_{key}"] = np.nan
                linear[f"log2_width_over_{key}"] = np.nan

        records.append(linear)

        audits.append({
            "file": path.name,
            "teacher": teacher,
            "dataset": dataset,
            "student": student,
            "epochs": epochs,
            "rows": len(linear),
            **{
                f"{key}_dim": value
                for key, value in geometry.items()
            },
        })

        print(
            f"Loaded: {path.name}\n"
            f"  Teacher: {teacher}\n"
            f"  Dataset: {dataset}\n"
            f"  Student: {student}\n"
            f"  Rows: {len(linear)}\n"
        )

    if not records:
        raise FileNotFoundError(
            f"No matching Pareto CSV files found in {folder}"
        )

    if skipped:
        print(
            f"Skipped {len(skipped)} nonmatching CSV file(s)."
        )

    data = pd.concat(
        records,
        ignore_index=True,
    )

    return data, pd.DataFrame(audits)






def make_pipeline(model_name, seed):

    numeric_steps = [
        (
            "impute",
            SimpleImputer(strategy="median")
        )
    ]

    
    if model_name == "RBF-SVM":
        numeric_steps.append(
            ("scale", StandardScaler())
        )

    
    encoder_kwargs = {
        "handle_unknown": "ignore"
    }

    if "sparse_output" in inspect.signature(
        OneHotEncoder
    ).parameters:
        encoder_kwargs["sparse_output"] = False
    else:
        encoder_kwargs["sparse"] = False

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(numeric_steps),
                NUMERIC,
            ),
            (
                "cat",
                OneHotEncoder(**encoder_kwargs),
                CATEGORICAL,
            ),
        ],
        remainder="drop",
    )

    
    if model_name == "Gradient Boosting":

        classifier = GradientBoostingClassifier(
            n_estimators=180,
            max_depth=2,
            learning_rate=0.05,
            random_state=seed,
        )

    
    elif model_name == "AdaBoost":

        classifier = AdaBoostClassifier(
            n_estimators=150,
            learning_rate=0.05,
            random_state=seed,
        )

    
    elif model_name == "Random Forest":

        classifier = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )

    
    elif model_name == "RBF-SVM":

        classifier = SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=seed,
        )

    
    elif model_name == "XGBoost":

        from xgboost import XGBClassifier

        classifier = XGBClassifier(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=5.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=4,
            tree_method="hist",
        )

    else:
        raise ValueError(
            f"Unknown classifier: {model_name}"
        )

    return Pipeline([
        ("features", preprocessing),
        ("model", classifier),
    ])






from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support


def load_or_train_model(name, stem, train):
    path = OUT_DIR / f"{stem}.joblib"

    if path.is_file():
        print(f"Loading existing model: {path}")
        model = joblib.load(path)  # Only load joblib files from a trusted source.

        if isinstance(model, dict) and "model" in model:
            model = model["model"]

        if not hasattr(model, "predict"):
            raise TypeError(f"{path}: saved object is not a predictor")

        saved_features = getattr(model, "feature_names_in_", None)
        if saved_features is not None and list(saved_features) != FEATURES:
            raise ValueError(
                f"{path}: expected input features {FEATURES}, "
                f"but saved model uses {list(saved_features)}"
            )

        try:
            model.predict(train[FEATURES].iloc[:1])
        except Exception as exc:
            raise ValueError(
                f"Existing model {path} is incompatible with the current data. "
                "Check the model rather than silently retraining it."
            ) from exc

        print(
            "WARNING: Loading a saved model does not verify which datasets, "
            "tolerance, or labels it was trained on. Confirm that CIFAR-100 "
            "was excluded from training before claiming held-out results."
        )
        return model, path, "loaded"

    print(f"Model not found; training {name}...")
    model = make_pipeline(name, SEED)
    model.fit(train[FEATURES], train["feasible"].astype(int))
    joblib.dump(model, path)
    print(f"Trained and saved: {path}")
    return model, path, "trained"


def main():
    data, audit = load_data(DATA_DIR, DELTA)
    train = data.loc[data["dataset"] != "cifar100"].copy()
    test = data.loc[data["dataset"] == "cifar100"].copy()

    if train.empty or test.empty:
        raise ValueError("Both training datasets and CIFAR-100 test files are required.")
    if train["feasible"].nunique() != 2:
        raise ValueError("Training requires both binary feasibility classes.")

    empty_features = [c for c in NUMERIC if train[c].notna().sum() == 0]
    if empty_features:
        raise ValueError(f"Training numeric features entirely missing: {empty_features}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    predictions = []

    print(f"Training datasets: {sorted(train['dataset'].unique().tolist())}")
    print(f"Testing dataset: CIFAR-100")
    print(f"Train rows: {len(train)} | Test rows: {len(test)}")
    print(f"Train files: {train['source_file'].nunique()} | Test files: {test['source_file'].nunique()}")
    print(f"Tolerance: {DELTA} percentage points")

    for name, stem in MODEL_NAMES.items():
        loaded, path, status = load_or_train_model(name, stem, train)
        y_true = test["feasible"].astype(int).to_numpy()
        y_pred = loaded.predict(test[FEATURES]).astype(int)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1], zero_division=0
        )
        result = {
            "classifier": name,
            "model_status": status,
            "train_rows": len(train),
            "test_rows": len(test),
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "precision_feasible": float(precision[1]),
            "recall_feasible": float(recall[1]),
            "f1_feasible": float(f1[1]),
            "test_feasible_rate": float(y_true.mean()),
            "predicted_feasible_rate": float(y_pred.mean()),
            "model_path": str(path),
        }
        results.append(result)
        frame = test[["source_file", "teacher", "dataset", "student", "mode", "bottleneck_d", "regret_pp", "feasible"]].copy()
        frame["classifier"] = name
        frame["prediction"] = y_pred
        if hasattr(loaded, "predict_proba"):
            probabilities = loaded.predict_proba(test[FEATURES])
            class_index = list(loaded.classes_).index(1)
            frame["probability_feasible"] = probabilities[:, class_index]
        predictions.append(frame)
        print(f"{name}: accuracy={result['accuracy']:.3f} | balanced accuracy={result['balanced_accuracy']:.3f} | F1={result['f1_feasible']:.3f}")
        print(confusion_matrix(y_true, y_pred, labels=[0, 1]))



if __name__ == "__main__":
    main()
