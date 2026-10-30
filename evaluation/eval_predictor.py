"""
eval_predictor_transition.py — Held-out evaluation of the per-patient transition
TIR predictor.

A path step moves the patient from behavior i to behavior j. The predictor is
trained on the same kind of move, taken from each patient's own consecutive
logged days (t → t+1, in date order; missing calendar days are not skipped).

    Input  : v_t (K)  +  Δ = v_{t+1} − v_t (K)  +  TIR_t (1)
    Target : TIR_{t+1}

For each patient, samples are sorted by date and split chronologically into
train (first 90%) and test (last 10%). One model is trained per patient on that
patient's train samples only.

Baselines
  Patient Mean : predicts the patient's mean train-target TIR
  Persistence  : predicts TIR_{t+1} = TIR_t

Metrics (RMSE and MAE in pp, R²) are pooled over all test samples of the cohort
and reported as mean ± std across seeds. Per-patient RMSE and MAE are averaged
across seeds.

Run from the repository root:
    python -m evaluation.eval_predictor_transition
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from config import DATASET, DATASET_FOLDER, FEATURE_KEYS

if DATASET == "ExActHealth":
    from preprocess import build_dataset as _build_fn
else:
    from preprocess_uom import build_dataset_uom as _build_fn


TARGET = "tir"
SPLIT_RATIO = 0.9
SEEDS = [42, 43, 44, 45, 46]
RESULTS_FOLDER = "results_ex" if DATASET == "ExActHealth" else "results_uom"
PERSISTENCE = "Persistence"

MODEL_CONFIG = {
    "Patient Mean": {
        "class": DummyRegressor,
        "params": {"strategy": "mean"}
    },
    PERSISTENCE: {
        "class": None,
        "params": {}
    },
    "Linear Regression": {
        "class": LinearRegression,
        "params": {}
    },
    "Ridge Regression": {
        "class": Ridge,
        "params": {"alpha": 1.0}
    },
    "Random Forest Regression": {
        "class": RandomForestRegressor,
        "params": {
            "n_estimators": 500,
            "max_depth": 10
        }
    },
    "XGBoost Regression": {
        "class": XGBRegressor,
        "params": {
            "n_estimators": 300,
            "max_depth": 6,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "learning_rate": 0.01,
            "verbosity": 0
        }
    },
    "MLP Regression": {
        "class": MLPRegressor,
        "params": {
            "hidden_layer_sizes": (256, 64),
            "activation": "relu",
            "solver": "adam",
            "max_iter": 2000
        }
    },
}


def get_model(model_name, seed):
    config = MODEL_CONFIG[model_name]
    params = config["params"].copy()

    # inject seed only if supported
    if "random_state" in config["class"]().get_params():
        params["random_state"] = seed

    return config["class"](**params)


def load_dataset():
    # same preprocessing and filters as graph construction
    return _build_fn(DATASET_FOLDER)


def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {"rmse": rmse, "mae": mae, "r2": r2}


def build_transitions(subject_df):
    df = subject_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    feats = df[FEATURE_KEYS].values
    tirs = df[TARGET].values

    X, y, target_dates = [], [], []

    # walk through consecutive logged days, t → t+1
    for t in range(len(df) - 1):
        v_t = feats[t]
        delta = feats[t + 1] - feats[t]

        X.append(np.concatenate([v_t, delta, [tirs[t]]]))
        y.append(tirs[t + 1])
        target_dates.append(df["date"].iloc[t + 1])

    X = np.array(X).reshape(-1, 2 * len(FEATURE_KEYS) + 1)

    return X, np.array(y), np.array(target_dates)


def split_transitions(X, y, split_ratio=SPLIT_RATIO):
    # samples are already in date order
    split = int(split_ratio * len(y))

    return X[:split], y[:split], X[split:], y[split:]


def train_and_predict(model_name, X_train, y_train, X_test, seed):
    # persistence: tomorrow's TIR = today's TIR (last input column)
    if model_name == PERSISTENCE:
        return X_test[:, -1]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # fresh model per patient
    model = get_model(model_name, seed)
    model.fit(X_train, y_train)

    return model.predict(X_test)


def evaluate_model(model_name, dataset, seed):
    y_true_all, y_pred_all = [], []
    per_patient_rows = []

    for subject in dataset["subject"].unique():
        subject_df = dataset[dataset["subject"] == subject]

        X, y, _ = build_transitions(subject_df)
        X_train, y_train, X_test, y_test = split_transitions(X, y)

        if len(y_train) < 2 or len(y_test) == 0:
            continue

        y_pred = train_and_predict(model_name, X_train, y_train, X_test, seed)

        y_true_all.append(y_test)
        y_pred_all.append(y_pred)

        per_patient_rows.append({"subject": subject,
                                 "n_train": len(y_train),
                                 "n_test": len(y_test),
                                 "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                                 "mae": mean_absolute_error(y_test, y_pred)})

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)

    pooled = compute_metrics(y_true_all, y_pred_all)

    return pooled, pd.DataFrame(per_patient_rows)


def run_experiments(seeds=SEEDS):
    dataset = load_dataset()

    def mean_std(arr):
        return f"{np.mean(arr):.2f} ± {np.std(arr):.2f}"

    summary_rows = []
    per_patient_all = []

    for model_name in MODEL_CONFIG.keys():
        rmse_list, mae_list, r2_list = [], [], []
        per_patient_seeds = []

        for seed in seeds:
            pooled, per_patient = evaluate_model(model_name, dataset, seed)

            rmse_list.append(pooled["rmse"])
            mae_list.append(pooled["mae"])
            r2_list.append(pooled["r2"])
            per_patient_seeds.append(per_patient)

        summary_rows.append({"Model": model_name,
                             "RMSE (pp)": mean_std(rmse_list),
                             "MAE (pp)": mean_std(mae_list),
                             "R2": mean_std(r2_list)})

        # average per-patient metrics across seeds
        df_pp = pd.concat(per_patient_seeds)
        df_pp = df_pp.groupby(["subject", "n_train", "n_test"], as_index=False)[["rmse", "mae"]].mean()
        df_pp.insert(0, "model", model_name)
        per_patient_all.append(df_pp)

    df_summary = pd.DataFrame(summary_rows)
    df_per_patient = pd.concat(per_patient_all).reset_index(drop=True)

    n_train = int(per_patient_all[0]["n_train"].sum())
    n_test = int(per_patient_all[0]["n_test"].sum())

    print(f"\nDataset : {DATASET}")
    print(f"Subjects: {per_patient_all[0]['subject'].nunique()}")
    print(f"Train   : {n_train} transitions")
    print(f"Test    : {n_test} transitions")

    print("\nTransition Predictor Performance (held-out, pooled over test transitions)")
    print("--------------------------------------------")
    print(df_summary.to_string(index=False))

    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    df_summary.to_csv(os.path.join(RESULTS_FOLDER, "predictor_summary.csv"), index=False)
    df_per_patient.to_csv(os.path.join(RESULTS_FOLDER, "predictor_per_patient.csv"), index=False)

    return df_summary, df_per_patient


def main():
    run_experiments(SEEDS)


if __name__ == "__main__":
    main()
