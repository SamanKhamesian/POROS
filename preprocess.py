import os

import numpy as np
import pandas as pd

from config import FEATURE_KEYS

FOOD_BOLUS_TYPES = {"Food", "Food / Correction", "Override"}


# ── Glycemic outcome helpers (NOT part of feature vector) ─────────────────────

def _tir_percent(values):
    return ((values >= 70) & (values <= 180)).mean() * 100


def _tbr_percent(values):
    return (values < 70).mean() * 100


def _tar_percent(values):
    return (values > 180).mean() * 100


def _cv_percent(values):
    return values.std() / values.mean() * 100 if values.mean() > 0 else np.nan


# ── Carb / meal features ──────────────────────────────────────────────────────

def _compute_total_carbs(day_df):
    """Total carbohydrate intake from pump-logged carb input (CarbSize)."""
    return day_df["CarbSize"].sum(skipna=True)


def _compute_avg_carbs_per_meal(day_df):
    """Average carbs per meal event."""
    meal_carbs = day_df.loc[day_df["CarbSize"] > 0, "CarbSize"]
    return float(meal_carbs.mean()) if not meal_carbs.empty else np.nan


def _compute_avg_time_between_meals(day_df):
    """
    Average time between consecutive meal events in minutes.
    Meals detected from pump-logged CarbSize > 0 events.
    """
    meal_times = day_df.loc[day_df["CarbSize"] > 0, "EventDateTime"].sort_values()

    if len(meal_times) < 2:
        return np.nan

    return float(meal_times.diff().dropna().dt.total_seconds().div(60).mean())


# ── Insulin features ──────────────────────────────────────────────────────────

def _compute_total_daily_insulin(day_df):
    """Total insulin delivered across all bolus types."""
    return day_df["InsulinDelivered"].sum(skipna=True)


def _compute_bolus_per_meal(day_df):
    """Average insulin delivered per meal event."""
    n_meals = (day_df["CarbSize"] > 0).sum()
    total_bolus = day_df["InsulinDelivered"].sum(skipna=True)
    return float(total_bolus / n_meals) if n_meals > 0 else np.nan


def _compute_total_correction_insulin(day_df):
    """
    Total insulin delivered in correction boluses only
    (pump rows where CarbSize == 0 and insulin > 0).
    Higher values indicate more glucose excursions requiring manual correction.
    Strongly correlated with TIR improvement in prior ASU-Mayo analysis (ρ = -0.685).
    """
    correction = day_df[(day_df["CarbSize"].fillna(0) == 0) & (day_df["InsulinDelivered"].notna()) & (day_df["InsulinDelivered"] > 0)]
    return float(correction["InsulinDelivered"].sum())


# ── Timing feature ────────────────────────────────────────────────────────────

def _compute_avg_meal_bolus_delta_minutes(day_df):
    """
    Average signed time delta between each meal and its nearest food bolus (minutes).

    Negative = pre-bolus (bolus given before eating — clinically desirable for T1D).
    Positive = post-bolus (bolus given after eating — leads to postprandial spikes).

    Meal times are taken from the food log (Carbs > 0, separate from pump events).
    Bolus times are taken from explicit food boluses (BolusType in FOOD_BOLUS_TYPES).
    Only boluses within a ±120-minute window around each meal are considered.
    """
    meal_mask = day_df["Carbs"].notna() & (day_df["Carbs"] > 0)
    bolus_mask = (day_df["InsulinDelivered"].notna() & (day_df["InsulinDelivered"] > 0) & day_df["BolusType"].isin(FOOD_BOLUS_TYPES))

    meal_times = day_df.loc[meal_mask, "EventDateTime"]
    bolus_times = day_df.loc[bolus_mask, "EventDateTime"]

    if meal_times.empty or bolus_times.empty:
        return np.nan

    window = pd.Timedelta(minutes=120)
    deltas = []

    for t_meal in meal_times:
        time_diffs = bolus_times - t_meal
        within_window = time_diffs[time_diffs.abs() <= window]

        if within_window.empty:
            continue

        nearest = within_window.iloc[within_window.abs().argmin()]
        deltas.append(nearest.total_seconds() / 60)

    return float(np.mean(deltas)) if deltas else np.nan


# ── Daily profile builder ─────────────────────────────────────────────────────

def _prepare_daily_profile(df):
    df = df.copy()
    df["EventDateTime"] = pd.to_datetime(df["EventDateTime"])
    df = df.sort_values("EventDateTime")
    df["Date"] = df["EventDateTime"].dt.date

    rows = []

    for date, day_df in df.groupby("Date"):
        glucose = day_df["BloodGlucoseLevel"].dropna()
        if len(glucose) == 0:
            continue

        rows.append({"date": date,

            # glycemic outcomes — excluded from CBTD feature vector
            "tir": _tir_percent(glucose),
            "tar": _tar_percent(glucose),
            "tbr": _tbr_percent(glucose),
            "cv": _cv_percent(glucose),

            # carb / meal features
            "total_carbs": _compute_total_carbs(day_df),
            "avg_carbs_per_meal": _compute_avg_carbs_per_meal(day_df),
            "avg_time_between_meals": _compute_avg_time_between_meals(day_df),

            # insulin features
            "total_daily_insulin": _compute_total_daily_insulin(day_df),
            "bolus_per_meal": _compute_bolus_per_meal(day_df),
            "total_correction_insulin": _compute_total_correction_insulin(day_df),

            # timing
            "avg_meal_bolus_delta_minutes": _compute_avg_meal_bolus_delta_minutes(day_df), })

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_dataset(folder_path):
    all_rows = []

    for subject in os.listdir(folder_path):
        subject_path = os.path.join(folder_path, subject)
        subject_name = subject.split(".")[0]

        df = pd.read_csv(subject_path)
        daily_df = _prepare_daily_profile(df)

        daily_df = daily_df[daily_df["total_carbs"] > 0].reset_index(drop=True)
        daily_df = daily_df.dropna(subset=FEATURE_KEYS).reset_index(drop=True)

        daily_df["subject"] = subject_name
        all_rows.append(daily_df)

    return pd.concat(all_rows, ignore_index=True)


if __name__ == "__main__":
    database = build_dataset("dataset/ExActHealth")
    print(database.to_string())