"""
preprocess_uom.py — T1D-UOM preprocessor for GraphCF
=====================================================

Produces a daily behavioral profile DataFrame compatible with GraphCF's
node construction pipeline. Six features are computed, matching the ExActHealth
feature set minus total_correction_insulin (not separable in T1D-UOM — the bolus
file carries no food/correction type label):

    Feature                      MCID    Notes
    ─────────────────────────────────────────────────────────────────────
    total_carbs                  δ=5     sum of carbs_g across all meal logs
    avg_carbs_per_meal           δ=5     mean carbs_g per logged meal event
    avg_time_between_meals       δ=30    mean inter-meal gap (minutes); NaN if <2 meals
    total_daily_insulin          δ=1     sum of all bolus_dose entries (food + correction)
    bolus_per_meal               δ=1     total_daily_insulin / n_meals
    avg_meal_bolus_delta_minutes δ=10    mean signed gap: bolus_time − meal_time (minutes)
                                         negative = pre-bolus (clinically preferred)
                                         positive = post-bolus (associated with spikes)

Outcome column: tir (Time in Range %, 70–180 mg/dL threshold)

Filtering mirrors ExActHealth pipeline:
    1. sleep_duration >= MIN_SLEEP_MINUTES (3 hours)
    2. total_carbs > 0  (day must have at least one logged carbohydrate)
    3. dropna() on all six feature columns

Usage
─────
    from preprocess_uom import build_dataset_uom, print_dataset_stats

    dataset = build_dataset_uom("path/to/T1D-UOM/Dataset")
    print_dataset_stats(dataset)

Expected directory layout (data_dir argument)
─────────────────────────────────────────────
    data_dir/
        Glucose Data/       UoMGlucose{PID}.csv
        Nutrition Data/     UoMNutrition{PID}.csv
        Insulin Data/
            Bolus Data/     UoMBolus{PID}.csv
        Sleep Data/         UoM{PID}sleeptime.csv   (preferred)
                            UoMsleep{PID}.csv        (fallback)

Changes needed in GraphCF node.py before using this output
───────────────────────────────────────────────────────────
    Remove "total_correction_insulin" from FEATURE_KEYS.
    Everything else (graph.py, distance.py, eval_*.py) is unchanged.
"""

import glob
import os

import numpy as np
import pandas as pd


# ── Constants ──────────────────────────────────────────────────────────────────

MMOL_TO_MGDL      = 18.0182   # glucose unit conversion (mmol/L → mg/dL)
MIN_SLEEP_MINUTES = 180        # minimum sleep per day to retain (3 hours)
MIN_CGM_READINGS  = 24         # minimum CGM readings per day (~2 hrs at 5-min intervals)
BOLUS_MEAL_WINDOW = 120        # ±minutes window for pairing a bolus to a meal event
TIR_THRESHOLD     = 70.0       # clinical threshold: well-controlled day (%)

GLUCOSE_SUBDIR    = "Glucose Data"
NUTRITION_SUBDIR  = "Nutrition Data"
BOLUS_SUBDIR      = os.path.join("Insulin Data", "Bolus Data")
SLEEP_SUBDIR      = "Sleep Data"

dataset_folder = "dataset/T1D-UOM"

FEATURE_COLS = [
    "total_carbs",
    "avg_carbs_per_meal",
    "avg_time_between_meals",
    "total_daily_insulin",
    "bolus_per_meal",
    "avg_meal_bolus_delta_minutes",
]

TIR_BINS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


# ── Utility helpers ────────────────────────────────────────────────────────────

def _clean_columns(df):
    """Strip BOM characters and surrounding whitespace from all column names."""
    df.columns = [c.strip().lstrip("\ufeff").strip() for c in df.columns]
    return df


def _parse_datetime(series):
    """Parse T1D-UOM timestamps — day-first convention (DD/MM/YYYY)."""
    return pd.to_datetime(series, dayfirst=True, format="mixed")


def _find_ts_column(df):
    """Return the first column whose name contains 'ts' (case-insensitive)."""
    for col in df.columns:
        if "ts" in col.lower():
            return col
    return None


def _get_patient_ids(data_dir):
    """
    Discover patient IDs by scanning glucose files (broadest coverage in dataset).
    Returns a sorted list of patient ID strings.
    """
    pattern = os.path.join(data_dir, GLUCOSE_SUBDIR, "UoMGlucose*.csv")
    files = sorted(glob.glob(pattern))
    return [
        os.path.basename(f).replace("UoMGlucose", "").replace(".csv", "")
        for f in files
    ]


# ── Glycemic outcome helpers ───────────────────────────────────────────────────

def _tir_percent(glucose_mgdl):
    return float(((glucose_mgdl >= 70) & (glucose_mgdl <= 180)).mean() * 100)

def _tar_percent(glucose_mgdl):
    return float((glucose_mgdl > 180).mean() * 100)

def _tbr_percent(glucose_mgdl):
    return float((glucose_mgdl < 70).mean() * 100)

def _cv_percent(glucose_mgdl):
    mean = glucose_mgdl.mean()
    return float(glucose_mgdl.std() / mean * 100) if mean > 0 else np.nan


# ── Per-modality daily aggregation ─────────────────────────────────────────────

def _compute_daily_glucose(pid, data_dir):
    """
    Load glucose data for one patient and compute daily glycemic outcomes.

    Glucose is converted from mmol/L to mg/dL before computing TIR so that
    the 70–180 mg/dL threshold applies without modification. Days with fewer
    than MIN_CGM_READINGS are excluded (~2 hours of CGM coverage minimum).

    Returns DataFrame: date | tir | tar | tbr | cv
    """
    fpath = os.path.join(data_dir, GLUCOSE_SUBDIR, f"UoMGlucose{pid}.csv")

    df = pd.read_csv(fpath)
    df = _clean_columns(df)
    df = df.dropna(subset=["value"])
    df["datetime"]     = _parse_datetime(df["bg_ts"])
    df["glucose_mgdl"] = df["value"] * MMOL_TO_MGDL
    df["date"]         = df["datetime"].dt.date

    rows = []
    for date, day_df in df.groupby("date"):
        g = day_df["glucose_mgdl"]
        if len(g) < MIN_CGM_READINGS:
            continue
        rows.append({
            "date": date,
            "tir":  _tir_percent(g),
            "tar":  _tar_percent(g),
            "tbr":  _tbr_percent(g),
            "cv":   _cv_percent(g),
        })

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _compute_daily_nutrition(pid, data_dir):
    """
    Load nutrition data for one patient and compute per-day meal features.

    Produces all meal-related features for GraphCF in a single pass over the
    nutrition file, avoiding repeated iteration in _prepare_daily_profile_uom().

    Returns DataFrame:
        date | total_carbs | avg_carbs_per_meal | avg_time_between_meals
            | n_meals | meal_timestamps

    avg_time_between_meals is NaN on days with fewer than 2 logged meals —
    those days will be dropped by the downstream dropna() filter.

    Returns empty DataFrame if nutrition file not found for this patient.
    """
    fpath = os.path.join(data_dir, NUTRITION_SUBDIR, f"UoMNutrition{pid}.csv")

    if not os.path.exists(fpath):
        return pd.DataFrame(columns=[
            "date", "total_carbs", "avg_carbs_per_meal",
            "avg_time_between_meals", "n_meals", "meal_timestamps",
        ])

    df = pd.read_csv(fpath)
    df = _clean_columns(df)
    ts_col = _find_ts_column(df)
    df["datetime"] = _parse_datetime(df[ts_col])
    df["date"]     = df["datetime"].dt.date

    rows = []
    for date, day_df in df.groupby("date"):
        carbs           = day_df["carbs_g"].dropna()
        n_meals         = len(carbs)
        meal_timestamps = day_df["datetime"].tolist()

        total_carbs        = float(carbs.sum())       if n_meals > 0 else np.nan
        avg_carbs_per_meal = float(carbs.mean())      if n_meals > 0 else np.nan

        if n_meals >= 2:
            sorted_ts = sorted(meal_timestamps)
            gaps = [
                (sorted_ts[i + 1] - sorted_ts[i]).total_seconds() / 60.0
                for i in range(len(sorted_ts) - 1)
            ]
            avg_time_between_meals = float(np.mean(gaps))
        else:
            avg_time_between_meals = np.nan  # will be dropped by dropna()

        rows.append({
            "date":                   date,
            "total_carbs":            total_carbs,
            "avg_carbs_per_meal":     avg_carbs_per_meal,
            "avg_time_between_meals": avg_time_between_meals,
            "n_meals":                n_meals,
            "meal_timestamps":        meal_timestamps,
        })

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _compute_daily_bolus(pid, data_dir):
    """
    Load bolus data for one patient and compute daily totals.

    T1D-UOM does not distinguish food bolus from correction bolus — the bolus
    file carries only bolus_dose with no type label. total_daily_insulin is
    therefore the sum of ALL bolus events in the day (food + correction combined).

    Returns DataFrame: date | total_daily_insulin | bolus_timestamps
    Returns empty DataFrame if bolus file not found for this patient.
    """
    fpath = os.path.join(data_dir, BOLUS_SUBDIR, f"UoMBolus{pid}.csv")

    if not os.path.exists(fpath):
        return pd.DataFrame(columns=["date", "total_daily_insulin", "bolus_timestamps"])

    df = pd.read_csv(fpath)
    df = _clean_columns(df)
    ts_col = _find_ts_column(df)
    df["datetime"] = _parse_datetime(df[ts_col])
    df["date"]     = df["datetime"].dt.date

    rows = []
    for date, day_df in df.groupby("date"):
        rows.append({
            "date":                date,
            "total_daily_insulin": float(day_df["bolus_dose"].sum(skipna=True)),
            "bolus_timestamps":    day_df["datetime"].tolist(),
        })

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _compute_daily_sleep(pid, data_dir):
    """
    Load sleep data for one patient and compute daily sleep duration (minutes).

    Prefers the per-night summary file (UoM{PID}sleeptime.csv) which already
    aggregates sleep session totals. Falls back to the minute-level file
    (UoMsleep{PID}.csv) where sleep_level == 1 at 5-minute granularity.

    Returns DataFrame: date | sleep_duration (minutes)
    Returns empty DataFrame if neither file exists for this patient.
    """
    # ── Preferred: per-night summary ─────────────────────────────────────────
    summary_path = os.path.join(data_dir, SLEEP_SUBDIR, f"UoM{pid}sleeptime.csv")

    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
        df = _clean_columns(df)

        # Compute total sleep from stage columns if duration_in_sec not present
        if "duration_in_sec" in df.columns:
            df["sleep_duration"] = df["duration_in_sec"] / 60.0
        else:
            stage_cols = [c for c in ["deep_sleep_s", "light_sleep_s", "rem_sleep_s"] if c in df.columns]
            if stage_cols:
                df["sleep_duration"] = df[stage_cols].sum(axis=1) / 60.0
            else:
                df["sleep_duration"] = np.nan

        df["date"] = pd.to_datetime(
            df["calendar_date"], dayfirst=True, format="mixed"
        ).dt.date

        daily = df.groupby("date", as_index=False)["sleep_duration"].sum()
        return daily.sort_values("date").reset_index(drop=True)

    # ── Fallback: minute-level file ───────────────────────────────────────────
    minute_path = os.path.join(data_dir, SLEEP_SUBDIR, f"UoMsleep{pid}.csv")

    if os.path.exists(minute_path):
        df = pd.read_csv(minute_path)
        df = _clean_columns(df)
        ts_col = _find_ts_column(df)
        df["datetime"] = _parse_datetime(df[ts_col])
        df["date"]     = df["datetime"].dt.date

        rows = []
        for date, day_df in df.groupby("date"):
            rows.append({
                "date":           date,
                "sleep_duration": float((day_df["sleep_level"] == 1).sum() * 5),
            })
        return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    return pd.DataFrame(columns=["date", "sleep_duration"])


# ── Derived feature helpers ────────────────────────────────────────────────────

def _compute_bolus_per_meal(total_daily_insulin, n_meals):
    """
    Average insulin delivered per logged meal event.
    Returns NaN if n_meals is zero or missing (day has no meal logs).
    """
    if pd.isna(total_daily_insulin) or pd.isna(n_meals) or n_meals <= 0:
        return np.nan
    return float(total_daily_insulin / n_meals)


def _compute_avg_meal_bolus_delta_minutes(meal_timestamps, bolus_timestamps):
    """
    Mean signed time delta (minutes) between each meal and its nearest bolus.

    Negative = pre-bolus  (bolus delivered before eating — clinically preferred
                           for T1D; prevents postprandial glucose spikes)
    Positive = post-bolus (bolus delivered after eating — associated with spikes)

    Only boluses within ±BOLUS_MEAL_WINDOW minutes of a meal are paired.
    Mirrors _compute_avg_meal_bolus_delta_minutes() in ExActHealth preprocess.py.

    Note: T1D-UOM cannot distinguish food bolus from correction bolus, so all
    bolus events in the day are candidates for meal pairing.
    """
    if not meal_timestamps or not bolus_timestamps:
        return np.nan

    bolus_series = pd.Series(bolus_timestamps)
    window       = pd.Timedelta(minutes=BOLUS_MEAL_WINDOW)
    deltas       = []

    for meal_time in meal_timestamps:
        time_diffs    = bolus_series - meal_time
        within_window = time_diffs[time_diffs.abs() <= window]
        if within_window.empty:
            continue
        nearest = within_window.iloc[within_window.abs().argmin()]
        deltas.append(nearest.total_seconds() / 60.0)

    return float(np.mean(deltas)) if deltas else np.nan


# ── Daily profile assembly ─────────────────────────────────────────────────────

def _prepare_daily_profile_uom(pid, data_dir):
    """
    Load and merge all modalities for one patient into a daily-level DataFrame.

    Glucose is the anchor — all other modalities are left-joined by date.
    Missing modality days produce NaN and are later dropped by the filter
    in build_dataset_uom().

    Output columns:
        date | tir | tar | tbr | cv | sleep_duration
        | total_carbs | avg_carbs_per_meal | avg_time_between_meals
        | total_daily_insulin | bolus_per_meal | avg_meal_bolus_delta_minutes
    """
    glucose_df   = _compute_daily_glucose(pid, data_dir)
    sleep_df     = _compute_daily_sleep(pid, data_dir)
    nutrition_df = _compute_daily_nutrition(pid, data_dir)
    bolus_df     = _compute_daily_bolus(pid, data_dir)

    # Merge on date — glucose as left anchor
    merged = glucose_df.copy()
    merged = merged.merge(sleep_df, on="date", how="left")

    nutrition_cols = [
        "date", "total_carbs", "avg_carbs_per_meal",
        "avg_time_between_meals", "n_meals", "meal_timestamps",
    ]
    merged = merged.merge(nutrition_df[nutrition_cols], on="date", how="left")

    bolus_cols = ["date", "total_daily_insulin", "bolus_timestamps"]
    merged = merged.merge(bolus_df[bolus_cols], on="date", how="left")

    # Compute row-level derived features
    bpm_list   = []
    delta_list = []

    for _, row in merged.iterrows():
        n_meals  = row.get("n_meals", np.nan)
        meal_ts  = row["meal_timestamps"]  if isinstance(row.get("meal_timestamps"),  list) else []
        bolus_ts = row["bolus_timestamps"] if isinstance(row.get("bolus_timestamps"), list) else []

        bpm_list.append(_compute_bolus_per_meal(row["total_daily_insulin"], n_meals))
        delta_list.append(_compute_avg_meal_bolus_delta_minutes(meal_ts, bolus_ts))

    merged["bolus_per_meal"]               = bpm_list
    merged["avg_meal_bolus_delta_minutes"] = delta_list

    # Drop intermediate columns not part of the GraphCF feature set
    merged = merged.drop(
        columns=["n_meals", "meal_timestamps", "bolus_timestamps"],
        errors="ignore",
    )

    return merged.sort_values("date").reset_index(drop=True)


# ── Dataset assembly ───────────────────────────────────────────────────────────

def build_dataset_uom(data_dir):
    """
    Process all T1D-UOM patients and return a pooled daily DataFrame ready
    for GraphCF's node construction.

    Filters applied in order (matching ExActHealth pipeline):
        1. sleep_duration >= MIN_SLEEP_MINUTES  (3 hours minimum)
        2. total_carbs > 0                       (at least one logged meal)
        3. dropna() on FEATURE_COLS              (all six features must be present)

    Filter attrition is printed per subject so you can see where days are lost.
    The main risk in T1D-UOM is nutrition sparsity (mobile-app logging) —
    days without ≥2 logged meals lose avg_time_between_meals and are dropped.

    Returns DataFrame with columns:
        subject | date | tir | tar | tbr | cv | sleep_duration
        | total_carbs | avg_carbs_per_meal | avg_time_between_meals
        | total_daily_insulin | bolus_per_meal | avg_meal_bolus_delta_minutes
    """
    patient_ids = _get_patient_ids(data_dir)
    all_rows    = []

    W = 12  # column width for attrition table

    print(f"  {'Subject':<{W}}  {'Glucose':>{W}}  {'Sleep':>{W}}  {'Carbs':>{W}}  {'Features':>{W}}  {'Kept':>{W}}")
    print(f"  {'─'*W}  {'─'*W}  {'─'*W}  {'─'*W}  {'─'*W}  {'─'*W}")

    for pid in patient_ids:
        daily_df = _prepare_daily_profile_uom(pid, data_dir)
        n_glucose = len(daily_df)

        # Filter 1: minimum sleep coverage
        daily_df = daily_df[
            daily_df["sleep_duration"].notna() &
            (daily_df["sleep_duration"] >= MIN_SLEEP_MINUTES)
        ].reset_index(drop=True)
        n_sleep = len(daily_df)

        # Filter 2: must have logged carbohydrate intake
        daily_df = daily_df[
            daily_df["total_carbs"].notna() &
            (daily_df["total_carbs"] > 0)
        ].reset_index(drop=True)
        n_carbs = len(daily_df)

        # Filter 3: all six feature columns present
        daily_df = daily_df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
        n_final = len(daily_df)

        print(f"  {pid:<{W}}  {n_glucose:>{W}}  {n_sleep:>{W}}  {n_carbs:>{W}}  {n_final:>{W}}  {n_final:>{W}}")

        if n_final == 0:
            continue

        daily_df["subject"] = pid
        all_rows.append(daily_df)

    print()

    if not all_rows:
        raise ValueError("No usable patient-days found after filtering. Check data_dir path.")

    dataset = pd.concat(all_rows, ignore_index=True)
    return dataset


# ── Stats ──────────────────────────────────────────────────────────────────────

def print_dataset_stats(dataset):
    """
    Print a comprehensive summary of the processed dataset.

    Covers: subject-level day counts and TIR breakdown, cohort-level TIR
    distribution, and per-feature descriptive statistics. Call this immediately
    after build_dataset_uom() to assess whether the dataset is viable for GraphCF.

    Decision benchmark: ExActHealth has 386 days across 19 subjects (71 RED,
    315 BLUE) with ~6% single-hop rate under d² weighting. A larger dataset
    here suggests better graph density and more multi-hop path opportunities.
    """
    n_subjects = dataset["subject"].nunique()
    n_days     = len(dataset)
    n_red      = int((dataset["tir"] <  TIR_THRESHOLD).sum())
    n_blue     = int((dataset["tir"] >= TIR_THRESHOLD).sum())

    print()
    print("═" * 68)
    print("  T1D-UOM  —  Dataset Statistics (post-filter)")
    print("═" * 68)
    print(f"  Subjects   : {n_subjects}")
    print(f"  Total days : {n_days}")
    print(f"  RED  (TIR <  {TIR_THRESHOLD:.0f}%) : {n_red:>4}  ({n_red  / n_days * 100:.1f}%)")
    print(f"  BLUE (TIR >= {TIR_THRESHOLD:.0f}%) : {n_blue:>4}  ({n_blue / n_days * 100:.1f}%)")
    print()

    # ── Per-subject table ────────────────────────────────────────────────────
    print(f"  {'Subject':<14}  {'Days':>5}  {'Mean TIR':>9}  {'Median':>7}  {'RED':>5}  {'BLUE':>5}  {'RED%':>6}")
    print(f"  {'─'*14}  {'─'*5}  {'─'*9}  {'─'*7}  {'─'*5}  {'─'*5}  {'─'*6}")

    for pid, grp in dataset.groupby("subject"):
        n   = len(grp)
        red = int((grp["tir"] <  TIR_THRESHOLD).sum())
        blu = int((grp["tir"] >= TIR_THRESHOLD).sum())
        print(f"  {str(pid):<14}  {n:>5}  {grp['tir'].mean():>8.1f}%  "
              f"{grp['tir'].median():>6.1f}%  {red:>5}  {blu:>5}  {red/n*100:>5.1f}%")

    print()

    # ── TIR distribution ─────────────────────────────────────────────────────
    print("  TIR distribution")
    print(f"  {'─'*50}")

    counts, _ = np.histogram(dataset["tir"], bins=TIR_BINS)
    max_count  = max(counts) if max(counts) > 0 else 1
    bar_width  = 24

    for i in range(len(TIR_BINS) - 1):
        lo, hi  = TIR_BINS[i], TIR_BINS[i + 1]
        label   = f"{lo:>3}–{hi}%"
        count   = int(counts[i])
        pct     = count / n_days * 100
        bar     = "█" * int(round(count / max_count * bar_width))
        marker  = "  ← 70% threshold" if lo == 70 else ""
        print(f"  {label}  {count:>4} ({pct:>5.1f}%)  {bar}{marker}")

    print()

    # ── Feature descriptive statistics ───────────────────────────────────────
    print("  Feature statistics (all post-filter days)")
    print(f"  {'─'*68}")
    print(f"  {'Feature':<35}  {'Mean':>8}  {'Median':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
    print(f"  {'─'*35}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")

    for col in FEATURE_COLS:
        s = dataset[col]
        print(f"  {col:<35}  {s.mean():>8.2f}  {s.median():>8.2f}  "
              f"{s.std():>8.2f}  {s.min():>8.2f}  {s.max():>8.2f}")

    print()

    # ── TIR summary ──────────────────────────────────────────────────────────
    print("  TIR summary (outcome variable)")
    print(f"  {'─'*50}")
    t = dataset["tir"]
    print(f"  Mean   : {t.mean():.1f}%")
    print(f"  Median : {t.median():.1f}%")
    print(f"  Std    : {t.std():.1f}%")
    print(f"  Min    : {t.min():.1f}%")
    print(f"  Max    : {t.max():.1f}%")
    print()

    # ── ExActHealth comparison benchmark ────────────────────────────────────
    print("  Comparison with ExActHealth (GraphCF baseline)")
    print(f"  {'─'*50}")
    print(f"  {'':20}  {'T1D-UOM':>12}  {'ExActHealth':>12}")
    print(f"  {'─'*20}  {'─'*12}  {'─'*12}")
    print(f"  {'Subjects':<20}  {n_subjects:>12}  {'19':>12}")
    print(f"  {'Total days':<20}  {n_days:>12}  {'386':>12}")
    print(f"  {'RED days':<20}  {n_red:>12}  {'71':>12}")
    print(f"  {'BLUE days':<20}  {n_blue:>12}  {'315':>12}")
    print(f"  {'RED %':<20}  {n_red/n_days*100:>11.1f}%  {'18.4%':>12}")
    print()
    print("═" * 68)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dataset = build_dataset_uom(dataset_folder)
    print_dataset_stats(dataset)