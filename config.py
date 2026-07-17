# "ExActHealth" | "T1D-UOM"
DATASET = "T1D-UOM"

DATASET_FOLDER = {
    "ExActHealth": "dataset/ExActHealth",
    "T1D-UOM":     "dataset/T1D-UOM",
}[DATASET]

TIR_THRESHOLD = 70.0

FEATURE_KEYS = [
    # carb / meal features
    "total_carbs",
    # "meal_frequency",
    "avg_carbs_per_meal",
    "avg_time_between_meals",
    # "first_meal_hour",
    # "last_meal_hour",

    # insulin features
    "total_daily_insulin",
    "bolus_per_meal",
    *(["total_correction_insulin"] if DATASET == "ExActHealth" else []),

    # timing
    "avg_meal_bolus_delta_minutes",

    # # activity / sleep
    # "total_steps",
    # "exercise_duration",
    # "sleep_duration",
]

DELTA = {
    # carb / meal features
    "total_carbs":                  5.0,
    # "meal_frequency":             1.0,
    "avg_carbs_per_meal":           5.0,
    "avg_time_between_meals":      30.0,
    # "first_meal_hour":            1.0,
    # "last_meal_hour":             1.0,

    # insulin features
    "total_daily_insulin":          1.0,
    "bolus_per_meal":               1.0,
    **( {"total_correction_insulin": 1.0} if DATASET == "ExActHealth" else {} ),

    # timing
    "avg_meal_bolus_delta_minutes": 10.0,

    # # activity / sleep
    # "total_steps":              1000.0,
    # "exercise_duration":          10.0,
    # "sleep_duration":             30.0,
}