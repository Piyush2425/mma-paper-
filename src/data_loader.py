import pandas as pd

REQUIRED_COLUMNS = [
    "timestamp",
    "action_type",
    "hand_speed",
    "leg_speed",
    "movement_intensity",
]


def load_events_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[REQUIRED_COLUMNS].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df
