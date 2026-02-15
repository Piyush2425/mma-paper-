import numpy as np
import pandas as pd


def compute_attack_frequency(timestamps: np.ndarray, index: int, window_seconds: float = 5.0) -> float:
    start = np.searchsorted(timestamps, timestamps[index] - window_seconds, side="left")
    return float(index - start)


def compute_window_stats(window_df: pd.DataFrame) -> dict:
    return {
        "avg_hand_speed": float(window_df["hand_speed"].mean()),
        "avg_leg_speed": float(window_df["leg_speed"].mean()),
        "avg_movement_intensity": float(window_df["movement_intensity"].mean()),
    }


def one_hot_actions(action_ids: np.ndarray, num_actions: int) -> np.ndarray:
    return np.eye(num_actions, dtype=float)[action_ids].reshape(-1)
