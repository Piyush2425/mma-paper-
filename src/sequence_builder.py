from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .feature_engineering import compute_attack_frequency, compute_window_stats, one_hot_actions


def build_action_mapping(df: pd.DataFrame) -> Dict[str, int]:
    actions = sorted(df["action_type"].unique().tolist())
    return {action: idx for idx, action in enumerate(actions)}


def build_sequences(
    df: pd.DataFrame,
    window_size: int,
    action_to_id: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    if window_size < 1:
        raise ValueError("window_size must be >= 1")

    timestamps = df["timestamp"].values
    num_actions = len(action_to_id)

    features = []
    labels = []

    for i in range(window_size, len(df)):
        window_df = df.iloc[i - window_size : i]
        window_action_ids = window_df["action_type"].map(action_to_id).values

        action_features = one_hot_actions(window_action_ids, num_actions)
        time_gaps = np.diff(window_df["timestamp"].values).astype(float)
        window_stats = compute_window_stats(window_df)
        attack_freq = compute_attack_frequency(timestamps, i, window_seconds=5.0)

        feature_vector = np.concatenate(
            [
                action_features,
                time_gaps,
                np.array(
                    [
                        window_stats["avg_hand_speed"],
                        window_stats["avg_leg_speed"],
                        window_stats["avg_movement_intensity"],
                        attack_freq,
                    ],
                    dtype=float,
                ),
            ]
        )

        features.append(feature_vector)
        labels.append(action_to_id[df.loc[i, "action_type"]])

    return np.vstack(features), np.array(labels)


def build_latest_feature_vector(
    df: pd.DataFrame,
    window_size: int,
    action_to_id: Dict[str, int],
) -> np.ndarray:
    if len(df) < window_size:
        raise ValueError("Not enough events to build the feature window")

    window_df = df.iloc[-window_size:]
    timestamps = df["timestamp"].values
    num_actions = len(action_to_id)

    window_action_ids = window_df["action_type"].map(action_to_id).values
    action_features = one_hot_actions(window_action_ids, num_actions)
    time_gaps = np.diff(window_df["timestamp"].values).astype(float)
    window_stats = compute_window_stats(window_df)
    attack_freq = compute_attack_frequency(timestamps, len(df) - 1, window_seconds=5.0)

    feature_vector = np.concatenate(
        [
            action_features,
            time_gaps,
            np.array(
                [
                    window_stats["avg_hand_speed"],
                    window_stats["avg_leg_speed"],
                    window_stats["avg_movement_intensity"],
                    attack_freq,
                ],
                dtype=float,
            ),
        ]
    )

    return feature_vector.reshape(1, -1)
