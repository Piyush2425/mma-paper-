import argparse
from collections import Counter
from typing import Dict, List

import numpy as np

from .data_loader import load_events_csv


STRIKE_ACTIONS = {"punch_left", "punch_right", "kick_high", "kick_low"}


def compute_style_profile(events_path: str) -> Dict[str, float]:
    df = load_events_csv(events_path)
    duration = max(df["timestamp"].iloc[-1] - df["timestamp"].iloc[0], 1.0)
    strikes = df[df["action_type"].isin(STRIKE_ACTIONS)]

    forward_moves = (df["action_type"] == "forward_movement").sum()
    backward_moves = (df["action_type"] == "backward_movement").sum()
    guard_moves = (df["action_type"] == "guard_position").sum()

    strike_frequency = float(len(strikes)) / (duration / 60.0)
    forward_ratio = float(forward_moves + 1) / float(backward_moves + 1)
    guard_ratio = float(guard_moves) / float(len(df))

    return {
        "strike_frequency_per_min": strike_frequency,
        "forward_to_backward_ratio": forward_ratio,
        "guard_ratio": guard_ratio,
    }


def top_combinations(actions: List[str], ngram: int = 2, top_k: int = 5):
    combos = Counter(
        tuple(actions[i : i + ngram])
        for i in range(len(actions) - ngram + 1)
    )
    return combos.most_common(top_k)


def compare_fighters(fighter1_path: str, fighter2_path: str):
    df1 = load_events_csv(fighter1_path)
    df2 = load_events_csv(fighter2_path)

    profile1 = compute_style_profile(fighter1_path)
    profile2 = compute_style_profile(fighter2_path)

    combos1 = top_combinations(df1["action_type"].tolist(), ngram=2, top_k=5)
    combos2 = top_combinations(df2["action_type"].tolist(), ngram=2, top_k=5)

    return profile1, profile2, combos1, combos2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fighter1", required=True, help="Path to fighter 1 CSV")
    parser.add_argument("--fighter2", required=True, help="Path to fighter 2 CSV")
    args = parser.parse_args()

    profile1, profile2, combos1, combos2 = compare_fighters(
        args.fighter1, args.fighter2
    )

    print("Fighter 1 profile:")
    for key, value in profile1.items():
        print(f"{key}: {value:.3f}")

    print("\nFighter 2 profile:")
    for key, value in profile2.items():
        print(f"{key}: {value:.3f}")

    print("\nFighter 1 top combinations:")
    for combo, count in combos1:
        print(f"{combo}: {count}")

    print("\nFighter 2 top combinations:")
    for combo, count in combos2:
        print(f"{combo}: {count}")


if __name__ == "__main__":
    main()
