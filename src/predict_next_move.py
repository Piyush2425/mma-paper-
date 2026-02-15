import argparse
import pickle

import numpy as np

from .data_loader import load_events_csv
from .sequence_builder import build_latest_feature_vector


def predict_next_move(model_path: str, data_path: str):
    with open(model_path, "rb") as f:
        artifact = pickle.load(f)

    model = artifact["model"]
    action_to_id = artifact["action_to_id"]
    window_size = artifact["window_size"]

    df = load_events_csv(data_path)
    X_latest = build_latest_feature_vector(df, window_size, action_to_id)

    probabilities = model.predict_proba(X_latest)[0]
    id_to_action = {idx: action for action, idx in action_to_id.items()}

    results = {
        id_to_action[idx]: float(prob)
        for idx, prob in enumerate(probabilities)
    }

    predicted_action = id_to_action[int(np.argmax(probabilities))]
    return predicted_action, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .pkl model")
    parser.add_argument("--data", required=True, help="Path to fighter events CSV")
    args = parser.parse_args()

    predicted_action, probabilities = predict_next_move(args.model, args.data)

    print("Next move probabilities:")
    for action, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
        print(f"{action}: {prob:.3f}")
    print(f"Predicted: {predicted_action}")


if __name__ == "__main__":
    main()
