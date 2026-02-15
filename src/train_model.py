import argparse
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from .data_loader import load_events_csv
from .sequence_builder import build_action_mapping, build_sequences


def build_model(model_name: str):
    name = model_name.lower()
    if name == "logistic_regression":
        return LogisticRegression(max_iter=200, multi_class="multinomial")
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=200, random_state=42)
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("xgboost is not installed") from exc
        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
        )
    raise ValueError("Unsupported model_name")


def train_model(data_path: str, fighter_id: str, window_size: int, model_name: str):
    df = load_events_csv(data_path)
    action_to_id = build_action_mapping(df)

    X, y = build_sequences(df, window_size=window_size, action_to_id=action_to_id)
    split_index = int(len(X) * 0.8)

    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    model = build_model(model_name)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    artifact = {
        "model": model,
        "action_to_id": action_to_id,
        "window_size": window_size,
    }

    output_path = Path("models") / f"{fighter_id}_next_move.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(artifact, f)

    return acc, report, output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to fighter events CSV")
    parser.add_argument("--fighter_id", required=True, help="Fighter identifier")
    parser.add_argument("--window", type=int, default=5, help="Sequence window size")
    parser.add_argument(
        "--model",
        default="random_forest",
        choices=["random_forest", "logistic_regression", "xgboost"],
        help="Model type",
    )
    args = parser.parse_args()

    acc, report, output_path = train_model(
        data_path=args.data,
        fighter_id=args.fighter_id,
        window_size=args.window,
        model_name=args.model,
    )

    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:\n" + report)
    print(f"Saved model: {output_path}")


if __name__ == "__main__":
    main()
