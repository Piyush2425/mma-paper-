# MMA AI - Next Move Prediction (Baseline)

This project trains a simple baseline model to predict the next action for each fighter based on their recent event history. It also provides a comparison script that summarizes fighter style traits.

## Folder Structure

- data/
  - fighter_F1_events.csv
  - fighter_F2_events.csv
- models/
- outputs/
- src/
  - data_loader.py
  - feature_engineering.py
  - sequence_builder.py
  - train_model.py
  - predict_next_move.py
  - compare_fighters.py
  - video_to_events.py

## Quick Start

1. Put the CSV files in the data/ folder.
2. Download the MediaPipe pose landmarker model file (pose_landmarker.task) and place it in models/.
3. Or, extract events from the video:

```bash
python -m src.video_to_events --video video/MMA.mp4 --out_dir data --show --model_path models/pose_landmarker.task
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Train models for both fighters:

```bash
python -m src.train_model --data data/fighter_F1_events.csv --fighter_id F1
python -m src.train_model --data data/fighter_F2_events.csv --fighter_id F2
```

6. Predict the next move from the most recent window:

```bash
python -m src.predict_next_move --model models/F1_next_move.pkl --data data/fighter_F1_events.csv
```

7. Compare fighter styles:

```bash
python -m src.compare_fighters --fighter1 data/fighter_F1_events.csv --fighter2 data/fighter_F2_events.csv
```

## Notes

- Baseline uses Random Forest by default.
- Models are trained per fighter and saved as .pkl files.
- For small datasets, accuracy will be limited and noisy.
- Video extraction uses helmet color (blue vs white) to assign fighters, with fallback to last known position when colors are occluded.
- Mapping: F1 = blue helmet, F2 = white helmet.
