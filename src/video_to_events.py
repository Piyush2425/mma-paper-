import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


ACTION_TYPES = {
    "punch_left",
    "punch_right",
    "kick_high",
    "kick_low",
    "guard_position",
    "backward_movement",
    "forward_movement",
}


@dataclass
class FighterState:
    last_left_wrist: Optional[Tuple[float, float]] = None
    last_right_wrist: Optional[Tuple[float, float]] = None
    last_left_ankle: Optional[Tuple[float, float]] = None
    last_right_ankle: Optional[Tuple[float, float]] = None
    last_hip_center: Optional[Tuple[float, float]] = None
    last_action: Optional[str] = None
    last_event_index: Optional[int] = None
    last_track_pos: Optional[Tuple[float, float]] = None
    last_forward_sign: float = 1.0


@dataclass
class PoseObservation:
    landmarks: Dict[str, Tuple[float, float]]
    hip_center_norm: Tuple[float, float]
    hip_center_px: Tuple[float, float]
    head_center_px: Tuple[float, float]
    color_label: str
    blue_score: float
    white_score: float


def create_landmarker(model_path: str) -> vision.PoseLandmarker:
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=2,
    )
    return vision.PoseLandmarker.create_from_options(options)


def to_pixel(point: Tuple[float, float], width: int, height: int) -> Tuple[float, float]:
    return (point[0] * width, point[1] * height)


def classify_helmet_color(
    hsv_frame: np.ndarray,
    head_center_px: Tuple[float, float],
    box_size: int = 16,
) -> Tuple[str, float, float]:
    x, y = int(head_center_px[0]), int(head_center_px[1])
    h, w = hsv_frame.shape[:2]
    x1 = max(x - box_size, 0)
    y1 = max(y - box_size, 0)
    x2 = min(x + box_size, w - 1)
    y2 = min(y + box_size, h - 1)
    if x2 <= x1 or y2 <= y1:
        return "unknown", 0.0, 0.0

    patch = hsv_frame[y1:y2, x1:x2]
    if patch.size == 0:
        return "unknown", 0.0, 0.0

    h_channel = patch[:, :, 0]
    s_channel = patch[:, :, 1]
    v_channel = patch[:, :, 2]

    blue_mask = (
        (h_channel >= 100)
        & (h_channel <= 140)
        & (s_channel >= 60)
        & (v_channel >= 60)
    )
    white_mask = (s_channel <= 40) & (v_channel >= 180)

    blue_score = float(np.mean(blue_mask))
    white_score = float(np.mean(white_mask))

    if blue_score > 0.08 and blue_score > white_score:
        return "blue", blue_score, white_score
    if white_score > 0.08 and white_score > blue_score:
        return "white", blue_score, white_score
    return "unknown", blue_score, white_score


def get_pose_observations(
    landmarker: vision.PoseLandmarker,
    frame: np.ndarray,
) -> List[PoseObservation]:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image(image_format=ImageFormat.SRGB, data=rgb)
    results = landmarker.detect(image)
    if not results.pose_landmarks:
        return []

    height, width = frame.shape[:2]
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    observations: List[PoseObservation] = []

    for landmarks in results.pose_landmarks:
        nose = (landmarks[0].x, landmarks[0].y)
        left_eye = (landmarks[2].x, landmarks[2].y)
        right_eye = (landmarks[5].x, landmarks[5].y)
        left_ear = (landmarks[7].x, landmarks[7].y)
        right_ear = (landmarks[8].x, landmarks[8].y)

        head_points = [nose, left_eye, right_eye, left_ear, right_ear]
        head_center_norm = (
            float(np.mean([p[0] for p in head_points])),
            float(np.mean([p[1] for p in head_points])),
        )

        landmarks_dict = {
            "nose": nose,
            "left_wrist": (landmarks[15].x, landmarks[15].y),
            "right_wrist": (landmarks[16].x, landmarks[16].y),
            "left_ankle": (landmarks[27].x, landmarks[27].y),
            "right_ankle": (landmarks[28].x, landmarks[28].y),
            "left_hip": (landmarks[23].x, landmarks[23].y),
            "right_hip": (landmarks[24].x, landmarks[24].y),
        }

        hip_center_norm = average_point(landmarks_dict["left_hip"], landmarks_dict["right_hip"])
        hip_center_px = to_pixel(hip_center_norm, width, height)
        head_center_px = to_pixel(head_center_norm, width, height)
        color_label, blue_score, white_score = classify_helmet_color(hsv_frame, head_center_px)

        observations.append(
            PoseObservation(
                landmarks=landmarks_dict,
                hip_center_norm=hip_center_norm,
                hip_center_px=hip_center_px,
                head_center_px=head_center_px,
                color_label=color_label,
                blue_score=blue_score,
                white_score=white_score,
            )
        )

    return observations


def compute_speed(
    last_point: Optional[Tuple[float, float]],
    current_point: Tuple[float, float],
    dt: float,
) -> float:
    if last_point is None:
        return 0.0
    dx = current_point[0] - last_point[0]
    dy = current_point[1] - last_point[1]
    return float(np.sqrt(dx * dx + dy * dy) / max(dt, 1e-6))


def average_point(p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float]:
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def detect_action(
    landmarks: Dict[str, Tuple[float, float]],
    state: FighterState,
    dt: float,
    forward_sign: float,
) -> Tuple[str, float, float, float, FighterState]:
    left_wrist = landmarks["left_wrist"]
    right_wrist = landmarks["right_wrist"]
    left_ankle = landmarks["left_ankle"]
    right_ankle = landmarks["right_ankle"]
    nose = landmarks["nose"]
    hip_center = average_point(landmarks["left_hip"], landmarks["right_hip"])

    left_hand_speed = compute_speed(state.last_left_wrist, left_wrist, dt)
    right_hand_speed = compute_speed(state.last_right_wrist, right_wrist, dt)
    left_leg_speed = compute_speed(state.last_left_ankle, left_ankle, dt)
    right_leg_speed = compute_speed(state.last_right_ankle, right_ankle, dt)

    hand_speed = max(left_hand_speed, right_hand_speed)
    leg_speed = max(left_leg_speed, right_leg_speed)
    movement_intensity = float((hand_speed + leg_speed) / 2.0)

    forward_movement = 0.0
    if state.last_hip_center is not None:
        forward_movement = (hip_center[0] - state.last_hip_center[0]) / max(dt, 1e-6)

    is_guard = (
        abs(left_wrist[1] - nose[1]) < 0.08
        and abs(right_wrist[1] - nose[1]) < 0.08
        and movement_intensity < 0.8
    )

    action = "guard_position"
    punch_threshold = 1.0
    kick_threshold = 1.0
    move_threshold = 0.2

    if hand_speed > punch_threshold and hand_speed > leg_speed * 1.2:
        action = "punch_left" if left_hand_speed >= right_hand_speed else "punch_right"
    elif leg_speed > kick_threshold and leg_speed > hand_speed * 1.2:
        highest_ankle = left_ankle if left_leg_speed >= right_leg_speed else right_ankle
        hip_y = hip_center[1]
        action = "kick_high" if highest_ankle[1] < hip_y else "kick_low"
    elif is_guard:
        action = "guard_position"
    elif forward_movement * forward_sign > move_threshold:
        action = "forward_movement"
    elif forward_movement * forward_sign < -move_threshold:
        action = "backward_movement"

    state.last_left_wrist = left_wrist
    state.last_right_wrist = right_wrist
    state.last_left_ankle = left_ankle
    state.last_right_ankle = right_ankle
    state.last_hip_center = hip_center

    return action, hand_speed, leg_speed, movement_intensity, state


def append_event(
    events: List[Dict[str, float]],
    timestamp: float,
    action: str,
    hand_speed: float,
    leg_speed: float,
    movement_intensity: float,
    state: FighterState,
):
    if state.last_action == action and state.last_event_index is not None:
        idx = state.last_event_index
        events[idx]["hand_speed"] = (events[idx]["hand_speed"] + hand_speed) / 2.0
        events[idx]["leg_speed"] = (events[idx]["leg_speed"] + leg_speed) / 2.0
        events[idx]["movement_intensity"] = (
            events[idx]["movement_intensity"] + movement_intensity
        ) / 2.0
        return

    events.append(
        {
            "timestamp": timestamp,
            "action_type": action,
            "hand_speed": hand_speed,
            "leg_speed": leg_speed,
            "movement_intensity": movement_intensity,
        }
    )
    state.last_action = action
    state.last_event_index = len(events) - 1


def draw_label(frame: np.ndarray, text: str, origin: Tuple[int, int], color: Tuple[int, int, int]):
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(
        frame,
        (x - 5, y - text_size[1] - 8),
        (x + text_size[0] + 5, y + 5),
        (0, 0, 0),
        -1,
    )
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def assign_fighters(
    observations: List[PoseObservation],
    f1_state: FighterState,
    f2_state: FighterState,
) -> Tuple[Optional[PoseObservation], Optional[PoseObservation]]:
    if not observations:
        return None, None

    blue_candidates = [obs for obs in observations if obs.color_label == "blue"]
    white_candidates = [obs for obs in observations if obs.color_label == "white"]

    if blue_candidates and white_candidates:
        return blue_candidates[0], white_candidates[0]

    if len(observations) >= 2 and blue_candidates and not white_candidates:
        blue_obs = blue_candidates[0]
        other_obs = observations[0] if observations[1] == blue_obs else observations[1]
        return blue_obs, other_obs

    if len(observations) >= 2 and white_candidates and not blue_candidates:
        white_obs = white_candidates[0]
        other_obs = observations[0] if observations[1] == white_obs else observations[1]
        return other_obs, white_obs

    if len(observations) == 1:
        obs = observations[0]
        if obs.color_label == "blue":
            return obs, None
        if obs.color_label == "white":
            return None, obs

        if f1_state.last_track_pos and f2_state.last_track_pos:
            d1 = np.linalg.norm(np.array(obs.hip_center_px) - np.array(f1_state.last_track_pos))
            d2 = np.linalg.norm(np.array(obs.hip_center_px) - np.array(f2_state.last_track_pos))
            return (obs, None) if d1 <= d2 else (None, obs)

        return obs, None

    if len(observations) >= 2:
        obs1, obs2 = observations[0], observations[1]
        if f1_state.last_track_pos and f2_state.last_track_pos:
            d11 = np.linalg.norm(np.array(obs1.hip_center_px) - np.array(f1_state.last_track_pos))
            d12 = np.linalg.norm(np.array(obs1.hip_center_px) - np.array(f2_state.last_track_pos))
            d21 = np.linalg.norm(np.array(obs2.hip_center_px) - np.array(f1_state.last_track_pos))
            d22 = np.linalg.norm(np.array(obs2.hip_center_px) - np.array(f2_state.last_track_pos))

            if d11 + d22 <= d12 + d21:
                return obs1, obs2
            return obs2, obs1

        obs_sorted = sorted(observations[:2], key=lambda o: o.hip_center_px[0])
        return obs_sorted[0], obs_sorted[1]

    return None, None


def extract_events(
    video_path: str,
    sample_fps: int,
    show: bool,
    model_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not Path(model_path).exists():
        raise ValueError(
            "Pose model not found. Download pose_landmarker.task and pass --model_path"
        )
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(int(round(native_fps / sample_fps)), 1)
    dt = stride / native_fps

    landmarker = create_landmarker(model_path)

    fighter_f1_events: List[Dict[str, float]] = []
    fighter_f2_events: List[Dict[str, float]] = []
    f1_state = FighterState()
    f2_state = FighterState()
    f1_action_text = "F1 no_pose"
    f2_action_text = "F2 no_pose"

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % stride != 0:
            frame_index += 1
            continue

        timestamp = frame_index / native_fps
        observations = get_pose_observations(landmarker, frame)
        f1_obs, f2_obs = assign_fighters(observations, f1_state, f2_state)

        if f1_obs and f2_obs:
            f1_forward_sign = 1.0 if f1_obs.hip_center_px[0] < f2_obs.hip_center_px[0] else -1.0
            f2_forward_sign = -f1_forward_sign
            f1_state.last_forward_sign = f1_forward_sign
            f2_state.last_forward_sign = f2_forward_sign

        if f1_obs:
            action, hand_speed, leg_speed, intensity, f1_state = detect_action(
                f1_obs.landmarks, f1_state, dt, forward_sign=f1_state.last_forward_sign
            )
            color_hint = f"blue {f1_obs.blue_score:.2f}" if f1_obs.color_label == "blue" else f"? {f1_obs.blue_score:.2f}"
            f1_action_text = (
                f"F1 {action} | hand {hand_speed:.2f} | leg {leg_speed:.2f} | intensity {intensity:.2f} | {color_hint}"
            )
            append_event(
                fighter_f1_events,
                timestamp,
                action,
                hand_speed,
                leg_speed,
                intensity,
                f1_state,
            )
            f1_state.last_track_pos = f1_obs.hip_center_px
        else:
            f1_action_text = "F1 no_pose"

        if f2_obs:
            action, hand_speed, leg_speed, intensity, f2_state = detect_action(
                f2_obs.landmarks, f2_state, dt, forward_sign=f2_state.last_forward_sign
            )
            color_hint = f"white {f2_obs.white_score:.2f}" if f2_obs.color_label == "white" else f"? {f2_obs.white_score:.2f}"
            f2_action_text = (
                f"F2 {action} | hand {hand_speed:.2f} | leg {leg_speed:.2f} | intensity {intensity:.2f} | {color_hint}"
            )
            append_event(
                fighter_f2_events,
                timestamp,
                action,
                hand_speed,
                leg_speed,
                intensity,
                f2_state,
            )
            f2_state.last_track_pos = f2_obs.hip_center_px
        else:
            f2_action_text = "F2 no_pose"

        if show:
            display_frame = frame.copy()
            draw_label(display_frame, f1_action_text, (15, 30), (0, 255, 0))
            draw_label(display_frame, f2_action_text, (15, 60), (255, 0, 0))
            draw_label(
                display_frame,
                f"t={timestamp:.2f}s | sample_fps={sample_fps}",
                (15, display_frame.shape[0] - 20),
                (255, 255, 255),
            )
            if f1_obs:
                cv2.circle(
                    display_frame,
                    (int(f1_obs.head_center_px[0]), int(f1_obs.head_center_px[1])),
                    8,
                    (0, 255, 0),
                    2,
                )
            if f2_obs:
                cv2.circle(
                    display_frame,
                    (int(f2_obs.head_center_px[0]), int(f2_obs.head_center_px[1])),
                    8,
                    (255, 0, 0),
                    2,
                )
            cv2.imshow("MMA Pose Data Collection", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_index += 1

    cap.release()
    landmarker.close()
    if show:
        cv2.destroyAllWindows()

    left_df = pd.DataFrame(fighter_f1_events)
    right_df = pd.DataFrame(fighter_f2_events)

    return left_df, right_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to the video file")
    parser.add_argument(
        "--sample_fps",
        type=int,
        default=10,
        help="Sampling rate for pose extraction",
    )
    parser.add_argument(
        "--out_dir",
        default="data",
        help="Output directory for fighter CSVs",
    )
    parser.add_argument(
        "--model_path",
        default="models/pose_landmarker.task",
        help="Path to pose_landmarker.task model",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display live data collection window",
    )
    args = parser.parse_args()

    left_df, right_df = extract_events(
        args.video,
        sample_fps=args.sample_fps,
        show=args.show,
        model_path=args.model_path,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    left_path = out_dir / "fighter_F1_events.csv"
    right_path = out_dir / "fighter_F2_events.csv"

    left_df.to_csv(left_path, index=False)
    right_df.to_csv(right_path, index=False)

    print(f"Saved: {left_path}")
    print(f"Saved: {right_path}")


if __name__ == "__main__":
    main()
