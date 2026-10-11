"""
Filename: data_inspector.py
Author: Jayendra Matarage
Created on: 10/11/2026 8:51 AM
Description: 
"""
import os
import cv2
import numpy as np


def inspect_dataset(dataset_path: str = "dataset/dino_data.npz") -> None:
  if not os.path.exists(dataset_path):
    print(f"Error: Dataset file '{dataset_path}' not found!")
    return

  # 1. Load Dataset
  data = np.load(dataset_path)
  frames = data["frames"]
  actions = data["actions"]

  total_frames = len(frames)
  print("=" * 50)
  print(f"DATASET SUMMARY: {dataset_path}")
  print("=" * 50)
  print(f"Total Captured Frames : {total_frames}")
  print(f"Frame Dimensions      : {frames.shape[1:]} (H x W)")
  print(f"Data Types            : Frames ({frames.dtype}), Actions ({actions.dtype})")

  # 2. Action Distribution
  action_names = {
      0: "RUN",
      1: "JUMP",
      2: "CROUCH/DUCK",
      3: "FAST DROP",
  }

  print("\nAction Distribution:")
  for action_id, name in action_names.items():
    count = np.sum(actions == action_id)
    percentage = (count / total_frames) * 100 if total_frames > 0 else 0
    print(f"  [{action_id}] {name:<12} : {count:6d} frames ({percentage:5.2f}%)")

  print("\n" + "=" * 50)
  print("INTERACTIVE PLAYER CONTROLS:")
  print("  [SPACE] : Pause / Play animation")
  print("  [D]     : Step forward 1 frame (when paused)")
  print("  [A]     : Step backward 1 frame (when paused)")
  print("  [Q]     : Quit viewer")
  print("=" * 50 + "\n")

  # 3. Interactive Playback
  idx = 0
  paused = False

  while True:
    frame = frames[idx]
    action = actions[idx]

    # Resize for readable preview
    preview = cv2.resize(frame, (384, 384), interpolation=cv2.INTER_NEAREST)
    preview_bgr = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)

    # Render metadata text overlay
    status_text = "PAUSED" if paused else "PLAYING"
    action_label = action_names.get(action, "UNKNOWN")

    cv2.putText(
        preview_bgr,
        f"Frame: {idx+1}/{total_frames} ({status_text})",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255) if paused else (0, 255, 0),
        2,
    )
    cv2.putText(
        preview_bgr,
        f"Action: [{action}] {action_label}",
        (15, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 165, 0),
        2,
    )

    cv2.imshow("Dino Dataset Visualizer", preview_bgr)

    # Key handling
    key = cv2.waitKey(0 if paused else 30) & 0xFF

    if key == ord("q"):
      break
    elif key == ord(" "):  # Toggle pause
      paused = not paused
    elif key == ord("d"):  # Step forward
      idx = (idx + 1) % total_frames
    elif key == ord("a"):  # Step backward
      idx = (idx - 1) % total_frames
    elif not paused:
      idx = (idx + 1) % total_frames

  cv2.destroyAllWindows()


if __name__ == "__main__":
  inspect_dataset()