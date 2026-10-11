"""
Filename: data_collection.py
Author: Jayendra Matarage
Created on: 10/10/2026 8:44 AM
Description: 
"""

import os
import random
import sys
import cv2
import numpy as np

# Ensure game_room directory is on the path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from game_room.dino import ChromeDinoEnv


class DinoDataCollector:
  """Automates gameplay data collection for VAE training."""

  def __init__(self, target_steps: int = 10000, frame_interval: int = 20):
    self.target_steps = target_steps
    self.frame_interval = frame_interval
    self.action_names = {
        0: "RUN",
        1: "JUMP",
        2: "CROUCH/DUCK",
        3: "FAST DROP",
    }

  def select_action(
      self, frame_idx: int, pre_action: int | None = None
  ) -> int:
    """Selects an action based on interval sampling and current state."""
    if pre_action is None:
      return 0

    if frame_idx % self.frame_interval == 0:
      if pre_action == 1:
        # Airborne actions: RUN, JUMP, FAST DROP
        actions_set = [0, 1, 3]
        weights = [0.50, 0.20, 0.30]
      else:
        # Ground actions: RUN, JUMP, CROUCH
        actions_set = [0, 1, 2]
        weights = [0.40, 0.30, 0.30]

      return random.choices(actions_set, weights=weights)[0]

    return pre_action

  def collect(self) -> None:
    os.makedirs("dataset", exist_ok=True)
    frames, actions = [], []

    print(f"Starting data collection ({self.target_steps} frames target)...")
    print("Press 'q' in the preview window to stop early.")

    with ChromeDinoEnv(frame_size=(64, 64)) as env:
      state = env.reset()
      current_action = 0

      for step in range(self.target_steps):
        current_action = self.select_action(step, current_action)

        frames.append(state)
        actions.append(current_action)

        # 1. Live Dual-Preview Rendering
        preview = cv2.resize(state, (256, 256), interpolation=cv2.INTER_NEAREST)
        preview_bgr = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)

        action_label = self.action_names.get(current_action, "UNKNOWN")
        cv2.putText(
            preview_bgr,
            f"Step: {step+1}/{self.target_steps}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        cv2.putText(
            preview_bgr,
            f"Action: {action_label}",
            (10, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )

        cv2.imshow("Dino Data Collection", preview_bgr)

        if cv2.waitKey(1) & 0xFF == ord("q"):
          print("\nCollection halted early by user.")
          break

        # Execute step
        next_state, _, done, _ = env.step(current_action)
        state = next_state

        if done:
          state = env.reset()
          current_action = 0

    cv2.destroyAllWindows()

    # 2. Save Dataset
    frames_arr = np.array(frames, dtype=np.uint8)
    actions_arr = np.array(actions, dtype=np.int64)

    dataset_path = "dataset/dino_data.npz"
    np.savez_compressed(dataset_path, frames=frames_arr, actions=actions_arr)
    print(f"\nSuccessfully collected {len(frames_arr)} steps!")
    print(f"Dataset saved to: {dataset_path}")

    # 3. Generate Visual Inspection Grid (5x5 Sample Matrix)
    self._save_preview_grid(frames_arr)

  def _save_preview_grid(self, frames: np.ndarray) -> None:
    if len(frames) >= 25:
      sample_indices = np.random.choice(len(frames), 25, replace=False)
      sample_frames = [frames[idx] for idx in sample_indices]

      rows = [np.hstack(sample_frames[r * 5 : (r + 1) * 5]) for r in range(5)]
      grid = np.vstack(rows)

      preview_path = "dataset/preview_grid.png"
      cv2.imwrite(preview_path, grid)
      print(f"Sample preview grid saved to: {preview_path}")


if __name__ == "__main__":
  collector = DinoDataCollector(target_steps=10000, frame_interval=20)
  collector.collect()