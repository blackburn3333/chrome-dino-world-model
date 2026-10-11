"""
Filename: data_collection.py
Author: Jayendra Matarage
Created on: 10/10/2026 8:44 AM
Description: Collects Chrome Dino observations and actions for world-model training.
"""

import os
import random
import sys
import cv2
import numpy as np

# Add the repository root to Python's import path when this module is launched
# directly, allowing the local ``game_room`` package to be resolved.
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from game_room.dino import ChromeDinoEnv


class DinoDataCollector:
  """Collect image observations and sampled actions from Chrome Dino.

  The collector runs the Selenium environment with a simple weighted-random
  policy. Each stored frame is paired with the action selected for that state,
  producing the observation/action sequences needed by later representation
  and dynamics-model training stages.
  """

  def __init__(self, target_steps: int = 10000, frame_interval: int = 20):
    """Configure the number of samples and action-selection frequency.

    Args:
      target_steps: Maximum number of frame/action pairs to collect.
      frame_interval: Number of environment steps for which an action is held
        before the policy samples another action.
    """
    self.target_steps = target_steps
    self.frame_interval = frame_interval
    # Human-readable names are used only in the live collection overlay.
    self.action_names = {
        0: "RUN",
        1: "JUMP",
        2: "CROUCH/DUCK",
        3: "FAST DROP",
    }

  def select_action(
      self, frame_idx: int, pre_action: int | None = None
  ) -> int:
    """Return a weighted-random action or retain the previous action.

    A new action is sampled only when ``frame_idx`` reaches an interval
    boundary. This avoids changing keyboard state on every captured frame and
    creates short, temporally consistent action sequences.

    Args:
      frame_idx: Zero-based index of the frame being collected.
      pre_action: Action active during the previous collection step.

    Returns:
      The action ID to associate with and apply after the current observation.
    """
    if pre_action is None:
      # Start in neutral RUN mode before introducing random controls.
      return 0

    if frame_idx % self.frame_interval == 0:
      if pre_action == 1:
        # After a jump, choose among RUN, JUMP, and FAST DROP. CROUCH is
        # excluded because ArrowDown acts as FAST DROP while airborne.
        actions_set = [0, 1, 3]
        weights = [0.50, 0.20, 0.30]
      else:
        # For a non-jump state, choose among RUN, JUMP, and CROUCH.
        actions_set = [0, 1, 2]
        weights = [0.40, 0.30, 0.30]

      # ``random.choices`` returns a list, so extract its single sampled ID.
      return random.choices(actions_set, weights=weights)[0]

    # Hold the previous input until the next interval boundary.
    return pre_action

  def collect(self) -> None:
    """Run collection, save the compressed dataset, and create a sample grid."""
    # Create the output directory without failing when it already exists.
    os.makedirs("dataset", exist_ok=True)
    # Lists allow samples to be appended efficiently before final conversion to
    # fixed-dtype NumPy arrays.
    frames, actions = [], []

    print(f"Starting data collection ({self.target_steps} frames target)...")
    print("Press 'q' in the preview window to stop early.")

    # The context manager closes Chrome even if collection raises an exception.
    with ChromeDinoEnv(frame_size=(64, 64)) as env:
      state = env.reset()
      current_action = 0

      for step in range(self.target_steps):
        # Select the action a future agent would take from the current state.
        current_action = self.select_action(step, current_action)

        # Store aligned (state_t, action_t) samples before advancing the game.
        frames.append(state)
        actions.append(current_action)

        # Step 1: Build a readable live preview of the 64 x 64 model input.
        # Nearest-neighbor scaling preserves the discrete source pixels.
        preview = cv2.resize(state, (256, 256), interpolation=cv2.INTER_NEAREST)
        # Convert the grayscale image to BGR so colored status text can be drawn.
        preview_bgr = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)

        # Overlay collection progress and the action paired with this frame.
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

        # Poll without intentionally delaying collection and allow an early,
        # graceful stop while retaining all samples collected so far.
        if cv2.waitKey(1) & 0xFF == ord("q"):
          print("\nCollection halted early by user.")
          break

        # Step 2: Apply action_t and advance to state_(t+1). Rewards are ignored
        # at this stage because the captured data is currently intended for VAE
        # and visual-dynamics development.
        next_state, _, done, _ = env.step(current_action)
        state = next_state

        if done:
          # Start a new episode after collision; reset the remembered action so
          # the next sample begins from neutral RUN input.
          state = env.reset()
          current_action = 0

    # Release the native preview window after the browser session is closed.
    cv2.destroyAllWindows()

    # Step 3: Convert to compact, training-friendly dtypes. Frames are 8-bit
    # grayscale pixels and actions are integer class IDs.
    frames_arr = np.array(frames, dtype=np.uint8)
    actions_arr = np.array(actions, dtype=np.int64)

    # Store related arrays in one compressed archive. Named keys make the file
    # straightforward to load with ``np.load`` in training and inspection code.
    dataset_path = "dataset/dino_data.npz"
    np.savez_compressed(dataset_path, frames=frames_arr, actions=actions_arr)
    print(f"\nSuccessfully collected {len(frames_arr)} steps!")
    print(f"Dataset saved to: {dataset_path}")

    # Step 4: Create a quick visual quality check from the saved observations.
    self._save_preview_grid(frames_arr)

  def _save_preview_grid(self, frames: np.ndarray) -> None:
    """Save a five-by-five grid of randomly selected observation frames.

    The preview is created only when at least 25 unique samples are available,
    because sampling is performed without replacement.
    """
    if len(frames) >= 25:
      # Random indices make the grid more representative than 25 adjacent
      # frames, which would usually show nearly identical game states.
      sample_indices = np.random.choice(len(frames), 25, replace=False)
      sample_frames = [frames[idx] for idx in sample_indices]

      # Join each group of five horizontally, then stack the five rows.
      rows = [np.hstack(sample_frames[r * 5 : (r + 1) * 5]) for r in range(5)]
      grid = np.vstack(rows)

      # Keep the original pixel values; the grid can be enlarged by a viewer.
      preview_path = "dataset/preview_grid.png"
      cv2.imwrite(preview_path, grid)
      print(f"Sample preview grid saved to: {preview_path}")


if __name__ == "__main__":
  # Use the default collection run when this module is executed as a script.
  collector = DinoDataCollector(target_steps=10000, frame_interval=20)
  collector.collect()
