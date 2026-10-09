"""
Filename: dino_run.py
Author: Jayendra Matarage
Created on: 10/9/2026 7:34 AM
Description: Runs and previews a visible Chrome Dino environment with sampled actions.
"""
import os
import sys
import time
import cv2

# Add the repository root to Python's module search path when this file is run
# directly instead of being imported as part of a package.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dino import ChromeDinoEnv


def main() -> None:
  """Run a demonstration episode and display its processed observations."""
  print("Initializing visible Chrome Dino environment...")

  # The context manager guarantees browser cleanup on normal exit or error.
  with ChromeDinoEnv(frame_size=(64, 64)) as env:
    # Reset starts the game and provides the initial grayscale observation.
    state = env.reset()
    print("Game started successfully!")

    pre_action = None
    for step in range(11500):
      # Reconsider the action every 20 steps, retaining it between boundaries.
      action = env.moment(step,20,pre_action)
      # Apply the chosen input and collect the next transition.
      state, reward, done, _ = env.step(action)
      # Feed the selected action back into ``moment`` on the next iteration.
      pre_action = action
      print(f"reward: {reward}")
      print(f"done: {done}")
      # Nearest-neighbor scaling makes the low-resolution agent input visible
      # without smoothing its pixels.
      preview = cv2.resize(state, (256, 256), interpolation=cv2.INTER_NEAREST)
      cv2.imshow("Environment Input Preview (64x64)", preview)

      if cv2.waitKey(30) & 0xFF == ord("q"):
        # Allow the user to stop the long demonstration loop interactively.
        print("Exiting test...")
        break

      if done:
        # A collision ends the current episode but not the demonstration.
        print("Collision detected! Resetting game...")
        state = env.reset()

  # Close the OpenCV preview window after WebDriver cleanup has completed.
  cv2.destroyAllWindows()


if __name__ == "__main__":
  # Execute the demo only when launched as a script, not when imported.
  main()
