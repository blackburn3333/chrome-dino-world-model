"""
Filename: manual_data_collection.py
Author: Jayendra Matarage
Created on: 11/10/2026 7:50 PM
Description: 
"""
import os
import sys
import time
import cv2
import numpy as np
from pynput import keyboard

# Ensure local modules can be imported
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from game_room.dino import ChromeDinoEnv

# Global action state tracking
# 0: RUN, 1: JUMP, 2: CROUCH, 3: FAST DROP
current_action = 0

def on_press(key):
    global current_action
    try:
        if key == keyboard.Key.space or key == keyboard.Key.up:
            current_action = 1
        elif key == keyboard.Key.down:
            current_action = 2
    except Exception:
        pass

def on_release(key):
    global current_action
    current_action = 0

def collect_manual_data(target_steps: int = 10000):
  global current_action  # <--- ADD THIS LINE HERE

  os.makedirs("dataset", exist_ok=True)
  action_names = {0: "RUN", 1: "JUMP", 2: "CROUCH/DUCK", 3: "FAST DROP"}

  # Start non-blocking global keyboard listener
  listener = keyboard.Listener(on_press=on_press, on_release=on_release)
  listener.start()

  frames, actions = [], []

  print("=" * 60)
  print("MANUAL DATA COLLECTION STARTED")
  print("1. Focus on the Chrome browser window.")
  print(
      "2. Play the game manually using SPACE/UP (Jump) and DOWN"
      " (Crouch)."
  )
  print(f"3. Target: {target_steps} valid gameplay frames.")
  print("4. Press 'q' in the OpenCV preview window to finish early.")
  print("=" * 60 + "\n")

  # Reusing ChromeDinoEnv context manager from game_room.dino
  with ChromeDinoEnv(frame_size=(64, 64)) as env:
    state = env.reset()

    while len(frames) < target_steps:
      # Capture action state right before execution
      act = current_action

      # Step environment with human action
      next_state, _, done, _ = env.step(act)

      if not done:
        frames.append(state)
        actions.append(act)
        state = next_state
      else:
        # Auto-restart on collision without appending crashed frame
        state = env.reset()
        current_action = 0
        time.sleep(0.3)
        continue

      # Render live preview telemetry
      preview = cv2.resize(
          state, (256, 256), interpolation=cv2.INTER_NEAREST
      )
      preview_bgr = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)

      action_label = action_names.get(act, "RUN")
      cv2.putText(
          preview_bgr,
          f"Collected: {len(frames)}/{target_steps}",
          (10, 25),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.5,
          (0, 255, 0),
          1,
      )
      cv2.putText(
          preview_bgr,
          f"Action: [{act}] {action_label}",
          (10, 45),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.5,
          (0, 255, 255),
          1,
      )

      cv2.imshow("Manual Dino Data Collection", preview_bgr)

      if cv2.waitKey(1) & 0xFF == ord("q"):
        print("\nCollection stopped early by user.")
        break

  cv2.destroyAllWindows()
  listener.stop()

  if len(frames) > 0:
    frames_arr = np.array(frames, dtype=np.uint8)
    actions_arr = np.array(actions, dtype=np.int64)

    dataset_path = "dataset/dino_data.npz"
    np.savez_compressed(dataset_path, frames=frames_arr, actions=actions_arr)
    print(f"\nSuccessfully collected {len(frames_arr)} human gameplay frames!")
    print(f"Saved dataset to: {dataset_path}")
  else:
    print("No frames were captured.")

if __name__ == "__main__":
    collect_manual_data(target_steps=10000)