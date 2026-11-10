"""
Filename: manual_data_collection.py
Author: Jayendra Matarage
Created on: 11/10/2026 7:50 PM
Description: Records fixed-rate, episode-aware human Chrome Dino gameplay.
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
    # Ignore unrelated key releases. Resetting on every released key can label
    # a held jump or crouch as RUN even though the gameplay key is still down.
    if key in (keyboard.Key.space, keyboard.Key.up) and current_action == 1:
        current_action = 0
    elif key == keyboard.Key.down and current_action == 2:
        current_action = 0

def collect_manual_data(target_steps: int = 10000, sample_hz: float = 20.0):
  global current_action  # <--- ADD THIS LINE HERE

  os.makedirs("dataset", exist_ok=True)
  action_names = {0: "RUN", 1: "JUMP", 2: "CROUCH/DUCK", 3: "FAST DROP"}

  # Start non-blocking global keyboard listener
  listener = keyboard.Listener(on_press=on_press, on_release=on_release)
  listener.start()

  frames, actions = [], []
  rewards, dones, episode_ids, timestamps = [], [], [], []
  episode_id = 0
  collection_started = time.monotonic()

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
      sample_started = time.monotonic()
      # The physical key press already controls the focused Chrome window. The
      # listener is used only to label that transition; do not inject it again.
      act = current_action
      # Use a stable sampling interval so transition duration is not determined
      # solely by screenshot and UI-rendering speed.
      remaining = (1.0 / sample_hz) - (time.monotonic() - sample_started)
      if remaining > 0:
        time.sleep(remaining)
      next_state, reward, done, _ = env.observe()

      frames.append(state)
      actions.append(act)
      rewards.append(reward)
      dones.append(done)
      # Every sample receives the episode it belongs to. The terminal sample is
      # stored before incrementing the ID, preserving the final transition.
      episode_ids.append(episode_id)
      timestamps.append(time.monotonic() - collection_started)

      if done:
        episode_id += 1
        state = env.reset()
        current_action = 0
      else:
        state = next_state

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
    rewards_arr = np.array(rewards, dtype=np.float32)
    dones_arr = np.array(dones, dtype=bool)
    episode_ids_arr = np.array(episode_ids, dtype=np.int64)
    timestamps_arr = np.array(timestamps, dtype=np.float64)

    # Save all arrays together so their indices remain aligned throughout VAE,
    # recurrent-model, and behavior-cloning preprocessing.
    dataset_path = "dataset/dino_data.npz"
    np.savez_compressed(
        dataset_path,
        frames=frames_arr,
        actions=actions_arr,
        rewards=rewards_arr,
        dones=dones_arr,
        episode_ids=episode_ids_arr,
        timestamps=timestamps_arr,
    )
    print(f"\nSuccessfully collected {len(frames_arr)} human gameplay frames!")
    print(f"Saved dataset to: {dataset_path}")
  else:
    print("No frames were captured.")

if __name__ == "__main__":
    collect_manual_data(target_steps=10000)
