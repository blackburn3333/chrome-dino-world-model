"""
Filename: evaluate_controller.py
Author: Jayendra Matarage
Created on: 10/13/2026 8:28 AM
Description: Evaluate the trained controller in the real Chrome Dino game.
"""
import os
import sys
import time
import cv2
import numpy as np
import torch

# Ensure local imports work properly
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from game_room.dino import ChromeDinoEnv
from train_controller import Controller
from train_rnn import MDNRNN
from train_vae import ConvVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_controller(num_episodes: int = 3):
  print(f"Using device: {device}")

  # 1. Load Vision Model (V)
  vae_path = "checkpoints/vae_dino.pth"
  if not os.path.exists(vae_path):
    raise FileNotFoundError(f"Missing VAE weights at {vae_path}")
  vae = ConvVAE(latent_dim=16).to(device)
  vae.load_state_dict(torch.load(vae_path, map_location=device))
  vae.eval()

  # 2. Load Memory Model (M)
  rnn_path = "checkpoints/rnn_dino.pth"
  if not os.path.exists(rnn_path):
    raise FileNotFoundError(f"Missing MDN-RNN weights at {rnn_path}")
  rnn = MDNRNN(z_dim=16, action_dim=4, hidden_dim=256, num_gaussians=5).to(
      device
  )
  rnn.load_state_dict(torch.load(rnn_path, map_location=device))
  rnn.eval()

  # 3. Load Controller (C)
  controller_path = "checkpoints/controller_dino.pth"
  if not os.path.exists(controller_path):
    raise FileNotFoundError(f"Missing Controller weights at {controller_path}")
  controller = Controller(z_dim=16, hidden_dim=256, action_dim=4).to(device)
  controller.load_state_dict(torch.load(controller_path, map_location=device))
  controller.eval()

  action_names = {0: "RUN", 1: "JUMP", 2: "CROUCH", 3: "FAST DROP"}

  print("\n" + "=" * 50)
  print("STARTING WORLD MODEL EVALUATION IN REAL GAME")
  print("=" * 50 + "\n")

  with ChromeDinoEnv(frame_size=(64, 64)) as env:
    for episode in range(1, num_episodes + 1):
      raw_frame = env.reset()

      # Reset RNN hidden states (h_0, c_0)
      h = torch.zeros(1, 1, 256).to(device)
      c = torch.zeros(1, 1, 256).to(device)

      done = False
      step = 0
      total_reward = 0.0

      print(f"--- Running Episode {episode}/{num_episodes} ---")

      while not done:
        # Preprocess frame tensor -> (1, 1, 64, 64)
        frame_tensor = (
            torch.tensor(raw_frame, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
            / 255.0
        )

        with torch.no_grad():
          # VAE Encode: Raw Frame -> Latent Vector z_t (dim 16)
          enc_h = vae.encoder(frame_tensor)
          z = vae.fc_mu(enc_h)

          # Controller Choice: C([z_t, h_t]) -> Action Logits
          dist = controller(z, h.squeeze(0))
          action = torch.argmax(dist.probs, dim=-1).item()

        # Execute Action in Chrome Dino
        next_frame, reward, done, _ = env.step(action)
        total_reward += reward
        step += 1

        # Update Memory Model (M): h_t -> h_{t+1}
        action_onehot = torch.zeros(1, 1, 4).to(device)
        action_onehot[0, 0, action] = 1.0
        rnn_input = torch.cat([z.unsqueeze(1), action_onehot], dim=-1)

        with torch.no_grad():
          _, _, _, (h, c) = rnn(rnn_input, (h, c))

        # Render Telemetry HUD
        preview = cv2.resize(
            raw_frame, (384, 384), interpolation=cv2.INTER_NEAREST
        )
        preview_bgr = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)

        cv2.putText(
            preview_bgr,
            f"Episode: {episode}/{num_episodes} | Step: {step}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            preview_bgr,
            f"Action: [{action}] {action_names[action]}",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 165, 0),
            2,
        )
        cv2.putText(
            preview_bgr,
            f"Score: {total_reward:.1f}",
            (15, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        cv2.imshow("World Model Controller Live Evaluation", preview_bgr)

        if cv2.waitKey(1) & 0xFF == ord("q"):
          print("Evaluation interrupted by user.")
          cv2.destroyAllWindows()
          return

        raw_frame = next_frame

      print(
          f"Episode {episode} Ended | Survived: {step} steps | Reward:"
          f" {total_reward:.1f}\n"
      )
      time.sleep(1.0)

  cv2.destroyAllWindows()


if __name__ == "__main__":
  evaluate_controller(num_episodes=5)
