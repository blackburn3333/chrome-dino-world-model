"""
Filename: dream_hallucination.py
Author: Jayendra Matarage
Created on: 10/13/2026 8:29 AM
Description: Generate a video from an MDN-RNN latent dream rollout.
"""
import os
import sys
import cv2
import numpy as np
import torch
from train_controller import Controller
from train_rnn import MDNRNN
from train_vae import ConvVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_next_z(pi, mu, logsigma):
  """Samples the next latent vector z_{t+1} from the MDN Gaussian Mixture parameters."""
  # Select Gaussian component based on pi mixture probabilities
  categorical = torch.distributions.Categorical(probs=pi)
  component_idx = categorical.sample()  # (batch, seq)

  # Gather corresponding mean and logsigma for sampled component
  # mu, logsigma shape: (batch, seq, num_gaussians, z_dim)
  batch_size, seq_len, _, z_dim = mu.shape

  idx_expanded = (
      component_idx.unsqueeze(-1)
      .unsqueeze(-1)
      .expand(batch_size, seq_len, 1, z_dim)
  )
  selected_mu = torch.gather(mu, 2, idx_expanded).squeeze(2)
  selected_logsigma = torch.gather(logsigma, 2, idx_expanded).squeeze(2)

  std = torch.exp(selected_logsigma) + 1e-6
  eps = torch.randn_like(std)

  return selected_mu + eps * std


def dream_gameplay(dream_steps: int = 200, output_video: str = "dream.mp4"):
  print(f"Using device: {device}")
  os.makedirs("dataset", exist_ok=True)

  # 1. Load VAE (V)
  vae = ConvVAE(latent_dim=16).to(device)
  vae.load_state_dict(
      torch.load("checkpoints/vae_dino.pth", map_location=device)
  )
  vae.eval()

  # 2. Load MDN-RNN (M)
  rnn = MDNRNN(z_dim=16, action_dim=4, hidden_dim=256, num_gaussians=5).to(
      device
  )
  rnn.load_state_dict(
      torch.load("checkpoints/rnn_dino.pth", map_location=device)
  )
  rnn.eval()

  # 3. Load Controller (C)
  controller = Controller(z_dim=16, hidden_dim=256, action_dim=4).to(device)
  controller.load_state_dict(
      torch.load("checkpoints/controller_dino.pth", map_location=device)
  )
  controller.eval()

  # 4. Load initial ground-truth starting latent z_0 from dataset
  latents_path = "dataset/dino_latents.npz"
  if not os.path.exists(latents_path):
    raise FileNotFoundError(
        f"Missing {latents_path}. Run 03_encode_latents.py first."
    )

  data = np.load(latents_path)
  z_0 = torch.tensor(data["latents"][0], dtype=torch.float32).unsqueeze(0).to(device)  # (1, 16)

  action_names = {0: "RUN", 1: "JUMP", 2: "CROUCH", 3: "FAST DROP"}

  # Video Writer Setup
  fourcc = cv2.VideoWriter_fourcc(*"mp4v")
  video_writer = cv2.VideoWriter(output_video, fourcc, 20.0, (384, 384))

  print(
      f"\nGenerating {dream_steps} steps of hallucinated 'dream' gameplay..."
  )

  current_z = z_0
  h = torch.zeros(1, 1, 256).to(device)
  c = torch.zeros(1, 1, 256).to(device)

  dreamed_frames = []

  with torch.no_grad():
    for step in range(dream_steps):
      # Decode z_t -> Visual Reconstruction Frame
      # decoder expects shape (batch, 16)
      recon_frame = vae.decoder_input(current_z)
      decoded_img = vae.decoder(recon_frame)  # (1, 1, 64, 64)

      img_np = (
          (decoded_img.squeeze().cpu().numpy() * 255)
          .astype(np.uint8)
      )

      # Render Telemetry
      preview = cv2.resize(img_np, (384, 384), interpolation=cv2.INTER_NEAREST)
      preview_bgr = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)

      # Select Action using Controller
      dist = controller(current_z, h.squeeze(0))
      action = torch.argmax(dist.probs, dim=-1).item()

      cv2.putText(
          preview_bgr,
          f"HALLUCINATED DREAM | Step: {step+1}/{dream_steps}",
          (10, 25),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.5,
          (0, 0, 255),
          2,
      )
      cv2.putText(
          preview_bgr,
          f"Action: [{action}] {action_names[action]}",
          (10, 50),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.5,
          (255, 165, 0),
          2,
      )

      video_writer.write(preview_bgr)
      dreamed_frames.append(preview_bgr)

      # Predict Next Latent z_{t+1} using Memory Model M
      action_onehot = torch.zeros(1, 1, 4).to(device)
      action_onehot[0, 0, action] = 1.0
      rnn_input = torch.cat([current_z.unsqueeze(1), action_onehot], dim=-1)

      pi, mu, logsigma, (h, c) = rnn(rnn_input, (h, c))

      # Sample next z from predicted Gaussian Mixture Distribution
      next_z = sample_next_z(pi, mu, logsigma)
      current_z = next_z.squeeze(1)

  video_writer.release()
  print(f"\nDream simulation complete! Saved video to: {output_video}")


if __name__ == "__main__":
  dream_gameplay(dream_steps=200, output_video="dino_dream.mp4")
