"""
Filename: encode_latents.py
Author: Jayendra Matarage
Created on: 10/12/2026 8:06 PM
Description: Encodes Dino frames into VAE latent vectors for MDN-RNN training.
"""
import os
import numpy as np
import torch
from train_vae import ConvVAE

# Keep inference on the GPU when CUDA is available while retaining a CPU
# fallback for machines without a compatible accelerator.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode_dataset() -> None:
  """Convert collected frames into deterministic VAE latent representations.

  The output preserves the original sample order and action alignment. A later
  dynamics pipeline can therefore form transitions such as
  ``(latent_t, action_t) -> latent_(t+1)`` for MDN-RNN training.

  Unlike VAE training, this preprocessing step uses the posterior mean ``mu``
  instead of randomly sampled ``z`` values. This produces a stable latent
  dataset that remains identical each time the same checkpoint is encoded.
  """
  # Ensure the destination exists when the encoder is run independently of the
  # earlier data-collection step.
  os.makedirs("dataset", exist_ok=True)

  # Step 1: Rebuild the same VAE architecture used during vision training. The
  # latent dimension must match the checkpoint's 16-feature projection heads.
  vae = ConvVAE(latent_dim=16).to(device)
  vae_path = "checkpoints/vae_dino.pth"

  # Stop with an actionable message instead of encoding with random weights.
  if not os.path.exists(vae_path):
    raise FileNotFoundError(f"Missing VAE weights at {vae_path}")

  # ``map_location`` makes a GPU-trained checkpoint loadable on a CPU machine.
  vae.load_state_dict(torch.load(vae_path, map_location=device))
  # Evaluation mode disables any training-specific layer behavior during
  # feature extraction.
  vae.eval()
  print(f"Loaded trained VAE from {vae_path}")

  # Step 2: Load observations and their aligned action IDs. The action sequence
  # is copied unchanged into the latent archive for later transition modeling.
  data = np.load("dataset/dino_data.npz")
  frames = data["frames"]  # (N, 64, 64)
  actions = data["actions"]  # (N,)

  # Convert uint8 pixels from [0, 255] to the [0, 1] range used to train the VAE,
  # and add the single image-channel axis: (N, 64, 64) -> (N, 1, 64, 64).
  frames_tensor = torch.tensor(frames, dtype=torch.float32).unsqueeze(1) / 255.0

  # Accumulate CPU NumPy batches to avoid keeping every encoded result in GPU
  # memory. A moderate batch size balances throughput and memory consumption.
  latents = []
  batch_size = 128

  print("Encoding frames into latent representations (z)...")
  # Gradient tracking is unnecessary because the frozen encoder is used only
  # for inference, reducing both memory usage and computation.
  with torch.no_grad():
    for i in range(0, len(frames_tensor), batch_size):
      batch = frames_tensor[i : i + batch_size].to(device)
      # Run only the convolutional encoder and posterior-mean head. The decoder,
      # log-variance head, and stochastic reparameterization path are not needed.
      h = vae.encoder(batch)
      mu = vae.fc_mu(h)
      latents.append(mu.cpu().numpy())

  # Restore a single sample-major matrix shaped (number_of_frames, latent_dim).
  latents_arr = np.concatenate(latents, axis=0)  # Shape: (N, 16)

  # Step 3: Save latent observations and matching actions together. Compression
  # reduces storage while named arrays keep downstream loading unambiguous.
  save_path = "dataset/dino_latents.npz"
  np.savez_compressed(save_path, latents=latents_arr, actions=actions)

  print(f"Successfully encoded {len(latents_arr)} frames into {latents_arr.shape}")
  print(f"Saved latent dataset to: {save_path}")


if __name__ == "__main__":
  # Perform encoding only for direct script execution, not when imported by a
  # future MDN-RNN training module or test suite.
  encode_dataset()
