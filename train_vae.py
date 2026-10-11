"""
Filename: train_vae.py
Author: Jayendra Matarage
Created on: 10/11/2026 9:18 AM
Description: Trains a convolutional VAE to encode and reconstruct Chrome Dino frames.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import cv2

# Seed PyTorch's random-number generator so parameter initialization, shuffled
# batches, and latent samples are more repeatable between training runs.
torch.manual_seed(42)
# Prefer a CUDA GPU when PyTorch can access one; otherwise train on the CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Step 1: VAE architecture -------------------------------------------------
class ConvVAE(nn.Module):
  """Convolutional variational autoencoder for 64 x 64 grayscale frames.

  The encoder compresses each observation into the parameters of a Gaussian
  latent distribution. A differentiable sample from that distribution is then
  expanded by the decoder into a reconstructed observation.
  """

  def __init__(self, latent_dim=16):
    """Build the encoder, latent projections, and mirrored decoder.

    Args:
      latent_dim: Number of features in the learned stochastic representation.
    """
    super(ConvVAE, self).__init__()
    self.latent_dim = latent_dim

    # Encoder: progressively reduce (1, 64, 64) to a flattened 4,096-feature
    # representation while increasing the number of learned feature channels.
    self.encoder = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # -> (32, 32, 32)
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> (64, 16, 16)
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # -> (128, 8, 8)
        nn.ReLU(),
        nn.Conv2d(
            128, 256, kernel_size=4, stride=2, padding=1
        ),  # -> (256, 4, 4)
        nn.ReLU(),
        nn.Flatten(),  # -> 4096
    )

    # Separate heads parameterize the approximate posterior q(z|x). Predicting
    # log-variance is numerically more stable than predicting variance directly.
    self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
    self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

    # Project a latent sample back to the decoder's (256, 4, 4) starting volume.
    self.decoder_input = nn.Linear(latent_dim, 256 * 4 * 4)

    # Decoder: mirror the encoder with transposed convolutions until the original
    # single-channel 64 x 64 resolution is restored.
    self.decoder = nn.Sequential(
        nn.Unflatten(1, (256, 4, 4)),
        nn.ConvTranspose2d(
            256, 128, kernel_size=4, stride=2, padding=1
        ),  # -> (128, 8, 8)
        nn.ReLU(),
        nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1
        ),  # -> (64, 16, 16)
        nn.ReLU(),
        nn.ConvTranspose2d(
            64, 32, kernel_size=4, stride=2, padding=1
        ),  # -> (32, 32, 32)
        nn.ReLU(),
        nn.ConvTranspose2d(
            32, 1, kernel_size=4, stride=2, padding=1
        ),  # -> (1, 64, 64)
        # Match the [0, 1] range used to normalize the training frames.
        nn.Sigmoid(),
    )

  def reparameterize(self, mu, logvar):
    """Sample a latent vector while preserving differentiability.

    The reparameterization trick expresses a stochastic sample as
    ``mu + epsilon * std``. Gradients can therefore flow through ``mu`` and
    ``logvar`` even though fresh Gaussian noise is drawn on each forward pass.
    """
    # Convert log-variance to standard deviation: std = exp(logvar / 2).
    std = torch.exp(0.5 * logvar)
    # Draw one standard-normal noise tensor with the same shape and device.
    eps = torch.randn_like(std)
    return mu + eps * std

  def forward(self, x):
    """Encode, sample, and reconstruct a batch of observations.

    Args:
      x: Float tensor shaped ``(batch, 1, 64, 64)`` with values in ``[0, 1]``.

    Returns:
      A tuple of reconstructed frames, latent means, and latent log-variances.
    """
    # Convert image features into the two posterior-distribution parameters.
    h = self.encoder(x)
    mu, logvar = self.fc_mu(h), self.fc_logvar(h)
    # Draw the stochastic latent representation used by the decoder.
    z = self.reparameterize(mu, logvar)

    # Expand the latent vector before the decoder reshapes it to feature maps.
    x_recon = self.decoder_input(z)
    recon_x = self.decoder(x_recon)

    return recon_x, mu, logvar


# --- Step 2: VAE objective (reconstruction + KL divergence) ------------------
def vae_loss_function(recon_x, x, mu, logvar, kl_tolerance=0.5):
    """Calculate the summed reconstruction and latent-regularization losses.

    Args:
      recon_x: Decoder output with the same shape as ``x``.
      x: Ground-truth normalized observation batch.
      mu: Mean of the approximate posterior for each latent dimension.
      logvar: Log-variance of the approximate posterior.
      kl_tolerance: Reserved for a future free-bits or KL-clamping strategy;
        the current objective does not apply this value.

    Returns:
      The total differentiable loss, reconstruction component, and KL component.
    """
    # Pixel-space mean squared error encourages accurate reconstructions. The
    # summed reduction makes every pixel in every batch item contribute.
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')

    # Closed-form KL divergence regularizes q(z|x) toward a standard Gaussian,
    # giving the latent space a smooth structure useful to a later world model.
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_loss, recon_loss, kl_loss


# --- Step 3: Training and artifact generation --------------------------------
def train_vae():
    """Train the VAE, save its weights, and export reconstruction samples."""
    print(f"Using device: {device}")
    # Create the checkpoint destination without failing on repeated runs.
    os.makedirs("checkpoints", exist_ok=True)

    # Load the frame/action archive produced by ``data_collection.py``. VAE
    # training uses observations only, so the stored action array is not loaded.
    data_path = "dataset/dino_data.npz"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run 01_collect_data.py first.")

    raw_data = np.load(data_path)
    frames = raw_data['frames']  # (N, 64, 64)

    # Convert uint8 pixels to [0, 1] floats and insert PyTorch's channel axis,
    # transforming (N, 64, 64) into (N, 1, 64, 64).
    frames_tensor = torch.tensor(frames, dtype=torch.float32).unsqueeze(1) / 255.0

    # TensorDataset provides one-element batches; shuffling prevents the model
    # from repeatedly seeing observations in their original gameplay order.
    dataset = TensorDataset(frames_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Move parameters to the selected device and optimize them with Adam.
    model = ConvVAE(latent_dim=16).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 20
    print(f"Loaded {len(frames)} frames. Starting VAE training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        # Enable training-specific module behavior for the epoch.
        model.train()
        train_loss = 0.0

        for batch_idx, (x,) in enumerate(dataloader):
            # Place the batch beside the model, then clear gradients left by the
            # preceding optimization step.
            x = x.to(device)
            optimizer.zero_grad()

            # Run the stochastic VAE pass and calculate both loss components.
            recon_x, mu, logvar = model(x)
            loss, recon_l, kl_l = vae_loss_function(recon_x, x, mu, logvar)

            # Backpropagate the total objective and update all model parameters.
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # The batch losses use sum reduction, so dividing by dataset size reports
        # an average total loss per observation for easier epoch comparison.
        avg_loss = train_loss / len(dataloader.dataset)
        print(f"Epoch [{epoch}/{epochs}] | Loss: {avg_loss:.4f}")

    # Save parameters rather than the full Python object so model construction
    # remains explicit when the checkpoint is loaded later.
    model_path = "checkpoints/vae_dino.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\nVAE training complete! Weights saved to: {model_path}")

    # Switch off training behavior and gradient tracking for visual evaluation.
    model.eval()
    with torch.no_grad():
        # Use five deterministic source frames; latent sampling remains
        # stochastic, so reconstructions can vary slightly across executions.
        sample_x = frames_tensor[:5].to(device)
        recon_x, _, _ = model(sample_x)

        # Convert normalized tensors back to displayable uint8 grayscale arrays.
        orig_imgs = (sample_x.cpu().numpy() * 255).astype(np.uint8).squeeze()
        recon_imgs = (recon_x.cpu().numpy() * 255).astype(np.uint8).squeeze()

        # Place each reconstruction beside its source, then stack all five pairs
        # vertically into one diagnostic image.
        comparison_rows = []
        for orig, recon in zip(orig_imgs, recon_imgs):
            combined = np.hstack([orig, recon])
            comparison_rows.append(combined)

        comparison_grid = np.vstack(comparison_rows)
        cv2.imwrite("dataset/vae_reconstruction_test.png", comparison_grid)
        print("Saved visual reconstruction test to dataset/vae_reconstruction_test.png!")


if __name__ == "__main__":
    # Start training only when invoked as a script, not when imported by tests or
    # future world-model components.
    train_vae()
