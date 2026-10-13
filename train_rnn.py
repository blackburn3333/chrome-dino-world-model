"""
Filename: train_rnn.py
Author: Jayendra Matarage
Created on: 10/12/2026 10:31 PM
Description: Trains an MDN-RNN to predict next-step Chrome Dino latent states.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Train on a CUDA GPU when available while retaining a CPU fallback.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Seed PyTorch so model initialization and DataLoader shuffling are more
# repeatable between runs.
torch.manual_seed(42)


# --- Step 1: Temporal sequence dataset ---------------------------------------
class DinoSequenceDataset(Dataset):
    """Create aligned current-state/action and next-state latent sequences.

    Each sample is an overlapping window of ``seq_len`` transitions. Inputs at
    time ``t`` contain ``[z_t, one_hot(a_t)]`` and targets contain ``z_(t+1)``.
    This teaches the recurrent model to predict how an action changes the latent
    game state while retaining temporal context through its LSTM.

    Note:
        The current latent archive does not record episode boundaries. Until
        boundary metadata is collected, some windows may span a collision/reset
        and present that discontinuity to the model as an ordinary transition.
    """

    def __init__(self, latents, actions, seq_len=32, num_actions=4):
        """Prepare contiguous training windows from encoded gameplay.

        Args:
            latents: Array shaped ``(N, latent_dim)`` in collection order.
            actions: Integer action IDs shaped ``(N,)`` and aligned with latents.
            seq_len: Number of consecutive transitions in each sample.
            num_actions: Size of the discrete Dino action space.
        """
        self.seq_len = seq_len
        self.num_actions = num_actions

        # Convert integer action IDs into categorical vectors shaped (N, 4).
        actions_onehot = np.eye(num_actions)[actions]

        # Concatenate z_t with a_t to form a 20-feature vector per time step:
        # 16 latent features plus four one-hot action features.
        self.inputs = np.hstack([latents, actions_onehot]).astype(np.float32)
        # Keep a float32 latent copy; ``__getitem__`` shifts this array by one
        # position so every input step is paired with its next latent z_(t+1).
        self.targets = latents.astype(np.float32)

        # A sequence needs ``seq_len`` inputs plus one following latent target.
        # Sliding the window by one produces N - seq_len overlapping samples.
        self.num_samples = len(self.inputs) - self.seq_len

    def __len__(self):
        """Return the number of valid temporal windows in the dataset."""
        return max(0, self.num_samples)

    def __getitem__(self, idx):
        """Return one input window and its one-step-shifted latent targets."""
        x_seq = self.inputs[idx: idx + self.seq_len]
        y_seq = self.targets[idx + 1: idx + self.seq_len + 1]
        # DataLoader will stack these into tensors shaped
        # (batch_size, seq_len, feature_dim).
        return torch.tensor(x_seq), torch.tensor(y_seq)


# --- Step 2: Mixture-density recurrent architecture --------------------------
class MDNRNN(nn.Module):
    """Model next-latent uncertainty with an LSTM and Gaussian mixtures.

    For every step, the LSTM summarizes the latent/action history. The MDN head
    converts that hidden state into mixture weights and the parameters of
    diagonal Gaussian distributions over the next 16-dimensional latent state.
    """

    def __init__(self, z_dim=16, action_dim=4, hidden_dim=256, num_gaussians=5):
        """Construct the recurrent core and mixture-density output projection.

        Args:
            z_dim: Number of features in each VAE latent state.
            action_dim: Width of the one-hot action representation.
            hidden_dim: Number of features in the LSTM hidden state.
            num_gaussians: Number of mixture components predicted per time step.
        """
        super(MDNRNN, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.num_gaussians = num_gaussians

        # The LSTM consumes the concatenated [z_t, a_t] vector. ``batch_first``
        # establishes the (batch, sequence, feature) tensor convention.
        input_dim = z_dim + action_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        # For each Gaussian, predict one mixture logit plus ``z_dim`` means and
        # ``z_dim`` log-standard-deviations for a diagonal covariance model.
        out_dim = num_gaussians * (1 + 2 * z_dim)
        self.fc_mdn = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, hidden=None):
        """Predict mixture parameters for every step of an input sequence.

        Args:
            x: Tensor shaped ``(batch, sequence, z_dim + action_dim)``.
            hidden: Optional initial ``(hidden_state, cell_state)`` LSTM tuple.

        Returns:
            Mixture probabilities, component means, component log-standard-
            deviations, and the final recurrent hidden-state tuple.
        """
        # The LSTM emits a context vector at every step plus its final state,
        # which can later be reused for stateful prediction or imagination.
        lstm_out, hidden = self.lstm(x, hidden)

        # Project each temporal context vector into one flat MDN parameter block.
        mdn_out = self.fc_mdn(lstm_out)

        # Split the flat output into K logits, K*z_dim means, and K*z_dim
        # log-standard-deviations.
        pi_logits = mdn_out[..., : self.num_gaussians]
        mu = mdn_out[
            ...,
            self.num_gaussians:
            self.num_gaussians + self.num_gaussians * self.z_dim,
        ]
        logsigma = mdn_out[
            ...,
            self.num_gaussians + self.num_gaussians * self.z_dim:,
        ]

        # Restore explicit mixture and latent axes. Softmax converts component
        # logits into probabilities that sum to one at each sequence step.
        batch, seq, _ = x.shape
        mu = mu.view(batch, seq, self.num_gaussians, self.z_dim)
        logsigma = logsigma.view(batch, seq, self.num_gaussians, self.z_dim)
        pi = torch.softmax(pi_logits, dim=-1)

        return pi, mu, logsigma, hidden


# --- Step 3: Mixture negative log-likelihood objective -----------------------
def mdn_loss_function(pi, mu, logsigma, y):
    """Return mean negative log-likelihood of the observed next latents.

    Args:
        pi: Mixture probabilities shaped ``(batch, sequence, mixtures)``.
        mu: Component means shaped ``(batch, sequence, mixtures, z_dim)``.
        logsigma: Component log-standard-deviations with the same shape as ``mu``.
        y: Observed next latents shaped ``(batch, sequence, z_dim)``.
    """
    # Add and broadcast a mixture axis so each target is scored under every
    # Gaussian component: (batch, sequence, mixtures, z_dim).
    y_expanded = y.unsqueeze(2).expand_as(mu)

    # Exponentiation guarantees positive standard deviations; epsilon prevents
    # degenerate zero scales from reaching the Normal distribution.
    sigma = torch.exp(logsigma) + 1e-6
    gaussian = torch.distributions.Normal(mu, sigma)

    # Sum dimension-wise log probabilities to score each diagonal multivariate
    # Gaussian component: (batch, sequence, mixtures).
    log_prob = gaussian.log_prob(y_expanded).sum(dim=-1)  # (batch, seq, num_gaussians)

    # Weight each component in log space; epsilon protects against log(0).
    weighted_log_prob = log_prob + torch.log(pi + 1e-6)

    # LogSumExp marginalizes the mixture components without unstable exponent
    # and sum operations. Negating it produces the NLL minimization objective.
    loss = -torch.logsumexp(weighted_log_prob, dim=-1)
    return loss.mean()


# --- Step 4: MDN-RNN training and checkpoint export --------------------------
def train_rnn() -> None:
    """Train the MDN-RNN on encoded gameplay and save its state dictionary."""
    print(f"Using device: {device}")
    # Create the checkpoint directory without failing on repeated runs.
    os.makedirs("checkpoints", exist_ok=True)

    # The latent archive must be produced from the trained VAE before recurrent
    # dynamics can be learned.
    latents_path = "dataset/dino_latents.npz"
    if not os.path.exists(latents_path):
        raise FileNotFoundError(f"Missing {latents_path}. Run 03_encode_latents.py first.")

    data = np.load(latents_path)
    latents, actions = data["latents"], data["actions"]

    # Build overlapping 32-transition windows and shuffle their order each epoch.
    dataset = DinoSequenceDataset(latents, actions, seq_len=32)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Predict five possible next-latent distributions from a 256-feature LSTM.
    model = MDNRNN(z_dim=16, action_dim=4, hidden_dim=256, num_gaussians=5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 30
    print(f"Loaded {len(latents)} latents ({len(dataset)} sequences). Training MDN-RNN for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        # Enable training behavior and reset the epoch's loss accumulator.
        model.train()
        total_loss = 0.0

        for x_seq, y_seq in dataloader:
            # Move both temporal tensors to the same device as the model and
            # clear gradients accumulated by the preceding optimization step.
            x_seq, y_seq = x_seq.to(device), y_seq.to(device)
            optimizer.zero_grad()

            # Hidden state starts from zeros for each shuffled window because no
            # state tuple is supplied. Score the observed next latents under the
            # predicted Gaussian mixture at every step.
            pi, mu, logsigma, _ = model(x_seq)
            loss = mdn_loss_function(pi, mu, logsigma, y_seq)

            # Backpropagate the mean NLL and update recurrent and MDN parameters.
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Average equally across batches for a compact epoch-level progress metric.
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch}/{epochs}] | MDN-RNN NLL Loss: {avg_loss:.4f}")

    # Persist parameters only; loading code must reconstruct the same dimensions.
    model_path = "checkpoints/rnn_dino.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\nMDN-RNN training complete! Weights saved to: {model_path}")


if __name__ == "__main__":
    # Start training only during direct execution, not on module import.
    train_rnn()
