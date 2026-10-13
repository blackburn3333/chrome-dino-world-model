"""
Filename: train_controller.py
Author: Jayendra Matarage
Created on: 10/13/2026 7:35 AM
Description: Trains a Dino policy with REINFORCE using VAE and MDN-RNN state features.
"""
import os
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Add the repository root to Python's module search path when this file is run
# directly, allowing local environment and model modules to be imported.
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from game_room.dino import ChromeDinoEnv
from train_vae import ConvVAE
from train_rnn import MDNRNN

# Run all three neural-network components on a CUDA GPU when available, with a
# CPU fallback for systems without a compatible accelerator.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Step 1: Controller policy architecture ----------------------------------
class Controller(nn.Module):
    """Map compact world-model state features to a categorical action policy.

    The controller observes both the VAE latent ``z_t`` for the current frame
    and the MDN-RNN hidden state ``h_t`` summarizing recent latent/action
    history. A deliberately small linear policy converts their concatenation
    into probabilities for the four environment actions.
    """

    def __init__(self, z_dim=16, hidden_dim=256, action_dim=4):
        """Create the linear action-logit projection.

        Args:
            z_dim: Width of the VAE observation latent.
            hidden_dim: Width of the MDN-RNN hidden state.
            action_dim: Number of discrete actions available to the controller.
        """
        super(Controller, self).__init__()
        # Default input width is 16 latent features plus 256 recurrent features.
        self.fc = nn.Linear(z_dim + hidden_dim, action_dim)

    def forward(self, z, h):
        """Return an action distribution conditioned on visual and memory state.

        Args:
            z: Current encoded observation shaped ``(batch, z_dim)``.
            h: Current recurrent hidden state shaped ``(batch, hidden_dim)``.
        """
        # Concatenate current visual evidence with the recurrent history summary.
        state = torch.cat([z, h], dim=-1)
        # Categorical accepts raw logits and normalizes them internally, retaining
        # a differentiable log-probability for the policy-gradient update.
        logits = self.fc(state)
        return Categorical(logits=logits)


# --- Step 2: Online policy-gradient training ---------------------------------
def train_controller() -> None:
    """Train the controller in the real browser environment with REINFORCE.

    The pretrained VAE and MDN-RNN act as frozen feature extractors. Only the
    small controller receives gradients. Training uses complete real-environment
    episodes; this stage does not yet optimize a policy inside imagined latent
    rollouts.
    """
    print(f"Using device: {device}")

    # Load the visual model V. It converts each normalized 64 x 64 frame into the
    # same deterministic posterior-mean representation used for RNN training.
    vae = ConvVAE(latent_dim=16).to(device)
    vae.load_state_dict(torch.load("checkpoints/vae_dino.pth", map_location=device))
    vae.eval()

    # Load the recurrent world model M. The controller uses its hidden state as
    # memory but does not consume or sample its Gaussian-mixture predictions.
    rnn = MDNRNN(z_dim=16, action_dim=4, hidden_dim=256, num_gaussians=5).to(device)
    rnn.load_state_dict(torch.load("checkpoints/rnn_dino.pth", map_location=device))
    rnn.eval()

    # Initialize controller C from scratch. The optimizer receives controller
    # parameters only, leaving the loaded VAE and MDN-RNN weights unchanged.
    controller = Controller(z_dim=16, hidden_dim=256, action_dim=4).to(device)
    optimizer = optim.Adam(controller.parameters(), lr=1e-3)

    # Train over complete episodes and discount later rewards by gamma per step.
    episodes = 50
    gamma = 0.99

    print("Starting Controller training in Chrome Dino environment...")

    # The environment context manager guarantees browser cleanup after training.
    with ChromeDinoEnv(frame_size=(64, 64)) as env:
        for ep in range(1, episodes + 1):
            # Start a new game and obtain its first processed observation.
            raw_frame = env.reset()

            # Reset both LSTM memory tensors at the beginning of every episode.
            # Shape is (layers, batch, hidden_dim) for a one-layer, one-game run.
            h = torch.zeros(1, 1, 256).to(device)
            c = torch.zeros(1, 1, 256).to(device)

            # Retain differentiable action log-probabilities and scalar rewards
            # until the episode ends and its Monte Carlo returns are available.
            log_probs = []
            rewards = []
            done = False
            step = 0

            while not done:
                # Step 1: Convert one uint8 frame to normalized NCHW format:
                # (64, 64) -> (batch=1, channel=1, 64, 64).
                frame_tensor = (
                    torch.tensor(raw_frame, dtype=torch.float32)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(device)
                    / 255.0
                )
                # Freeze the VAE path and use its posterior mean for stable state
                # features rather than drawing a stochastic latent sample.
                with torch.no_grad():
                    enc_h = vae.encoder(frame_tensor)
                    z = vae.fc_mu(enc_h)  # (1, 16)

                # Step 2: Build C([z_t, h_t]), sample an exploratory action, and
                # retain its log-probability for the later REINFORCE objective.
                dist = controller(z, h.squeeze(0))
                action = dist.sample()
                log_prob = dist.log_prob(action)

                # Step 3: Apply the sampled discrete action to the real Chrome
                # game and observe its reward and terminal collision flag.
                action_idx = action.item()
                next_frame, reward, done, _ = env.step(action_idx)

                # Step 4: Combine z_t with one-hot a_t using the same input format
                # seen during MDN-RNN training, then advance recurrent memory.
                action_onehot = torch.zeros(1, 1, 4).to(device)
                action_onehot[0, 0, action_idx] = 1.0
                rnn_input = torch.cat([z.unsqueeze(1), action_onehot], dim=-1)

                # The world model is frozen: only its updated hidden/cell state is
                # needed as context for the controller's next decision.
                with torch.no_grad():
                    _, _, _, (h, c) = rnn(rnn_input, (h, c))

                # Store the trajectory information required once the full return
                # from this episode can be calculated.
                log_probs.append(log_prob)
                rewards.append(reward)

                raw_frame = next_frame
                step += 1

            # Step 5: Calculate reward-to-go in reverse order. Each action is
            # weighted by all discounted rewards that followed it in the episode.
            discounted_rewards = []
            R = 0
            for r in reversed(rewards):
                R = r + gamma * R
                discounted_rewards.insert(0, R)

            discounted_rewards = torch.tensor(discounted_rewards).to(device)
            # Normalize returns within sufficiently long episodes to reduce
            # gradient scale variance; epsilon prevents division by zero.
            if len(discounted_rewards) > 1:
                discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
                            discounted_rewards.std() + 1e-8)

            # REINFORCE minimizes negative log-probability times return, thereby
            # increasing the likelihood of actions associated with high returns.
            policy_loss = []
            for log_p, r in zip(log_probs, discounted_rewards):
                policy_loss.append(-log_p * r)

            # Update the controller once using the complete episode trajectory.
            optimizer.zero_grad()
            if policy_loss:
                total_loss = torch.cat(policy_loss).sum()
                total_loss.backward()
                optimizer.step()

            total_score = sum(rewards)
            print(f"Episode [{ep}/{episodes}] | Steps: {step} | Total Reward: {total_score:.2f}")

    # Save controller parameters only; inference must recreate the same linear
    # architecture and load compatible VAE/MDN-RNN checkpoints separately.
    torch.save(controller.state_dict(), "checkpoints/controller_dino.pth")
    print("Controller training complete! Weights saved to checkpoints/controller_dino.pth")


if __name__ == "__main__":
    # Start browser-based training only during direct script execution.
    train_controller()
