"""Train the Dino controller to imitate actions recorded during manual play."""

import argparse
import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from train_controller import Controller
from train_rnn import MDNRNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)


def load_demonstrations():
    """Load latent/action pairs and any available episode metadata."""
    latent_data = np.load("dataset/dino_latents.npz")
    latents = latent_data["latents"].astype(np.float32)
    actions = latent_data["actions"].astype(np.int64)

    if len(latents) != len(actions):
        raise ValueError("Latent and action arrays have different lengths.")
    if len(latents) < 2:
        raise ValueError("At least two demonstration samples are required.")
    if actions.min() < 0 or actions.max() >= 4:
        raise ValueError("Recorded actions must be integer IDs in [0, 3].")

    episode_ids = None
    dones = None

    # New collectors can preserve metadata in the latent archive. For backward
    # compatibility, also look in the source frame archive.
    for archive_path in ("dataset/dino_latents.npz", "dataset/dino_data.npz"):
        archive = np.load(archive_path)
        if "episode_ids" in archive.files and len(archive["episode_ids"]) == len(latents):
            episode_ids = archive["episode_ids"].astype(np.int64)
        if "dones" in archive.files and len(archive["dones"]) == len(latents):
            dones = archive["dones"].astype(bool)
        if episode_ids is not None or dones is not None:
            break

    return latents, actions, episode_ids, dones


def build_memory_features(latents, actions, episode_ids, dones):
    """Recreate the RNN state preceding each demonstrated action."""
    hidden_features = np.zeros((len(latents), 256), dtype=np.float32)
    has_boundaries = episode_ids is not None or dones is not None

    if not has_boundaries:
        # RNN state cannot be reconstructed safely when separate games are
        # concatenated. Zero memory features turn the existing controller into
        # a latent-only policy while keeping checkpoint shapes compatible.
        print(
            "WARNING: Demonstrations contain no episode boundaries. "
            "Training a safe latent-only controller with RNN weights disabled."
        )
        return hidden_features, False

    rnn = MDNRNN(z_dim=16, action_dim=4, hidden_dim=256, num_gaussians=5).to(device)
    rnn.load_state_dict(
        torch.load("checkpoints/rnn_dino.pth", map_location=device)
    )
    rnn.eval()

    h = torch.zeros(1, 1, 256, device=device)
    c = torch.zeros(1, 1, 256, device=device)

    with torch.no_grad():
        for idx, (latent, action) in enumerate(zip(latents, actions)):
            starts_episode = idx == 0
            if idx > 0 and episode_ids is not None:
                starts_episode = episode_ids[idx] != episode_ids[idx - 1]
            if idx > 0 and dones is not None and dones[idx - 1]:
                starts_episode = True

            if starts_episode:
                # Human demonstrations from different games must never share
                # recurrent memory, even though they occupy one archive.
                h.zero_()
                c.zero_()

            hidden_features[idx] = h.squeeze(0).squeeze(0).cpu().numpy()

            z = torch.from_numpy(latent).view(1, 1, 16).to(device)
            action_onehot = torch.zeros(1, 1, 4, device=device)
            action_onehot[0, 0, int(action)] = 1.0
            rnn_input = torch.cat([z, action_onehot], dim=-1)
            _, _, _, (h, c) = rnn(rnn_input, (h, c))

    return hidden_features, True


def split_indices(sample_count, episode_ids):
    """Create a validation split without mixing frames from one episode."""
    if episode_ids is not None:
        episodes = np.unique(episode_ids)
        if len(episodes) >= 2:
            validation_count = max(1, round(len(episodes) * 0.2))
            validation_episodes = set(episodes[-validation_count:].tolist())
            validation_mask = np.array(
                [episode in validation_episodes for episode in episode_ids]
            )
            return np.flatnonzero(~validation_mask), np.flatnonzero(validation_mask)

    # Older datasets have no episode information. Use a chronological holdout
    # instead of a random frame split, which would leak near-duplicate frames.
    split_at = max(1, int(sample_count * 0.8))
    split_at = min(split_at, sample_count - 1)
    return np.arange(split_at), np.arange(split_at, sample_count)


def make_class_weights(actions):
    """Reduce RUN dominance without assigning weight to absent action classes."""
    counts = np.bincount(actions, minlength=4).astype(np.float32)
    weights = np.zeros(4, dtype=np.float32)
    present = counts > 0
    # Square-root inverse frequency is less aggressive than full inverse class
    # weighting, while still preventing the abundant RUN class from dominating.
    weights[present] = np.sqrt(
        counts[present].sum() / (present.sum() * counts[present])
    )
    weights[present] /= weights[present].mean()
    print("Training action counts:", counts.astype(int).tolist())
    print("Cross-entropy weights:", np.round(weights, 3).tolist())
    return torch.from_numpy(weights).to(device)


def evaluate(model, loader):
    """Return loss, overall accuracy, and per-action accuracy."""
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    class_correct = np.zeros(4, dtype=np.int64)
    class_total = np.zeros(4, dtype=np.int64)

    model.eval()
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model.fc(features)
            total_loss += criterion(logits, labels).item() * len(labels)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += len(labels)
            for action in range(4):
                mask = labels == action
                class_total[action] += mask.sum().item()
                class_correct[action] += (
                    (predictions == labels) & mask
                ).sum().item()

    per_action = [
        class_correct[i] / class_total[i] if class_total[i] else None
        for i in range(4)
    ]
    return total_loss / max(total, 1), correct / max(total, 1), per_action


def train_behavior_clone(epochs=40, batch_size=128, learning_rate=1e-3):
    """Fit the controller to demonstrated actions and save its best checkpoint."""
    print(f"Using device: {device}")
    latents, actions, episode_ids, dones = load_demonstrations()
    hidden, uses_memory = build_memory_features(
        latents, actions, episode_ids, dones
    )
    features = np.concatenate([latents, hidden], axis=1).astype(np.float32)
    train_indices, validation_indices = split_indices(len(features), episode_ids)

    train_dataset = TensorDataset(
        torch.from_numpy(features[train_indices]),
        torch.from_numpy(actions[train_indices]),
    )
    validation_dataset = TensorDataset(
        torch.from_numpy(features[validation_indices]),
        torch.from_numpy(actions[validation_indices]),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False
    )

    controller = Controller(z_dim=16, hidden_dim=256, action_dim=4).to(device)
    if not uses_memory:
        # Initializing these columns to zero ensures live RNN values cannot
        # influence a checkpoint trained only with zero memory features.
        with torch.no_grad():
            controller.fc.weight[:, 16:].zero_()

    criterion = nn.CrossEntropyLoss(
        weight=make_class_weights(actions[train_indices])
    )
    optimizer = optim.Adam(controller.parameters(), lr=learning_rate)
    best_state = copy.deepcopy(controller.state_dict())
    best_validation_loss = float("inf")

    print(
        f"Training behavior clone on {len(train_dataset)} samples; "
        f"validating on {len(validation_dataset)} samples."
    )
    for epoch in range(1, epochs + 1):
        controller.train()
        for batch_features, batch_actions in train_loader:
            batch_features = batch_features.to(device)
            batch_actions = batch_actions.to(device)
            logits = controller.fc(batch_features)
            loss = criterion(logits, batch_actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Without boundary metadata, memory features are deliberately
            # disabled both during training and later real-game evaluation.
            if not uses_memory:
                with torch.no_grad():
                    controller.fc.weight[:, 16:].zero_()

        validation_loss, accuracy, per_action = evaluate(
            controller, validation_loader
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(controller.state_dict())

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            readable_accuracy = [
                "n/a" if value is None else f"{value:.1%}"
                for value in per_action
            ]
            print(
                f"Epoch [{epoch}/{epochs}] | validation loss: "
                f"{validation_loss:.4f} | accuracy: {accuracy:.1%} | "
                f"per action: {readable_accuracy}"
            )

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = "checkpoints/controller_bc_dino.pth"
    torch.save(best_state, checkpoint_path)
    print(f"Saved best demonstrated controller to: {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    train_behavior_clone(args.epochs, args.batch_size, args.learning_rate)
