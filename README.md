# Chrome Dino World Model

An experimental model-based reinforcement learning project that aims to teach
an agent to play the Chrome Dino game by learning a compact model of the game
world from pixels.

The long-term goal is to build the learning system from scratch in PyTorch. The
planned agent will use a Variational Autoencoder (VAE) to compress game frames,
a Recurrent State-Space Model (RSSM) to predict future latent states, and latent
imagination (or "dreaming") to improve its policy without interacting with the
real game for every training step.

> [!WARNING]
> **Early development / pre-alpha:** this repository is public for development
> visibility. It is not yet a complete or trained world-model agent, and its
> APIs, project structure, and setup process may change frequently.

## Current status

As of October 11, 2026, the environment, offline-data workflow, and first
vision-model training stage are operational. World-dynamics and policy training
have not started yet.

| Area | Status | Details |
| --- | --- | --- |
| Chrome Dino environment | Implemented | Launches `chrome://dino` through Selenium and exposes `reset()` and `step()` methods. |
| Visual observations | Implemented | Captures the active runner area and normalizes it into thresholded grayscale NumPy frames (64 x 64 by default). |
| Action controls | Implemented | Supports run, jump, crouch, and fast-drop inputs. |
| Rewards and episode endings | Prototype | Gives a small survival reward and detects collisions through the game's JavaScript state. |
| Demo loop | Implemented | Samples weighted random actions and displays the processed observation with OpenCV. |
| Data collector | Prototype | Runs a weighted-random policy, pairs observations with actions, resets after collisions, and saves compressed NumPy data. |
| Collected dataset | Available | Includes 10,000 aligned frame/action samples in `dataset/dino_data.npz` and a 5 x 5 preview grid. |
| Dataset inspector | Implemented | Reports shapes, dtypes, and action distribution and provides interactive frame-by-frame playback. |
| Replay and sequence pipeline | Planned | Rewards, terminal flags, episode boundaries, replay sampling, and train/validation sequences are not implemented yet. |
| VAE representation model | Prototype | A convolutional VAE compresses 64 x 64 frames into a 16-dimensional stochastic latent space and reconstructs them. |
| VAE training | Implemented | Trains for 20 epochs with Adam using reconstruction and KL-divergence losses, with automatic CUDA/CPU selection. |
| Vision artifacts | Generated | Training produces `checkpoints/vae_dino.pth` and a source-versus-reconstruction comparison image. |
| RSSM world model | Planned | Latent dynamics, reward, and continuation models are not implemented yet. |
| Latent policy learning | Planned | Actor-critic training in imagined trajectories is not implemented yet. |
| Evaluation and tests | Planned | Reproducible benchmarks and automated tests still need to be added. |

The current random controller and collector generate exploratory data only.
They do not learn from experience and should not be interpreted as the final
agent.

## How the finished system is intended to work

1. Collect game frames, actions, rewards, and terminal signals from Chrome Dino.
2. Encode image observations into compact latent representations with a VAE.
3. Train an RSSM to predict how latent states evolve after each action.
4. Imagine possible future trajectories inside the learned world model.
5. Train a policy from those imagined outcomes and evaluate it in the real game.

## Repository structure

```text
chrome-dino-world-model/
├── data_collection.py  # Random-policy observation/action data collector
├── data_inspector.py   # Dataset summary and interactive frame player
├── dataset/
│   ├── dino_data.npz   # 10,000 collected frame/action pairs
│   ├── preview_grid.png
│   └── vae_reconstruction_test.png
├── checkpoints/
│   └── vae_dino.pth    # Trained VAE state dictionary
├── game_room/
│   ├── __init__.py     # Package definition
│   ├── dino.py         # Selenium environment and frame preprocessing
│   └── dino_run.py     # Random-action environment demonstration
├── requirements.txt    # Pinned dependencies for the current prototype
├── train_vae.py         # Convolutional VAE architecture and training pipeline
└── README.md
```

More model, training, configuration, and evaluation modules will be introduced
as development progresses.

## Running the current prototype

### Setup

- Python 3.10 or newer
- Google Chrome
- A desktop session capable of displaying the Chrome and OpenCV windows

Create and activate a virtual environment, then install the pinned dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Environment demonstration

From the repository root, run:

```bash
python game_room/dino_run.py
```

The script opens Chrome Dino, selects a new weighted random action every 20
steps, prints rewards and collision state, and shows the 64 x 64 observation
provided to a future agent. Press `q` in the OpenCV preview window to stop.

`webdriver-manager` may download a compatible ChromeDriver the first time the
prototype starts, so an internet connection can be required on the first run.

### Collecting data

Run the current weighted-random collector:

```bash
python data_collection.py
```

The default run collects up to 10,000 `(frame, action)` pairs. It displays the
processed model input while running and automatically resets the environment
after a collision. Press `q` in the preview window to stop early and save the
samples collected so far.

The collector writes:

- `dataset/dino_data.npz`, containing `frames` (`uint8`) and `actions` (`int64`)
- `dataset/preview_grid.png`, containing 25 randomly selected observations

Running the collector again overwrites these files.

### Inspecting collected data

Launch the dataset summary and interactive player:

```bash
python data_inspector.py
```

The inspector prints dataset metadata and action counts, then replays frames in
collection order. Use `Space` to pause or resume, `A` and `D` to move backward
or forward while paused, and `Q` to exit.

### Training the vision model

Train the convolutional variational autoencoder on the collected frames:

```bash
python train_vae.py
```

The current training configuration:

- Normalizes `uint8` frames from `[0, 255]` to `[0, 1]`
- Uses shuffled batches of 64 observations
- Encodes each frame into a 16-dimensional Gaussian latent representation
- Optimizes summed pixel reconstruction error plus KL divergence
- Trains for 20 epochs with Adam at a learning rate of `1e-3`
- Uses CUDA when available and falls back to the CPU

After training, the script writes:

- `checkpoints/vae_dino.pth`, containing the model state dictionary
- `dataset/vae_reconstruction_test.png`, comparing five source frames with their
  reconstructions

The current reconstruction diagnostic is shown below. Each row contains an
original observation on the left and its VAE reconstruction on the right.

![VAE source and reconstruction comparison](dataset/vae_reconstruction_test.png)

## Limitations

- No trained reinforcement learning agent or world model exists yet.
- The environment relies on Chrome's internal `runnerInstance` JavaScript API,
  which may change between Chrome versions.
- Browser and keyboard behavior has not yet been tested across operating
  systems or in headless mode.
- The collector stores frames and actions only; rewards, terminal flags, and
  explicit episode boundaries are not preserved in the dataset yet.
- The current collector overwrites its fixed output paths instead of versioning
  collection runs.
- VAE training uses the full dataset without a train/validation split, early
  stopping, checkpoint metadata, or quantitative validation metrics.
- VAE hyperparameters are currently hard-coded, and the declared KL-tolerance
  argument is reserved for future use rather than applied by the loss function.
- Reconstruction quality has only been checked on a small sample from the
  training dataset; generalization has not been evaluated yet.
- Error handling, logging, configuration, and automated test coverage are still
  minimal.

## Roadmap

- Stabilize and test the game environment across platforms.
- Complete reproducible dependency and runtime configuration.
- Extend collection to store rewards, terminal flags, and episode boundaries.
- Add replay sampling and train/validation sequence generation.
- Add VAE validation, configurable hyperparameters, and quantitative metrics.
- Evaluate and improve latent representations for downstream dynamics learning.
- Implement the RSSM latent dynamics model.
- Add reward and continuation prediction heads.
- Train an actor-critic policy using imagined latent trajectories.
- Compare the learned policy against random and model-free baselines.
- Add checkpoints, metrics, visualizations, and reproducible evaluations.

## Contributing

The project is currently evolving quickly. Issues, design suggestions, and
small focused pull requests are welcome, but please expect breaking changes
while the initial architecture is being established.

## License

See [LICENSE](LICENSE) for the repository's license terms.
