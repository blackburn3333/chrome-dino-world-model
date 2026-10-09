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

As of October 2026, the project contains the first environment prototype:

| Area | Status | Details |
| --- | --- | --- |
| Chrome Dino environment | Implemented | Launches `chrome://dino` through Selenium and exposes `reset()` and `step()` methods. |
| Visual observations | Implemented | Captures the game canvas and converts it to a configurable grayscale NumPy frame (64 x 64 by default). |
| Action controls | Implemented | Supports run, jump, crouch, and fast-drop inputs. |
| Rewards and episode endings | Prototype | Gives a small survival reward and detects collisions through the game's JavaScript state. |
| Demo loop | Implemented | Samples weighted random actions and displays the processed observation with OpenCV. |
| Data collection pipeline | Planned | Replay storage and training-sequence generation are not implemented yet. |
| VAE representation model | Planned | Frame encoder/decoder training is not implemented yet. |
| RSSM world model | Planned | Latent dynamics, reward, and continuation models are not implemented yet. |
| Latent policy learning | Planned | Actor-critic training in imagined trajectories is not implemented yet. |
| Evaluation and tests | Planned | Reproducible benchmarks and automated tests still need to be added. |

The current random controller is an environment test only. It does not learn
from experience and should not be interpreted as the final agent.

## How the finished system is intended to work

1. Collect game frames, actions, rewards, and terminal signals from Chrome Dino.
2. Encode image observations into compact latent representations with a VAE.
3. Train an RSSM to predict how latent states evolve after each action.
4. Imagine possible future trajectories inside the learned world model.
5. Train a policy from those imagined outcomes and evaluate it in the real game.

## Repository structure

```text
game_room/
├── __init__.py   # Package definition
├── dino.py       # Selenium environment and frame preprocessing
└── dino_run.py   # Random-action environment demonstration
```

More model, training, configuration, and evaluation modules will be introduced
as development progresses.

## Running the current prototype

### Requirements

- Python 3.9 or newer
- Google Chrome
- A desktop session capable of displaying the Chrome and OpenCV windows

Install the current Python dependencies:

```bash
python -m pip install numpy opencv-python selenium webdriver-manager
```

From the repository root, run:

```bash
python game_room/dino_run.py
```

The script opens Chrome Dino, selects a new weighted random action every 20
steps, prints rewards and collision state, and shows the 64 x 64 observation
provided to a future agent. Press `q` in the OpenCV preview window to stop.

`webdriver-manager` may download a compatible ChromeDriver the first time the
prototype starts, so an internet connection can be required on the first run.

## Limitations

- No trained reinforcement learning agent or world model exists yet.
- The environment relies on Chrome's internal `runnerInstance` JavaScript API,
  which may change between Chrome versions.
- Browser and keyboard behavior has not yet been tested across operating
  systems or in headless mode.
- Dependency versions are not pinned yet.
- Error handling, logging, configuration, and automated test coverage are still
  minimal.

## Roadmap

- Stabilize and test the game environment.
- Add reproducible configuration and dependency management.
- Build an experience replay and data collection pipeline.
- Implement and train the VAE observation model.
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
