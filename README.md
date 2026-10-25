# Geometry Dash, Played by a DQN

A deep Q-network that learns to play **Geometry Dash** from the pixels on screen, the way a person
does. It doesn't read the game's memory and doesn't use a mod or API. It watches a 128×128 grayscale
copy of the window, decides up to 20 times a second whether to hold the jump button, and learns from dying.

https://github.com/user-attachments/assets/e30906f1-97d7-46c2-a727-508c697aa81f

## The loop

```
screen ──► grayscale 128×128 ──► CNN ──► Q(release), Q(hold) ──► mouse down / up ──► did it die?
  ▲                                                                                     │
  └──────────────────────── replay buffer (20,000 transitions) ◄───────── +1 alive, −100 dead
```

| Piece | How |
|---|---|
| **Seeing** (`screen_capture.py`) | Finds the game window with `xdotool` and grabs it with `mss` |
| **Acting** (`env.py`) | `pyautogui` presses or releases the left mouse button in the window, then waits 50 ms |
| **Dying** (`env.py`) | The level stops scrolling when you crash, so five near-identical frames in a row (MSE < 5) mean death. Matching the "Attempt" text is a fallback |
| **Restarting** (`env.py`) | Clicks Retry and presses space, then a new episode starts |
| **Brain** (`dqn_model.py`) | 3 conv layers (32·8×8/4, 64·4×4/2, 64·3×3/1) → 512 → 2 Q-values. 1 × 128 × 128 in, about 4.8M parameters |
| **Learning** (`train.py`) | DQN with a target network synced every 50 steps, batches of 64, γ = 0.99, Adam at 1e-4, MSE loss on rewards scaled to [−1, 1], gradient norm clipped at 1. Exploration decays from 100% to 5% over about 10,000 steps |

## Run it

Needs Geometry Dash (Steam) visible on screen, Linux with `xdotool`, and Python 3.8+.

```bash
pip install -r requirements.txt
python capture_death_screen.py   # optional: save the "Attempt" text as a death template
python train.py                  # then switch to the game within 3 seconds
```

Training prints one line per step and a summary per episode. Survival time (`Steps`) climbing from
episode to episode is the sign it's learning:

```
Ep   1 | Step   42 | Action: Jump  | Reward:    1.0 | Loss: 0.0412
Episode 1 | Steps: 43 | Reward: -57 | Loss: 0.8111 | Epsilon: 0.985
```

`Ctrl+C` stops training and saves the network to `gd_dqn_model.pth`. A checkpoint is included.

Keep the game window where it is during training: the Retry click is computed from the window's
position. On Windows or macOS, replace the `xdotool` lookup in `screen_capture.py` with fixed screen
coordinates.

## Files

| File | |
|---|---|
| `train.py` | Training loop and hyperparameters |
| `env.py` | Gym-style wrapper: `reset()`, `step(action)`, death detection |
| `dqn_model.py` | The CNN |
| `replay_buffer.py` | Experience replay |
| `screen_capture.py` | Window lookup and capture |
| `capture_death_screen.py` | Saves the death template |
| [ARCHITECTURE.md](ARCHITECTURE.md), [TRAINING_GUIDE.md](TRAINING_GUIDE.md) | Design notes and hyperparameter guide |

Built with PyTorch, OpenCV and a lot of crashes.
