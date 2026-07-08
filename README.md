# LeRobot DOT Policy 3D Pick and Place (Red Cube to Yellow Container)

Team development repository for validating the **Decoder-Only Transformer (DOT) Policy** inside the **LeRobot** ecosystem.

Project goal: train and evaluate an imitation-learning policy that drives an SO-101 arm to pick up a small red cube, carry it across a 3D workspace, and drop it inside a designated yellow container.

Project members :
* 5024231008 - THEO KAWALISA PINEM
* 5024231009 - SULTAN SYAFIQ RAKAN
* 5024231010 - SEBASTIAN ADIRIAN NUGRAHA
* 5024231023 - EDWARD NATASAPUTRA
---

## Project Resources & Hub Links

* **Trained Model:** [sebry4n/my_policy_dot_3](https://huggingface.co/sebry4n/my_policy_dot_3)
* **Demonstration Dataset:** [sebry4n/lerobot_rian_dataset_4_20260707_235902](https://huggingface.co/datasets/sebry4n/lerobot_rian_dataset_4_20260707_235902)
* **Dataset Visualizer:** [LeRobot Dataset Visualizer](https://huggingface.co/spaces/lerobot/visualize_dataset)
* **Demo Video:** [Model Implementation Video](https://drive.google.com/file/d/1MtTwR5n2S1YIG7vqhDzjkfisknKnvhnG/view?usp=drive_link)

---

## Repository Architecture & Upstream Tracking

Fork of the original work by Ilia Larchenko.

* **Upstream (`IliaLarchenko/dot_policy`):** parent repo. Fetch framework bugfixes from here.
* **Origin (team fork):** our workspace. Push custom files and this documentation here.

Original author docs preserved as `ORIGINAL_README.md`.

---

## Task Breakdown

Network takes camera frames + state vectors, outputs continuous joint-action tracks across three phases:

1. **Approach & Grasp** — align gripper, clamp on the red cube.
2. **Workspace Trajectory** — lift and move payload across the 3D target coordinates.
3. **Zone Deposit** — precise release inside the rectangular box boundary.

---

## Environment Setup

### 1. Activate the LeRobot virtual environment

Activate the pre-built Python 3.12 venv. Pick the line for your shell:

```bash
# Linux / macOS (bash, zsh)
source "$HOME/lerobot/bin/activate"
```

```powershell
# Windows PowerShell
& "$HOME\lerobot\Scripts\Activate.ps1"
```

```bat
:: Windows CMD
%USERPROFILE%\lerobot\Scripts\activate.bat
```

If your venv lives elsewhere, replace `$HOME/lerobot` with your actual path.

### 2. Install DOT policy files into LeRobot

DOT runs inside LeRobot's policy registry. Copy the four policy files from this folder into the LeRobot source tree so `--policy.type=dot` resolves:

```bash
# Run from this repo folder. Adjust the destination to your LeRobot checkout.
DEST="$HOME/lerobot_rian/code/lerobot/src/lerobot/policies/dot"
mkdir -p "$DEST"
cp configuration_dot.py modeling_dot.py processor_dot.py __init__.py "$DEST/"
```

Final layout must be:

```
lerobot/src/lerobot/policies/dot/
├── __init__.py
├── configuration_dot.py
├── modeling_dot.py
└── processor_dot.py
```

### 3. Reinstall LeRobot in editable mode

```bash
pip install -e .
```

Run from the LeRobot repo root so it picks up the new `dot` policy.

---

## Step 1 — Record the Dataset

Teleoperate the SO-101 leader arm to drive the follower and record demonstration episodes. Command taken from `testy.cmd`:

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{top : {type: opencv, index_or_path: '/dev/video4', width: 640, height: 480, fps: 30}, wrist : {type: opencv, index_or_path: '/dev/video0', width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true \
    --dataset.repo_id=sebry4n/lerobot_rian_dataset_3 \
    --dataset.num_episodes=50 \
    --dataset.single_task="Pick up redblock and place it in the yellow container" \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=2 \
    --dataset.episode_time_s=20 \
    --dataset.reset_time_s=7
```

Key flags:

| Flag | Meaning |
|------|---------|
| `--robot.port` / `--teleop.port` | Serial ports for follower / leader arms (`lerobot-find-port` to discover). |
| `--robot.cameras` | Two OpenCV cameras: `top` and `wrist`. Match `index_or_path` to your `/dev/videoN`. |
| `--dataset.repo_id` | Hugging Face dataset destination. |
| `--dataset.num_episodes` | Number of demonstration episodes to record. |
| `--dataset.single_task` | Natural-language task label stored in the dataset. |
| `--dataset.episode_time_s` | Seconds per episode. |
| `--dataset.reset_time_s` | Reset window between episodes. |

Dataset auto-pushes to the Hub on finish. Terminal prints the hotkeys for early-exit / re-record / stop during recording.

Calibrate both arms first if not already done:

```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_awesome_follower_arm
lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=my_awesome_leader_arm
```

---

## Step 2 — Train the DOT Policy

Train on the recorded dataset. Command from `testy.cmd`:

```bash
lerobot-train \
  --dataset.repo_id=sebry4n/lerobot_rian_dataset_1_20260624_165116 \
  --policy.type=dot \
  --output_dir=outputs/train/dot_so101_test \
  --job_name=act_so101_test \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.repo_id=sebry4n/my_policy
```

Notes:

* `--policy.type=dot` selects the DOT policy installed in Step 2 of setup.
* `--policy.device=cuda` for GPU; use `cpu` if no GPU (much slower).
* `--policy.repo_id` — where the trained checkpoint pushes on the Hub.
* `--output_dir` — local checkpoint directory.
* Set `--wandb.enable=true` to log metrics to Weights & Biases.

---

## DOT Parameter Changes

DOT defaults were tuned for the real SO-101 pick-and-place task (2 cameras, 30 fps, 640×480) in `configuration_dot.py`. Override any of these on the `lerobot-train` command with `--policy.<name>=<value>`.

| Parameter | Value | What it does |
|-----------|-------|--------------|
| `n_obs_steps` | `3` | Recent observation frames fed to the model. |
| `train_horizon` | `75` | Action steps predicted during training. Must be ≥ `inference_horizon`. |
| `inference_horizon` | `50` | Action steps predicted at inference. |
| `lookback_obs_steps` | `30` | How far back in history the model also samples (long-horizon context). |
| `lookback_aug` | `5` | Jitter window around the lookback point for augmentation. |
| `vision_backbone` | `resnet18` | Image encoder, ImageNet-pretrained (`ResNet18_Weights.IMAGENET1K_V1`). |
| `lora_rank` | `20` | LoRA adapter rank on the vision backbone. Higher = more capacity + compute. |
| `dim_model` | `128` | Transformer hidden width. |
| `n_heads` | `8` | Attention heads. |
| `dim_feedforward` | `512` | Feed-forward width per decoder layer. |
| `n_decoder_layers` | `8` | Depth of the causal decoder. |
| `rescale_shape` | `(240, 320)` | Camera frames resized to this before the backbone. |
| `crop_scale` | `0.8` | Random-crop augmentation scale. |
| `state_noise` | `0.01` | Gaussian noise added to state input during training. |
| `noise_decay` | `0.999995` | Per-step decay of that noise. |
| `dropout` | `0.1` | Decoder dropout. |
| `alpha` / `train_alpha` | `0.98` / `0.99` | Exponential weighting of chunked actions (inference / training). |
| `predict_every_n` / `return_every_n` | `1` / `1` | Inference-speed knobs: predict/return every N steps. |
| `optimizer_lr` | `1.0e-4` | AdamW learning rate. |
| `optimizer_weight_decay` | `1e-5` | AdamW weight decay. |

**Constraints** (enforced in `__post_init__`): `train_horizon ≥ inference_horizon`; `predict_every_n` and `return_every_n` must both be `≤ inference_horizon`.

Example override:

```bash
lerobot-train --policy.type=dot --dataset.repo_id=<your_dataset> \
  --policy.train_horizon=100 --policy.inference_horizon=60 --policy.device=cuda
```

For PushT / sim environments, set `--policy.override_dataset_stats=true` to use `new_dataset_stats`.

---

## Step 3 — Run Inference (Rollout)

Deploy the trained policy on the real arm. Command from `testy.cmd`:

```bash
lerobot-rollout \
    --strategy.type=base \
    --policy.path=sebry4n/my_policy_dot_2 \
    --robot.type=so101_follower \
    --robot.id=my_followerclear6 \
    --robot.port=/dev/ttyACM0 \
    --robot.cameras="{front : {type: opencv, index_or_path: '/dev/video4', width: 640, height: 480, fps: 30}, wrist : {type: opencv, index_or_path: '/dev/video2', width: 640, height: 480, fps: 30}}" \
    --task="move red cube to the brown area" \
    --duration=300 \
    --display_data=true \
    --fps=5
```

Notes:

* `--policy.path` — trained checkpoint (Hub repo id or local path).
* `--robot.cameras` — camera names must match those used during recording.
* `--task` — natural-language task string.
* `--duration` — rollout length in seconds.
* `--fps` — control rate; lower for CPU inference.
* Add `--device=cpu` if running without a GPU.

---

## Dataset Inspection

Paste the dataset handle `sebry4n/lerobot_rian_dataset_1_20260623_213100` into the [LeRobot Dataset Visualizer](https://huggingface.co/spaces/lerobot/visualize_dataset) to inspect frames, coordinates, and joint values.

---

## References & Credits

1. Model modules adapted from [IliaLarchenko/dot_policy](https://github.com/IliaLarchenko/dot_policy).
2. Built on [huggingface/lerobot](https://github.com/huggingface/lerobot).
</content>
