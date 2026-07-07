# LeRobot DOT Policy - 3D Pick and Place (Red Cube to Rectangle)

Welcome to our team's development repository for validating the **Decoder-Only Transformer (DOT) Policy** within the **LeRobot** ecosystem. 

Our current project objective is to train and evaluate an imitation learning policy that autonomously controls a robotic arm to pick up a small red cube, transfer it cleanly over a 3D workspace, and drop it securely inside a designated 3D rectangular zone.

---

## Project Resources & Hub Links

* **Our Trained Model:** [sebry4n/my_policy_dot_2 on Hugging Face](https://huggingface.co/sebry4n/my_policy_dot_2)
* **Our Custom Demonstration Dataset:** [sebry4n/lerobot_rian_dataset_1_20260623_213100](https://huggingface.co/datasets/sebry4n/lerobot_rian_dataset_1_20260623_213100)
* **Dataset Visualization Space:** [LeRobot Dataset Visualizer](https://huggingface.co/spaces/lerobot/visualize_dataset)
* **Demo Video of Model Implementation:** [LeRobot Model Implementation Video](https://drive.google.com/file/d/1og-42bjEB8PcujE0k-aXTGUBgd46ZnhX/view?usp=sharing)

---

## Repository Architecture & Upstream Tracking

This repository is a **fork** of the original work by Ilia Larchenko. To understand how changes flow, keep these definitions in mind:

* **Upstream Remote (`IliaLarchenko/dot_policy`):** The parent repository we cloned this from. If the original author releases framework bugfixes, we fetch them from *upstream*.
* **Origin Remote (Our Team Fork):** Our personal playground. We can push modifications, custom files, and this documentation file here safely without breaking the original author's codebase.

*Note: The original repository documentation has been renamed to `lerobot_instructions.md` for historical engineering reference.*

---

## Task Breakdown: Pick & Place Sequence

The imitation network processes visual camera data and state vectors to output continuous joint action tracks across three distinct manipulation phases:
1. **Approach & Grasp:** Align the gripper and clamp down securely on the small red cube.
2. **Workspace Trajectory:** Lift and navigate the payload across the 3D target coordinates.
3. **Zone Deposit:** Execute precision release inside the specified 3D rectangular box boundary.

---

## Environment Setup

Our workspace operates inside a local virtual environment

### 1. Source LeRobot
Open PowerShell, move to your local code directory, and activate the pre-configured Python 3.12 layer:
```powershell
source ${HOME}$\lerobot\bin\activate
```
### 2. File Verification Layout
The DOT submodules operate directly within LeRobot's internal runtime architecture. Verify your local workspace has these files in plac
Configuration Layer: `lerobot/src/policies/dot/configuration_dot.py`
Model Code: `lerobot/src/policies/dot/modeling_dot.py`
Process Code: `lerobot/src/policies/dot/processor_dot.py`


### 3. Resolving `pyav` & `conda` Dependency Issues

```powershell

```

### 4. 

```powershell
pip install -e .
```

### Running Policy Evaluation
To verify the model weights locally and run simulation evaluation batches using our custom baseline metrics, execute the following script from the root folder:
```powershell
python lerobot/scripts/eval.py `
  --policy.path=sebry4n/my_policy_dot_2 `
  --env.type=pusht `
  --env.task=PushT-v0 `
  --eval.n_episodes=10 `
  --eval.batch_size=10 `
  --policy.override_dataset_stats=True
```

### Dataset Inspections: 
If you want to analyze the exact video frames, coordinates, and joints captured inside our custom demonstration rows, copy our dataset handle sebry4n/lerobot_rian_dataset_1_20260623_213100 directly into the [LeRobot Dataset Visualizer Space](https://huggingface.co/spaces/lerobot/visualize_dataset).


### References & Credits
1. Model wrapper modules adapted from [IliaLarchenko/dot_policy](https://github.com/IliaLarchenko/dot_policy).
2. Built upon the open-source [huggingface/lerobot](https://github.com/huggingface/lerobot) robotics repository.
