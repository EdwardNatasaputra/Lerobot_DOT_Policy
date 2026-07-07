# LeRobot DOT Policy - 3D Pick and Place (Red Cube to Rectangle)

Welcome to our team's development repository for validating the **Decoder-Only Transformer (DOT) Policy** within the **LeRobot** ecosystem. 

Our current project objective is to train and evaluate an imitation learning policy that autonomously controls a robotic arm to pick up a small red cube, transfer it cleanly over a 3D workspace, and drop it securely inside a designated 3D rectangular zone.

---

## 🔗 Project Resources & Hub Links

* **🤖 Our Trained Model:** [sebry4n/my_policy_dot_2 on Hugging Face](https://huggingface.co/sebry4n/my_policy_dot_2)
* **📊 Our Custom Demonstration Dataset:** [sebry4n/lerobot_rian_dataset_1_20260623_213100](https://huggingface.co/datasets/sebry4n/lerobot_rian_dataset_1_20260623_213100)
* **📺 Dataset Visualization Space:** [LeRobot Dataset Visualizer](https://huggingface.co/spaces/lerobot/visualize_dataset)
* **📽️ Demo Video of Model Implementation:** [LeRobot Model Implementation Video](https://drive.google.com/file/d/1MHl58Vlw251wVrGCPqCBgenpCS8_rJu0/view?usp=sharing)

---

## 🗺️ Repository Architecture & Upstream Tracking

This repository is a **fork** of the original work by Ilia Larchenko. To understand how changes flow, keep these definitions in mind:

* **Upstream Remote (`IliaLarchenko/dot_policy`):** The parent repository we cloned this from. If the original author releases framework bugfixes, we fetch them from *upstream*.
* **Origin Remote (Our Team Fork):** Our personal playground. We can push modifications, custom files, and this documentation file here safely without breaking the original author's codebase.

*Note: The original repository documentation has been renamed to `lerobot_instructions.md` for historical engineering reference.*

---

## 🎯 Task Breakdown: Pick & Place Sequence

The imitation network processes visual camera data and state vectors to output continuous joint action tracks across three distinct manipulation phases:
1. **Approach & Grasp:** Align the gripper and clamp down securely on the small red cube.
2. **Workspace Trajectory:** Lift and navigate the payload across the 3D target coordinates.
3. **Zone Deposit:** Execute precision release inside the specified 3D rectangular box boundary.

---

## 🛠️ Team Local Environment Setup

Our workspace operates inside a local virtual environment explicitly configured to bypass standard Windows binary conflicts (such as native `pyav` / FFmpeg compilation problems).

### 1. Activating the Sandbox Environment
Open PowerShell, move to your local code directory, and activate the pre-configured Python 3.12 layer:
```powershell
cd "C:\CODE PROJECTS\lerobot"
.\venv_312\Scripts\activate
```
### 2. File Verification Layout
The DOT submodules operate directly within LeRobot's internal runtime architecture. Verify your local workspace has these files in place:
📁 Configuration Layer: `lerobot/common/policies/dot/configuration_dot.py`
📁 Model Code: `lerobot/common/policies/dot/modeling_dot.py`


### 3. Resolving pyav & conda Dependency Issues
If you attempt to run a generic `pip install -e .`, Windows will fail looking for a requirement package named `pyav>=12.0.5.` Additionally, native conda commands will return a CommandNotFoundException if it isn't explicitly in your system paths.

To bypass this safely, force install the pre-compiled wheel files under the official package name av, and bypass checking it during initialization:
```powershell
# Pre-install the binary package directly
pip install av==12.3.0

# Register LeRobot directly bypassing the strict check
pip install -e . --no-dependencies

# Bulk download the required runtime dependencies
pip install datasets deepdiff diffusers einops flask gdown gymnasium h5py jsonlines numba opencv-python torch torchvision draccus cmake pymunk rerun-sdk termcolor wandb zarr
```
