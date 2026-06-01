# Probabilistic Biped Navigation — MuJoCo Starting Repository

> **Based on:** *Probabilistically-Safe Bipedal Navigation over Uncertain Terrain via Conformal Prediction and Contraction Analysis*

This repository is the starting point for implementing a MuJoCo-based bipedal navigation project. Since the Digit MuJoCo model is not available, this repository starts with a custom simplified MuJoCo humanoid. The purpose is to build the simulation, state-reading, and control infrastructure first, then gradually layer in the paper's components.

---

## Repository Structure

```
probabilistic_biped_nav/
│
├── README.md
├── requirements.txt
├── simple_humanoid.xml
└── test_humanoid.py
```

| File | Description |
|------|-------------|
| `simple_humanoid.xml` | Custom MuJoCo humanoid — torso, head, two legs, feet, hinge joints, actuators |
| `test_humanoid.py` | Loads the XML model, opens the MuJoCo viewer, runs the simulation |
| `requirements.txt` | Python packages for the current stage |

---

## Setup from Scratch

### 1. Install Python

Download from [python.org/downloads](https://www.python.org/downloads/)

> ⚠️ On Windows: tick **"Add Python to PATH"** during installation.

```powershell
python --version
# Expected: Python 3.10.x / 3.11.x / 3.12.x
```

### 2. Create the project folder

```powershell
cd "D:\GNC_LAB\LAB_WORK"
mkdir probabilistic_biped_nav
cd probabilistic_biped_nav
```

### 3. Create and activate a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\activate
```

Your prompt should now show `(venv)` — this means Python is using the isolated project environment.

### 4. Install dependencies

```powershell
pip install --upgrade pip
pip install mujoco numpy matplotlib
```

### 5. Verify MuJoCo installation

```powershell
python -c "import mujoco; print(mujoco.__version__)"
# Expected: 3.3.0 (or similar)
```

### 6. Run the humanoid test

```powershell
python test_humanoid.py
```

Expected: the MuJoCo viewer opens, a simple humanoid appears, simulation starts. The humanoid may fall — that is fine at this stage.

---

## Daily Workflow

Every time you open a new terminal:

```powershell
cd "D:\GNC_LAB\LAB_WORK\probabilistic_biped_nav"
.\venv\Scripts\activate
python test_humanoid.py
```

MuJoCo does not need reinstalling — only the virtual environment activation is needed each session.

---

## Common Issues

### `ModuleNotFoundError: No module named 'mujoco'`
Virtual environment is not active. Run:
```powershell
.\venv\Scripts\activate
pip install mujoco
```

### PowerShell blocks activation
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\venv\Scripts\activate
```

### Prompt shows `(base)` but not `(venv)`
You are in conda base, not the project venv. Run the activate command above.

---

## Project Roadmap

```
Phase 1 — Simulation Foundation (current)
  ├── MuJoCo humanoid model
  ├── Viewer and simulation loop
  └── Robot state reading (torso pos/vel, foot pos, contact)

Phase 2 — Stabilization
  └── Simple PD standing controller

Phase 3 — Terrain
  ├── Procedural terrain generation
  └── Gaussian Process terrain estimation

Phase 4 — Uncertainty Quantification
  └── Conformal prediction terrain interval

Phase 5 — Planning
  ├── CP-safe footstep planning
  └── LIPM / Aug-LIPM reference dynamics

Phase 6 — Robust Control
  ├── Robust control invariant tube
  └── Flywheel torque correction

Phase 7 — Full Pipeline
  └── Simulation + tracking plots
```

The paper's philosophy: uncertain terrain is estimated → uncertainty is converted into planning/control constraints → full-order robot simulation tests whether CoM motion stays safe and trackable.

---

## Current Next Step

Read robot states from MuJoCo:

```python
# Target state variables
torso_position    # 3D CoM position
torso_velocity    # 3D CoM velocity
left_foot_pos     # 3D left foot position
right_foot_pos    # 3D right foot position
contact_count     # number of active ground contacts
```

After state reading, a simple PD stabilizer will be added so the humanoid does not immediately collapse.

---

## Reference

Probabilistically-Safe Bipedal Navigation over Uncertain Terrain via Conformal Prediction and Contraction Analysis