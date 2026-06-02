# Probabilistic Biped Navigation — MuJoCo Implementation

> **Based on:** *Probabilistically-Safe Bipedal Navigation over Uncertain Terrain via Conformal Prediction and Contraction Analysis*

This repository implements a simplified MuJoCo-based bipedal navigation pipeline inspired by the above paper. The original paper uses the Digit robot model and an existing full-body walking controller. Since the Digit MuJoCo model is not available here, this repository uses a custom simplified humanoid and focuses on reproducing the main algorithmic structure:

```text
uncertain terrain estimation
→ conformal prediction safety margin
→ CP-safe global route planning
→ waypoint-guided LIPM-MPC
→ disturbed Aug-LIPM tracking
→ CCM-like flywheel torque correction
→ RCI tube verification
→ MuJoCo uneven-terrain visualization
```

The MuJoCo humanoid visualization is currently **kinematic**, not a dynamically stable full-body walking controller. It is used to visualize the reduced-order planned trajectory over uneven terrain.

---

## Repository Structure

```text
probabilistic_biped_nav/
│
├── README.md
├── requirements.txt
├── simple_humanoid.xml
│
├── test_humanoid.py
├── read_humanoid_state.py
├── mujoco_ccm_torso_torque.py
│
├── terrain_gp_cp_demo.py
├── cp_safe_footstep_demo.py
├── cp_mpc_parameter_study.py
│
├── aug_lipm_ccm_demo.py
├── lipm_discrete_planner_demo.py
├── lipm_cp_safe_planner_demo.py
├── lipm_cp_safe_mpc_demo.py
├── lipm_cp_safe_waypoint_mpc_demo.py
├── lipm_cp_safe_waypoint_mpc_fast.py
│
├── rci_tube_tracking_demo.py
├── run_full_pipeline.py
│
├── visualize_humanoid_on_terrain.py
└── visualize_humanoid_walking_on_terrain.py
```

---

## Main Files

| File                                       | Purpose                                                                    |
| ------------------------------------------ | -------------------------------------------------------------------------- |
| `simple_humanoid.xml`                      | Custom MuJoCo humanoid model with torso, legs, feet, joints, and actuators |
| `test_humanoid.py`                         | Loads the simple humanoid and opens the MuJoCo viewer                      |
| `read_humanoid_state.py`                   | Reads torso position, velocity proxy, foot positions, and contacts         |
| `mujoco_ccm_torso_torque.py`               | Applies a CCM-like external torso torque in MuJoCo                         |
| `terrain_gp_cp_demo.py`                    | Demonstrates GP terrain estimation and split conformal prediction          |
| `cp_safe_footstep_demo.py`                 | Demonstrates CP-safe footstep constraint                                   |
| `cp_mpc_parameter_study.py`                | Studies confidence level vs reachability/safety tradeoff                   |
| `aug_lipm_ccm_demo.py`                     | Reduced-order Aug-LIPM + CCM tracking demo                                 |
| `lipm_discrete_planner_demo.py`            | Discrete LIPM footstep dynamics demo                                       |
| `lipm_cp_safe_waypoint_mpc_fast.py`        | Fast global CP-safe route + waypoint-guided LIPM-MPC                       |
| `rci_tube_tracking_demo.py`                | Disturbed Aug-LIPM tracking with RCI tube check                            |
| `run_full_pipeline.py`                     | Main full algorithmic pipeline                                             |
| `visualize_humanoid_walking_on_terrain.py` | MuJoCo animated humanoid visualization over uneven terrain                 |

---

## Setup from Scratch
## Main commands

Activate the virtual environment first:

```powershell
cd "D:\GNC_LAB\LAB_WORK\probabilistic_biped_nav"
.\venv\Scripts\activate
### 1. Install Python

Download Python from:

```text
https://www.python.org/downloads/
```

On Windows, tick:

```text
Add Python to PATH
```

Check installation:

```powershell
python --version
```

Expected:

```text
Python 3.10.x / 3.11.x / 3.12.x
```

---

### 2. Create the project folder

```powershell
cd "D:\GNC_LAB\LAB_WORK"
mkdir probabilistic_biped_nav
cd probabilistic_biped_nav
```

---

### 3. Create and activate a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\activate
```

The terminal should show:

```text
(venv)
```

This means Python is using the isolated project environment.

---

### 4. Install dependencies

```powershell
pip install --upgrade pip
pip install mujoco numpy matplotlib
```

Or install from `requirements.txt`:

```powershell
pip install -r requirements.txt
```

---

### 5. Verify MuJoCo installation

```powershell
python -c "import mujoco; print(mujoco.__version__)"
```

Expected output:

```text
3.x.x
```

---

## Daily Workflow

Every time a new terminal is opened:

```powershell
cd "D:\GNC_LAB\LAB_WORK\probabilistic_biped_nav"
.\venv\Scripts\activate
```

Then run whichever script is needed.

---

## Main Commands

### Run the full algorithmic pipeline

```powershell
python run_full_pipeline.py
```

This runs:

```text
GP + conformal prediction terrain uncertainty
→ global CP-safe route planning
→ waypoint-guided LIPM-MPC
→ terrain safety evaluation
→ disturbed Aug-LIPM tracking
→ RCI / CCM tube verification
```

A typical successful run gives:

```text
goal reached                     = True
final distance to goal           ≈ 0.21 m
true terrain violations          = 0
max true |delta h|               ≈ 0.068 m
tube violation fraction          = 0.00%
final tracking error             ≈ 0.013 m
```

---

### Run the MuJoCo humanoid visualization

```powershell
python visualize_humanoid_walking_on_terrain.py
```

This opens MuJoCo and shows:

```text
uneven block terrain
blue dots     = global CP-safe route
orange dots   = executed LIPM-MPC path
humanoid      = kinematic walking animation
green sphere  = goal
```

The humanoid body follows the planned CP-safe LIPM-MPC trajectory. The legs and arms are animated kinematically so the robot looks like it is walking, but this is not yet a dynamically stable physics-based walking controller.

---

### Basic MuJoCo humanoid test

```powershell
python test_humanoid.py
```

This simply loads the custom humanoid XML and runs the simulation.

---

### Read MuJoCo humanoid state

```powershell
python read_humanoid_state.py
```

This prints:

```text
torso position
root velocity proxy
left foot position
right foot position
number of contacts
foot-ground contact pairs
```

---

## Current Full Pipeline

The current working implementation contains the following stages.

### 1. Terrain Estimation

A synthetic uneven terrain map is generated:

```text
z = g(x, y)
```

Sparse noisy terrain samples are collected and used to train a Gaussian Process terrain estimator.

The GP gives:

```text
mean terrain estimate      μ(x, y)
posterior uncertainty      σ(x, y)
```

---

### 2. Conformal Prediction

Split conformal prediction is used to compute a terrain-height safety margin:

```text
C = quantile of calibration residuals |z - μ(x,y)|
```

This gives a calibrated terrain interval:

```text
Iδ(x,y) = [ μ(x,y) - C, μ(x,y) + C ]
```

The safety confidence is controlled by:

```text
1 - δ
```

Higher confidence gives a larger conformal margin, which improves safety but makes the planner more conservative.

---

### 3. CP-Safe Terrain Constraint

The terrain step safety constraint is:

```text
Δhmax - |z_next - z_current| ≥ 0
```

Since the next true height is unknown, the planner enforces the CP-safe version:

```text
Δhmax - | μ(x_next, y_next) - z_current | ≥ C
```

This is the key uncertainty-aware safety constraint.

---

### 4. Global CP-Safe Route

A high-level global route is generated using CP-safe terrain transitions.

This route gives safe intermediate waypoints over the terrain. It prevents the local LIPM planner from repeatedly trying to cross steep or unsafe terrain directly.

---

### 5. Waypoint-Guided LIPM-MPC

The reduced-order walking state is:

```text
state = [x, y, z, v_loc, theta]
```

where:

```text
x, y, z   = global position on terrain
v_loc     = local sagittal velocity
theta     = heading angle
```

The control is:

```text
control = [u_f, delta_theta]
```

where:

```text
u_f          = sagittal foot placement control
delta_theta  = heading change
```

The planner uses discrete LIPM dynamics and tracks a local waypoint from the global CP-safe route.

---

### 6. Disturbed Aug-LIPM Tracking

The planned LIPM trajectory is treated as the nominal trajectory.

The disturbed Aug-LIPM tracking model is:

```text
e_dot = A e + B τ_y + B_w w
```

where:

```text
e       = tracking error
τ_y     = flywheel torque correction
w       = bounded terrain/model disturbance
```

The CCM-like torque law is:

```text
τ_y = -1/2 ρ Bᵀ M e
```

---

### 7. RCI Tube Check

A robust tube radius is computed around the nominal trajectory:

```text
Ω(x*, t) = { x : ||x - x*|| ≤ ε(t) }
```

The simulation checks whether the disturbed trajectory remains inside the tube.

Typical result:

```text
tube violation fraction = 0.00%
```

---

### 8. MuJoCo Visualization

The final planned path is visualized in MuJoCo on an uneven block terrain.

The current visualization is kinematic:

```text
the humanoid body moves along the planned path
legs and arms swing with a walking-like gait
terrain is uneven
route and path markers are shown
```

This is useful for visualizing the full algorithmic result, but it is not yet a physics-valid full-body walking controller.

---

## Common Issues

### `ModuleNotFoundError: No module named 'mujoco'`

The virtual environment is probably not active.

Fix:

```powershell
cd "D:\GNC_LAB\LAB_WORK\probabilistic_biped_nav"
.\venv\Scripts\activate
pip install mujoco
```

---

### PowerShell blocks virtual environment activation

Run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\venv\Scripts\activate
```

---

### Prompt shows `(base)` but not `(venv)`

You are in conda base, not the project virtual environment.

Run:

```powershell
.\venv\Scripts\activate
```

The prompt may show both:

```text
(venv) (base)
```

That is okay as long as `(venv)` appears.

---

### The MuJoCo humanoid falls in `test_humanoid.py`

That is expected. `test_humanoid.py` only checks that the model loads and the simulation runs. It does not include a stabilizing walking controller.

---

### The MuJoCo visualization humanoid is not dynamically walking

Correct. The current visualization is kinematic. It is intended to show the planned CP-safe LIPM-MPC path over uneven terrain. Full dynamic walking requires a low-level controller, swing-foot control, contact switching, and balance stabilization.

---

## Project Roadmap

```text
Phase 1 — MuJoCo foundation
  ✓ custom humanoid XML
  ✓ viewer test
  ✓ state reading
  ✓ torso torque interface

Phase 2 — Reduced-order control
  ✓ Aug-LIPM dynamics
  ✓ CCM-like flywheel torque demo

Phase 3 — Terrain uncertainty
  ✓ synthetic rough terrain
  ✓ sparse samples
  ✓ GP terrain estimation
  ✓ split conformal prediction margin

Phase 4 — CP-safe planning
  ✓ CP-safe footstep constraint
  ✓ parameter study
  ✓ global CP-safe waypoint route

Phase 5 — LIPM-MPC
  ✓ discrete LIPM dynamics
  ✓ waypoint-guided short-horizon LIPM-MPC
  ✓ goal reaching with zero true terrain violations

Phase 6 — Robust tracking
  ✓ disturbed Aug-LIPM tracking
  ✓ CCM-like correction
  ✓ RCI tube verification

Phase 7 — MuJoCo visualization
  ✓ uneven terrain visualization
  ✓ kinematic humanoid walking animation

Phase 8 — Future dynamic walking
  □ full-body humanoid balance controller
  □ swing-foot trajectory generation
  □ stance/contact switching
  □ physical tracking of LIPM-MPC references
  □ integration of CCM torque with physics walking
```

---

## Current Status

The repository currently implements **Version 0.1** of a simplified paper-faithful pipeline.

Working:

```text
✓ full Python algorithmic pipeline
✓ CP-safe global route reaches goal
✓ waypoint-guided LIPM-MPC reaches goal
✓ zero true terrain safety violations
✓ disturbed Aug-LIPM tracking remains inside RCI tube
✓ MuJoCo uneven-terrain humanoid visualization
```

Not yet implemented:

```text
✗ dynamically stable full-body humanoid walking
✗ physical footstep contact control
✗ swing-foot trajectory tracking
✗ Digit robot model
✗ passivity-based whole-body controller
```

---

## Reference

**Probabilistically-Safe Bipedal Navigation over Uncertain Terrain via Conformal Prediction and Contraction Analysis**
