\# Probabilistic Biped Navigation — MuJoCo Starting Repository



This repository is the starting point for implementing a MuJoCo-based bipedal navigation project inspired by the paper:



\*\*Probabilistically-Safe Bipedal Navigation over Uncertain Terrain via Conformal Prediction and Contraction Analysis\*\*



The original paper uses the Digit biped robot in MuJoCo. Since the Digit MuJoCo model is not available here, this repository starts with a custom simplified MuJoCo humanoid model. The purpose is to first build the simulation, state-reading, and control infrastructure, and then gradually add the paper components: terrain uncertainty, conformal prediction, LIPM planning, MPC, reachable tubes, and flywheel torque correction.



\---



\## 1. What this repository currently contains



```text

probabilistic\_biped\_nav/

│

├── README.md

├── requirements.txt

├── simple\_humanoid.xml

└── test\_humanoid.py

```



\### File descriptions



```text

simple\_humanoid.xml

```



A simple custom MuJoCo humanoid model with torso, head, two legs, feet, hinge joints, and actuators.



```text

test\_humanoid.py

```



Python script that loads the humanoid XML model, opens the MuJoCo viewer, and runs the simulation.



```text

requirements.txt

```



Python packages required for the current stage.



\---



\## 2. Installation from zero knowledge



This section explains how to set up MuJoCo for someone starting from scratch.



MuJoCo is a physics simulator used for robotics, control, locomotion, reinforcement learning, and contact-rich simulation. In this project, MuJoCo is used as the full-order robot simulation environment.



The basic installation flow is:



```text

Install Python

→ create a project folder

→ create a virtual environment

→ install MuJoCo

→ run a small simulation

```



\---



\## 3. Install Python



First install Python from:



```text

https://www.python.org/downloads/

```



During installation on Windows, make sure to tick:



```text

Add Python to PATH

```



After installation, open PowerShell and check:



```powershell

python --version

```



You should see something like:



```text

Python 3.10.x

```



or



```text

Python 3.11.x

```



Python 3.10, 3.11, or 3.12 is recommended for this project.



\---



\## 4. Create the project folder



In PowerShell:



```powershell

cd "D:\\GNC\_LAB\\LAB\_WORK"

mkdir probabilistic\_biped\_nav

cd probabilistic\_biped\_nav

```



This becomes the project directory.



\---



\## 5. Create a virtual environment



A virtual environment keeps the project packages separate from the rest of the computer.



Run:



```powershell

python -m venv venv

```



This creates a folder called:



```text

venv

```



inside the project.



\---



\## 6. Activate the virtual environment



On Windows PowerShell:



```powershell

.\\venv\\Scripts\\activate

```



After activation, the terminal should show:



```text

(venv) PS D:\\GNC\_LAB\\LAB\_WORK\\probabilistic\_biped\_nav>

```



The `(venv)` is important. It means Python is now using the project environment.



If `(venv)` is not visible, the environment is not active.



\---



\## 7. Install MuJoCo and required packages



After activating the virtual environment, run:



```powershell

pip install --upgrade pip

pip install mujoco numpy matplotlib

```



The package `mujoco` provides the MuJoCo Python interface and viewer.



The package `numpy` is used for numerical computation.



The package `matplotlib` will be used later for plotting trajectories, terrain, phase portraits, and tracking errors.



\---



\## 8. Check that MuJoCo installed correctly



Run:



```powershell

python -c "import mujoco; print(mujoco.\_\_version\_\_)"

```



If MuJoCo is installed correctly, this prints the installed MuJoCo version.



Example:



```text

3.3.0

```



If this gives:



```text

ModuleNotFoundError: No module named 'mujoco'

```



then the virtual environment is probably not activated.



Fix it by running:



```powershell

.\\venv\\Scripts\\activate

pip install mujoco

```



\---



\## 9. Run the humanoid test



After installing the packages, run:



```powershell

python test\_humanoid.py

```



Expected result:



1\. The MuJoCo viewer opens.

2\. A simple humanoid appears.

3\. The simulation starts running.

4\. The humanoid may fall, which is fine at this stage.



At this stage, the goal is only to verify:



```text

MuJoCo installed

→ model loads

→ viewer opens

→ simulation runs

```



Control will be added later.



\---



\## 10. Daily workflow



Every time a new PowerShell terminal is opened, go to the project folder and activate the environment:



```powershell

cd "D:\\GNC\_LAB\\LAB\_WORK\\probabilistic\_biped\_nav"

.\\venv\\Scripts\\activate

python test\_humanoid.py

```



MuJoCo does not need to be installed again every time. Only the virtual environment needs to be activated.



\---



\## 11. Common beginner mistakes



\### Mistake 1: Running from base instead of venv



Wrong:



```text

(base) PS D:\\GNC\_LAB\\LAB\_WORK\\probabilistic\_biped\_nav>

```



Correct:



```text

(venv) (base) PS D:\\GNC\_LAB\\LAB\_WORK\\probabilistic\_biped\_nav>

```



If the prompt only shows `(base)` and not `(venv)`, activate the virtual environment:



```powershell

.\\venv\\Scripts\\activate

```



\---



\### Mistake 2: MuJoCo not found



Error:



```text

ModuleNotFoundError: No module named 'mujoco'

```



Fix:



```powershell

cd "D:\\GNC\_LAB\\LAB\_WORK\\probabilistic\_biped\_nav"

.\\venv\\Scripts\\activate

pip install mujoco

python test\_humanoid.py

```



\---



\### Mistake 3: PowerShell blocks activation



If PowerShell refuses to activate the environment, run:



```powershell

Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

```



Then try again:



```powershell

.\\venv\\Scripts\\activate

```



\---



\## 12. Project roadmap



The final goal is to build the following pipeline:



```text

MuJoCo humanoid simulation

→ robot state reading

→ foot/contact detection

→ simple standing controller

→ terrain generation

→ Gaussian Process terrain estimation

→ conformal prediction terrain interval

→ CP-safe footstep planning

→ LIPM / Aug-LIPM reference dynamics

→ robust control invariant tube

→ flywheel torque correction

→ full simulation and plots

```



The paper uses this same high-level philosophy: uncertain terrain is estimated, uncertainty is converted into planning and control constraints, and a full-order robot simulation is used to test whether the desired center-of-mass motion remains safe and trackable.



\---



\## 13. Current next step



The next coding step is to read robot states from MuJoCo:



```text

torso position

torso velocity

left foot position

right foot position

contact count

```



After that, a simple PD stabilizer will be added so that the humanoid does not immediately collapse.



