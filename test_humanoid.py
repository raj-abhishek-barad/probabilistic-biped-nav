import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


def get_body_position(model, data, body_name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body not found: {body_name}")
    return data.xpos[body_id].copy()


def main():
    model_path = Path("simple_humanoid.xml")

    if not model_path.exists():
        raise FileNotError(f"Could not find {model_path.resolve()}")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    left_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_foot")
    right_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_foot")

    print("State-reading test started.")
    print(f"torso_id = {torso_id}")
    print(f"left_foot_id = {left_foot_id}")
    print(f"right_foot_id = {right_foot_id}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        last_print = 0.0

        while viewer.is_running() and time.time() - start < 20:
            mujoco.mj_step(model, data)

            sim_time = data.time

            if sim_time - last_print > 0.5:
                torso_pos = data.xpos[torso_id].copy()
                left_foot_pos = data.xpos[left_foot_id].copy()
                right_foot_pos = data.xpos[right_foot_id].copy()

                # qvel[0:3] is approximately the floating-base linear velocity.
                # For now, we use it as a practical torso/CoM velocity proxy.
                root_lin_vel = data.qvel[0:3].copy()

                print("\n----------------------------")
                print(f"t = {sim_time:.2f} s")
                print(f"torso position      = {np.round(torso_pos, 3)}")
                print(f"root linear velocity = {np.round(root_lin_vel, 3)}")
                print(f"left foot position  = {np.round(left_foot_pos, 3)}")
                print(f"right foot position = {np.round(right_foot_pos, 3)}")
                print(f"number of contacts  = {data.ncon}")

                last_print = sim_time

            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()