import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


def get_body_id(model, body_name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        raise ValueError(f"Body '{body_name}' not found in model.")
    return body_id


def main():
    model_path = Path("simple_humanoid.xml")

    if not model_path.exists():
        raise FileNotFoundError(f"Could not find {model_path.resolve()}")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    torso_id = get_body_id(model, "torso")
    left_foot_id = get_body_id(model, "left_foot")
    right_foot_id = get_body_id(model, "right_foot")

    print("Humanoid state-reading script started.")
    print(f"Model loaded from: {model_path.resolve()}")
    print(f"torso_id     = {torso_id}")
    print(f"left_foot_id = {left_foot_id}")
    print(f"right_foot_id = {right_foot_id}")
    print("\nThe script will print robot state every 0.5 seconds.\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        last_print_time = -1.0

        while viewer.is_running():
            step_start = time.time()

            mujoco.mj_step(model, data)

            sim_time = data.time

            if sim_time - last_print_time >= 0.5:
                torso_pos = data.xpos[torso_id].copy()
                left_foot_pos = data.xpos[left_foot_id].copy()
                right_foot_pos = data.xpos[right_foot_id].copy()

                # For a free-floating body, qvel[0:3] gives translational velocity
                # of the root body in MuJoCo generalized coordinates.
                root_lin_vel = data.qvel[0:3].copy()

                print("-" * 60)
                print(f"simulation time       : {sim_time:.3f} s")
                print(f"torso position        : {np.round(torso_pos, 4)}")
                print(f"root linear velocity  : {np.round(root_lin_vel, 4)}")
                print(f"left foot position    : {np.round(left_foot_pos, 4)}")
                print(f"right foot position   : {np.round(right_foot_pos, 4)}")
                print(f"number of contacts    : {data.ncon}")

                if data.ncon > 0:
                    print("contacts:")
                    for i in range(data.ncon):
                        contact = data.contact[i]
                        geom1_name = mujoco.mj_id2name(
                            model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1
                        )
                        geom2_name = mujoco.mj_id2name(
                            model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2
                        )
                        print(f"  {i}: {geom1_name} <--> {geom2_name}")

                last_print_time = sim_time

            viewer.sync()

            sleep_time = model.opt.timestep - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()