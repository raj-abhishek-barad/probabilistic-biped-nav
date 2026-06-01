import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


def get_body_id(model, body_name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        raise ValueError(f"Body '{body_name}' not found.")
    return body_id


def get_geom_id(model, geom_name: str) -> int:
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if geom_id == -1:
        raise ValueError(f"Geom '{geom_name}' not found.")
    return geom_id


def lipm_reference(t, x0, v0, uf, omega):
    """
    Nominal LIPM sagittal reference from the paper.
    x_ref is local sagittal CoM position relative to the stance foot.
    """
    x_ref = x0 + np.sinh(omega * t) / omega * v0 + (1.0 - np.cosh(omega * t)) * uf
    v_ref = np.cosh(omega * t) * v0 - omega * np.sinh(omega * t) * uf
    return np.array([x_ref, v_ref])


def main():
    model_path = Path("simple_humanoid.xml")

    if not model_path.exists():
        raise FileNotFoundError(f"Could not find {model_path.resolve()}")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    torso_id = get_body_id(model, "torso")
    left_foot_id = get_body_id(model, "left_foot")
    right_foot_id = get_body_id(model, "right_foot")

    left_foot_geom_id = get_geom_id(model, "left_foot_geom")
    right_foot_geom_id = get_geom_id(model, "right_foot_geom")
    ground_geom_id = get_geom_id(model, "ground")

    # ------------------------------------------------------------
    # LIPM / CCM parameters
    # ------------------------------------------------------------
    g = 9.81
    zH = 0.9
    omega = np.sqrt(g / zH)

    m = 50.0

    # Desired one-step reference.
    # These are deliberately gentle because our humanoid is not a real walking robot.
    x0_ref = 0.0
    v0_ref = 0.0
    uf = 0.0

    # CCM-like parameters from the reduced-order demo.
    M = np.array([
        [40.0, 4.0],
        [4.0, 8.0]
    ])

    B = np.array([
        [0.0],
        [-omega**2 / (m * g)]
    ])

    rho = 5000.0

    # Safety saturation for external torque.
    tau_limit = 120.0

    # We apply torque around MuJoCo y-axis.
    # For our model, this acts like a pitch/flywheel torque.
    torque_axis = np.array([0.0, 1.0, 0.0])

    print("MuJoCo CCM torso torque demo started.")
    print(f"omega = {omega:.3f} rad/s")
    print(f"torso_id = {torso_id}")
    print(f"left_foot_id = {left_foot_id}")
    print(f"right_foot_id = {right_foot_id}")
    print("Press ESC or close the viewer to stop.")

    # We choose the stance foot at startup.
    # Initially both feet are close to the ground; choose the lower one.
    mujoco.mj_forward(model, data)
    left_foot_pos0 = data.xpos[left_foot_id].copy()
    right_foot_pos0 = data.xpos[right_foot_id].copy()

    if left_foot_pos0[2] <= right_foot_pos0[2]:
        stance_foot_id = left_foot_id
        stance_name = "left_foot"
    else:
        stance_foot_id = right_foot_id
        stance_name = "right_foot"

    print(f"Initial stance foot = {stance_name}")

    # Root x-velocity proxy from qvel[0].
    # Later we will replace this with a cleaner CoM velocity estimate.
    last_print_time = -1.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_wall_time = time.time()

        while viewer.is_running():
            step_start = time.time()
            sim_time = data.time

            # ------------------------------------------------------------
            # 1. Read MuJoCo full-order state
            # ------------------------------------------------------------
            torso_pos = data.xpos[torso_id].copy()
            stance_pos = data.xpos[stance_foot_id].copy()

            # Local sagittal coordinate: torso x relative to stance foot x.
            x_loc = torso_pos[0] - stance_pos[0]

            # Approximate sagittal velocity using floating base x velocity.
            v_loc = data.qvel[0]

            x_actual = np.array([x_loc, v_loc])

            # ------------------------------------------------------------
            # 2. Desired reduced-order LIPM reference
            # ------------------------------------------------------------
            x_ref = lipm_reference(sim_time, x0_ref, v0_ref, uf, omega)

            # ------------------------------------------------------------
            # 3. Compute CCM torque
            # ------------------------------------------------------------
            e = x_actual - x_ref

            tau_y = -0.5 * rho * (B.T @ M @ e.reshape(2, 1)).item()
            tau_y = np.clip(tau_y, -tau_limit, tau_limit)

            # ------------------------------------------------------------
            # 4. Apply external flywheel torque to torso
            # ------------------------------------------------------------
            # xfrc_applied[body_id, 0:3] = force
            # xfrc_applied[body_id, 3:6] = torque
            data.xfrc_applied[:, :] = 0.0
            data.xfrc_applied[torso_id, 3:6] = tau_y * torque_axis

            # ------------------------------------------------------------
            # 5. Step physics
            # ------------------------------------------------------------
            mujoco.mj_step(model, data)

            # ------------------------------------------------------------
            # 6. Print debug information
            # ------------------------------------------------------------
            if sim_time - last_print_time >= 0.5:
                print("-" * 70)
                print(f"t                  = {sim_time:.3f} s")
                print(f"stance foot         = {stance_name}")
                print(f"torso position      = {np.round(torso_pos, 4)}")
                print(f"stance foot pos     = {np.round(stance_pos, 4)}")
                print(f"x_loc actual        = {x_loc:.4f} m")
                print(f"v_loc actual        = {v_loc:.4f} m/s")
                print(f"x_ref               = {x_ref[0]:.4f} m")
                print(f"v_ref               = {x_ref[1]:.4f} m/s")
                print(f"tracking error norm = {np.linalg.norm(e):.4f}")
                print(f"tau_y applied       = {tau_y:.4f} N m")
                print(f"number of contacts  = {data.ncon}")

                # Contact summary
                for i in range(data.ncon):
                    contact = data.contact[i]
                    g1 = contact.geom1
                    g2 = contact.geom2

                    name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1)
                    name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2)

                    if (g1 in [left_foot_geom_id, right_foot_geom_id] and g2 == ground_geom_id) or \
                       (g2 in [left_foot_geom_id, right_foot_geom_id] and g1 == ground_geom_id):
                        print(f"foot-ground contact : {name1} <--> {name2}")

                last_print_time = sim_time

            viewer.sync()

            sleep_time = model.opt.timestep - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()