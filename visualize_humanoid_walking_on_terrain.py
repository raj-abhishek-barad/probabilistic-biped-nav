import time
import numpy as np
import mujoco
import mujoco.viewer

from run_full_pipeline import run_full_pipeline
from lipm_cp_safe_waypoint_mpc_fast import true_terrain


def yaw_to_quat(yaw):
    return np.array([
        np.cos(yaw / 2.0),
        0.0,
        0.0,
        np.sin(yaw / 2.0),
    ])


def make_scene_xml(states, route, start, goal):
    xml = """
<mujoco model="humanoid_walking_pipeline_visualization">
  <compiler angle="degree" coordinate="local"/>
  <option timestep="0.01" gravity="0 0 -9.81"/>

  <asset>
    <material name="robot_blue" rgba="0.2 0.45 1.0 1"/>
    <material name="head_mat" rgba="0.9 0.75 0.55 1"/>
    <material name="left_leg_mat" rgba="0.1 0.8 0.35 1"/>
    <material name="right_leg_mat" rgba="0.9 0.25 0.25 1"/>
    <material name="route_mat" rgba="0.1 0.45 1.0 1"/>
    <material name="path_mat" rgba="1.0 0.45 0.05 1"/>
    <material name="start_mat" rgba="0.0 0.7 1.0 1"/>
    <material name="goal_mat" rgba="0.0 0.9 0.1 1"/>
  </asset>

  <worldbody>
    <light pos="0 0 8" dir="0 0 -1"/>
"""

    # ------------------------------------------------------------
    # Uneven terrain as boxes
    # ------------------------------------------------------------
    x_vals = np.linspace(-5.0, 5.0, 31)
    y_vals = np.linspace(-5.0, 5.0, 31)

    dx = x_vals[1] - x_vals[0]
    dy = y_vals[1] - y_vals[0]
    base_z = -0.55

    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            z_top = true_terrain(x, y)
            z_center = 0.5 * (base_z + z_top)
            z_size = max(0.01, 0.5 * (z_top - base_z))

            h = (z_top + 0.4) / 0.8
            h = min(1.0, max(0.0, h))
            r = 0.15 + 0.45 * h
            g = 0.35 + 0.45 * h
            b = 0.25

            xml += f"""
    <geom name="terrain_{i}_{j}" type="box"
          pos="{x:.4f} {y:.4f} {z_center:.4f}"
          size="{dx/2:.4f} {dy/2:.4f} {z_size:.4f}"
          rgba="{r:.3f} {g:.3f} {b:.3f} 1"/>
"""

    z_start = true_terrain(start[0], start[1]) + 0.10
    z_goal = true_terrain(goal[0], goal[1]) + 0.15

    xml += f"""
    <body name="start_marker" pos="{start[0]:.4f} {start[1]:.4f} {z_start:.4f}">
      <geom type="box" size="0.16 0.16 0.08" material="start_mat"/>
    </body>

    <body name="goal_marker" pos="{goal[0]:.4f} {goal[1]:.4f} {z_goal:.4f}">
      <geom type="sphere" size="0.18" material="goal_mat"/>
    </body>
"""

    for i, p in enumerate(route):
        z = true_terrain(p[0], p[1]) + 0.08
        xml += f"""
    <body name="route_{i}" pos="{p[0]:.4f} {p[1]:.4f} {z:.4f}">
      <geom type="sphere" size="0.045" material="route_mat"/>
    </body>
"""

    for i, s in enumerate(states):
        x, y = s[0], s[1]
        z = true_terrain(x, y) + 0.14
        xml += f"""
    <body name="path_{i}" pos="{x:.4f} {y:.4f} {z:.4f}">
      <geom type="sphere" size="0.055" material="path_mat"/>
    </body>
"""

    # ------------------------------------------------------------
    # Kinematic humanoid:
    # root, head, arms, and legs are separate mocap bodies.
    # This lets us animate leg swing.
    # ------------------------------------------------------------
    xml += """
    <body name="torso_root" mocap="true" pos="0 0 1.0">
      <geom name="torso" type="capsule"
            fromto="0 0 -0.25 0 0 0.25"
            size="0.13" material="robot_blue"/>
    </body>

    <body name="head_root" mocap="true" pos="0 0 1.45">
      <geom name="head" type="sphere" size="0.12" material="head_mat"/>
    </body>

    <body name="left_leg_root" mocap="true" pos="0 0.1 0.5">
      <geom name="left_leg" type="capsule"
            fromto="0 0 0 0 0 -0.75"
            size="0.045" material="left_leg_mat"/>
    </body>

    <body name="right_leg_root" mocap="true" pos="0 -0.1 0.5">
      <geom name="right_leg" type="capsule"
            fromto="0 0 0 0 0 -0.75"
            size="0.045" material="right_leg_mat"/>
    </body>

    <body name="left_arm_root" mocap="true" pos="0 0.22 1.1">
      <geom name="left_arm" type="capsule"
            fromto="0 0 0 0 0 -0.45"
            size="0.035" rgba="0.2 0.8 0.9 1"/>
    </body>

    <body name="right_arm_root" mocap="true" pos="0 -0.22 1.1">
      <geom name="right_arm" type="capsule"
            fromto="0 0 0 0 0 -0.45"
            size="0.035" rgba="0.2 0.8 0.9 1"/>
    </body>
"""

    xml += """
  </worldbody>
</mujoco>
"""
    return xml


def local_to_world(base_pos, yaw, local_offset):
    c = np.cos(yaw)
    s = np.sin(yaw)

    R = np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])

    return base_pos + R @ local_offset


def build_animation(states, samples_per_step=8):
    """
    Interpolate the planned LIPM path into a smoother animation.
    """
    path = []

    for i in range(len(states) - 1):
        p0 = states[i, :2]
        p1 = states[i + 1, :2]

        for a in np.linspace(0.0, 1.0, samples_per_step, endpoint=False):
            xy = (1.0 - a) * p0 + a * p1
            path.append(xy)

    path.append(states[-1, :2])
    path = np.array(path)

    yaw_list = []
    for i in range(len(path)):
        if i < len(path) - 1:
            d = path[i + 1] - path[i]
        else:
            d = path[i] - path[i - 1]

        yaw_list.append(np.arctan2(d[1], d[0]))

    return path, np.array(yaw_list)


def main():
    print("Running full pipeline...")
    result = run_full_pipeline()

    states = result["states"]
    route = result["route"]
    start = result["start"]
    goal = result["goal"]

    print("\nBuilding MuJoCo animated humanoid walking visualization...")
    xml = make_scene_xml(states, route, start, goal)

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    path_xy, yaw_list = build_animation(states, samples_per_step=8)

    # Mocap bodies are created in XML order.
    torso_id = 0
    head_id = 1
    left_leg_id = 2
    right_leg_id = 3
    left_arm_id = 4
    right_arm_id = 5

    print("\nVisualization legend:")
    print("  uneven boxes  = terrain height map")
    print("  blue dots     = global CP-safe route")
    print("  orange dots   = executed LIPM-MPC path")
    print("  humanoid      = kinematic walking animation")
    print("  green sphere  = goal")
    print("\nClose the MuJoCo viewer to stop.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        k = 0
        last_switch = time.time()
        switch_period = 0.045

        while viewer.is_running():
            now = time.time()

            if now - last_switch > switch_period:
                k = (k + 1) % len(path_xy)
                last_switch = now

            x, y = path_xy[k]
            yaw = yaw_list[k]

            terrain_z = true_terrain(x, y)

            # Gait phase
            phase = 2.0 * np.pi * (k % 16) / 16.0

            # Body motion
            body_bob = 0.035 * np.sin(2.0 * phase)
            torso_pos = np.array([x, y, terrain_z + 1.05 + body_bob])

            # Leg swing offsets in local sagittal direction.
            # Left/right legs move out of phase.
            stride_amp = 0.18
            lift_amp = 0.10

            left_swing = stride_amp * np.sin(phase)
            right_swing = stride_amp * np.sin(phase + np.pi)

            left_lift = lift_amp * max(0.0, np.sin(phase))
            right_lift = lift_amp * max(0.0, np.sin(phase + np.pi))

            # Local offsets from torso/root.
            left_leg_local = np.array([left_swing, 0.11, -0.35 + left_lift])
            right_leg_local = np.array([right_swing, -0.11, -0.35 + right_lift])

            left_arm_local = np.array([-right_swing * 0.7, 0.24, 0.10])
            right_arm_local = np.array([-left_swing * 0.7, -0.24, 0.10])

            head_local = np.array([0.0, 0.0, 0.45])

            # Convert offsets to world.
            head_pos = local_to_world(torso_pos, yaw, head_local)
            left_leg_pos = local_to_world(torso_pos, yaw, left_leg_local)
            right_leg_pos = local_to_world(torso_pos, yaw, right_leg_local)
            left_arm_pos = local_to_world(torso_pos, yaw, left_arm_local)
            right_arm_pos = local_to_world(torso_pos, yaw, right_arm_local)

            quat = yaw_to_quat(yaw)

            data.mocap_pos[torso_id] = torso_pos
            data.mocap_quat[torso_id] = quat

            data.mocap_pos[head_id] = head_pos
            data.mocap_quat[head_id] = quat

            data.mocap_pos[left_leg_id] = left_leg_pos
            data.mocap_quat[left_leg_id] = quat

            data.mocap_pos[right_leg_id] = right_leg_pos
            data.mocap_quat[right_leg_id] = quat

            data.mocap_pos[left_arm_id] = left_arm_pos
            data.mocap_quat[left_arm_id] = quat

            data.mocap_pos[right_arm_id] = right_arm_pos
            data.mocap_quat[right_arm_id] = quat

            mujoco.mj_step(model, data)
            viewer.sync()

            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()