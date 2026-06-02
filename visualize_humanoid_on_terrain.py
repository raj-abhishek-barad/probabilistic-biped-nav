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
<mujoco model="humanoid_pipeline_visualization">
  <compiler angle="degree" coordinate="local"/>
  <option timestep="0.01" gravity="0 0 -9.81"/>

  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.12 0.12 0.12" rgb2="0.22 0.22 0.22"/>
    <material name="robot_blue" rgba="0.2 0.45 1.0 1"/>
    <material name="head_mat" rgba="0.9 0.75 0.55 1"/>
    <material name="route_mat" rgba="0.1 0.45 1.0 1"/>
    <material name="path_mat" rgba="1.0 0.45 0.05 1"/>
    <material name="start_mat" rgba="0.0 0.7 1.0 1"/>
    <material name="goal_mat" rgba="0.0 0.9 0.1 1"/>
  </asset>

  <worldbody>
    <light pos="0 0 8" dir="0 0 -1"/>
"""

    # ------------------------------------------------------------
    # Uneven terrain as many small boxes.
    # This is a visual terrain approximation of z = true_terrain(x,y).
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

            # simple height-based color
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

    # ------------------------------------------------------------
    # Start and goal markers
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # Global CP-safe route markers
    # ------------------------------------------------------------
    for i, p in enumerate(route):
        z = true_terrain(p[0], p[1]) + 0.08
        xml += f"""
    <body name="route_{i}" pos="{p[0]:.4f} {p[1]:.4f} {z:.4f}">
      <geom type="sphere" size="0.045" material="route_mat"/>
    </body>
"""

    # ------------------------------------------------------------
    # Executed LIPM-MPC path markers
    # ------------------------------------------------------------
    for i, s in enumerate(states):
        x, y = s[0], s[1]
        z = true_terrain(x, y) + 0.14
        xml += f"""
    <body name="path_{i}" pos="{x:.4f} {y:.4f} {z:.4f}">
      <geom type="sphere" size="0.055" material="path_mat"/>
    </body>
"""

    # ------------------------------------------------------------
    # Kinematic humanoid body.
    # This is moved using mocap_pos/mocap_quat along the LIPM path.
    # ------------------------------------------------------------
    xml += """
    <body name="humanoid_root" mocap="true" pos="0 0 1.0">
      <geom name="torso" type="capsule"
            fromto="0 0 -0.25 0 0 0.25"
            size="0.13" material="robot_blue"/>

      <body name="head" pos="0 0 0.43">
        <geom type="sphere" size="0.12" material="head_mat"/>
      </body>

      <body name="left_leg" pos="0 0.10 -0.28">
        <geom type="capsule" fromto="0 0 0 0 0 -0.75"
              size="0.045" rgba="0.1 0.8 0.35 1"/>
      </body>

      <body name="right_leg" pos="0 -0.10 -0.28">
        <geom type="capsule" fromto="0 0 0 0 0 -0.75"
              size="0.045" rgba="0.9 0.25 0.25 1"/>
      </body>

      <body name="left_arm" pos="0 0.20 0.15">
        <geom type="capsule" fromto="0 0 0 0 0 -0.45"
              size="0.035" rgba="0.2 0.8 0.9 1"/>
      </body>

      <body name="right_arm" pos="0 -0.20 0.15">
        <geom type="capsule" fromto="0 0 0 0 0 -0.45"
              size="0.035" rgba="0.2 0.8 0.9 1"/>
      </body>
    </body>
"""

    xml += """
  </worldbody>
</mujoco>
"""
    return xml


def build_animation(states):
    path_xyz = []
    yaw_list = []

    for i, s in enumerate(states):
        x, y = s[0], s[1]
        terrain_z = true_terrain(x, y)

        # Humanoid root height above terrain.
        z = terrain_z + 1.05

        # small walking-like vertical bob
        z += 0.04 * np.sin(2.0 * np.pi * i / 4.0)

        path_xyz.append([x, y, z])

        if i < len(states) - 1:
            dx = states[i + 1, 0] - states[i, 0]
            dy = states[i + 1, 1] - states[i, 1]
        else:
            dx = states[i, 0] - states[i - 1, 0]
            dy = states[i, 1] - states[i - 1, 1]

        yaw = np.arctan2(dy, dx)
        yaw_list.append(yaw)

    return np.array(path_xyz), np.array(yaw_list)


def main():
    print("Running full pipeline...")
    result = run_full_pipeline()

    states = result["states"]
    route = result["route"]
    start = result["start"]
    goal = result["goal"]

    print("\nBuilding MuJoCo uneven-terrain humanoid visualization...")
    xml = make_scene_xml(states, route, start, goal)

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    path_xyz, yaw_list = build_animation(states)

    print("\nVisualization legend:")
    print("  uneven boxes  = terrain height map")
    print("  blue dots     = global CP-safe route")
    print("  orange dots   = executed LIPM-MPC path")
    print("  humanoid      = kinematic robot visualization")
    print("  green sphere  = goal")
    print("\nClose the MuJoCo viewer to stop.")

    mocap_id = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        k = 0
        last_switch = time.time()
        switch_period = 0.18

        while viewer.is_running():
            now = time.time()

            if now - last_switch > switch_period:
                k = (k + 1) % len(path_xyz)
                last_switch = now

            data.mocap_pos[mocap_id] = path_xyz[k]
            data.mocap_quat[mocap_id] = yaw_to_quat(yaw_list[k])

            mujoco.mj_step(model, data)
            viewer.sync()

            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()