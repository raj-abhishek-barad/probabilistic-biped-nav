import time
import numpy as np
import mujoco
import mujoco.viewer

from run_full_pipeline import run_full_pipeline
from lipm_cp_safe_waypoint_mpc_fast import true_terrain


def make_mujoco_scene(states, route, start, goal):
    """
    Build a MuJoCo XML scene that visualizes:
      - ground plane
      - global CP-safe route
      - executed LIPM-MPC path
      - moving CoM proxy sphere
      - start and goal markers

    This does not yet simulate a full walking humanoid.
    It visualizes the planned reduced-order motion in MuJoCo.
    """

    xml = """
<mujoco model="pipeline_visualization">
  <compiler angle="degree" coordinate="local"/>
  <option timestep="0.01" gravity="0 0 -9.81"/>

  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.1 0.1 0.1" rgb2="0.25 0.25 0.25"/>
    <material name="gridmat" texture="grid" texrepeat="10 10" reflectance="0.2"/>
    <material name="route_mat" rgba="0.1 0.4 1.0 1"/>
    <material name="path_mat" rgba="1.0 0.45 0.05 1"/>
    <material name="com_mat" rgba="1.0 0.1 0.1 1"/>
    <material name="start_mat" rgba="0.0 0.7 1.0 1"/>
    <material name="goal_mat" rgba="0.0 0.9 0.1 1"/>
  </asset>

  <worldbody>
    <light pos="0 0 6" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="6 6 0.1" material="gridmat"/>

    <body name="com_proxy" mocap="true" pos="0 0 0.4">
      <geom name="com_proxy_geom" type="sphere" size="0.12" material="com_mat"/>
    </body>
"""

    # Start marker
    z_start = true_terrain(start[0], start[1]) + 0.05
    xml += f"""
    <body name="start_marker" pos="{start[0]} {start[1]} {z_start}">
      <geom type="box" size="0.16 0.16 0.08" material="start_mat"/>
    </body>
"""

    # Goal marker
    z_goal = true_terrain(goal[0], goal[1]) + 0.08
    xml += f"""
    <body name="goal_marker" pos="{goal[0]} {goal[1]} {z_goal}">
      <geom type="sphere" size="0.18" material="goal_mat"/>
    </body>
"""

    # Global CP-safe route markers
    for i, p in enumerate(route):
        z = true_terrain(p[0], p[1]) + 0.04
        xml += f"""
    <body name="route_{i}" pos="{p[0]} {p[1]} {z}">
      <geom type="sphere" size="0.045" material="route_mat"/>
    </body>
"""

    # Executed LIPM-MPC path markers
    for i, s in enumerate(states):
        x, y = s[0], s[1]
        z = true_terrain(x, y) + 0.08
        xml += f"""
    <body name="path_{i}" pos="{x} {y} {z}">
      <geom type="sphere" size="0.06" material="path_mat"/>
    </body>
"""

    xml += """
  </worldbody>
</mujoco>
"""
    return xml


def main():
    print("Running full pipeline first...")
    result = run_full_pipeline()

    states = result["states"]
    route = result["route"]
    start = result["start"]
    goal = result["goal"]

    print("\nOpening MuJoCo visualization...")
    xml = make_mujoco_scene(states, route, start, goal)

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # Mocap body id for moving red CoM proxy.
    mocap_id = 0

    # Build animation points from executed LIPM-MPC states.
    path_xyz = []
    for s in states:
        x, y = s[0], s[1]
        z = true_terrain(x, y) + 0.35
        path_xyz.append([x, y, z])

    path_xyz = np.array(path_xyz)

    print("MuJoCo visualization ready.")
    print("Blue dots  = global CP-safe route")
    print("Orange dots = executed LIPM-MPC path")
    print("Red sphere = moving CoM / robot proxy")
    print("Green sphere = goal")
    print("Close the viewer to stop.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        k = 0
        last_switch = time.time()
        switch_period = 0.20

        while viewer.is_running():
            now = time.time()

            if now - last_switch > switch_period:
                k = (k + 1) % len(path_xyz)
                last_switch = now

            data.mocap_pos[mocap_id] = path_xyz[k]
            data.mocap_quat[mocap_id] = np.array([1.0, 0.0, 0.0, 0.0])

            mujoco.mj_step(model, data)
            viewer.sync()

            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()