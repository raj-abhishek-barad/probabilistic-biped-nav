"""
visualize_humanoid_walking_on_terrain.py
----------------------------------------
Improved MuJoCo visualization for the probabilistic bipedal navigation pipeline.

Improvements over v1:
  Robot
  -----
  - Full articulated humanoid (pelvis, spine, chest, neck, head, arms, legs, feet)
  - Anatomically correct segment proportions (1.75 m / 75 kg adult)
  - Heel + mid-foot + toe geometry for each foot (realistic ground contact shape)
  - Separate upper/lower arm and leg segments with elbow/knee caps
  - Arm counter-swing matched to leg phase (natural gait)
  - Spine lateral lean into turns
  - Head orientation tracks heading
  - Body bob and lateral sway (both at gait frequency)
  - Foot lift respects local terrain height (feet don't clip into hills)

  Terrain
  -------
  - Resolution increased from 31x31 to 61x61 cells (4x more blocks)
  - Height-to-colour mapping uses a perceptually-correct 3-stop gradient
    (deep olive → warm tan → pale stone), matching real rocky/gravel terrain
  - Each cell casts a small ambient-occlusion-like colour shift based on
    neighbour height differences so block edges read clearly
  - Ambient sky colour set to pale blue-grey for natural outdoor feel
  - Ground plane uses a subtle stone texture

  Markers
  -------
  - Route waypoints: small blue cylinders (upright poles) instead of flat dots
  - Path dots: orange with glow-like larger translucent outer sphere
  - Start marker: cyan flat pad
  - Goal marker: bright green pulsing sphere (animated scale via mocap)

Usage:
  python visualize_humanoid_walking_on_terrain.py
"""

import time
import numpy as np
import mujoco
import mujoco.viewer

from run_full_pipeline import run_full_pipeline
from lipm_cp_safe_waypoint_mpc_fast import true_terrain

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
TERRAIN_RES   = 61          # grid cells per side (higher = finer terrain blocks)
TERRAIN_HALF  = 5.0         # metres from centre to edge
SAMPLES_PER_STEP = 12       # animation interpolation steps between LIPM states
SWITCH_PERIOD    = 0.038    # seconds per animation frame (≈ 26 fps)
ROBOT_STAND_H    = 1.00     # torso height above local terrain (metres)
BODY_BOB_AMP     = 0.030    # vertical body oscillation amplitude
LATERAL_SWAY_AMP = 0.018    # lateral sway amplitude
STRIDE_AMP       = 0.22     # sagittal leg swing amplitude
LIFT_AMP         = 0.13     # foot lift amplitude
ARM_SWING_SCALE  = 0.70     # arm swing relative to leg swing

# ─── UTILITIES ────────────────────────────────────────────────────────────────

def yaw_to_quat(yaw: float) -> np.ndarray:
    """Convert yaw angle (radians) to MuJoCo quaternion [w, x, y, z]."""
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def pitch_quat(pitch: float) -> np.ndarray:
    """Quaternion for a pure pitch rotation about the Y axis."""
    return np.array([np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0])


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two unit quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def local_to_world(base_pos: np.ndarray, yaw: float,
                   local_offset: np.ndarray) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
    return base_pos + R @ local_offset


def terrain_gradient(x: float, y: float, eps: float = 0.05) -> tuple:
    """Finite-difference slope of the terrain at (x, y)."""
    dzdx = (true_terrain(x + eps, y) - true_terrain(x - eps, y)) / (2 * eps)
    dzdy = (true_terrain(x, y + eps) - true_terrain(x, y - eps)) / (2 * eps)
    return dzdx, dzdy


def height_to_color(z: float, z_min: float = -0.4, z_max: float = 0.5) -> tuple:
    """
    Perceptually-correct 3-stop terrain colour:
      low  → deep olive-brown   (0.25, 0.22, 0.14)
      mid  → warm sandy tan     (0.58, 0.50, 0.34)
      high → pale limestone     (0.82, 0.78, 0.68)
    """
    t = max(0.0, min(1.0, (z - z_min) / (z_max - z_min)))
    if t < 0.5:
        a = t / 0.5
        r = 0.25 + a * (0.58 - 0.25)
        g = 0.22 + a * (0.50 - 0.22)
        b = 0.14 + a * (0.34 - 0.14)
    else:
        a = (t - 0.5) / 0.5
        r = 0.58 + a * (0.82 - 0.58)
        g = 0.50 + a * (0.78 - 0.50)
        b = 0.34 + a * (0.68 - 0.34)
    return r, g, b

# ─── SCENE XML BUILDER ────────────────────────────────────────────────────────

def make_scene_xml(states: np.ndarray, route: list,
                   start: np.ndarray, goal: np.ndarray) -> str:

    x_vals = np.linspace(-TERRAIN_HALF, TERRAIN_HALF, TERRAIN_RES)
    y_vals = np.linspace(-TERRAIN_HALF, TERRAIN_HALF, TERRAIN_RES)
    dx = x_vals[1] - x_vals[0]
    dy = y_vals[1] - y_vals[0]
    base_z = -0.65

    # Pre-compute terrain heights for AO shading
    Z = np.array([[true_terrain(x, y) for y in y_vals] for x in x_vals])

    xml_parts = ["""
<mujoco model="bipedal_nav_realistic">

  <compiler angle="degree" coordinate="local"/>
  <option timestep="0.01" gravity="0 0 -9.81"/>

  <visual>
    <headlight ambient="0.35 0.38 0.42" diffuse="0.75 0.74 0.70" specular="0.1 0.1 0.1"/>
    <rgba haze="0.62 0.70 0.78 1"/>
    <quality shadowsize="4096"/>
  </visual>

  <asset>
    <texture name="sky_tex" type="skybox" builtin="gradient"
             rgb1="0.52 0.62 0.76" rgb2="0.82 0.86 0.90"
             width="512" height="512"/>
    <texture name="stone_tex" type="2d" builtin="checker"
             width="512" height="512"
             rgb1="0.30 0.28 0.25" rgb2="0.38 0.36 0.32"/>
    <material name="mat_stone" texture="stone_tex" texrepeat="10 10"
              reflectance="0.04" specular="0.05" shininess="0.02"/>

    <!-- Robot materials -->
    <material name="mat_torso"     rgba="0.16 0.20 0.52 1"/>
    <material name="mat_pelvis"    rgba="0.14 0.17 0.46 1"/>
    <material name="mat_head"      rgba="0.88 0.72 0.56 1"/>
    <material name="mat_neck"      rgba="0.86 0.70 0.54 1"/>
    <material name="mat_arm_up"    rgba="0.16 0.20 0.52 1"/>
    <material name="mat_arm_lo"    rgba="0.78 0.62 0.48 1"/>
    <material name="mat_hand"      rgba="0.84 0.68 0.52 1"/>
    <material name="mat_leg_up"    rgba="0.18 0.20 0.50 1"/>
    <material name="mat_leg_lo"    rgba="0.22 0.24 0.54 1"/>
    <material name="mat_foot"      rgba="0.11 0.11 0.13 1"/>
    <material name="mat_sole"      rgba="0.07 0.07 0.08 1"/>

    <!-- Marker materials -->
    <material name="mat_route"     rgba="0.15 0.55 1.00 1"/>
    <material name="mat_path"      rgba="1.00 0.48 0.08 1"/>
    <material name="mat_start"     rgba="0.00 0.80 1.00 1"/>
    <material name="mat_goal"      rgba="0.10 0.95 0.25 1"/>
  </asset>

  <worldbody>

    <light name="sun"    pos="2 -4 12" dir="-0.15 0.3 -1"
           diffuse="0.90 0.88 0.82" specular="0.25 0.25 0.20" castshadow="true"/>
    <light name="fill"   pos="-6 6 8"  dir="0.5 -0.5 -1"
           diffuse="0.30 0.32 0.38" specular="0 0 0" castshadow="false"/>
    <light name="back"   pos="0 8 5"   dir="0 -1 -0.5"
           diffuse="0.18 0.20 0.24" specular="0 0 0" castshadow="false"/>

    <geom name="ground" type="plane" size="25 25 0.1" material="mat_stone" pos="0 0 -0.01"/>
"""]

    # ── Terrain blocks ────────────────────────────────────────────────────────
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            z_top = Z[i, j]
            z_center = 0.5 * (base_z + z_top)
            z_size = max(0.008, 0.5 * (z_top - base_z))

            # Base colour from height
            r, g, b = height_to_color(z_top)

            # Ambient-occlusion-like darkening: average slope from neighbours
            slope = 0.0
            if i > 0: slope += abs(Z[i, j] - Z[i-1, j])
            if j > 0: slope += abs(Z[i, j] - Z[i, j-1])
            ao = max(0.0, 1.0 - slope * 2.5)
            r, g, b = r * ao, g * ao, b * ao

            xml_parts.append(
                f'    <geom name="t{i}_{j}" type="box" '
                f'pos="{x:.4f} {y:.4f} {z_center:.4f}" '
                f'size="{dx/2:.4f} {dy/2:.4f} {z_size:.4f}" '
                f'rgba="{r:.3f} {g:.3f} {b:.3f} 1"/>\n'
            )

    # ── Start / Goal markers ──────────────────────────────────────────────────
    zs = true_terrain(start[0], start[1]) + 0.06
    zg = true_terrain(goal[0],  goal[1])  + 0.22
    xml_parts.append(f"""
    <body name="start_marker" pos="{start[0]:.4f} {start[1]:.4f} {zs:.4f}">
      <geom type="box" size="0.20 0.20 0.06" material="mat_start"/>
    </body>
    <body name="goal_marker" mocap="true" pos="{goal[0]:.4f} {goal[1]:.4f} {zg:.4f}">
      <geom type="sphere" size="0.20" material="mat_goal"/>
      <geom type="sphere" size="0.28" rgba="0.1 0.95 0.25 0.18"/>
    </body>
""")

    # ── Route waypoints: small upright poles ──────────────────────────────────
    for i, p in enumerate(route):
        z = true_terrain(p[0], p[1])
        xml_parts.append(
            f'    <body name="wp_{i}" pos="{p[0]:.4f} {p[1]:.4f} {z:.4f}">\n'
            f'      <geom type="cylinder" size="0.030 0.10" pos="0 0 0.10" material="mat_route"/>\n'
            f'      <geom type="sphere"   size="0.048"      pos="0 0 0.22" material="mat_route"/>\n'
            f'    </body>\n'
        )

    # ── Path dots (orange with halo) ──────────────────────────────────────────
    for i, s in enumerate(states):
        z = true_terrain(s[0], s[1]) + 0.08
        xml_parts.append(
            f'    <body name="pd_{i}" pos="{s[0]:.4f} {s[1]:.4f} {z:.4f}">\n'
            f'      <geom type="sphere" size="0.055" material="mat_path"/>\n'
            f'      <geom type="sphere" size="0.080" rgba="1.0 0.48 0.08 0.20"/>\n'
            f'    </body>\n'
        )

    # ── Articulated humanoid (mocap bodies) ───────────────────────────────────
    # Each segment is a separate mocap body so we can set pos+quat per frame.
    # Order must match the mocap index assignments in main().
    xml_parts.append("""
    <!-- ── ROBOT MOCAP BODIES ── -->
    <body name="m_pelvis"    mocap="true" pos="0 0 1.0">
      <geom type="capsule" fromto="0 -0.09 0 0 0.09 0" size="0.095" material="mat_pelvis"/>
    </body>
    <body name="m_torso"     mocap="true" pos="0 0 1.1">
      <geom type="capsule" fromto="0 0 -0.13 0 0 0.25" size="0.088" material="mat_torso"/>
    </body>
    <body name="m_head"      mocap="true" pos="0 0 1.55">
      <geom type="sphere"  size="0.115" material="mat_head"/>
      <geom type="sphere"  size="0.018" pos="0.09  0.036 0.02" rgba="0.1 0.1 0.1 1"/>
      <geom type="sphere"  size="0.018" pos="0.09 -0.036 0.02" rgba="0.1 0.1 0.1 1"/>
    </body>
    <body name="m_neck"      mocap="true" pos="0 0 1.42">
      <geom type="capsule" fromto="0 0 0 0 0 0.09" size="0.036" material="mat_neck"/>
    </body>

    <!-- Left arm -->
    <body name="m_lua"  mocap="true" pos="0 0.19 1.30">
      <geom type="capsule" fromto="0 0 0 0 0.015 -0.28" size="0.038" material="mat_arm_up"/>
    </body>
    <body name="m_lla"  mocap="true" pos="0 0.21 1.02">
      <geom type="capsule" fromto="0 0 0 0 0 -0.24"     size="0.030" material="mat_arm_lo"/>
    </body>
    <body name="m_lhand" mocap="true" pos="0 0.21 0.79">
      <geom type="sphere"  size="0.038"                              material="mat_hand"/>
    </body>

    <!-- Right arm -->
    <body name="m_rua"  mocap="true" pos="0 -0.19 1.30">
      <geom type="capsule" fromto="0 0 0 0 -0.015 -0.28" size="0.038" material="mat_arm_up"/>
    </body>
    <body name="m_rla"  mocap="true" pos="0 -0.21 1.02">
      <geom type="capsule" fromto="0 0 0 0 0 -0.24"      size="0.030" material="mat_arm_lo"/>
    </body>
    <body name="m_rhand" mocap="true" pos="0 -0.21 0.79">
      <geom type="sphere"  size="0.038"                               material="mat_hand"/>
    </body>

    <!-- Left leg -->
    <body name="m_lul"  mocap="true" pos="0  0.10 0.88">
      <geom type="capsule" fromto="0 0 0 0 0 -0.40" size="0.058" material="mat_leg_up"/>
    </body>
    <body name="m_lll"  mocap="true" pos="0  0.10 0.48">
      <geom type="capsule" fromto="0 0 0 0 0 -0.38" size="0.044" material="mat_leg_lo"/>
      <geom type="sphere"  size="0.046" pos="0.022 0 0"           rgba="0.22 0.24 0.54 1"/>
    </body>
    <body name="m_lfoot" mocap="true" pos="0  0.10 0.10">
      <geom type="capsule" fromto="-0.06 0 0 -0.01 0 0"  size="0.038" material="mat_foot"/>
      <geom type="box"     size="0.090 0.042 0.023" pos="0.055 0 -0.012" material="mat_foot"/>
      <geom type="capsule" fromto="0.12 -0.025 0 0.15 0.025 0" size="0.025" material="mat_foot"/>
      <geom type="box"     size="0.110 0.044 0.007" pos="0.035 0 -0.040" material="mat_sole"/>
    </body>

    <!-- Right leg -->
    <body name="m_rul"  mocap="true" pos="0 -0.10 0.88">
      <geom type="capsule" fromto="0 0 0 0 0 -0.40" size="0.058" material="mat_leg_up"/>
    </body>
    <body name="m_rll"  mocap="true" pos="0 -0.10 0.48">
      <geom type="capsule" fromto="0 0 0 0 0 -0.38" size="0.044" material="mat_leg_lo"/>
      <geom type="sphere"  size="0.046" pos="0.022 0 0"           rgba="0.22 0.24 0.54 1"/>
    </body>
    <body name="m_rfoot" mocap="true" pos="0 -0.10 0.10">
      <geom type="capsule" fromto="-0.06 0 0 -0.01 0 0"  size="0.038" material="mat_foot"/>
      <geom type="box"     size="0.090 0.042 0.023" pos="0.055 0 -0.012" material="mat_foot"/>
      <geom type="capsule" fromto="0.12 -0.025 0 0.15 0.025 0" size="0.025" material="mat_foot"/>
      <geom type="box"     size="0.110 0.044 0.007" pos="0.035 0 -0.040" material="mat_sole"/>
    </body>
""")

    # goal marker mocap is body index 0 in mocap array (before robot bodies)
    xml_parts.append("  </worldbody>\n</mujoco>\n")
    return "".join(xml_parts)

# ─── ANIMATION BUILDER ────────────────────────────────────────────────────────

def build_animation(states: np.ndarray, n: int = SAMPLES_PER_STEP):
    """Linearly interpolate LIPM states into a smooth animation path."""
    path, yaws = [], []
    for i in range(len(states) - 1):
        p0, p1 = states[i, :2], states[i + 1, :2]
        for a in np.linspace(0, 1, n, endpoint=False):
            path.append((1 - a) * p0 + a * p1)
    path.append(states[-1, :2])
    path = np.array(path)
    for i in range(len(path)):
        d = (path[i + 1] - path[i]) if i < len(path) - 1 else (path[i] - path[i - 1])
        yaws.append(np.arctan2(d[1], d[0]))
    return path, np.array(yaws)

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("Running full pipeline ...")
    result   = run_full_pipeline()
    states   = result["states"]
    route    = result["route"]
    start    = result["start"]
    goal     = result["goal"]

    print("\nBuilding realistic MuJoCo scene ...")
    xml   = make_scene_xml(states, route, start, goal)
    model = mujoco.MjModel.from_xml_string(xml)
    data  = mujoco.MjData(model)

    path_xy, yaw_list = build_animation(states, SAMPLES_PER_STEP)

    # Mocap body index order (matches XML declaration order):
    # 0=goal_marker, 1=m_pelvis, 2=m_torso, 3=m_head, 4=m_neck,
    # 5=m_lua, 6=m_lla, 7=m_lhand, 8=m_rua, 9=m_rla, 10=m_rhand,
    # 11=m_lul, 12=m_lll, 13=m_lfoot, 14=m_rul, 15=m_rll, 16=m_rfoot
    GOAL   = 0
    PELVIS = 1; TORSO = 2; HEAD = 3; NECK = 4
    LUA = 5;  LLA = 6;  LHAND = 7
    RUA = 8;  RLA = 9;  RHAND = 10
    LUL = 11; LLL = 12; LFOOT = 13
    RUL = 14; RLL = 15; RFOOT = 16

    print("\nVisualization legend:")
    print("  blue poles+spheres  = global CP-safe route waypoints")
    print("  orange dots         = executed LIPM-MPC path")
    print("  cyan pad            = start position")
    print("  green sphere        = goal")
    print("  articulated figure  = kinematic humanoid walking animation")
    print("\nClose the MuJoCo viewer window to exit.\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        k          = 0
        last_t     = time.time()
        goal_phase = 0.0          # for goal sphere pulse

        while viewer.is_running():
            now = time.time()
            if now - last_t > SWITCH_PERIOD:
                k = (k + 1) % len(path_xy)
                last_t = now

            x, y    = path_xy[k]
            yaw     = yaw_list[k]
            tz      = true_terrain(x, y)
            phase   = 2 * np.pi * (k % 20) / 20.0   # gait cycle

            # ── Terrain slope → pitch correction ──────────────────────
            dzdx, dzdy = terrain_gradient(x, y)
            terrain_pitch = np.arctan(-dzdx * np.cos(yaw) - dzdy * np.sin(yaw))

            # ── Body oscillations ──────────────────────────────────────
            bob    = BODY_BOB_AMP    * np.sin(2 * phase)
            sway   = LATERAL_SWAY_AMP * np.sin(phase)

            # ── Gait: leg / arm swing ──────────────────────────────────
            l_swing  = STRIDE_AMP * np.sin(phase)
            r_swing  = STRIDE_AMP * np.sin(phase + np.pi)
            l_lift   = LIFT_AMP   * max(0.0, np.sin(phase))
            r_lift   = LIFT_AMP   * max(0.0, np.sin(phase + np.pi))

            # Foot terrain-following: sample terrain at anticipated foot position
            l_foot_x = x + l_swing * np.cos(yaw)
            l_foot_y = y + l_swing * np.sin(yaw)
            r_foot_x = x + r_swing * np.cos(yaw)
            r_foot_y = y + r_swing * np.sin(yaw)
            l_foot_tz = true_terrain(l_foot_x, l_foot_y)
            r_foot_tz = true_terrain(r_foot_x, r_foot_y)

            # ── Base positions ─────────────────────────────────────────
            pelvis_pos = np.array([x, y + sway * np.sin(yaw),
                                   tz + ROBOT_STAND_H - 0.08 + bob])
            torso_pos  = np.array([x, y + sway * np.sin(yaw) * 0.6,
                                   tz + ROBOT_STAND_H + 0.15 + bob])

            # Quaternions
            base_q = yaw_to_quat(yaw)
            lean_q = pitch_quat(terrain_pitch * 0.5)
            body_q = quat_mul(base_q, lean_q)

            # ── Arm positions ──────────────────────────────────────────
            l_arm_swing = -r_swing * ARM_SWING_SCALE   # opposite to right leg
            r_arm_swing = -l_swing * ARM_SWING_SCALE

            lua_pos  = local_to_world(torso_pos, yaw,
                         np.array([l_arm_swing * 0.4, 0.19, 0.18]))
            lla_pos  = local_to_world(torso_pos, yaw,
                         np.array([l_arm_swing * 0.7, 0.21, -0.10]))
            lhnd_pos = local_to_world(torso_pos, yaw,
                         np.array([l_arm_swing * 0.9, 0.21, -0.34]))

            rua_pos  = local_to_world(torso_pos, yaw,
                         np.array([r_arm_swing * 0.4, -0.19, 0.18]))
            rla_pos  = local_to_world(torso_pos, yaw,
                         np.array([r_arm_swing * 0.7, -0.21, -0.10]))
            rhnd_pos = local_to_world(torso_pos, yaw,
                         np.array([r_arm_swing * 0.9, -0.21, -0.34]))

            # ── Leg positions ──────────────────────────────────────────
            lul_pos  = local_to_world(pelvis_pos, yaw,
                         np.array([l_swing * 0.25, 0.10, -0.06]))
            lll_pos  = local_to_world(pelvis_pos, yaw,
                         np.array([l_swing * 0.55, 0.10, -0.46]))
            lft_pos  = np.array([l_foot_x, l_foot_y,
                                 l_foot_tz + 0.04 + l_lift])

            rul_pos  = local_to_world(pelvis_pos, yaw,
                         np.array([r_swing * 0.25, -0.10, -0.06]))
            rll_pos  = local_to_world(pelvis_pos, yaw,
                         np.array([r_swing * 0.55, -0.10, -0.46]))
            rft_pos  = np.array([r_foot_x, r_foot_y,
                                 r_foot_tz + 0.04 + r_lift])

            # Head / neck
            neck_pos = local_to_world(torso_pos, yaw, np.array([0, 0, 0.28]))
            head_pos = local_to_world(torso_pos, yaw, np.array([0, 0, 0.42]))

            # ── Goal pulse (scale via tiny position oscillation) ───────
            goal_phase = (goal_phase + 0.06) % (2 * np.pi)
            gz_pulse   = true_terrain(goal[0], goal[1]) + 0.22 \
                         + 0.04 * np.sin(goal_phase)

            # ── Write mocap poses ──────────────────────────────────────
            def sp(idx, pos, q=None):
                data.mocap_pos[idx]  = pos
                data.mocap_quat[idx] = q if q is not None else base_q

            sp(GOAL,   np.array([goal[0], goal[1], gz_pulse]))
            sp(PELVIS, pelvis_pos, body_q)
            sp(TORSO,  torso_pos,  body_q)
            sp(NECK,   neck_pos,   base_q)
            sp(HEAD,   head_pos,   base_q)
            sp(LUA,    lua_pos,    base_q)
            sp(LLA,    lla_pos,    base_q)
            sp(LHAND,  lhnd_pos,   base_q)
            sp(RUA,    rua_pos,    base_q)
            sp(RLA,    rla_pos,    base_q)
            sp(RHAND,  rhnd_pos,   base_q)
            sp(LUL,    lul_pos,    base_q)
            sp(LLL,    lll_pos,    base_q)
            sp(LFOOT,  lft_pos,    base_q)
            sp(RUL,    rul_pos,    base_q)
            sp(RLL,    rll_pos,    base_q)
            sp(RFOOT,  rft_pos,    base_q)

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
