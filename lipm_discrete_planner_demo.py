import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Discrete LIPM footstep planner demo
#
# Paper connection:
# The paper uses the Linear Inverted Pendulum Model (LIPM)
# for high-level bipedal CoM/footstep planning.
#
# State:
#   x_q = (p_q, v_q^loc, theta_q)
# where
#   p_q = (x_q, y_q, z_q)
#   v_q^loc = local sagittal velocity
#   theta_q = heading angle
#
# Control:
#   u_q = (u_f_q, u_delta_theta_q)
#
# This script implements the paper-style discrete LIPM update:
#
#   x_loc_{q+1} = x_loc_q + sinh(wT)/w v_loc_q
#                 + (1 - cosh(wT)) u_f_q
#
#   v_loc_{q+1} = cosh(wT) v_loc_q
#                 - w sinh(wT) u_f_q
#
# and maps the local sagittal displacement to global x-y motion.
# ============================================================


def true_terrain(x, y):
    """
    Same synthetic terrain used in the CP-safe planning demos.
    """
    return (
        0.18 * np.sin(0.8 * x)
        + 0.12 * np.cos(0.7 * y)
        + 0.08 * np.sin(1.5 * x + 0.6 * y)
        + 0.04 * np.cos(2.0 * x - 1.2 * y)
    )


def terrain_gradient(x, y, eps=1e-4):
    """
    Numerical terrain gradient.
    Later this will be replaced by GP mean gradient ∇mu(x,y).
    """
    dz_dx = (true_terrain(x + eps, y) - true_terrain(x - eps, y)) / (2.0 * eps)
    dz_dy = (true_terrain(x, y + eps) - true_terrain(x, y - eps)) / (2.0 * eps)
    return np.array([dz_dx, dz_dy])


def lipm_step_local(x_loc, v_loc, u_f, omega, T_step):
    """
    One-step local sagittal LIPM update.

    x_loc is the local sagittal CoM position relative to stance foot.
    v_loc is the local sagittal velocity.
    u_f is the sagittal foot placement relative to CoM.

    This follows the closed-form update used in the paper.
    """
    s = np.sinh(omega * T_step)
    c = np.cosh(omega * T_step)

    x_next_loc = x_loc + (s / omega) * v_loc + (1.0 - c) * u_f
    v_next_loc = c * v_loc - omega * s * u_f

    delta_x_loc = x_next_loc - x_loc

    return x_next_loc, v_next_loc, delta_x_loc


def global_lipm_update(state, control, params):
    """
    Paper-style global discrete LIPM update.

    state = [x, y, z, v_loc, theta]
    control = [u_f, delta_theta]
    """
    x, y, z, v_loc, theta = state
    u_f, delta_theta = control

    omega = params["omega"]
    T_step = params["T_step"]

    # We use x_loc = 0 at each stance switching instant.
    # This is a simplified convention for this demo.
    x_loc = 0.0

    x_next_loc, v_next_loc, delta_x_loc = lipm_step_local(
        x_loc=x_loc,
        v_loc=v_loc,
        u_f=u_f,
        omega=omega,
        T_step=T_step,
    )

    x_next = x + delta_x_loc * np.cos(theta)
    y_next = y + delta_x_loc * np.sin(theta)

    # Paper uses z update from terrain slope / GP terrain estimate.
    # Here we set z to the true terrain height for visualization.
    z_next = true_terrain(x_next, y_next)

    theta_next = theta + delta_theta

    return np.array([x_next, y_next, z_next, v_next_loc, theta_next])


def candidate_controls():
    """
    Candidate controls for the reduced-order footstep planner.

    u_f:
      sagittal foot placement relative to CoM.
      Larger u_f changes the next sagittal velocity.

    delta_theta:
      heading change.
    """
    u_f_candidates = np.array([-0.20, -0.10, 0.0, 0.10, 0.20, 0.30, 0.40])
    dtheta_candidates = np.deg2rad(np.array([-25.0, -12.5, 0.0, 12.5, 25.0]))

    controls = []
    for u_f in u_f_candidates:
        for dtheta in dtheta_candidates:
            controls.append(np.array([u_f, dtheta]))

    return controls


def control_is_feasible(next_state, bounds):
    x, y, z, v_loc, theta = next_state

    if not (bounds["x"][0] <= x <= bounds["x"][1]):
        return False

    if not (bounds["y"][0] <= y <= bounds["y"][1]):
        return False

    if not (bounds["v_loc"][0] <= v_loc <= bounds["v_loc"][1]):
        return False

    return True


def stage_cost(state, next_state, goal, control, weights):
    """
    Simplified MPC-style cost inspired by the paper:

      terminal/goal progress
      + terrain slope penalty
      + heading-to-goal penalty
      + control effort
    """
    x, y, z, v_loc, theta = state
    xn, yn, zn, vn, thetan = next_state
    u_f, dtheta = control

    p_next = np.array([xn, yn])
    p_goal = np.array(goal)

    dist_cost = weights["goal"] * np.linalg.norm(p_next - p_goal) ** 2

    grad = terrain_gradient(xn, yn)
    slope_cost = weights["slope"] * np.dot(grad, grad)

    theta_goal = np.arctan2(goal[1] - yn, goal[0] - xn)
    heading_error = wrap_angle(thetan - theta_goal)
    heading_cost = weights["heading"] * heading_error**2

    control_cost = weights["control"] * (u_f**2 + dtheta**2)

    # Encourage forward progress toward the goal.
    old_dist = np.linalg.norm(np.array([x, y]) - p_goal)
    new_dist = np.linalg.norm(p_next - p_goal)
    progress_reward = weights["progress"] * (old_dist - new_dist)

    return dist_cost + slope_cost + heading_cost + control_cost - progress_reward


def wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def plan_lipm_path(start_xy, goal_xy, params):
    """
    Greedy one-step LIPM planner.

    This is not yet full MPC over a horizon.
    It is the first paper-style dynamics demo:
    candidate controls are passed through the LIPM transition,
    then the best dynamically feasible next state is selected.
    """
    z0 = true_terrain(start_xy[0], start_xy[1])

    theta0 = np.arctan2(goal_xy[1] - start_xy[1], goal_xy[0] - start_xy[0])

    # state = [x, y, z, v_loc, theta]
    state = np.array([
        start_xy[0],
        start_xy[1],
        z0,
        params["v0"],
        theta0,
    ])

    states = [state.copy()]
    controls_used = []

    controls = candidate_controls()

    max_steps = params["max_steps"]
    goal_tol = params["goal_tol"]

    for k in range(max_steps):
        current_xy = state[:2]
        dist = np.linalg.norm(current_xy - goal_xy)

        if dist < goal_tol:
            print(f"Goal reached at step {k}.")
            break

        best_cost = np.inf
        best_next = None
        best_control = None

        for control in controls:
            next_state = global_lipm_update(state, control, params)

            if not control_is_feasible(next_state, params["bounds"]):
                continue

            cost = stage_cost(
                state=state,
                next_state=next_state,
                goal=goal_xy,
                control=control,
                weights=params["weights"],
            )

            if cost < best_cost:
                best_cost = cost
                best_next = next_state
                best_control = control

        if best_next is None:
            print(f"No feasible LIPM control found at step {k}.")
            break

        state = best_next
        states.append(state.copy())
        controls_used.append(best_control.copy())

    return np.array(states), np.array(controls_used)


def main():
    # ------------------------------------------------------------
    # LIPM parameters
    # ------------------------------------------------------------
    g = 9.81
    zH = 0.9
    omega = np.sqrt(g / zH)

    params = {
        "g": g,
        "zH": zH,
        "omega": omega,
        "T_step": 0.25,
        "v0": 0.65,
        "max_steps": 80,
        "goal_tol": 0.35,
        "bounds": {
            "x": (-5.0, 5.0),
            "y": (-5.0, 5.0),
            "v_loc": (0.05, 2.50),
        },
        "weights": {
            "goal": 1.0,
            "slope": 4.0,
            "heading": 0.5,
            "control": 0.05,
            "progress": 5.0,
        },
    }

    start = np.array([-4.3, -4.2])
    goal = np.array([4.3, 4.0])

    states, controls = plan_lipm_path(start, goal, params)

    final_dist = np.linalg.norm(states[-1, :2] - goal)

    print("Discrete LIPM planner demo completed.")
    print(f"omega              = {omega:.4f} rad/s")
    print(f"T_step             = {params['T_step']:.3f} s")
    print(f"number of steps    = {len(states) - 1}")
    print(f"final distance     = {final_dist:.4f} m")

    if len(controls) > 0:
        print(f"u_f range          = [{controls[:, 0].min():.3f}, {controls[:, 0].max():.3f}] m")
        print(
            "delta_theta range  = "
            f"[{np.rad2deg(controls[:, 1].min()):.2f}, "
            f"{np.rad2deg(controls[:, 1].max()):.2f}] deg"
        )

    # ------------------------------------------------------------
    # Terrain plot
    # ------------------------------------------------------------
    x_grid = np.linspace(-5, 5, 150)
    y_grid = np.linspace(-5, 5, 150)
    Xg, Yg = np.meshgrid(x_grid, y_grid)
    Zg = true_terrain(Xg, Yg)

    plt.figure(figsize=(8, 7))
    plt.contourf(Xg, Yg, Zg, levels=35)
    plt.colorbar(label="terrain height [m]")

    plt.plot(
        states[:, 0],
        states[:, 1],
        "o-",
        linewidth=2.0,
        markersize=4,
        label="LIPM planner path",
    )

    # heading arrows every few steps
    skip = max(1, len(states) // 12)
    for i in range(0, len(states), skip):
        x, y, z, v, theta = states[i]
        plt.arrow(
            x,
            y,
            0.25 * np.cos(theta),
            0.25 * np.sin(theta),
            head_width=0.08,
            alpha=0.7,
        )

    plt.scatter(start[0], start[1], s=100, marker="s", label="start")
    plt.scatter(goal[0], goal[1], s=140, marker="*", label="goal")

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Discrete LIPM footstep planner over terrain")
    plt.legend()
    plt.tight_layout()

    # ------------------------------------------------------------
    # State histories
    # ------------------------------------------------------------
    step_idx = np.arange(len(states))

    plt.figure(figsize=(8, 4.5))
    plt.plot(step_idx, states[:, 3], "o-")
    plt.xlabel("step index")
    plt.ylabel("local sagittal velocity v_loc [m/s]")
    plt.title("LIPM sagittal velocity across steps")
    plt.grid(True)
    plt.tight_layout()

    plt.figure(figsize=(8, 4.5))
    plt.plot(step_idx, np.rad2deg(states[:, 4]), "o-")
    plt.xlabel("step index")
    plt.ylabel("heading theta [deg]")
    plt.title("Heading evolution")
    plt.grid(True)
    plt.tight_layout()

    if len(controls) > 0:
        control_idx = np.arange(len(controls))

        plt.figure(figsize=(8, 4.5))
        plt.plot(control_idx, controls[:, 0], "o-")
        plt.xlabel("step index")
        plt.ylabel("u_f [m]")
        plt.title("Sagittal foot placement control")
        plt.grid(True)
        plt.tight_layout()

        plt.figure(figsize=(8, 4.5))
        plt.plot(control_idx, np.rad2deg(controls[:, 1]), "o-")
        plt.xlabel("step index")
        plt.ylabel("delta theta [deg]")
        plt.title("Heading-change control")
        plt.grid(True)
        plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()