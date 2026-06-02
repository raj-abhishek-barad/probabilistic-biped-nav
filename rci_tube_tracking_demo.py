import numpy as np
import matplotlib.pyplot as plt

from lipm_cp_safe_waypoint_mpc_fast import (
    build_gp_and_cp,
    plan_global_cp_safe_route,
    plan_waypoint_lipm_mpc,
    candidate_controls,
    true_terrain,
)


# ============================================================
# RCI tube / disturbed Aug-LIPM tracking demo
#
# Paper connection:
# The paper does not only plan safe footsteps. It also constructs
# a robust tube around the desired reduced-order trajectory and
# uses a CCM/flywheel torque law to keep the actual robot state
# near the desired LIPM trajectory under bounded disturbances.
#
# Here:
#   nominal trajectory = waypoint-guided LIPM-MPC path
#   actual trajectory  = disturbed Aug-LIPM dynamics
#   controller         = CCM-like flywheel torque
#   tube               = approximate robust invariant tube radius
# ============================================================


def build_nominal_plan():
    confidence = 0.85

    gp, C, X_train, X_cal = build_gp_and_cp(
        seed=4,
        n_samples=120,
        confidence=confidence,
    )

    g = 9.81
    zH = 0.9
    omega = np.sqrt(g / zH)

    start = np.array([-4.3, -4.2])
    goal = np.array([4.3, 4.0])

    delta_h_max = 0.18

    route = plan_global_cp_safe_route(
        start=start,
        goal=goal,
        gp=gp,
        C=C,
        delta_h_max=delta_h_max,
        step_length=0.45,
        n_heading_candidates=31,
        max_steps=160,
        goal_tol=0.35,
        x_bounds=(-5.0, 5.0),
        y_bounds=(-5.0, 5.0),
    )

    params = {
        "g": g,
        "zH": zH,
        "omega": omega,
        "T_step": 0.25,
        "v0": 0.65,
        "horizon": 2,
        "max_steps": 100,
        "goal_tol": 0.35,
        "lookahead_distance": 1.2,
        "bounds": {
            "x": (-5.0, 5.0),
            "y": (-5.0, 5.0),
            "v_loc": (0.05, 2.70),
        },
        "weights": {
            "waypoint": 8.0,
            "final_goal": 0.15,
            "progress": 8.0,
            "slope": 25.0,
            "height": 50.0,
            "heading": 1.0,
            "control": 0.02,
            "velocity": 0.4,
            "v_ref": 1.6,
        },
    }

    states, controls, active_waypoints, planned_sequences = plan_waypoint_lipm_mpc(
        start_xy=start,
        goal_xy=goal,
        route=route,
        gp=gp,
        C=C,
        delta_h_max=delta_h_max,
        params=params,
    )

    return {
        "gp": gp,
        "C": C,
        "confidence": confidence,
        "delta_h_max": delta_h_max,
        "route": route,
        "states": states,
        "controls": controls,
        "params": params,
        "start": start,
        "goal": goal,
    }


def cumulative_path_length(states):
    xy = states[:, :2]
    s = np.zeros(len(xy))

    for k in range(1, len(xy)):
        s[k] = s[k - 1] + np.linalg.norm(xy[k] - xy[k - 1])

    return s


def interpolate_nominal(t, t_nodes, s_nodes, v_nodes):
    s_ref = np.interp(t, t_nodes, s_nodes)
    v_ref = np.interp(t, t_nodes, v_nodes)
    return np.array([s_ref, v_ref])


def bounded_disturbance(t, w_bar):
    """
    Bounded terrain/model disturbance proxy.

    In the paper, terrain uncertainty and reduced-order mismatch
    enter as bounded disturbances. Here we use a deterministic
    bounded signal for repeatability.
    """
    return (
        0.55 * w_bar * np.sin(2.0 * np.pi * 0.9 * t)
        + 0.35 * w_bar * np.sin(2.0 * np.pi * 2.1 * t + 0.7)
    )


def simulate_tracking_with_rci_tube(plan):
    states = plan["states"]
    params = plan["params"]

    g = params["g"]
    omega = params["omega"]
    T_step = params["T_step"]

    # Mass proxy for simplified humanoid / reduced model.
    m = 50.0

    # Build nominal trajectory nodes.
    s_nodes = cumulative_path_length(states)
    v_nodes = states[:, 3]

    t_nodes = np.arange(len(states)) * T_step
    T_final = t_nodes[-1]

    # Error dynamics for Aug-LIPM:
    #
    # e = x_actual - x_ref
    #
    # e_dot = A e + B tau_y + Bw w
    #
    # This is the correct object for tube tracking.
    # Do NOT use global path coordinate s inside A @ x_actual.
    A = np.array([
        [0.0, 1.0],
        [omega**2, 0.0],
    ])

    B = np.array([
        [0.0],
        [-omega**2 / (m * g)],
    ])

    Bw = np.array([
        [0.0],
        [1.0],
    ])

    # CCM-like metric.
    M = np.array([
        [40.0, 4.0],
        [4.0, 8.0],
    ])

    # Larger gain because LIPM has an unstable open-loop mode.
    rho = 18000.0

    # Let the flywheel torque act strongly enough.
    tau_limit = 1200.0

    # Bounded disturbance.
    w_bar = 0.20

    # Approximate tube parameters.
    lambda_contraction = 2.2
    tube_gain = 0.45

    # Initial tracking error.
    e = np.array([
        -0.04,
        0.15,
    ])

    e0 = np.linalg.norm(e)

    tube_inf = tube_gain * w_bar / lambda_contraction

    dt = 0.002
    t_grid = np.arange(0.0, T_final + dt, dt)

    x_actual_hist = []
    xref_hist = []
    e_hist = []
    tube_hist = []
    tau_hist = []
    w_hist = []

    violations = 0

    for t in t_grid:
        x_ref = interpolate_nominal(t, t_nodes, s_nodes, v_nodes)

        # CCM-like torque based on tracking error.
        tau_y = -0.5 * rho * (B.T @ M @ e.reshape(2, 1)).item()
        tau_y = np.clip(tau_y, -tau_limit, tau_limit)

        w = bounded_disturbance(t, w_bar)

        # Corrected: propagate error dynamics, not absolute dynamics.
        e_dot = A @ e + B.flatten() * tau_y + Bw.flatten() * w

        e = e + dt * e_dot

        x_actual = x_ref + e

        err_norm = np.linalg.norm(e)

        tube_radius = tube_inf + max(0.0, e0 - tube_inf) * np.exp(
            -lambda_contraction * t
        )

        if err_norm > tube_radius:
            violations += 1

        x_actual_hist.append(x_actual.copy())
        xref_hist.append(x_ref.copy())
        e_hist.append(err_norm)
        tube_hist.append(tube_radius)
        tau_hist.append(tau_y)
        w_hist.append(w)

    x_actual_hist = np.array(x_actual_hist)
    xref_hist = np.array(xref_hist)
    e_hist = np.array(e_hist)
    tube_hist = np.array(tube_hist)
    tau_hist = np.array(tau_hist)
    w_hist = np.array(w_hist)

    violation_fraction = violations / len(t_grid)

    return {
        "t": t_grid,
        "x": x_actual_hist,
        "xref": xref_hist,
        "error": e_hist,
        "tube": tube_hist,
        "tau": tau_hist,
        "w": w_hist,
        "violation_fraction": violation_fraction,
        "s_nodes": s_nodes,
        "v_nodes": v_nodes,
        "t_nodes": t_nodes,
        "lambda_contraction": lambda_contraction,
        "w_bar": w_bar,
        "tube_inf": tube_inf,
    }

def main():
    print("\nBuilding nominal CP-safe waypoint LIPM-MPC plan...")
    plan = build_nominal_plan()

    print("\nSimulating disturbed Aug-LIPM tracking with RCI tube...")
    result = simulate_tracking_with_rci_tube(plan)

    states = plan["states"]
    final_dist = np.linalg.norm(states[-1, :2] - plan["goal"])

    print("\nRCI tube tracking demo completed.")
    print("=" * 72)
    print(f"nominal LIPM steps             = {len(states) - 1}")
    print(f"nominal final distance         = {final_dist:.4f} m")
    print(f"confidence level               = {plan['confidence'] * 100:.1f}%")
    print(f"conformal margin C             = {plan['C']:.5f} m")
    print(f"delta_h_max                    = {plan['delta_h_max']:.5f} m")
    print(f"disturbance bound w_bar        = {result['w_bar']:.4f} m/s^2")
    print(f"contraction rate lambda        = {result['lambda_contraction']:.4f}")
    print(f"steady tube radius estimate    = {result['tube_inf']:.5f}")
    print(f"max tracking error             = {np.max(result['error']):.5f}")
    print(f"final tracking error           = {result['error'][-1]:.5f}")
    print(f"tube violation fraction        = {100 * result['violation_fraction']:.2f}%")
    print(f"max |tau_y|                    = {np.max(np.abs(result['tau'])):.4f} N m")

    t = result["t"]

    # ------------------------------------------------------------
    # Plot 1: nominal vs actual path coordinate
    # ------------------------------------------------------------
    plt.figure(figsize=(8, 4.5))
    plt.plot(t, result["xref"][:, 0], label="nominal path coordinate s*(t)")
    plt.plot(t, result["x"][:, 0], label="actual disturbed s(t)")
    plt.xlabel("time [s]")
    plt.ylabel("path coordinate s [m]")
    plt.title("Aug-LIPM path-coordinate tracking")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # ------------------------------------------------------------
    # Plot 2: velocity tracking
    # ------------------------------------------------------------
    plt.figure(figsize=(8, 4.5))
    plt.plot(t, result["xref"][:, 1], label="nominal v*(t)")
    plt.plot(t, result["x"][:, 1], label="actual disturbed v(t)")
    plt.xlabel("time [s]")
    plt.ylabel("local sagittal velocity [m/s]")
    plt.title("Aug-LIPM velocity tracking")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # ------------------------------------------------------------
    # Plot 3: tube
    # ------------------------------------------------------------
    plt.figure(figsize=(8, 4.5))
    plt.plot(t, result["error"], label="tracking error ||x-x*||")
    plt.plot(t, result["tube"], "--", linewidth=2, label="RCI tube radius")
    plt.xlabel("time [s]")
    plt.ylabel("tracking error / tube radius")
    plt.title("Robust tube condition")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # ------------------------------------------------------------
    # Plot 4: torque
    # ------------------------------------------------------------
    plt.figure(figsize=(8, 4.5))
    plt.plot(t, result["tau"], label="CCM flywheel torque")
    plt.xlabel("time [s]")
    plt.ylabel("tau_y [N m]")
    plt.title("CCM flywheel torque command")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # ------------------------------------------------------------
    # Plot 5: disturbance
    # ------------------------------------------------------------
    plt.figure(figsize=(8, 4.5))
    plt.plot(t, result["w"], label="bounded disturbance w(t)")
    plt.xlabel("time [s]")
    plt.ylabel("disturbance acceleration [m/s²]")
    plt.title("Bounded terrain/model disturbance proxy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # ------------------------------------------------------------
    # Plot 6: nominal 2D terrain route for context
    # ------------------------------------------------------------
    x_grid = np.linspace(-5, 5, 150)
    y_grid = np.linspace(-5, 5, 150)
    Xg, Yg = np.meshgrid(x_grid, y_grid)
    Zg = true_terrain(Xg, Yg)

    plt.figure(figsize=(8, 7))
    plt.contourf(Xg, Yg, Zg, levels=35)
    plt.colorbar(label="true terrain height [m]")

    route = plan["route"]
    plt.plot(route[:, 0], route[:, 1], "--", label="global CP-safe route")
    plt.plot(states[:, 0], states[:, 1], "o-", label="nominal LIPM-MPC path")

    plt.scatter(plan["start"][0], plan["start"][1], s=100, marker="s", label="start")
    plt.scatter(plan["goal"][0], plan["goal"][1], s=140, marker="*", label="goal")

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Nominal CP-safe LIPM-MPC path used for tube tracking")
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()