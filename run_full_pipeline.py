import numpy as np
import matplotlib.pyplot as plt

from lipm_cp_safe_waypoint_mpc_fast import (
    build_gp_and_cp,
    plan_global_cp_safe_route,
    plan_waypoint_lipm_mpc,
    true_terrain,
    candidate_controls,
)

from rci_tube_certified import simulate_tracking_with_rci_tube


def evaluate_true_step_safety(states, delta_h_max):
    if len(states) < 2:
        return 0, 0, np.array([])

    violations = []
    height_changes = []

    for q in range(len(states) - 1):
        p0 = states[q, :2]
        p1 = states[q + 1, :2]

        z0 = true_terrain(p0[0], p0[1])
        z1 = true_terrain(p1[0], p1[1])

        dh = abs(z1 - z0)
        height_changes.append(dh)

        if dh > delta_h_max:
            violations.append(q)

    return len(violations), len(states) - 1, np.array(height_changes)


def run_full_pipeline():
    # ============================================================
    # 1. Build GP + conformal prediction terrain model
    # ============================================================
    confidence = 0.85

    gp, C, X_train, X_cal = build_gp_and_cp(
        seed=4,
        n_samples=120,
        confidence=confidence,
    )

    # ============================================================
    # 2. Plan global CP-safe route
    # ============================================================
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

    # ============================================================
    # 3. Run waypoint-guided LIPM-MPC
    # ============================================================
    g = 9.81
    zH = 0.9
    omega = np.sqrt(g / zH)

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

    # ============================================================
    # 4. Evaluate terrain safety
    # ============================================================
    terrain_violations, terrain_steps, dh = evaluate_true_step_safety(
        states,
        delta_h_max,
    )

    final_distance = np.linalg.norm(states[-1, :2] - goal)
    reached_goal = final_distance < params["goal_tol"]

    # ============================================================
    # 5. Run RCI tube tracking
    # ============================================================
    plan = {
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

    tube_result = simulate_tracking_with_rci_tube(plan)

    # ============================================================
    # 6. Report final metrics
    # ============================================================
    print("\nFULL PIPELINE RESULT")
    print("=" * 80)

    print("\n[1] Terrain uncertainty")
    print(f"confidence level                 = {confidence * 100:.1f}%")
    print(f"conformal margin C               = {C:.5f} m")
    print(f"delta_h_max                      = {delta_h_max:.5f} m")

    print("\n[2] Global CP-safe route")
    print(f"global route points              = {len(route)}")

    print("\n[3] Waypoint-guided LIPM-MPC")
    print(f"LIPM steps                       = {len(states) - 1}")
    print(f"goal reached                     = {reached_goal}")
    print(f"final distance to goal           = {final_distance:.4f} m")
    print(f"horizon                          = {params['horizon']}")
    print(f"candidate controls               = {len(candidate_controls())}")

    if len(controls) > 0:
        print(f"u_f range                        = [{controls[:, 0].min():.3f}, {controls[:, 0].max():.3f}] m")
        print(
            "delta_theta range                = "
            f"[{np.rad2deg(controls[:, 1].min()):.2f}, "
            f"{np.rad2deg(controls[:, 1].max()):.2f}] deg"
        )

    print("\n[4] True terrain safety")
    print(f"true terrain violations          = {terrain_violations}")
    print(f"max true |delta h|               = {np.max(dh) if len(dh) else 0.0:.5f} m")
    print(f"mean true |delta h|              = {np.mean(dh) if len(dh) else 0.0:.5f} m")

    print("\n[5] RCI / CCM tracking")
    print(f"disturbance bound w_bar          = {tube_result['w_bar']:.4f} m/s^2")
    print(f"contraction rate lambda          = {tube_result['lambda_contraction']:.4f}")
    print(f"steady tube radius estimate      = {tube_result['tube_inf']:.5f}")
    print(f"max tracking error               = {np.max(tube_result['error']):.5f}")
    print(f"final tracking error             = {tube_result['error'][-1]:.5f}")
    print(f"tube violation fraction          = {100 * tube_result['violation_fraction']:.2f}%")
    print(f"max |tau_y|                      = {np.max(np.abs(tube_result['tau'])):.4f} N m")

    return {
        "gp": gp,
        "C": C,
        "route": route,
        "states": states,
        "controls": controls,
        "active_waypoints": active_waypoints,
        "planned_sequences": planned_sequences,
        "terrain_height_changes": dh,
        "tube_result": tube_result,
        "params": params,
        "start": start,
        "goal": goal,
        "delta_h_max": delta_h_max,
        "confidence": confidence,
    }


def plot_full_pipeline(result):
    route = result["route"]
    states = result["states"]
    active_waypoints = result["active_waypoints"]
    planned_sequences = result["planned_sequences"]
    dh = result["terrain_height_changes"]
    tube_result = result["tube_result"]
    start = result["start"]
    goal = result["goal"]
    delta_h_max = result["delta_h_max"]

    # ------------------------------------------------------------
    # Plot 1: terrain route and LIPM-MPC path
    # ------------------------------------------------------------
    x_grid = np.linspace(-5, 5, 150)
    y_grid = np.linspace(-5, 5, 150)
    Xg, Yg = np.meshgrid(x_grid, y_grid)
    Zg = true_terrain(Xg, Yg)

    plt.figure(figsize=(8, 7))
    plt.contourf(Xg, Yg, Zg, levels=35)
    plt.colorbar(label="true terrain height [m]")

    plt.plot(
        route[:, 0],
        route[:, 1],
        "--",
        linewidth=1.8,
        label="global CP-safe route",
    )

    plt.plot(
        states[:, 0],
        states[:, 1],
        "o-",
        linewidth=2.2,
        markersize=4,
        label="waypoint-guided LIPM-MPC path",
    )

    if len(active_waypoints) > 0:
        plt.scatter(
            active_waypoints[:, 0],
            active_waypoints[:, 1],
            s=25,
            marker="x",
            label="active local waypoints",
        )

    if len(planned_sequences) > 0:
        stride = max(1, len(planned_sequences) // 10)
        for seq in planned_sequences[::stride]:
            plt.plot(seq[:, 0], seq[:, 1], "-", linewidth=1.0, alpha=0.35)

    plt.scatter(start[0], start[1], s=100, marker="s", label="start")
    plt.scatter(goal[0], goal[1], s=140, marker="*", label="goal")

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Full pipeline: CP-safe route + LIPM-MPC path")
    plt.legend()
    plt.tight_layout()

    # ------------------------------------------------------------
    # Plot 2: true terrain height changes
    # ------------------------------------------------------------
    plt.figure(figsize=(8, 4.5))
    plt.plot(dh, "o-", label="true |Δh|")
    plt.axhline(delta_h_max, linestyle="--", linewidth=2, label="Δh max")
    plt.xlabel("step index")
    plt.ylabel("true terrain height change [m]")
    plt.title("True terrain safety along executed path")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # ------------------------------------------------------------
    # Plot 3: LIPM local velocity
    # ------------------------------------------------------------
    plt.figure(figsize=(8, 4.5))
    plt.plot(states[:, 3], "o-")
    plt.xlabel("step index")
    plt.ylabel("v_loc [m/s]")
    plt.title("Local sagittal velocity from LIPM-MPC")
    plt.grid(True)
    plt.tight_layout()

    # ------------------------------------------------------------
    # Plot 4: RCI tube
    # ------------------------------------------------------------
    t = tube_result["t"]

    plt.figure(figsize=(8, 4.5))
    plt.plot(t, tube_result["error"], label="tracking error ||x-x*||")
    plt.plot(t, tube_result["tube"], "--", linewidth=2, label="RCI tube radius")
    plt.xlabel("time [s]")
    plt.ylabel("tracking error / tube radius")
    plt.title("RCI tube condition for disturbed Aug-LIPM tracking")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # ------------------------------------------------------------
    # Plot 5: CCM torque
    # ------------------------------------------------------------
    plt.figure(figsize=(8, 4.5))
    plt.plot(t, tube_result["tau"])
    plt.xlabel("time [s]")
    plt.ylabel("tau_y [N m]")
    plt.title("CCM flywheel torque command")
    plt.grid(True)
    plt.tight_layout()

    plt.show()


def main():
    result = run_full_pipeline()
    plot_full_pipeline(result)


if __name__ == "__main__":
    main()