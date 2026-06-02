import itertools
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Global CP-safe waypoint route + local LIPM-MPC tracking
#
# Pipeline:
#   1. Build GP + conformal prediction terrain model
#   2. Generate a global CP-safe waypoint route over terrain
#   3. Run local LIPM-MPC toward a nearby waypoint, not directly
#      toward the final goal
#
# This fixes the previous local-planner failure where the LIPM-MPC
# repeatedly tried to cross the high terrain region directly.
# ============================================================


# ------------------------------------------------------------
# Terrain and GP
# ------------------------------------------------------------

def true_terrain(x, y):
    return (
        0.18 * np.sin(0.8 * x)
        + 0.12 * np.cos(0.7 * y)
        + 0.08 * np.sin(1.5 * x + 0.6 * y)
        + 0.04 * np.cos(2.0 * x - 1.2 * y)
    )


def rbf_kernel(X1, X2, length_scale=1.2, sigma_f=1.0):
    X1_sq = np.sum(X1**2, axis=1).reshape(-1, 1)
    X2_sq = np.sum(X2**2, axis=1).reshape(1, -1)
    sqdist = X1_sq + X2_sq - 2.0 * X1 @ X2.T
    return sigma_f**2 * np.exp(-0.5 * sqdist / length_scale**2)


class GaussianProcessRegressorFromScratch:
    def __init__(self, length_scale=1.1, sigma_f=0.35, sigma_n=0.025):
        self.length_scale = length_scale
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

        K = rbf_kernel(
            X_train,
            X_train,
            length_scale=self.length_scale,
            sigma_f=self.sigma_f,
        )
        K = K + (self.sigma_n**2 + 1e-8) * np.eye(len(X_train))

        self.L = np.linalg.cholesky(K)
        tmp = np.linalg.solve(self.L, y_train)
        self.alpha = np.linalg.solve(self.L.T, tmp)

    def predict(self, X_test):
        K_star = rbf_kernel(
            X_test,
            self.X_train,
            length_scale=self.length_scale,
            sigma_f=self.sigma_f,
        )

        mu = K_star @ self.alpha
        v = np.linalg.solve(self.L, K_star.T)

        K_test_diag = np.full(X_test.shape[0], self.sigma_f**2)
        var = K_test_diag - np.sum(v**2, axis=0)
        var = np.maximum(var, 1e-12)

        return mu, var


def split_train_calibration(X, y, train_fraction=0.7, seed=1):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_train = int(train_fraction * len(X))
    return X[idx[:n_train]], y[idx[:n_train]], X[idx[n_train:]], y[idx[n_train:]]


def conformal_margin(y_cal_true, y_cal_pred, delta):
    scores = np.abs(y_cal_true - y_cal_pred)
    scores_sorted = np.sort(scores)
    k = len(scores_sorted)
    p = int(np.ceil((k + 1) * (1.0 - delta)))
    idx = min(p - 1, k - 1)
    return scores_sorted[idx], scores


def gp_mean(gp, point):
    mu, _ = gp.predict(np.asarray(point).reshape(1, 2))
    return float(mu.item())


def gp_slope_norm(gp, point, eps=1e-3):
    p = np.asarray(point, dtype=float)

    dmu_dx = (
        gp_mean(gp, p + np.array([eps, 0.0]))
        - gp_mean(gp, p - np.array([eps, 0.0]))
    ) / (2.0 * eps)

    dmu_dy = (
        gp_mean(gp, p + np.array([0.0, eps]))
        - gp_mean(gp, p - np.array([0.0, eps]))
    ) / (2.0 * eps)

    return np.sqrt(dmu_dx**2 + dmu_dy**2)


def build_gp_and_cp(seed=4, n_samples=120, confidence=0.85):
    rng = np.random.default_rng(seed)

    x_min, x_max = -5.0, 5.0
    y_min, y_max = -5.0, 5.0
    noise_std = 0.025

    X_samples = np.column_stack([
        rng.uniform(x_min, x_max, n_samples),
        rng.uniform(y_min, y_max, n_samples),
    ])

    z_clean = true_terrain(X_samples[:, 0], X_samples[:, 1])
    z_samples = z_clean + rng.normal(0.0, noise_std, size=n_samples)

    X_train, y_train, X_cal, y_cal = split_train_calibration(
        X_samples,
        z_samples,
        train_fraction=0.7,
        seed=10,
    )

    gp = GaussianProcessRegressorFromScratch(
        length_scale=1.1,
        sigma_f=0.35,
        sigma_n=noise_std,
    )
    gp.fit(X_train, y_train)

    y_cal_pred, _ = gp.predict(X_cal)

    delta = 1.0 - confidence
    C, scores = conformal_margin(y_cal, y_cal_pred, delta)

    return gp, C, X_train, X_cal


# ------------------------------------------------------------
# Global CP-safe route
# ------------------------------------------------------------

def inside_bounds_xy(point, x_bounds, y_bounds):
    return (
        x_bounds[0] <= point[0] <= x_bounds[1]
        and y_bounds[0] <= point[1] <= y_bounds[1]
    )


def cp_safe_xy_step(gp, z_current_reference, next_xy, delta_h_max, C):
    mu_next = gp_mean(gp, next_xy)
    return (delta_h_max - abs(mu_next - z_current_reference)) >= C


def generate_route_candidates(current, goal, step_length, n_heading_candidates):
    direction = np.arctan2(goal[1] - current[1], goal[0] - current[0])

    # Wide fan. This allows going around terrain instead of always going direct.
    headings = np.linspace(direction - np.pi, direction + np.pi, n_heading_candidates)

    candidates = []
    for h in headings:
        candidates.append(current + step_length * np.array([np.cos(h), np.sin(h)]))

    return candidates


def plan_global_cp_safe_route(
    start,
    goal,
    gp,
    C,
    delta_h_max,
    step_length=0.45,
    n_heading_candidates=61,
    max_steps=160,
    goal_tol=0.35,
    x_bounds=(-5.0, 5.0),
    y_bounds=(-5.0, 5.0),
):
    """
    Greedy global route over terrain with CP-safe step constraints.

    This is the high-level route provider.
    It is not a full optimal planner, but it is enough to guide
    the local LIPM-MPC around bad terrain regions.
    """
    route = [np.array(start, dtype=float)]
    z_current_true = true_terrain(start[0], start[1])

    for k in range(max_steps):
        current = route[-1]

        if np.linalg.norm(current - goal) < goal_tol:
            print(f"Global route reached goal at step {k}.")
            break

        candidates = generate_route_candidates(
            current=current,
            goal=goal,
            step_length=step_length,
            n_heading_candidates=n_heading_candidates,
        )

        feasible = []

        for cand in candidates:
            if not inside_bounds_xy(cand, x_bounds, y_bounds):
                continue

            if not cp_safe_xy_step(
                gp=gp,
                z_current_reference=z_current_true,
                next_xy=cand,
                delta_h_max=delta_h_max,
                C=C,
            ):
                continue

            dist_to_goal = np.linalg.norm(cand - goal)
            slope_cost = gp_slope_norm(gp, cand)
            height_cost = abs(gp_mean(gp, cand) - z_current_true)

            cost = (
                1.0 * dist_to_goal
                + 2.0 * slope_cost
                + 3.0 * height_cost
            )

            feasible.append((cost, cand))

        if len(feasible) == 0:
            print(f"No CP-safe global route candidate at step {k}.")
            break

        feasible.sort(key=lambda item: item[0])
        next_point = feasible[0][1]

        route.append(next_point)
        z_current_true = true_terrain(next_point[0], next_point[1])

    return np.array(route)


def get_active_waypoint(state_xy, route, current_index, lookahead_distance):
    """
    Advance along the global route until the active waypoint is far enough ahead.
    """
    idx = current_index

    while idx < len(route) - 1:
        if np.linalg.norm(route[idx] - state_xy) >= lookahead_distance:
            break
        idx += 1

    return route[idx], idx


# ------------------------------------------------------------
# LIPM dynamics and local MPC
# ------------------------------------------------------------

def lipm_step_local(x_loc, v_loc, u_f, omega, T_step):
    s = np.sinh(omega * T_step)
    c = np.cosh(omega * T_step)

    x_next_loc = x_loc + (s / omega) * v_loc + (1.0 - c) * u_f
    v_next_loc = c * v_loc - omega * s * u_f

    delta_x_loc = x_next_loc - x_loc

    return x_next_loc, v_next_loc, delta_x_loc


def global_lipm_update(state, control, params, gp):
    x, y, z, v_loc, theta = state
    u_f, delta_theta = control

    omega = params["omega"]
    T_step = params["T_step"]

    x_loc = 0.0

    _, v_next_loc, delta_x_loc = lipm_step_local(
        x_loc=x_loc,
        v_loc=v_loc,
        u_f=u_f,
        omega=omega,
        T_step=T_step,
    )

    x_next = x + delta_x_loc * np.cos(theta)
    y_next = y + delta_x_loc * np.sin(theta)
    z_next = gp_mean(gp, np.array([x_next, y_next]))
    theta_next = theta + delta_theta

    return np.array([x_next, y_next, z_next, v_next_loc, theta_next])


def candidate_controls():
    u_f_candidates = np.array([-0.20, 0.20, 0.35, 0.45])
    dtheta_candidates = np.deg2rad(np.array([-30.0, 0.0, 30.0]))

    controls = []
    for u_f in u_f_candidates:
        for dtheta in dtheta_candidates:
            controls.append(np.array([u_f, dtheta]))

    return controls


def inside_bounds_state(state, bounds):
    x, y, z, v_loc, theta = state

    if not (bounds["x"][0] <= x <= bounds["x"][1]):
        return False

    if not (bounds["y"][0] <= y <= bounds["y"][1]):
        return False

    if not (bounds["v_loc"][0] <= v_loc <= bounds["v_loc"][1]):
        return False

    return True


def wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def sequence_cost(sequence, controls, waypoint, final_goal, gp, weights):
    waypoint = np.array(waypoint)
    final_goal = np.array(final_goal)

    terminal_xy = sequence[-1, :2]
    initial_xy = sequence[0, :2]

    cost = 0.0

    # Local waypoint tracking is the key difference.
    cost += weights["waypoint"] * np.linalg.norm(terminal_xy - waypoint) ** 2

    # Still weakly care about final goal.
    cost += weights["final_goal"] * np.linalg.norm(terminal_xy - final_goal) ** 2

    progress = np.linalg.norm(initial_xy - waypoint) - np.linalg.norm(terminal_xy - waypoint)
    cost -= weights["progress"] * progress

    for q in range(len(controls)):
        state_q = sequence[q]
        state_next = sequence[q + 1]
        control_q = controls[q]

        x_next, y_next, z_next, v_next, theta_next = state_next
        u_f, dtheta = control_q

        #slope = gp_slope_norm(gp, state_next[:2])
        #cost += weights["slope"] * slope**2

        dz_pred = state_next[2] - state_q[2]
        cost += weights["height"] * dz_pred**2

        theta_wp = np.arctan2(waypoint[1] - y_next, waypoint[0] - x_next)
        heading_error = wrap_angle(theta_next - theta_wp)
        cost += weights["heading"] * heading_error**2

        cost += weights["control"] * (u_f**2 + dtheta**2)

        cost += weights["velocity"] * (v_next - weights["v_ref"]) ** 2

    return cost


def find_best_local_sequence(current_state, waypoint, final_goal, gp, C, delta_h_max, params):
    controls_all = candidate_controls()
    horizon = params["horizon"]

    best_cost = np.inf
    best_sequence = None
    best_controls = None

    current_true_height = true_terrain(current_state[0], current_state[1])
    z_reference = current_true_height

    for control_indices in itertools.product(range(len(controls_all)), repeat=horizon):
        sequence = [current_state.copy()]
        controls_seq = []

        feasible = True
        z_reference = current_true_height

        for idx in control_indices:
            control = controls_all[idx]
            state_now = sequence[-1]

            state_next = global_lipm_update(
                state=state_now,
                control=control,
                params=params,
                gp=gp,
            )

            if not inside_bounds_state(state_next, params["bounds"]):
                feasible = False
                break

            if not cp_safe_xy_step(
                gp=gp,
                z_current_reference=z_reference,
                next_xy=state_next[:2],
                delta_h_max=delta_h_max,
                C=C,
            ):
                feasible = False
                break

            sequence.append(state_next)
            controls_seq.append(control)

            # Use GP mean internally inside prediction horizon.
            z_reference = state_next[2]

        if not feasible:
            continue

        sequence = np.array(sequence)
        controls_seq = np.array(controls_seq)

        cost = sequence_cost(
            sequence=sequence,
            controls=controls_seq,
            waypoint=waypoint,
            final_goal=final_goal,
            gp=gp,
            weights=params["weights"],
        )

        if cost < best_cost:
            best_cost = cost
            best_sequence = sequence
            best_controls = controls_seq

    return best_sequence, best_controls, best_cost


def plan_waypoint_lipm_mpc(start_xy, goal_xy, route, gp, C, delta_h_max, params):
    z0_true = true_terrain(start_xy[0], start_xy[1])

    # Start heading points toward first meaningful route point.
    first_wp = route[min(3, len(route) - 1)]
    theta0 = np.arctan2(first_wp[1] - start_xy[1], first_wp[0] - start_xy[0])

    state = np.array([
        start_xy[0],
        start_xy[1],
        z0_true,
        params["v0"],
        theta0,
    ])

    states = [state.copy()]
    controls_executed = []
    active_waypoints = []
    planned_sequences = []

    route_index = 1

    for step in range(params["max_steps"]):
        if np.linalg.norm(state[:2] - goal_xy) < params["goal_tol"]:
            print(f"Waypoint LIPM-MPC reached final goal at step {step}.")
            break

        waypoint, route_index = get_active_waypoint(
            state_xy=state[:2],
            route=route,
            current_index=route_index,
            lookahead_distance=params["lookahead_distance"],
        )

        best_sequence, best_controls, best_cost = find_best_local_sequence(
            current_state=state,
            waypoint=waypoint,
            final_goal=goal_xy,
            gp=gp,
            C=C,
            delta_h_max=delta_h_max,
            params=params,
        )

        if best_sequence is None:
            print(f"No feasible waypoint LIPM-MPC sequence found at step {step}.")
            break

        next_state = best_sequence[1].copy()
        executed_control = best_controls[0].copy()

        # Execution: true terrain height becomes measured.
        next_state[2] = true_terrain(next_state[0], next_state[1])

        state = next_state

        states.append(state.copy())
        controls_executed.append(executed_control)
        active_waypoints.append(waypoint.copy())
        planned_sequences.append(best_sequence.copy())

    return (
        np.array(states),
        np.array(controls_executed),
        np.array(active_waypoints),
        planned_sequences,
    )


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


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    confidence = 0.85
    gp, C, X_train, X_cal = build_gp_and_cp(
        seed=4,
        n_samples=180,
        confidence=confidence,
    )

    g = 9.81
    zH = 0.9
    omega = np.sqrt(g / zH)

    start = np.array([-4.3, -4.2])
    goal = np.array([4.3, 4.0])

    delta_h_max = 0.18

    print("\nGenerating global CP-safe waypoint route...")
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

    print("\nRunning waypoint-guided LIPM-MPC...")
    states, controls, active_waypoints, planned_sequences = plan_waypoint_lipm_mpc(
        start_xy=start,
        goal_xy=goal,
        route=route,
        gp=gp,
        C=C,
        delta_h_max=delta_h_max,
        params=params,
    )

    violations, steps, dh = evaluate_true_step_safety(states, delta_h_max)
    final_dist = np.linalg.norm(states[-1, :2] - goal)

    print("\nWaypoint-guided LIPM + CP-safe MPC demo completed.")
    print("=" * 72)
    print(f"confidence level              = {confidence * 100:.1f}%")
    print(f"conformal margin C            = {C:.5f} m")
    print(f"delta_h_max                   = {delta_h_max:.5f} m")
    print(f"global route points           = {len(route)}")
    print(f"executed LIPM steps           = {steps}")
    print(f"final distance                = {final_dist:.4f} m")
    print(f"true terrain violations       = {violations}")
    print(f"max true |dh|                 = {np.max(dh) if len(dh) else 0.0:.5f} m")
    print(f"horizon                       = {params['horizon']}")
    print(f"candidate controls            = {len(candidate_controls())}")

    if len(controls) > 0:
        print(f"u_f range                     = [{controls[:, 0].min():.3f}, {controls[:, 0].max():.3f}] m")
        print(
            "delta_theta range             = "
            f"[{np.rad2deg(controls[:, 1].min()):.2f}, "
            f"{np.rad2deg(controls[:, 1].max()):.2f}] deg"
        )

    # ------------------------------------------------------------
    # Plots
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
        label="global CP-safe waypoint route",
    )

    plt.plot(
        states[:, 0],
        states[:, 1],
        "o-",
        linewidth=2.2,
        markersize=4,
        label="executed waypoint LIPM-MPC path",
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
    plt.title("Global CP-safe route + waypoint-guided LIPM-MPC")
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(8, 4.5))
    plt.plot(dh, "o-", label="true |Δh|")
    plt.axhline(delta_h_max, linestyle="--", linewidth=2, label="Δh max")
    plt.xlabel("step index")
    plt.ylabel("true terrain height change [m]")
    plt.title("True height change per executed waypoint-LIPM step")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(8, 4.5))
    plt.plot(states[:, 3], "o-")
    plt.xlabel("step index")
    plt.ylabel("v_loc [m/s]")
    plt.title("Local sagittal velocity")
    plt.grid(True)
    plt.tight_layout()

    if len(controls) > 0:
        plt.figure(figsize=(8, 4.5))
        plt.plot(controls[:, 0], "o-")
        plt.xlabel("step index")
        plt.ylabel("u_f [m]")
        plt.title("Executed sagittal foot-placement control")
        plt.grid(True)
        plt.tight_layout()

        plt.figure(figsize=(8, 4.5))
        plt.plot(np.rad2deg(controls[:, 1]), "o-")
        plt.xlabel("step index")
        plt.ylabel("delta theta [deg]")
        plt.title("Executed heading-change control")
        plt.grid(True)
        plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()