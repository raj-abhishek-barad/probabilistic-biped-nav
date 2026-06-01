import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# LIPM + CP-safe terrain planner demo
#
# This combines:
#   1. GP + conformal prediction terrain uncertainty
#   2. paper-style discrete LIPM walking dynamics
#   3. CP-safe footstep height constraint
#
# This is the closest simplified version so far to the paper's
# uncertainty-informed LIPM/MPC planner.
# ============================================================


# ------------------------------------------------------------
# Terrain and GP tools
# ------------------------------------------------------------

def true_terrain(x, y):
    return (
        0.18 * np.sin(0.8 * x)
        + 0.12 * np.cos(0.7 * y)
        + 0.08 * np.sin(1.5 * x + 0.6 * y)
        + 0.04 * np.cos(2.0 * x - 1.2 * y)
    )


def terrain_gradient_true(x, y, eps=1e-4):
    dz_dx = (true_terrain(x + eps, y) - true_terrain(x - eps, y)) / (2.0 * eps)
    dz_dy = (true_terrain(x, y + eps) - true_terrain(x, y - eps)) / (2.0 * eps)
    return np.array([dz_dx, dz_dy])


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
        self.X_train = None
        self.y_train = None
        self.L = None
        self.alpha = None

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
    n = len(X)
    indices = rng.permutation(n)

    n_train = int(train_fraction * n)

    train_idx = indices[:n_train]
    cal_idx = indices[n_train:]

    return X[train_idx], y[train_idx], X[cal_idx], y[cal_idx]


def conformal_margin(y_cal_true, y_cal_pred, delta):
    scores = np.abs(y_cal_true - y_cal_pred)
    scores_sorted = np.sort(scores)

    k = len(scores_sorted)
    p = int(np.ceil((k + 1) * (1.0 - delta)))
    idx = min(p - 1, k - 1)

    C = scores_sorted[idx]
    return C, scores


def gp_mean(gp, point):
    mu, _ = gp.predict(np.asarray(point).reshape(1, 2))
    return float(mu.item())


def gp_slope_norm(gp, point, eps=1e-3):
    p = np.asarray(point, dtype=float)

    pxp = p + np.array([eps, 0.0])
    pxm = p - np.array([eps, 0.0])
    pyp = p + np.array([0.0, eps])
    pym = p - np.array([0.0, eps])

    dmu_dx = (gp_mean(gp, pxp) - gp_mean(gp, pxm)) / (2.0 * eps)
    dmu_dy = (gp_mean(gp, pyp) - gp_mean(gp, pym)) / (2.0 * eps)

    return np.sqrt(dmu_dx**2 + dmu_dy**2)


# ------------------------------------------------------------
# LIPM dynamics
# ------------------------------------------------------------

def lipm_step_local(x_loc, v_loc, u_f, omega, T_step):
    s = np.sinh(omega * T_step)
    c = np.cosh(omega * T_step)

    x_next_loc = x_loc + (s / omega) * v_loc + (1.0 - c) * u_f
    v_next_loc = c * v_loc - omega * s * u_f

    delta_x_loc = x_next_loc - x_loc

    return x_next_loc, v_next_loc, delta_x_loc


def global_lipm_update(state, control, params, terrain_model="true", gp=None):
    """
    state = [x, y, z, v_loc, theta]
    control = [u_f, delta_theta]
    """
    x, y, z, v_loc, theta = state
    u_f, delta_theta = control

    omega = params["omega"]
    T_step = params["T_step"]

    # Simplified stance-switch convention:
    # local CoM position reset to 0 at each stance phase.
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

    if terrain_model == "gp":
        if gp is None:
            raise ValueError("gp must be provided when terrain_model='gp'")
        z_next = gp_mean(gp, np.array([x_next, y_next]))
    else:
        z_next = true_terrain(x_next, y_next)

    theta_next = theta + delta_theta

    return np.array([x_next, y_next, z_next, v_next_loc, theta_next])


def candidate_controls():
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


def cp_safe_transition(gp, current_true_height, next_xy, delta_h_max, C):
    """
    Paper-style CP-safe height transition:

        Delta_h_max - |mu(next) - z_current_true| >= C

    current_true_height is measured after the current stance foot.
    next height is uncertain, represented by GP mean + CP margin.
    """
    mu_next = gp_mean(gp, next_xy)
    return (delta_h_max - abs(mu_next - current_true_height)) >= C


def wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def stage_cost(state, next_state, goal, control, gp, weights):
    x, y, z, v_loc, theta = state
    xn, yn, zn, vn, thetan = next_state
    u_f, dtheta = control

    p_now = np.array([x, y])
    p_next = np.array([xn, yn])
    p_goal = np.array(goal)

    dist_cost = weights["goal"] * np.linalg.norm(p_next - p_goal) ** 2

    slope_cost = weights["slope"] * gp_slope_norm(gp, p_next) ** 2

    theta_goal = np.arctan2(goal[1] - yn, goal[0] - xn)
    heading_error = wrap_angle(thetan - theta_goal)
    heading_cost = weights["heading"] * heading_error**2

    control_cost = weights["control"] * (u_f**2 + dtheta**2)

    old_dist = np.linalg.norm(p_now - p_goal)
    new_dist = np.linalg.norm(p_next - p_goal)
    progress_reward = weights["progress"] * (old_dist - new_dist)

    height_change_cost = weights["height"] * (zn - z) ** 2

    return (
        dist_cost
        + slope_cost
        + heading_cost
        + control_cost
        + height_change_cost
        - progress_reward
    )


def plan_lipm_path(
    start_xy,
    goal_xy,
    params,
    gp,
    use_cp_constraint,
    C=None,
    delta_h_max=None,
):
    """
    Greedy one-step LIPM planner with optional CP-safe terrain constraint.

    This is a stepping stone before full horizon MPC.
    """
    z0_true = true_terrain(start_xy[0], start_xy[1])

    theta0 = np.arctan2(goal_xy[1] - start_xy[1], goal_xy[0] - start_xy[0])

    state = np.array([
        start_xy[0],
        start_xy[1],
        z0_true,
        params["v0"],
        theta0,
    ])

    states = [state.copy()]
    controls_used = []
    cp_feasible_flags = []

    controls = candidate_controls()

    max_steps = params["max_steps"]
    goal_tol = params["goal_tol"]

    current_true_height = z0_true

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
            # For planning, use GP terrain height estimate in z update.
            next_state = global_lipm_update(
                state,
                control,
                params,
                terrain_model="gp",
                gp=gp,
            )

            if not control_is_feasible(next_state, params["bounds"]):
                continue

            next_xy = next_state[:2]

            if use_cp_constraint:
                if C is None or delta_h_max is None:
                    raise ValueError("C and delta_h_max are required for CP-safe planning")

                if not cp_safe_transition(
                    gp=gp,
                    current_true_height=current_true_height,
                    next_xy=next_xy,
                    delta_h_max=delta_h_max,
                    C=C,
                ):
                    continue

            cost = stage_cost(
                state=state,
                next_state=next_state,
                goal=goal_xy,
                control=control,
                gp=gp,
                weights=params["weights"],
            )

            if cost < best_cost:
                best_cost = cost
                best_next = next_state
                best_control = control

        if best_next is None:
            if use_cp_constraint:
                print(f"No CP-safe feasible LIPM control found at step {k}.")
            else:
                print(f"No feasible LIPM control found at step {k}.")
            break

        # Execute step. In real robot: next stance height becomes measured.
        true_z_next = true_terrain(best_next[0], best_next[1])
        best_next[2] = true_z_next
        current_true_height = true_z_next

        state = best_next
        states.append(state.copy())
        controls_used.append(best_control.copy())
        cp_feasible_flags.append(True)

    return np.array(states), np.array(controls_used)


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


def build_gp_and_cp(seed=4, n_samples=180, confidence=0.85):
    rng = np.random.default_rng(seed)

    x_min, x_max = -5.0, 5.0
    y_min, y_max = -5.0, 5.0
    noise_std = 0.025

    X_samples = np.column_stack([
        rng.uniform(x_min, x_max, n_samples),
        rng.uniform(y_min, y_max, n_samples),
    ])

    z_samples_clean = true_terrain(X_samples[:, 0], X_samples[:, 1])
    z_samples = z_samples_clean + rng.normal(0.0, noise_std, size=n_samples)

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


def main():
    # ------------------------------------------------------------
    # Build GP + CP terrain model
    # ------------------------------------------------------------
    confidence = 0.85
    gp, C, X_train, X_cal = build_gp_and_cp(
        seed=4,
        n_samples=180,
        confidence=confidence,
    )

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
            "height": 20.0,
        },
    }

    start = np.array([-4.3, -4.2])
    goal = np.array([4.3, 4.0])

    delta_h_max = 0.15

    # ------------------------------------------------------------
    # Plan without CP constraint
    # ------------------------------------------------------------
    print("\nPlanning LIPM path without CP-safe terrain constraint...")
    states_free, controls_free = plan_lipm_path(
        start_xy=start,
        goal_xy=goal,
        params=params,
        gp=gp,
        use_cp_constraint=False,
    )

    # ------------------------------------------------------------
    # Plan with CP constraint
    # ------------------------------------------------------------
    print("\nPlanning LIPM path with CP-safe terrain constraint...")
    states_cp, controls_cp = plan_lipm_path(
        start_xy=start,
        goal_xy=goal,
        params=params,
        gp=gp,
        use_cp_constraint=True,
        C=C,
        delta_h_max=delta_h_max,
    )

    free_viol, free_steps, free_dh = evaluate_true_step_safety(
        states_free,
        delta_h_max,
    )
    cp_viol, cp_steps, cp_dh = evaluate_true_step_safety(
        states_cp,
        delta_h_max,
    )

    free_final_dist = np.linalg.norm(states_free[-1, :2] - goal)
    cp_final_dist = np.linalg.norm(states_cp[-1, :2] - goal)

    print("\nLIPM + CP-safe planner demo completed.")
    print("=" * 72)
    print(f"confidence level              = {confidence * 100:.1f}%")
    print(f"conformal margin C            = {C:.5f} m")
    print(f"delta_h_max                   = {delta_h_max:.5f} m")
    print(f"omega                         = {omega:.4f} rad/s")
    print(f"T_step                        = {params['T_step']:.3f} s")
    print()
    print("LIPM without CP:")
    print(f"  steps                       = {free_steps}")
    print(f"  final distance              = {free_final_dist:.4f} m")
    print(f"  true terrain violations     = {free_viol}")
    print(f"  max true |dh|               = {np.max(free_dh) if len(free_dh) else 0.0:.5f} m")
    print()
    print("LIPM with CP-safe constraint:")
    print(f"  steps                       = {cp_steps}")
    print(f"  final distance              = {cp_final_dist:.4f} m")
    print(f"  true terrain violations     = {cp_viol}")
    print(f"  max true |dh|               = {np.max(cp_dh) if len(cp_dh) else 0.0:.5f} m")

    if len(controls_cp) > 0:
        print(f"  u_f range                   = [{controls_cp[:, 0].min():.3f}, {controls_cp[:, 0].max():.3f}] m")
        print(
            "  delta_theta range           = "
            f"[{np.rad2deg(controls_cp[:, 1].min()):.2f}, "
            f"{np.rad2deg(controls_cp[:, 1].max()):.2f}] deg"
        )

    # ------------------------------------------------------------
    # Plot terrain and paths
    # ------------------------------------------------------------
    x_grid = np.linspace(-5, 5, 150)
    y_grid = np.linspace(-5, 5, 150)
    Xg, Yg = np.meshgrid(x_grid, y_grid)
    Zg = true_terrain(Xg, Yg)

    plt.figure(figsize=(8, 7))
    plt.contourf(Xg, Yg, Zg, levels=35)
    plt.colorbar(label="true terrain height [m]")

    plt.plot(
        states_free[:, 0],
        states_free[:, 1],
        "o--",
        linewidth=1.5,
        markersize=3.5,
        label="LIPM without CP",
    )

    plt.plot(
        states_cp[:, 0],
        states_cp[:, 1],
        "o-",
        linewidth=2.2,
        markersize=4,
        label="LIPM with CP-safe constraint",
    )

    plt.scatter(start[0], start[1], s=100, marker="s", label="start")
    plt.scatter(goal[0], goal[1], s=140, marker="*", label="goal")

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("LIPM planner with and without CP-safe terrain constraint")
    plt.legend()
    plt.tight_layout()

    # ------------------------------------------------------------
    # True height change comparison
    # ------------------------------------------------------------
    plt.figure(figsize=(8, 4.5))
    plt.plot(free_dh, "o--", label="without CP true |Δh|")
    plt.plot(cp_dh, "o-", label="with CP true |Δh|")
    plt.axhline(delta_h_max, linestyle="--", linewidth=2, label="Δh max")
    plt.xlabel("step index")
    plt.ylabel("true terrain height change [m]")
    plt.title("True terrain height change per LIPM step")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # ------------------------------------------------------------
    # Velocity comparison
    # ------------------------------------------------------------
    plt.figure(figsize=(8, 4.5))
    plt.plot(states_free[:, 3], "o--", label="without CP")
    plt.plot(states_cp[:, 3], "o-", label="with CP")
    plt.xlabel("step index")
    plt.ylabel("v_loc [m/s]")
    plt.title("Local sagittal velocity")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # ------------------------------------------------------------
    # Heading comparison
    # ------------------------------------------------------------
    plt.figure(figsize=(8, 4.5))
    plt.plot(np.rad2deg(states_free[:, 4]), "o--", label="without CP")
    plt.plot(np.rad2deg(states_cp[:, 4]), "o-", label="with CP")
    plt.xlabel("step index")
    plt.ylabel("heading θ [deg]")
    plt.title("Heading evolution")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()