import itertools
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# LIPM + CP-safe short-horizon MPC demo
#
# Paper-faithful idea:
#   GP + conformal prediction terrain uncertainty
#   + discrete LIPM walking dynamics
#   + CP-safe footstep height constraint
#   + short-horizon receding MPC
#
# This upgrades lipm_cp_safe_planner_demo.py from greedy one-step
# selection to horizon-based control sequence selection.
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
    indices = rng.permutation(len(X))
    n_train = int(train_fraction * len(X))

    train_idx = indices[:n_train]
    cal_idx = indices[n_train:]

    return X[train_idx], y[train_idx], X[cal_idx], y[cal_idx]


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

    pxp = p + np.array([eps, 0.0])
    pxm = p - np.array([eps, 0.0])
    pyp = p + np.array([0.0, eps])
    pym = p - np.array([0.0, eps])

    dmu_dx = (gp_mean(gp, pxp) - gp_mean(gp, pxm)) / (2.0 * eps)
    dmu_dy = (gp_mean(gp, pyp) - gp_mean(gp, pym)) / (2.0 * eps)

    return np.sqrt(dmu_dx**2 + dmu_dy**2)


def build_gp_and_cp(seed=4, n_samples=180, confidence=0.85):
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
# LIPM dynamics
# ------------------------------------------------------------

def lipm_step_local(x_loc, v_loc, u_f, omega, T_step):
    s = np.sinh(omega * T_step)
    c = np.cosh(omega * T_step)

    x_next_loc = x_loc + (s / omega) * v_loc + (1.0 - c) * u_f
    v_next_loc = c * v_loc - omega * s * u_f

    delta_x_loc = x_next_loc - x_loc

    return x_next_loc, v_next_loc, delta_x_loc


def global_lipm_update(state, control, params, gp):
    """
    state = [x, y, z, v_loc, theta]
    control = [u_f, delta_theta]

    During planning, terrain height is GP mean.
    During execution, we overwrite z by true terrain height after accepting a step.
    """
    x, y, z, v_loc, theta = state
    u_f, delta_theta = control

    omega = params["omega"]
    T_step = params["T_step"]

    # Simplified stance-switch convention:
    # local CoM position reset to zero at each step.
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
    """
    Reduced candidate set for horizon search.

    Too many candidates make brute-force MPC slow.
    These are enough to demonstrate the principle.
    """
    u_f_candidates = np.array([-0.20, 0.0, 0.20, 0.35, 0.45])
    dtheta_candidates = np.deg2rad(np.array([-30.0, -15.0, 0.0, 15.0, 30.0]))

    controls = []
    for u_f in u_f_candidates:
        for dtheta in dtheta_candidates:
            controls.append(np.array([u_f, dtheta]))

    return controls


def inside_bounds(state, bounds):
    x, y, z, v_loc, theta = state

    if not (bounds["x"][0] <= x <= bounds["x"][1]):
        return False

    if not (bounds["y"][0] <= y <= bounds["y"][1]):
        return False

    if not (bounds["v_loc"][0] <= v_loc <= bounds["v_loc"][1]):
        return False

    return True


def cp_safe_transition(gp, z_current_reference, next_xy, delta_h_max, C):
    """
    CP-safe terrain transition:

        Delta_h_max - |mu(next) - z_current| >= C

    z_current_reference is:
      - true measured height at the first predicted step
      - GP mean height for internal predicted horizon nodes
    """
    mu_next = gp_mean(gp, next_xy)
    return (delta_h_max - abs(mu_next - z_current_reference)) >= C


def wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def sequence_cost(sequence, controls, goal, gp, weights):
    """
    MPC-style cost:
      terminal goal distance
      + terrain slope cost
      + heading error cost
      + height smoothness cost
      + control cost
      - progress reward
    """
    p_goal = np.array(goal)

    terminal_xy = sequence[-1, :2]
    initial_xy = sequence[0, :2]

    cost = weights["terminal_goal"] * np.linalg.norm(terminal_xy - p_goal) ** 2

    progress = np.linalg.norm(initial_xy - p_goal) - np.linalg.norm(terminal_xy - p_goal)
    cost -= weights["progress"] * progress

    for q in range(len(controls)):
        state_q = sequence[q]
        state_next = sequence[q + 1]
        control_q = controls[q]

        x_next, y_next, z_next, v_next, theta_next = state_next
        u_f, dtheta = control_q

        slope = gp_slope_norm(gp, state_next[:2])
        cost += weights["slope"] * slope**2

        dz_pred = state_next[2] - state_q[2]
        cost += weights["height"] * dz_pred**2

        theta_goal = np.arctan2(goal[1] - y_next, goal[0] - x_next)
        heading_error = wrap_angle(theta_next - theta_goal)
        cost += weights["heading"] * heading_error**2

        cost += weights["control"] * (u_f**2 + dtheta**2)

        # Soft preference for velocities away from the upper bound.
        v_ref = weights["v_ref"]
        cost += weights["velocity"] * (v_next - v_ref) ** 2

    return cost


def find_best_sequence(current_state, goal, gp, C, delta_h_max, params):
    controls_all = candidate_controls()
    horizon = params["horizon"]

    best_cost = np.inf
    best_sequence = None
    best_controls = None

    current_true_height = true_terrain(current_state[0], current_state[1])

    for control_indices in itertools.product(range(len(controls_all)), repeat=horizon):
        sequence = [current_state.copy()]
        controls_seq = []

        feasible = True

        # First predicted transition uses measured current true height.
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

            if not inside_bounds(state_next, params["bounds"]):
                feasible = False
                break

            next_xy = state_next[:2]

            if not cp_safe_transition(
                gp=gp,
                z_current_reference=z_reference,
                next_xy=next_xy,
                delta_h_max=delta_h_max,
                C=C,
            ):
                feasible = False
                break

            sequence.append(state_next)
            controls_seq.append(control)

            # Internal horizon nodes use predicted mean height.
            z_reference = state_next[2]

        if not feasible:
            continue

        sequence = np.array(sequence)
        controls_seq = np.array(controls_seq)

        cost = sequence_cost(
            sequence=sequence,
            controls=controls_seq,
            goal=goal,
            gp=gp,
            weights=params["weights"],
        )

        if cost < best_cost:
            best_cost = cost
            best_sequence = sequence
            best_controls = controls_seq

    return best_sequence, best_controls, best_cost


def plan_lipm_cp_safe_mpc(start_xy, goal_xy, gp, C, delta_h_max, params):
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
    controls_executed = []
    planned_sequences = []

    for step in range(params["max_steps"]):
        dist = np.linalg.norm(state[:2] - goal_xy)

        if dist < params["goal_tol"]:
            print(f"Goal reached at step {step}.")
            break

        best_sequence, best_controls, best_cost = find_best_sequence(
            current_state=state,
            goal=goal_xy,
            gp=gp,
            C=C,
            delta_h_max=delta_h_max,
            params=params,
        )

        if best_sequence is None:
            print(f"No feasible LIPM+CP MPC sequence found at step {step}.")
            break

        # Receding horizon: execute only first control.
        next_state = best_sequence[1].copy()
        executed_control = best_controls[0].copy()

        # Execution: true terrain height becomes measured.
        next_state[2] = true_terrain(next_state[0], next_state[1])

        state = next_state

        states.append(state.copy())
        controls_executed.append(executed_control)
        planned_sequences.append(best_sequence.copy())

    return np.array(states), np.array(controls_executed), planned_sequences


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


def main():
    # ------------------------------------------------------------
    # Terrain uncertainty model
    # ------------------------------------------------------------
    confidence = 0.85
    gp, C, X_train, X_cal = build_gp_and_cp(
        seed=4,
        n_samples=180,
        confidence=confidence,
    )

    # ------------------------------------------------------------
    # LIPM + MPC parameters
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
        "horizon": 2,
        "max_steps": 80,
        "goal_tol": 0.35,
        "bounds": {
            "x": (-5.0, 5.0),
            "y": (-5.0, 5.0),
            "v_loc": (0.05, 2.70),
        },
        "weights": {
            "terminal_goal": 1.0,
            "progress": 4.0,
            "slope": 40.0,
            "height": 80.0,
            "heading": 0.3,
            "control": 0.03,
            "velocity": 0.4,
            "v_ref": 1.6,
        },
    }

    start = np.array([-4.3, -4.2])
    goal = np.array([4.3, 4.0])

    # The greedy CP-LIPM planner got stuck at 0.15.
    # The MPC version should do better, but this value still shows
    # conservatism clearly.
    delta_h_max = 0.18

    states, controls, planned_sequences = plan_lipm_cp_safe_mpc(
        start_xy=start,
        goal_xy=goal,
        gp=gp,
        C=C,
        delta_h_max=delta_h_max,
        params=params,
    )

    violations, steps, dh = evaluate_true_step_safety(states, delta_h_max)
    final_dist = np.linalg.norm(states[-1, :2] - goal)

    print("\nLIPM + CP-safe short-horizon MPC demo completed.")
    print("=" * 72)
    print(f"confidence level              = {confidence * 100:.1f}%")
    print(f"conformal margin C            = {C:.5f} m")
    print(f"delta_h_max                   = {delta_h_max:.5f} m")
    print(f"omega                         = {omega:.4f} rad/s")
    print(f"T_step                        = {params['T_step']:.3f} s")
    print(f"horizon                       = {params['horizon']}")
    print(f"candidate controls            = {len(candidate_controls())}")
    print()
    print(f"steps                         = {steps}")
    print(f"final distance                = {final_dist:.4f} m")
    print(f"true terrain violations       = {violations}")
    print(f"max true |dh|                 = {np.max(dh) if len(dh) else 0.0:.5f} m")

    if len(controls) > 0:
        print(f"u_f range                     = [{controls[:, 0].min():.3f}, {controls[:, 0].max():.3f}] m")
        print(
            "delta_theta range             = "
            f"[{np.rad2deg(controls[:, 1].min()):.2f}, "
            f"{np.rad2deg(controls[:, 1].max()):.2f}] deg"
        )

    # ------------------------------------------------------------
    # Plot terrain and path
    # ------------------------------------------------------------
    x_grid = np.linspace(-5, 5, 150)
    y_grid = np.linspace(-5, 5, 150)
    Xg, Yg = np.meshgrid(x_grid, y_grid)
    Zg = true_terrain(Xg, Yg)

    plt.figure(figsize=(8, 7))
    plt.contourf(Xg, Yg, Zg, levels=35)
    plt.colorbar(label="true terrain height [m]")

    plt.plot(
        states[:, 0],
        states[:, 1],
        "o-",
        linewidth=2.2,
        markersize=4,
        label="LIPM + CP-safe MPC path",
    )

    # Plot selected predicted horizons
    if len(planned_sequences) > 0:
        stride = max(1, len(planned_sequences) // 10)
        for seq in planned_sequences[::stride]:
            plt.plot(seq[:, 0], seq[:, 1], "-", linewidth=1.0, alpha=0.35)

    plt.scatter(start[0], start[1], s=100, marker="s", label="start")
    plt.scatter(goal[0], goal[1], s=140, marker="*", label="goal")

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("LIPM + CP-safe short-horizon MPC over terrain")
    plt.legend()
    plt.tight_layout()

    # ------------------------------------------------------------
    # True height change
    # ------------------------------------------------------------
    plt.figure(figsize=(8, 4.5))
    plt.plot(dh, "o-", label="true |Δh|")
    plt.axhline(delta_h_max, linestyle="--", linewidth=2, label="Δh max")
    plt.xlabel("step index")
    plt.ylabel("true terrain height change [m]")
    plt.title("True height change per executed LIPM-MPC step")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # ------------------------------------------------------------
    # Velocity
    # ------------------------------------------------------------
    plt.figure(figsize=(8, 4.5))
    plt.plot(states[:, 3], "o-")
    plt.xlabel("step index")
    plt.ylabel("v_loc [m/s]")
    plt.title("Local sagittal velocity under LIPM + CP-safe MPC")
    plt.grid(True)
    plt.tight_layout()

    # ------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------
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