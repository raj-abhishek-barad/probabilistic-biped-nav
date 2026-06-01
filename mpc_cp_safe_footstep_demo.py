import itertools
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Short-horizon MPC-style CP-safe footstep planner
#
# This extends the previous greedy planner.
#
# Paper connection:
# The paper uses an uncertainty-informed MPC that penalizes
# terminal goal error and terrain slope, while enforcing
# CP-safe footstep height constraints.
#
# This file implements a simplified receding-horizon version:
#
#   choose a sequence of H future footsteps
#   minimize goal distance + slope cost + height-change cost
#   enforce CP-safe height constraint at every step
#   execute only the first step
#   repeat
# ============================================================


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
    """
    Numerical gradient of GP mean terrain estimate.
    This approximates ||∇ mu(x,y)||.
    """
    p = np.asarray(point, dtype=float)

    pxp = p + np.array([eps, 0.0])
    pxm = p - np.array([eps, 0.0])
    pyp = p + np.array([0.0, eps])
    pym = p - np.array([0.0, eps])

    dmu_dx = (gp_mean(gp, pxp) - gp_mean(gp, pxm)) / (2.0 * eps)
    dmu_dy = (gp_mean(gp, pyp) - gp_mean(gp, pym)) / (2.0 * eps)

    return np.sqrt(dmu_dx**2 + dmu_dy**2)


def generate_candidate_steps(current, goal, step_length, n_heading_candidates):
    """
    Generate candidate next footsteps in a fan pointing roughly toward the goal.
    """
    direction_to_goal = np.arctan2(goal[1] - current[1], goal[0] - current[0])

    headings = np.linspace(
        direction_to_goal - np.pi / 2,
        direction_to_goal + np.pi / 2,
        n_heading_candidates,
    )

    candidates = []
    for heading in headings:
        candidate = current + step_length * np.array([np.cos(heading), np.sin(heading)])
        candidates.append(candidate)

    return candidates


def inside_bounds(point, x_bounds, y_bounds):
    return (
        x_bounds[0] <= point[0] <= x_bounds[1]
        and y_bounds[0] <= point[1] <= y_bounds[1]
    )


def is_cp_safe_step(gp, current_true_height, next_point, delta_h_max, C):
    """
    CP-safe footstep constraint:
        Delta_h_max - |mu(next) - z_current_true| >= C
    """
    mu_next = gp_mean(gp, next_point)
    return (delta_h_max - abs(mu_next - current_true_height)) >= C


def sequence_cost(
    sequence,
    goal,
    gp,
    w_goal,
    w_slope,
    w_height,
    w_progress,
):
    """
    MPC-style objective:
      terminal goal distance
      + sum terrain slope penalties
      + sum predicted height-change penalties
      - progress reward
    """
    terminal = sequence[-1]
    cost = w_goal * np.linalg.norm(terminal - goal) ** 2

    total_slope = 0.0
    total_height_change = 0.0

    for i in range(len(sequence) - 1):
        p0 = sequence[i]
        p1 = sequence[i + 1]

        total_slope += gp_slope_norm(gp, p1) ** 2

        mu0 = gp_mean(gp, p0)
        mu1 = gp_mean(gp, p1)
        total_height_change += (mu1 - mu0) ** 2

    progress = np.linalg.norm(sequence[0] - goal) - np.linalg.norm(terminal - goal)

    cost += w_slope * total_slope
    cost += w_height * total_height_change
    cost -= w_progress * progress

    return cost


def find_best_horizon_sequence(
    current,
    goal,
    gp,
    C,
    delta_h_max,
    step_length,
    horizon,
    n_heading_candidates,
    x_bounds,
    y_bounds,
    weights,
):
    """
    Brute-force short-horizon MPC over candidate heading sequences.

    Since this is a demo, we use enumeration. Later, this can be replaced
    by scipy.optimize or CasADi.
    """
    candidate_heading_indices = range(n_heading_candidates)

    best_cost = np.inf
    best_sequence = None

    # Current foot height is measured after each executed step.
    # This matches the paper's receding-horizon assumption that the
    # current stance foot height is known.
    current_true_height = true_terrain(current[0], current[1])

    # Enumerate sequences of heading choices.
    for heading_sequence in itertools.product(candidate_heading_indices, repeat=horizon):
        sequence = [current.copy()]
        feasible = True

        z_measured = current_true_height

        for heading_idx in heading_sequence:
            p_now = sequence[-1]

            candidates = generate_candidate_steps(
                p_now,
                goal,
                step_length,
                n_heading_candidates,
            )

            p_next = candidates[heading_idx]

            if not inside_bounds(p_next, x_bounds, y_bounds):
                feasible = False
                break

            if not is_cp_safe_step(gp, z_measured, p_next, delta_h_max, C):
                feasible = False
                break

            sequence.append(p_next)

            # In the real receding-horizon implementation, only the first
            # next foot height is truly measured. For this simplified planner,
            # we use GP mean for hypothetical future internal nodes after the
            # first candidate to avoid cheating with true terrain.
            z_measured = gp_mean(gp, p_next)

        if not feasible:
            continue

        sequence = np.array(sequence)

        cost = sequence_cost(
            sequence=sequence,
            goal=goal,
            gp=gp,
            w_goal=weights["goal"],
            w_slope=weights["slope"],
            w_height=weights["height"],
            w_progress=weights["progress"],
        )

        if cost < best_cost:
            best_cost = cost
            best_sequence = sequence

    return best_sequence, best_cost


def plan_mpc_cp_safe_path(
    start,
    goal,
    gp,
    C,
    delta_h_max,
    step_length,
    horizon,
    n_heading_candidates,
    x_bounds,
    y_bounds,
    weights,
):
    path = [np.array(start, dtype=float)]

    max_steps = 60
    goal_tol = 0.35

    horizon_sequences = []

    for step in range(max_steps):
        current = path[-1]

        best_sequence, best_cost = find_best_horizon_sequence(
            current=current,
            goal=goal,
            gp=gp,
            C=C,
            delta_h_max=delta_h_max,
            step_length=step_length,
            horizon=horizon,
            n_heading_candidates=n_heading_candidates,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            weights=weights,
        )

        if best_sequence is None:
            print(f"No feasible MPC sequence found at step {step}. Stopping.")
            break

        # Receding horizon: execute only the first step.
        next_step = best_sequence[1]
        path.append(next_step)

        horizon_sequences.append(best_sequence)

        if np.linalg.norm(next_step - goal) < goal_tol:
            print("Goal reached.")
            break

    return np.array(path), horizon_sequences


def plan_greedy_cp_safe_path(
    start,
    goal,
    gp,
    C,
    delta_h_max,
    step_length,
    n_heading_candidates,
    x_bounds,
    y_bounds,
):
    """
    Previous greedy CP-safe planner for comparison.
    """
    path = [np.array(start, dtype=float)]

    z_current_true = true_terrain(start[0], start[1])

    max_steps = 140
    goal_tol = 0.35

    for _ in range(max_steps):
        current = path[-1]

        candidates = generate_candidate_steps(
            current,
            goal,
            step_length,
            n_heading_candidates,
        )

        feasible_candidates = []

        for candidate in candidates:
            if not inside_bounds(candidate, x_bounds, y_bounds):
                continue

            mu_next = gp_mean(gp, candidate)
            cp_margin_value = delta_h_max - abs(mu_next - z_current_true)

            if cp_margin_value < C:
                continue

            dist_to_goal = np.linalg.norm(candidate - goal)
            height_cost = abs(mu_next - z_current_true)

            cost = dist_to_goal + 2.0 * height_cost
            feasible_candidates.append((cost, candidate))

        if len(feasible_candidates) == 0:
            print("No CP-safe greedy candidate found. Stopping greedy path.")
            break

        feasible_candidates.sort(key=lambda item: item[0])
        _, best_candidate = feasible_candidates[0]

        path.append(best_candidate)
        z_current_true = true_terrain(best_candidate[0], best_candidate[1])

        if np.linalg.norm(best_candidate - goal) < goal_tol:
            break

    return np.array(path)


def evaluate_true_step_safety(path, delta_h_max):
    if len(path) < 2:
        return 0, 0, np.array([])

    violations = []
    height_changes = []

    for q in range(len(path) - 1):
        p0 = path[q]
        p1 = path[q + 1]

        z0 = true_terrain(p0[0], p0[1])
        z1 = true_terrain(p1[0], p1[1])

        dh = abs(z1 - z0)
        height_changes.append(dh)

        if dh > delta_h_max:
            violations.append(q)

    return len(violations), len(path) - 1, np.array(height_changes)


def main():
    rng = np.random.default_rng(4)

    # ------------------------------------------------------------
    # Terrain and sparse data
    # ------------------------------------------------------------
    x_min, x_max = -5.0, 5.0
    y_min, y_max = -5.0, 5.0

    n_samples = 180
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

    delta = 0.15
    confidence = 1.0 - delta
    C, scores = conformal_margin(y_cal, y_cal_pred, delta)

    # ------------------------------------------------------------
    # Planner parameters
    # ------------------------------------------------------------
    start = np.array([-4.3, -4.2])
    goal = np.array([4.3, 4.0])

    step_length = 0.45

    # Make this slightly larger than previous demo.
    # The previous value was so conservative that CP-safe path got stuck.
    delta_h_max = 0.095

    horizon = 2
    n_heading_candidates = 7

    weights = {
        "goal": 1.0,
        "slope": 1.5,
        "height": 20.0,
        "progress": 1.0,
    }

    mpc_path, horizon_sequences = plan_mpc_cp_safe_path(
        start=start,
        goal=goal,
        gp=gp,
        C=C,
        delta_h_max=delta_h_max,
        step_length=step_length,
        horizon=horizon,
        n_heading_candidates=n_heading_candidates,
        x_bounds=(x_min, x_max),
        y_bounds=(y_min, y_max),
        weights=weights,
    )

    greedy_path = plan_greedy_cp_safe_path(
        start=start,
        goal=goal,
        gp=gp,
        C=C,
        delta_h_max=delta_h_max,
        step_length=step_length,
        n_heading_candidates=41,
        x_bounds=(x_min, x_max),
        y_bounds=(y_min, y_max),
    )

    mpc_violations, mpc_steps, mpc_dh = evaluate_true_step_safety(mpc_path, delta_h_max)
    greedy_violations, greedy_steps, greedy_dh = evaluate_true_step_safety(greedy_path, delta_h_max)

    print("MPC-style CP-safe footstep demo completed.")
    print(f"Target confidence             = {confidence * 100:.1f}%")
    print(f"Conformal margin C            = {C:.5f} m")
    print(f"Delta h max                   = {delta_h_max:.5f} m")
    print(f"Horizon                       = {horizon}")
    print(f"Heading candidates / step     = {n_heading_candidates}")
    print()
    print("MPC-style CP-safe planner:")
    print(f"  number of steps             = {mpc_steps}")
    print(f"  final distance to goal      = {np.linalg.norm(mpc_path[-1] - goal):.4f} m")
    print(f"  true safety violations      = {mpc_violations}")
    print(f"  max true height change      = {np.max(mpc_dh) if len(mpc_dh) else 0.0:.5f} m")
    print()
    print("Greedy CP-safe planner:")
    print(f"  number of steps             = {greedy_steps}")
    print(f"  final distance to goal      = {np.linalg.norm(greedy_path[-1] - goal):.4f} m")
    print(f"  true safety violations      = {greedy_violations}")
    print(f"  max true height change      = {np.max(greedy_dh) if len(greedy_dh) else 0.0:.5f} m")

    # ------------------------------------------------------------
    # Plot terrain and paths
    # ------------------------------------------------------------
    grid_n = 120
    x_grid = np.linspace(x_min, x_max, grid_n)
    y_grid = np.linspace(y_min, y_max, grid_n)
    Xg, Yg = np.meshgrid(x_grid, y_grid)
    Z_true = true_terrain(Xg, Yg)

    plt.figure(figsize=(8, 7))
    plt.contourf(Xg, Yg, Z_true, levels=35)
    plt.colorbar(label="true terrain height [m]")

    plt.plot(
        greedy_path[:, 0],
        greedy_path[:, 1],
        "o--",
        linewidth=1.3,
        markersize=3.5,
        label="greedy CP-safe path",
    )

    plt.plot(
        mpc_path[:, 0],
        mpc_path[:, 1],
        "o-",
        linewidth=2.2,
        markersize=4,
        label="MPC-style CP-safe path",
    )

    plt.scatter(start[0], start[1], s=100, marker="s", label="start")
    plt.scatter(goal[0], goal[1], s=120, marker="*", label="goal")

    # Show a few predicted horizon sequences
    for seq in horizon_sequences[::max(1, len(horizon_sequences) // 8)]:
        plt.plot(seq[:, 0], seq[:, 1], linewidth=1.0, alpha=0.35)

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("MPC-style CP-safe footstep planning over true terrain")
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(8, 4.5))
    plt.plot(mpc_dh, "o-", label="MPC-style CP-safe true |Δh|")
    plt.plot(greedy_dh, "o--", label="greedy CP-safe true |Δh|")
    plt.axhline(delta_h_max, linestyle="--", linewidth=2, label="Δh max")
    plt.xlabel("step index")
    plt.ylabel("true height change [m]")
    plt.title("True terrain height change per step")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # ------------------------------------------------------------
    # Plot distance-to-goal history
    # ------------------------------------------------------------
    mpc_dist = np.linalg.norm(mpc_path - goal, axis=1)
    greedy_dist = np.linalg.norm(greedy_path - goal, axis=1)

    plt.figure(figsize=(8, 4.5))
    plt.plot(mpc_dist, "o-", label="MPC-style distance to goal")
    plt.plot(greedy_dist, "o--", label="greedy distance to goal")
    plt.xlabel("step index")
    plt.ylabel("distance to goal [m]")
    plt.title("Receding-horizon planning improves goal progress")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()