import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CP-safe footstep planning demo
#
# This implements the paper's Lemma 1 idea:
#
#   c(z_q, z_{q+1}) = Delta_h_max - |z_{q+1} - z_q|
#
# Since the future true height is unknown, enforce:
#
#   Delta_h_max - |mu(x_{q+1}, y_{q+1}) - z_q| >= C
#
# where C is the conformal prediction margin.
# ============================================================


def true_terrain(x, y):
    z = (
        0.18 * np.sin(0.8 * x)
        + 0.12 * np.cos(0.7 * y)
        + 0.08 * np.sin(1.5 * x + 0.6 * y)
        + 0.04 * np.cos(2.0 * x - 1.2 * y)
    )
    return z


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


def nearest_grid_index(x_grid, y_grid, point):
    x, y = point
    ix = np.argmin(np.abs(x_grid - x))
    iy = np.argmin(np.abs(y_grid - y))
    return ix, iy


def plan_cp_safe_path(
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
    Greedy CP-safe footstep planner.

    This is not yet full MPC. It is a direct implementation of the
    CP-safe step constraint. At each step, we sample candidate headings,
    reject CP-unsafe steps, and choose the candidate closest to the goal.
    """

    path = [np.array(start, dtype=float)]

    z_current_true = true_terrain(start[0], start[1])

    max_steps = 120
    goal_tol = 0.35

    for _ in range(max_steps):
        current = path[-1]
        direction_to_goal = np.arctan2(goal[1] - current[1], goal[0] - current[0])

        headings = np.linspace(
            direction_to_goal - np.pi / 2,
            direction_to_goal + np.pi / 2,
            n_heading_candidates,
        )

        candidates = []

        for heading in headings:
            candidate = current + step_length * np.array([np.cos(heading), np.sin(heading)])

            if not (x_bounds[0] <= candidate[0] <= x_bounds[1]):
                continue

            if not (y_bounds[0] <= candidate[1] <= y_bounds[1]):
                continue

            mu_next, _ = gp.predict(candidate.reshape(1, 2))
            mu_next = mu_next.item()

            # Paper-style CP-safe constraint:
            # Delta_h_max - |mu_next - z_current_true| >= C
            cp_margin_value = delta_h_max - abs(mu_next - z_current_true)

            is_cp_safe = cp_margin_value >= C

            if not is_cp_safe:
                continue

            dist_to_goal = np.linalg.norm(candidate - goal)

            # Mild slope/height penalty through predicted height difference
            height_cost = abs(mu_next - z_current_true)

            cost = dist_to_goal + 2.0 * height_cost

            candidates.append((cost, candidate, mu_next, cp_margin_value))

        if len(candidates) == 0:
            print("No CP-safe candidate found. Stopping path.")
            break

        candidates.sort(key=lambda item: item[0])
        _, best_candidate, best_mu, best_margin = candidates[0]

        path.append(best_candidate)

        # In receding-horizon walking, after taking the step,
        # the current foot height is measured.
        z_current_true = true_terrain(best_candidate[0], best_candidate[1])

        if np.linalg.norm(best_candidate - goal) < goal_tol:
            print("Goal reached.")
            break

    return np.array(path)


def plan_naive_gp_path(
    start,
    goal,
    gp,
    delta_h_max,
    step_length,
    n_heading_candidates,
    x_bounds,
    y_bounds,
):
    """
    Naive planner using only GP mean.
    It ignores the CP margin C.
    """

    path = [np.array(start, dtype=float)]

    z_current_true = true_terrain(start[0], start[1])

    max_steps = 120
    goal_tol = 0.35

    for _ in range(max_steps):
        current = path[-1]
        direction_to_goal = np.arctan2(goal[1] - current[1], goal[0] - current[0])

        headings = np.linspace(
            direction_to_goal - np.pi / 2,
            direction_to_goal + np.pi / 2,
            n_heading_candidates,
        )

        candidates = []

        for heading in headings:
            candidate = current + step_length * np.array([np.cos(heading), np.sin(heading)])

            if not (x_bounds[0] <= candidate[0] <= x_bounds[1]):
                continue

            if not (y_bounds[0] <= candidate[1] <= y_bounds[1]):
                continue

            mu_next, _ = gp.predict(candidate.reshape(1, 2))
            mu_next = mu_next.item()

            # Naive predicted safety:
            # Delta_h_max - |mu_next - z_current_true| >= 0
            naive_margin_value = delta_h_max - abs(mu_next - z_current_true)

            is_naive_safe = naive_margin_value >= 0.0

            if not is_naive_safe:
                continue

            dist_to_goal = np.linalg.norm(candidate - goal)
            height_cost = abs(mu_next - z_current_true)

            cost = dist_to_goal + 2.0 * height_cost

            candidates.append((cost, candidate, mu_next, naive_margin_value))

        if len(candidates) == 0:
            print("No naive-safe candidate found. Stopping path.")
            break

        candidates.sort(key=lambda item: item[0])
        _, best_candidate, best_mu, best_margin = candidates[0]

        path.append(best_candidate)

        z_current_true = true_terrain(best_candidate[0], best_candidate[1])

        if np.linalg.norm(best_candidate - goal) < goal_tol:
            break

    return np.array(path)


def evaluate_true_step_safety(path, delta_h_max):
    """
    Check true safety using the unknown true terrain.
    This is only for evaluation.
    """

    if len(path) < 2:
        return 0, 0, []

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

    n_samples = 300
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
    # Footstep planning parameters
    # ------------------------------------------------------------
    start = np.array([-4.3, -4.2])
    goal = np.array([4.3, 4.0])

    step_length = 0.45
    delta_h_max = 0.075
    n_heading_candidates = 41

    cp_path = plan_cp_safe_path(
        start=start,
        goal=goal,
        gp=gp,
        C=C,
        delta_h_max=delta_h_max,
        step_length=step_length,
        n_heading_candidates=n_heading_candidates,
        x_bounds=(x_min, x_max),
        y_bounds=(y_min, y_max),
    )

    naive_path = plan_naive_gp_path(
        start=start,
        goal=goal,
        gp=gp,
        delta_h_max=delta_h_max,
        step_length=step_length,
        n_heading_candidates=n_heading_candidates,
        x_bounds=(x_min, x_max),
        y_bounds=(y_min, y_max),
    )

    cp_violations, cp_steps, cp_dh = evaluate_true_step_safety(cp_path, delta_h_max)
    naive_violations, naive_steps, naive_dh = evaluate_true_step_safety(naive_path, delta_h_max)

    print("CP-safe footstep demo completed.")
    print(f"Target confidence             = {confidence * 100:.1f}%")
    print(f"Conformal margin C            = {C:.5f} m")
    print(f"Delta h max                   = {delta_h_max:.5f} m")
    print()
    print("CP-safe planner:")
    print(f"  number of steps             = {cp_steps}")
    print(f"  true safety violations      = {cp_violations}")
    print(f"  max true height change      = {np.max(cp_dh) if len(cp_dh) else 0.0:.5f} m")
    print()
    print("Naive GP planner:")
    print(f"  number of steps             = {naive_steps}")
    print(f"  true safety violations      = {naive_violations}")
    print(f"  max true height change      = {np.max(naive_dh) if len(naive_dh) else 0.0:.5f} m")

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
        naive_path[:, 0],
        naive_path[:, 1],
        "o--",
        linewidth=1.5,
        markersize=4,
        label="naive GP path",
    )

    plt.plot(
        cp_path[:, 0],
        cp_path[:, 1],
        "o-",
        linewidth=2.2,
        markersize=4,
        label="CP-safe path",
    )

    plt.scatter(start[0], start[1], s=100, marker="s", label="start")
    plt.scatter(goal[0], goal[1], s=120, marker="*", label="goal")

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Footstep paths over true terrain")
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(8, 4.5))
    plt.plot(cp_dh, "o-", label="CP-safe true |Δh|")
    plt.plot(naive_dh, "o--", label="naive GP true |Δh|")
    plt.axhline(delta_h_max, linestyle="--", linewidth=2, label="Δh max")
    plt.xlabel("step index")
    plt.ylabel("true height change [m]")
    plt.title("True terrain height change per step")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()