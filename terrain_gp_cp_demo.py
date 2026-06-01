import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# GP + Split Conformal Prediction Terrain Demo
# Inspired by:
# "Probabilistically-Safe Bipedal Navigation over Uncertain
#  Terrain via Conformal Prediction and Contraction Analysis"
#
# This implements the paper's first block:
# sparse terrain data -> GP mean -> CP interval
# ============================================================


def true_terrain(x, y):
    """
    Synthetic rough terrain height map z = g(x, y).

    This plays the role of the unknown true terrain.
    In the paper, the terrain is learned from sparse samples.
    """
    z = (
        0.18 * np.sin(0.8 * x)
        + 0.12 * np.cos(0.7 * y)
        + 0.08 * np.sin(1.5 * x + 0.6 * y)
        + 0.04 * np.cos(2.0 * x - 1.2 * y)
    )
    return z


def rbf_kernel(X1, X2, length_scale=1.2, sigma_f=1.0):
    """
    Squared exponential / RBF kernel.

    X1: shape (N1, 2)
    X2: shape (N2, 2)

    k(x, x') = sigma_f^2 exp(-||x-x'||^2 / (2 l^2))
    """
    X1_sq = np.sum(X1**2, axis=1).reshape(-1, 1)
    X2_sq = np.sum(X2**2, axis=1).reshape(1, -1)

    sqdist = X1_sq + X2_sq - 2.0 * X1 @ X2.T

    return sigma_f**2 * np.exp(-0.5 * sqdist / length_scale**2)


class GaussianProcessRegressorFromScratch:
    """
    Minimal GP regression from scratch.

    We avoid sklearn here so that the math is visible.
    """

    def __init__(self, length_scale=1.2, sigma_f=1.0, sigma_n=0.03):
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

        # Cholesky factorization for numerical stability
        self.L = np.linalg.cholesky(K)

        # alpha = K^{-1} y using triangular solves
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
    """
    Split conformal margin.

    Nonconformity score:
        R_i = |z_i - mu(x_i)|

    Paper form:
        C = R_(p), p = ceil((k+1)(1-delta))

    where k is number of calibration samples.
    """
    scores = np.abs(y_cal_true - y_cal_pred)
    scores_sorted = np.sort(scores)

    k = len(scores_sorted)
    p = int(np.ceil((k + 1) * (1.0 - delta)))

    # Convert 1-based p to Python index.
    # If p > k, use the largest calibration score.
    idx = min(p - 1, k - 1)

    C = scores_sorted[idx]

    return C, scores


def main():
    rng = np.random.default_rng(4)

    # ------------------------------------------------------------
    # 1. Create true terrain grid
    # ------------------------------------------------------------
    x_min, x_max = -5.0, 5.0
    y_min, y_max = -5.0, 5.0

    grid_n = 80

    x_grid = np.linspace(x_min, x_max, grid_n)
    y_grid = np.linspace(y_min, y_max, grid_n)

    Xg, Yg = np.meshgrid(x_grid, y_grid)

    Z_true = true_terrain(Xg, Yg)

    X_query = np.column_stack([Xg.ravel(), Yg.ravel()])

    # ------------------------------------------------------------
    # 2. Sparse noisy terrain measurements
    # ------------------------------------------------------------
    n_samples = 300

    X_samples = np.column_stack([
        rng.uniform(x_min, x_max, n_samples),
        rng.uniform(y_min, y_max, n_samples),
    ])

    noise_std = 0.025
    z_samples_clean = true_terrain(X_samples[:, 0], X_samples[:, 1])
    z_samples = z_samples_clean + rng.normal(0.0, noise_std, size=n_samples)

    # ------------------------------------------------------------
    # 3. Train/calibration split
    # ------------------------------------------------------------
    X_train, y_train, X_cal, y_cal = split_train_calibration(
        X_samples, z_samples, train_fraction=0.7, seed=10
    )

    # ------------------------------------------------------------
    # 4. GP fit on training set
    # ------------------------------------------------------------
    gp = GaussianProcessRegressorFromScratch(
        length_scale=1.1,
        sigma_f=0.35,
        sigma_n=noise_std,
    )

    gp.fit(X_train, y_train)

    # ------------------------------------------------------------
    # 5. Conformal calibration
    # ------------------------------------------------------------
    y_cal_pred, y_cal_var = gp.predict(X_cal)

    delta = 0.15
    confidence = 1.0 - delta

    C, scores = conformal_margin(y_cal, y_cal_pred, delta)

    # ------------------------------------------------------------
    # 6. Predict on dense terrain grid
    # ------------------------------------------------------------
    mu_grid, var_grid = gp.predict(X_query)

    Z_mu = mu_grid.reshape(grid_n, grid_n)
    Z_std = np.sqrt(var_grid).reshape(grid_n, grid_n)

    Z_lower_cp = Z_mu - C
    Z_upper_cp = Z_mu + C

    # For comparison: ordinary GP 85% Gaussian interval
    # Normal quantile approx for 85% two-sided interval is about 1.44.
    gaussian_factor = 1.44
    Z_lower_gp = Z_mu - gaussian_factor * Z_std
    Z_upper_gp = Z_mu + gaussian_factor * Z_std

    # ------------------------------------------------------------
    # 7. Empirical coverage test on grid
    # ------------------------------------------------------------
    cp_inside = np.logical_and(Z_true >= Z_lower_cp, Z_true <= Z_upper_cp)
    gp_inside = np.logical_and(Z_true >= Z_lower_gp, Z_true <= Z_upper_gp)

    cp_coverage = np.mean(cp_inside)
    gp_coverage = np.mean(gp_inside)

    rmse = np.sqrt(np.mean((Z_true - Z_mu) ** 2))

    print("GP + Split Conformal Prediction terrain demo completed.")
    print(f"Number of samples          = {n_samples}")
    print(f"Training samples           = {len(X_train)}")
    print(f"Calibration samples        = {len(X_cal)}")
    print(f"Target confidence          = {confidence * 100:.1f}%")
    print(f"Conformal margin C         = {C:.5f} m")
    print(f"Grid RMSE of GP mean       = {rmse:.5f} m")
    print(f"Empirical CP coverage      = {cp_coverage * 100:.2f}%")
    print(f"Empirical GP coverage      = {gp_coverage * 100:.2f}%")

    # ------------------------------------------------------------
    # 8. Plot terrain maps
    # ------------------------------------------------------------
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(Xg, Yg, Z_true, linewidth=0, alpha=0.85)
    ax.scatter(
        X_samples[:, 0],
        X_samples[:, 1],
        z_samples,
        s=10,
        label="sparse noisy samples",
    )
    ax.set_title("True terrain with sparse measurements")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("terrain height z [m]")
    ax.legend()
    plt.tight_layout()

    plt.figure(figsize=(8, 6))
    plt.contourf(Xg, Yg, Z_true, levels=30)
    plt.scatter(X_train[:, 0], X_train[:, 1], s=10, label="train samples")
    plt.scatter(X_cal[:, 0], X_cal[:, 1], s=10, label="calibration samples")
    plt.colorbar(label="true terrain height [m]")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("True terrain and sparse train/calibration samples")
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(8, 6))
    plt.contourf(Xg, Yg, Z_mu, levels=30)
    plt.colorbar(label="GP mean height [m]")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("GP mean terrain estimate")
    plt.tight_layout()

    plt.figure(figsize=(8, 6))
    plt.contourf(Xg, Yg, np.abs(Z_true - Z_mu), levels=30)
    plt.colorbar(label="absolute prediction error [m]")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("GP terrain estimation error")
    plt.tight_layout()

    plt.figure(figsize=(8, 6))
    plt.contourf(Xg, Yg, Z_std, levels=30)
    plt.colorbar(label="GP posterior std [m]")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("GP posterior uncertainty")
    plt.tight_layout()

    # ------------------------------------------------------------
    # 9. Slice plot like the paper's terrain confidence section
    # ------------------------------------------------------------
    slice_idx = grid_n // 2

    plt.figure(figsize=(9, 5))
    plt.plot(x_grid, Z_true[slice_idx, :], label="true terrain")
    plt.plot(x_grid, Z_mu[slice_idx, :], label="GP mean")

    plt.fill_between(
        x_grid,
        Z_lower_cp[slice_idx, :],
        Z_upper_cp[slice_idx, :],
        alpha=0.25,
        label=f"CP interval, {confidence * 100:.0f}%",
    )

    plt.fill_between(
        x_grid,
        Z_lower_gp[slice_idx, :],
        Z_upper_gp[slice_idx, :],
        alpha=0.18,
        label="GP Gaussian interval",
    )

    plt.xlabel("x [m] at fixed y = 0")
    plt.ylabel("terrain height z [m]")
    plt.title("Terrain height prediction interval: GP vs CP")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # ------------------------------------------------------------
    # 10. Calibration score distribution
    # ------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.hist(scores, bins=25, alpha=0.75)
    plt.axvline(C, linestyle="--", linewidth=2, label=f"C = {C:.4f} m")
    plt.xlabel("nonconformity score |z - mu(x,y)| [m]")
    plt.ylabel("count")
    plt.title("Calibration residuals used for conformal prediction")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()