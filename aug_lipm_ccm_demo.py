import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Augmented LIPM + CCM-like flywheel torque demo
# Inspired by:
# Probabilistically-Safe Bipedal Navigation over Uncertain Terrain
# via Conformal Prediction and Contraction Analysis
#
# This is NOT yet MuJoCo.
# This is the reduced-order sagittal dynamics layer.
# ============================================================


def lipm_reference(t, x0, v0, uf, omega):
    """
    Nominal LIPM closed-form trajectory.

    Paper form:
        x(t) = x0 + sinh(omega t)/omega * v0
               + (1 - cosh(omega t)) * uf

        v(t) = cosh(omega t) * v0
               - omega sinh(omega t) * uf

    Here x is sagittal CoM position relative to stance foot.
    uf is the sagittal foot placement relative to CoM.
    """

    x_ref = x0 + np.sinh(omega * t) / omega * v0 + (1.0 - np.cosh(omega * t)) * uf
    v_ref = np.cosh(omega * t) * v0 - omega * np.sinh(omega * t) * uf

    return np.array([x_ref, v_ref])


def disturbance(t):
    """
    Bounded terrain/model disturbance.

    In the paper, terrain uncertainty enters the Aug-LIPM
    through w_terrain. Here we use a bounded sinusoidal disturbance
    to imitate terrain-induced sagittal acceleration mismatch.
    """
    return 0.6 * np.sin(2.0 * np.pi * 1.2 * t)


def simulate(use_ccm=True):
    # -----------------------------
    # Physical / LIPM parameters
    # -----------------------------
    g = 9.81
    zH = 0.9
    omega = np.sqrt(g / zH)

    m = 50.0  # simplified humanoid mass proxy [kg]

    # One walking step duration
    T_step = 0.8

    # Integration
    dt = 0.001
    t_grid = np.arange(0.0, T_step + dt, dt)

    # Desired initial local sagittal state
    x0_ref = -0.12
    v0_ref = 0.55

    # Foot placement input
    uf = -0.02

    # Actual initial state is perturbed from nominal
    x = np.array([-0.18, 0.25], dtype=float)

    # -----------------------------
    # Aug-LIPM matrices
    # xdot = A x + B tau_y + Bw w
    # -----------------------------
    A = np.array([
        [0.0, 1.0],
        [omega**2, 0.0]
    ])

    B = np.array([
        [0.0],
        [-omega**2 / (m * g)]
    ])

    Bw = np.array([
        [0.0],
        [1.0]
    ])

    # -----------------------------
    # CCM-like gain
    # -----------------------------
    # In the paper:
    # tau_y = -1/2 rho B^T M (x - x*)
    #
    # Here we choose M and rho manually first.
    # Later, we can compute M from an SDP.
    M = np.array([
        [40.0, 4.0],
        [4.0, 8.0]
    ])

    rho = 5000.0

    xs = []
    xrefs = []
    taus = []
    errors = []
    ws = []

    for t in t_grid:
        x_ref = lipm_reference(t, x0_ref, v0_ref, uf, omega)
        e = x - x_ref

        if use_ccm:
            tau_y = -0.5 * rho * (B.T @ M @ e.reshape(2, 1)).item()
        else:
            tau_y = 0.0

        w = disturbance(t)

        xdot = A @ x + (B.flatten() * tau_y) + (Bw.flatten() * w)

        # Euler integration is enough for this first reduced-order demo
        x = x + dt * xdot

        xs.append(x.copy())
        xrefs.append(x_ref.copy())
        taus.append(tau_y)
        errors.append(np.linalg.norm(e))
        ws.append(w)

    return {
        "t": t_grid,
        "x": np.array(xs),
        "xref": np.array(xrefs),
        "tau": np.array(taus),
        "error": np.array(errors),
        "w": np.array(ws),
        "omega": omega,
        "T_step": T_step,
    }


def main():
    no_ccm = simulate(use_ccm=False)
    with_ccm = simulate(use_ccm=True)

    t = with_ccm["t"]

    print("Aug-LIPM + CCM demo completed.")
    print(f"omega = {with_ccm['omega']:.3f} rad/s")
    print(f"T_step = {with_ccm['T_step']:.3f} s")
    print()
    print(f"Final tracking error without CCM = {no_ccm['error'][-1]:.4f}")
    print(f"Final tracking error with CCM    = {with_ccm['error'][-1]:.4f}")
    print(f"Max |tau_y| with CCM             = {np.max(np.abs(with_ccm['tau'])):.4f} N m")

    # ------------------------------------------------------------
    # Plot 1: sagittal position
    # ------------------------------------------------------------
    plt.figure(figsize=(8, 4.5))
    plt.plot(t, with_ccm["xref"][:, 0], label="desired LIPM x*(t)")
    plt.plot(t, no_ccm["x"][:, 0], label="actual x(t), no CCM")
    plt.plot(t, with_ccm["x"][:, 0], label="actual x(t), with CCM")
    plt.xlabel("time [s]")
    plt.ylabel("sagittal CoM position x_loc [m]")
    plt.title("Aug-LIPM sagittal position tracking")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # ------------------------------------------------------------
    # Plot 2: sagittal velocity
    # ------------------------------------------------------------
    plt.figure(figsize=(8, 4.5))
    plt.plot(t, with_ccm["xref"][:, 1], label="desired LIPM v*(t)")
    plt.plot(t, no_ccm["x"][:, 1], label="actual v(t), no CCM")
    plt.plot(t, with_ccm["x"][:, 1], label="actual v(t), with CCM")
    plt.xlabel("time [s]")
    plt.ylabel("sagittal CoM velocity v_loc [m/s]")
    plt.title("Aug-LIPM sagittal velocity tracking")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # ------------------------------------------------------------
    # Plot 3: phase portrait
    # ------------------------------------------------------------
    plt.figure(figsize=(6, 5))
    plt.plot(with_ccm["xref"][:, 0], with_ccm["xref"][:, 1], label="desired phase trajectory")
    plt.plot(no_ccm["x"][:, 0], no_ccm["x"][:, 1], label="actual phase, no CCM")
    plt.plot(with_ccm["x"][:, 0], with_ccm["x"][:, 1], label="actual phase, with CCM")
    plt.xlabel("x_loc [m]")
    plt.ylabel("v_loc [m/s]")
    plt.title("Sagittal phase portrait")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # ------------------------------------------------------------
    # Plot 4: tracking error
    # ------------------------------------------------------------
    plt.figure(figsize=(8, 4.5))
    plt.plot(t, no_ccm["error"], label="tracking error, no CCM")
    plt.plot(t, with_ccm["error"], label="tracking error, with CCM")
    plt.xlabel("time [s]")
    plt.ylabel("||x - x*||")
    plt.title("Tracking error comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # ------------------------------------------------------------
    # Plot 5: CCM torque and disturbance
    # ------------------------------------------------------------
    plt.figure(figsize=(8, 4.5))
    plt.plot(t, with_ccm["tau"], label="CCM flywheel torque tau_y")
    plt.xlabel("time [s]")
    plt.ylabel("torque [N m]")
    plt.title("CCM flywheel torque command")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(8, 4.5))
    plt.plot(t, with_ccm["w"], label="bounded disturbance w(t)")
    plt.xlabel("time [s]")
    plt.ylabel("disturbance acceleration [m/s²]")
    plt.title("Terrain/model disturbance proxy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()