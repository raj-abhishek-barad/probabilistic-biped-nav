"""
ccm_sdp_solver.py
-----------------
SDP-certified Control Contraction Metric (CCM) for the Augmented LIPM.

Paper: Muenprasitivej et al., "Probabilistically-Safe Bipedal Navigation
       over Uncertain Terrain via Conformal Prediction and Contraction
       Analysis", arXiv:2510.07725, 2025.

This module provides a formally certified CCM (Eq. 9 of the paper) solved
via a semidefinite program (SDP):

    A'M + MA − ρ·M B B' M + 2λ M ≼ 0,   M ≻ 0        (Paper Eq. 9)

Equivalently in the dual variable W = M⁻¹, with B scaled for numerical
conditioning:

    A W + W A' − ρ · B̃ B̃' + 2λ W ≼ 0,   W ≻ 0

where B̃ = α · B  (α is a scaling constant).

The CCM torque law (Paper Eq. 17):

    τ_y^CCM(t) = − ½ρ̃ B̃' M (x(t) − x*(t))  =  K · e(t)

The full Jacobian saltation matrix Ξ (Paper Sec. III-D.2) is computed
analytically at each foot-switch for certified tube propagation.

Design notes
------------
  - B = [0, −ω²/(mg)]' is very small for typical humanoid parameters
    (≈ −0.02). The SDP is numerically conditioned by scaling B → B̃ = α B
    so that B̃ entries are O(1). The resulting K is unscaled before use.
  - ρ is optimised jointly with W in the SDP to find the tightest tube.
  - The LMI residual (max eigenvalue of the SDP constraint at the solution)
    is stored and should be ≤ 0 for a certified result.

Usage
-----
    from ccm_sdp_solver import default_ccm, CCMResult
    ccm = default_ccm()
    assert ccm.verify()
    tau = ccm.torque(error)           # error = x_actual - x_ref
    eps = ccm.tube_radius(t, e0, w_bar)
    Xi  = ccm.saltation_matrix(x_Tm, v_Tm, uf)

Requires: cvxpy ≥ 1.3, numpy, scipy
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import cvxpy as cp
from scipy import linalg


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CCMResult:
    """
    Certified CCM for the Augmented LIPM.

    All matrices are in the original (unscaled) coordinate system.
    """
    M: np.ndarray           # (2,2) PD CCM metric
    K: np.ndarray           # (1,2) feedback gain  K = −½ρ̃ B̃'M
    lambda_rate: float      # certified contraction rate λ
    rho: float              # ρ used in SDP (in scaled coordinates)
    B_scale: float          # α: B̃ = α·B
    omega: float
    m_robot: float
    A: np.ndarray           # (2,2) state matrix
    B: np.ndarray           # (2,1) unscaled control matrix
    Bw: np.ndarray          # (2,1) disturbance matrix
    sdp_status: str
    sdp_residual: float     # max eigenvalue of LMI (≤0 required)
    _W: np.ndarray = field(repr=False)   # dual variable M⁻¹

    # -----------------------------------------------------------------
    # Torque law  (Paper Eq. 17)
    # -----------------------------------------------------------------

    def torque(self, error: np.ndarray) -> float:
        """
        Compute CCM flywheel torque  τ_y = K · e.

        Parameters
        ----------
        error : array-like (2,)
            Tracking error e = x_actual − x_reference.

        Returns
        -------
        float
            Flywheel torque [N·m].
        """
        return float((self.K @ np.asarray(error, dtype=float).reshape(2)).item())

    # -----------------------------------------------------------------
    # Tube bound  (Paper Eq. 10 + Def. 3)
    # -----------------------------------------------------------------

    def tube_radius(self, t: float, e0_norm: float, w_bar: float) -> float:
        """
        Euclidean RCI tube radius converted from the M-metric energy bound.

        The CCM certificate gives a bound in the Riemannian energy

            V = e.T @ M @ e.

        The differential inequality gives

            sqrt(V(t)) <= sqrt(V(0)) exp(-lambda t)
                        + d_bar_M (1 - exp(-lambda t)),

        where

            d_bar_M = || M^{1/2} Bw || w_bar / lambda.

        To compare this bound with the Euclidean tracking error ||e||_2,
        use

            ||e||_2 <= sqrt(V) / sqrt(lambda_min(M)).
        """
        eigvals = np.linalg.eigvalsh(self.M)
        lam_min = max(float(eigvals.min()), 1e-12)
        lam_max = max(float(eigvals.max()), 1e-12)

        M_sqrt = linalg.sqrtm(self.M).real

        d_bar_M = (
            np.linalg.norm(M_sqrt @ self.Bw, ord=2)
            * w_bar
            / self.lambda_rate
        )

        sqrtV0 = np.sqrt(lam_max) * e0_norm

        sqrtV_t = (
            sqrtV0 * np.exp(-self.lambda_rate * t)
            + d_bar_M * (1.0 - np.exp(-self.lambda_rate * t))
        )

        return float(sqrtV_t / np.sqrt(lam_min))
    def steady_tube_radius(self, w_bar: float) -> float:
        """
        Asymptotic Euclidean tube radius converted from the M-metric bound.
        """
        eigvals = np.linalg.eigvalsh(self.M)
        lam_min = max(float(eigvals.min()), 1e-12)

        M_sqrt = linalg.sqrtm(self.M).real

        d_bar_M = (
            np.linalg.norm(M_sqrt @ self.Bw, ord=2)
            * w_bar
            / self.lambda_rate
        )

        return float(d_bar_M / np.sqrt(lam_min))

    # -----------------------------------------------------------------
    # Saltation matrix  (Paper Sec. III-D.2)
    # -----------------------------------------------------------------

    def saltation_matrix(
        self,
        x_loc_Tm: float,
        v_loc_Tm: float,
        uf: float,
        T_switch: float = 0.01,
    ) -> np.ndarray:
        """
        Full Jacobian saltation matrix Ξ at the foot-switching event.

        Paper Sec. III-D.2 (Eq. before Def. 3):

            Ξ = J_Δ + (F⁺ − J_Δ F⁻) J_g' / (J_g' F⁻)

        Guard condition g:
            x_loc(0⁺) = x̃_loc(T⁻_step)    [continuous guard on sagittal pos]

        Reset map Δ:
            x_loc(0⁺) = x_loc(T⁻) + v_loc(T⁻)·T_sw − u_f
            v_loc(0⁺) = v_loc(T⁻)

        Parameters
        ----------
        x_loc_Tm   : sagittal CoM position at step end  x_loc(T⁻)
        v_loc_Tm   : sagittal CoM velocity at step end  v_loc(T⁻)
        uf         : foot placement control u_f
        T_switch   : double-contact transition duration [s]

        Returns
        -------
        Ξ : (2,2) ndarray.  ‖Ξ‖_2 > 1 (expansive).
        """
        J_g     = np.array([[1.0, 0.0]])
        J_delta = np.array([[1.0, T_switch], [0.0, 1.0]])

        F_minus = np.array([v_loc_Tm, self.omega**2 * x_loc_Tm])
        x_post  = x_loc_Tm + v_loc_Tm * T_switch - uf
        F_plus  = np.array([v_loc_Tm, self.omega**2 * x_post])

        denom = float((J_g @ F_minus).item())
        if abs(denom) < 1e-10:
            warnings.warn(
                "Saltation: J_g'F⁻ ≈ 0 (near apex). Returning identity.",
                RuntimeWarning,
            )
            return np.eye(2)

        return J_delta + np.outer(F_plus - J_delta @ F_minus, J_g) / denom

    def propagate_tube_through_impact(
        self,
        eps_Tm: float,
        x_loc_Tm: float,
        v_loc_Tm: float,
        uf: float,
        T_switch: float = 0.01,
    ) -> float:
        """
        Propagate RCI tube bound across foot-switch  (Paper Eq. 18).

            ε̄(0⁺) = ‖Ξ‖_M · ε̄(T⁻)

        where  ‖Ξ‖_M = √λ_max(Ξ'MΞ · M⁻¹)  (M-induced operator norm).
        """
        Xi         = self.saltation_matrix(x_loc_Tm, v_loc_Tm, uf, T_switch)
        Xi_M       = Xi.T @ self.M @ Xi
        norm_Xi_M  = float(np.sqrt(np.linalg.eigvalsh(Xi_M @ self._W).max()))
        return norm_Xi_M * eps_Tm

    # -----------------------------------------------------------------
    # Verification
    # -----------------------------------------------------------------

    def verify(self, tol: float = 1e-3) -> bool:
        """
        Numerical check of the CCM LMI condition (Paper Eq. 9).

        Returns True iff  A'M + MA − ρ M B B' M + 2λ M ≼ tol·I.
        """
        B_tilde = self.B_scale * self.B
        lmi = (
            self.A.T @ self.M
            + self.M @ self.A
            - self.rho * self.M @ B_tilde @ B_tilde.T @ self.M
            + 2.0 * self.lambda_rate * self.M
        )
        return bool(np.linalg.eigvalsh(lmi).max() <= tol)

    def closed_loop_eigenvalues(self) -> np.ndarray:
        """Eigenvalues of A + B·K (closed-loop state matrix)."""
        return np.linalg.eigvals(self.A + self.B @ self.K)

    def summary(self) -> str:
        lines = [
            "CCM (SDP-certified) Summary",
            "=" * 52,
            f"  SDP status            : {self.sdp_status}",
            f"  SDP LMI residual      : {self.sdp_residual:+.6f}  (≤0 required)",
            f"  LMI verified (tol 1e-3): {self.verify()}",
            f"  λ (contraction rate)  : {self.lambda_rate:.4f}",
            f"  ρ (gain, scaled)      : {self.rho:.2f}",
            f"  B scale α             : {self.B_scale:.1f}",
            f"  ω (LIPM frequency)    : {self.omega:.4f} rad/s",
            f"  m (robot mass)        : {self.m_robot:.1f} kg",
            "",
            f"  M = [{self.M[0,0]:.6e}  {self.M[0,1]:.6e}]",
            f"      [{self.M[1,0]:.6e}  {self.M[1,1]:.6e}]",
            "",
            f"  K = [{self.K[0,0]:.4f}  {self.K[0,1]:.4f}]",
            "",
            "  Closed-loop eigenvalues (A + BK):",
        ]
        for ev in self.closed_loop_eigenvalues():
            lines.append(f"    {ev.real:+.4f}{ev.imag:+.4f}j")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# SDP solver
# ---------------------------------------------------------------------------

def solve_ccm(
    omega: float,
    m_robot: float,
    lambda_rate: float = 2.5,
    B_scale: float = 1e4,
    solver: str = "SCS",
    verbose: bool = False,
) -> CCMResult:
    """
    Solve the CCM SDP (Paper Eq. 9) for the Augmented LIPM.

    The SDP is formulated in scaled coordinates to overcome the poor
    numerical conditioning caused by B ≈ [0, −0.02]':

        Dual variables: W = M⁻¹ (2×2 SPD), ρ ≥ 0 (scalar)

        minimise    −log det(W) + γ · ρ
        subject to
            A W + W A' − ρ · B̃ B̃' + 2λ W ≼ −ε I
            W ≻ δ I

    where B̃ = B_scale · B, ε = 1e-3, δ = 1e-2.

    The feedback gain is  K = −½ρ B̃' M  in scaled coordinates.

    Parameters
    ----------
    omega       : LIPM natural frequency √(g/z_H) [rad/s]
    m_robot     : robot mass [kg]
    lambda_rate : desired contraction rate λ > 0
    B_scale     : scaling constant α so B̃ = α·B has O(1) entries
    solver      : CVXPY solver ('SCS' default; 'MOSEK' if licensed)
    verbose     : print solver output

    Returns
    -------
    CCMResult with certified M, K, and metadata.
    """
    A = np.array([[0.0,       1.0    ],
                  [omega**2,  0.0    ]])
    B = np.array([[0.0              ],
                  [-omega**2 / (m_robot * 9.81)]])
    Bw = np.array([[0.0], [1.0]])
    B_tilde = B_scale * B

    n = 2
    W   = cp.Variable((n, n), symmetric=True)
    rho = cp.Variable(pos=True)

    lmi = A @ W + W @ A.T - rho * B_tilde @ B_tilde.T + 2.0 * lambda_rate * W

    constraints = [
        lmi << -1e-3 * np.eye(n),
        W >> 1e-2 * np.eye(n),
        rho >= 1.0,
    ]
    # Regularise: minimise −log det W + small rho penalty
    objective = cp.Minimize(-cp.log_det(W) + 1e-2 * rho)
    prob = cp.Problem(objective, constraints)

    solve_kwargs = dict(verbose=verbose, eps=1e-7, max_iters=50000)
    if solver == "SCS":
        solve_kwargs["acceleration_lookback"] = 10
    prob.solve(solver=getattr(cp, solver), **solve_kwargs)

    if W.value is None:
        raise RuntimeError(
            f"CCM SDP failed with status '{prob.status}'. "
            "Try increasing B_scale or switching solver to MOSEK."
        )

    W_val   = W.value
    rho_val = float(rho.value)
    M_val   = np.linalg.inv(W_val)

    # K in original coordinates: τ = K e
    K_virtual = -0.5 * rho_val * B_tilde.T @ M_val
    K_val = B_scale * K_virtual

    # Verify LMI
    lmi_val = (
        A.T @ M_val + M_val @ A
        - rho_val * M_val @ B_tilde @ B_tilde.T @ M_val
        + 2.0 * lambda_rate * M_val
    )
    residual = float(np.linalg.eigvalsh(lmi_val).max())

    return CCMResult(
        M=M_val, K=K_val,
        lambda_rate=lambda_rate, rho=rho_val, B_scale=B_scale,
        omega=omega, m_robot=m_robot,
        A=A, B=B, Bw=Bw,
        sdp_status=prob.status, sdp_residual=residual,
        _W=W_val,
    )


def default_ccm(
    g: float = 9.81,
    z_H: float = 0.9,
    m_robot: float = 50.0,
    lambda_rate: float = 2.5,
    B_scale: float = 1e4,
) -> CCMResult:
    """Build a certified CCM with the paper's default LIPM parameters."""
    omega = np.sqrt(g / z_H)
    return solve_ccm(omega, m_robot, lambda_rate=lambda_rate, B_scale=B_scale)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Solving CCM SDP for Aug-LIPM system ...")
    ccm = default_ccm()
    print(ccm.summary())
    print()
    print(f"LMI verified (tol=1e-3) : {ccm.verify()}")
    print(f"Steady tube (w=0.20 m/s²): {ccm.steady_tube_radius(0.20):.5f} m")

    Xi = ccm.saltation_matrix(-0.05, 0.60, -0.02)
    print(f"Saltation ||Xi||_2       : {np.linalg.norm(Xi, 2):.4f}")

    eps_after = ccm.propagate_tube_through_impact(0.012, -0.05, 0.60, -0.02)
    print(f"Tube after impact        : {eps_after:.5f} m")

    e_test = np.array([0.04, 0.15])
    print(f"Torque (e=[0.04,0.15])   : {ccm.torque(e_test):.2f} N·m")
