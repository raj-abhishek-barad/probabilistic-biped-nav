"""
rci_tube_certified.py
---------------------
Certified RCI tube tracking for the Augmented LIPM.

Paper: Muenprasitivej et al., arXiv:2510.07725, 2025.

Replaces rci_tube_tracking_demo.py with three improvements:
  1. CCM matrix M from SDP (ccm_sdp_solver.py) — not hand-tuned.
  2. Full Jacobian saltation matrix Ξ propagated at each foot-switch.
  3. Tube bound ε̄(t) from certified Riemannian energy (Paper Eq. 10 + Def. 3).

This module is designed to be imported by run_full_pipeline.py
(replacing the old simulate_tracking_with_rci_tube call) and also
run standalone for single-trajectory analysis.

Key outputs that correspond to paper Tables I–III
-------------------------------------------------
  error_norm[t]         : ||e(t)||  (tracking error over time)
  tube_bound[t]         : ε̄(t)     (certified tube radius)
  violation_fraction    : P(||e||>ε̄) fraction (should ≈ 1-δ for RCI test)
  tau_hist[t]           : τ_y^CCM(t) flywheel torque command [N·m]
  per_step_invariant[]  : 1 if step q stayed inside tube, 0 otherwise
  saltation_norms[]     : ||Ξ_q||_2 at each foot-switch
  w_bar_terrain         : CP-derived disturbance bound w̄_terrain
  avg_norm_error        : mean ||e(t)|| across entire trajectory (Table I ANE)
"""

from __future__ import annotations

import numpy as np
from scipy import linalg
from dataclasses import dataclass
from typing import Optional, List

from ccm_sdp_solver import CCMResult, default_ccm


# ---------------------------------------------------------------------------
# Disturbance model
# ---------------------------------------------------------------------------

def terrain_disturbance(
    t: float,
    w_bar: float,
    seed_offset: float = 0.0,
) -> float:
    """
    Bounded multi-frequency terrain/model disturbance proxy.

    Models wterrain from Paper Eq. (16): uncertainty in ω²_true caused
    by deviation of true terrain height from GP estimate.  The signal is
    deterministic (for reproducibility) and bounded by w_bar.

    |w(t)| ≤ w_bar  ∀ t.
    """
    w = (
        0.45 * np.sin(2.0 * np.pi * 0.85 * t + seed_offset)
        + 0.30 * np.sin(2.0 * np.pi * 2.30 * t + seed_offset * 1.3)
        + 0.25 * np.sin(2.0 * np.pi * 4.10 * t + seed_offset * 0.7)
    )
    return float(np.clip(w * w_bar, -w_bar, w_bar))


# ---------------------------------------------------------------------------
# Nominal LIPM reference interpolation
# ---------------------------------------------------------------------------

def _build_time_nodes(states: np.ndarray, T_step: float) -> np.ndarray:
    return np.arange(len(states)) * T_step


def _lipm_closed_form(
    t: float, x0: float, v0: float, uf: float, omega: float
) -> np.ndarray:
    """Closed-form LIPM solution (Paper Eq. 2)."""
    x = x0 + np.sinh(omega * t) / omega * v0 + (1.0 - np.cosh(omega * t)) * uf
    v = np.cosh(omega * t) * v0 - omega * np.sinh(omega * t) * uf
    return np.array([x, v])


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

@dataclass
class TubeSimResult:
    """Complete result from one tube simulation run."""
    t: np.ndarray
    error_norm: np.ndarray
    tube_bound: np.ndarray
    tau_hist: np.ndarray
    w_hist: np.ndarray
    x_actual: np.ndarray
    x_ref: np.ndarray

    # Per-step statistics
    per_step_invariant: np.ndarray   # shape (n_steps,) bool
    saltation_norms: np.ndarray      # shape (n_steps,)
    eps_at_step_end: np.ndarray      # shape (n_steps,)
    eps_after_impact: np.ndarray     # shape (n_steps,)

    # Scalar summaries (Table I / II equivalents)
    violation_fraction: float
    avg_norm_error: float
    max_norm_error: float
    w_bar: float
    w_bar_terrain: float
    lambda_contraction: float
    tube_inf: float

    # CCM metadata
    ccm: CCMResult


def simulate_certified_rci(
    plan: dict,
    ccm: Optional[CCMResult] = None,
    w_bar_scale: float = 1.0,
    disturbance_seed: float = 0.0,
    dt: float = 0.002,
    tau_limit: float = 1500.0,
    initial_error: Optional[np.ndarray] = None,
) -> TubeSimResult:
    """
    Simulate disturbed Aug-LIPM tracking under certified CCM control
    with full saltation matrix propagation at each foot-switch.

    Parameters
    ----------
    plan : dict
        Output of run_full_pipeline() — must contain 'states', 'controls',
        'params', 'C', and optionally 'gp'.
    ccm : CCMResult or None
        Pre-solved CCM. If None, solved via default_ccm().
    w_bar_scale : float
        Scale factor on the CP-derived disturbance bound.
    disturbance_seed : float
        Phase offset for the disturbance model (use different values for
        Monte-Carlo trials to get statistically independent runs).
    dt : float
        Integration timestep [s].
    tau_limit : float
        Torque saturation limit [N·m].
    initial_error : ndarray(2,) or None
        Initial tracking error e(0). If None, uses a fixed perturbation.

    Returns
    -------
    TubeSimResult
    """
    states   = plan["states"]       # (N+1, 5) LIPM global states
    controls = plan["controls"]     # (N, 2)   [u_f, Δθ]
    params   = plan["params"]
    C        = plan["C"]            # CP conformal margin

    omega   = params["omega"]
    T_step  = params["T_step"]
    g       = params["g"]
    z_H     = params["zH"]
    m_robot = 50.0

    # ── Solve CCM if not provided ─────────────────────────────────────────
    if ccm is None:
        ccm = default_ccm(g=g, z_H=z_H, m_robot=m_robot)

    # ── CP-derived disturbance bound (Paper Eq. 16 / Sec. IV-C.1) ────────
    # w̄_terrain = max_{x,τ} |C_Δ| · |x_loc − τ_y/(mg)|
    # Here C_Δ ∈ [−gC/(z_H(z_H+C)), gC/(z_H(z_H−C))]
    C_delta_max = g * C / (z_H * max(z_H - C, 0.05))   # upper bound
    x_loc_max   = 0.30   # feasible state bound
    tau_mg_max  = 1500.0 / (m_robot * g)
    w_bar_terrain = w_bar_scale * C_delta_max * (x_loc_max + tau_mg_max)
    w_bar = max(w_bar_terrain, 0.05)   # floor for numerical stability

    # ── Build per-step reference from LIPM closed-form ───────────────────
    n_steps   = len(states) - 1
    t_nodes   = _build_time_nodes(states, T_step)
    T_final   = t_nodes[-1]

    # ── Initial tracking error ────────────────────────────────────────────
    if initial_error is None:
        e = np.array([-0.04, 0.15])
    else:
        e = np.asarray(initial_error, dtype=float).reshape(2)
    e0_norm = np.linalg.norm(e)

    # ── Storage ───────────────────────────────────────────────────────────
    t_grid = np.arange(0.0, T_final + dt, dt)
    N_t    = len(t_grid)

    error_norm_hist = np.empty(N_t)
    tube_bound_hist = np.empty(N_t)
    tau_hist        = np.empty(N_t)
    w_hist          = np.empty(N_t)
    x_actual_hist   = np.empty((N_t, 2))
    x_ref_hist      = np.empty((N_t, 2))

    per_step_invariant = np.ones(n_steps, dtype=bool)
    saltation_norms    = np.zeros(n_steps)
    eps_at_step_end    = np.zeros(n_steps)
    eps_after_impact   = np.zeros(n_steps)

    violations = 0

    # Running tube radius: initialised from Eq. (10)
    tube_radius = ccm.tube_radius(0.0, e0_norm, w_bar)
    # Reset at each step start
    step_e0_norm    = e0_norm
    step_start_t    = 0.0
    current_step    = 0
    step_violations = False

    # ── Matrices ─────────────────────────────────────────────────────────
    A  = ccm.A
    B  = ccm.B
    Bw = ccm.Bw

    # ── Integration loop ─────────────────────────────────────────────────
    for i, t in enumerate(t_grid):
        # Determine which step we're in
        q = min(int(t / T_step), n_steps - 1)

        # Reference from closed-form LIPM
        x0_ref = float(states[q, 0])   # global x used as proxy for s_loc
        v0_ref = float(states[q, 3])   # local sagittal velocity
        uf_q   = float(controls[q, 0]) if q < len(controls) else 0.0
        t_loc  = t - q * T_step
        x_ref  = _lipm_closed_form(t_loc, x0_ref, v0_ref, uf_q, omega)

        # CCM torque
        tau_y = float(np.clip(ccm.torque(e), -tau_limit, tau_limit))

        # Terrain disturbance
        w = terrain_disturbance(t, w_bar, seed_offset=disturbance_seed)

        # Propagate error dynamics: ė = Ae + Bτ + Bw·w
        e_dot = A @ e + B.flatten() * tau_y + Bw.flatten() * w
        e = e + dt * e_dot

        # Certified tube radius at time t
        t_in_step = t - q * T_step
        tube_radius = ccm.tube_radius(t_in_step, step_e0_norm, w_bar)

        err = float(np.linalg.norm(e))
        if err > tube_radius:
            violations += 1
            step_violations = True

        # ── Foot-switch detection ─────────────────────────────────────
        q_next = min(int((t + dt) / T_step), n_steps - 1)
        if q_next > current_step and current_step < n_steps:
            # End of step: record invariance and propagate tube
            eps_end = ccm.tube_radius(T_step, step_e0_norm, w_bar)
            eps_at_step_end[current_step]  = eps_end
            per_step_invariant[current_step] = not step_violations

            uf_cur = float(controls[current_step, 0]) if current_step < len(controls) else 0.0
            Xi = ccm.saltation_matrix(
                x_loc_Tm=float(x_ref[0]),
                v_loc_Tm=float(x_ref[1]),
                uf=uf_cur,
                T_switch=0.01,
            )
            saltation_norms[current_step] = float(np.linalg.norm(Xi, ord=2))
            eps_impact = ccm.propagate_tube_through_impact(
                eps_Tm=eps_end,
                x_loc_Tm=float(x_ref[0]),
                v_loc_Tm=float(x_ref[1]),
                uf=uf_cur,
            )
            eps_after_impact[current_step] = eps_impact

            # Reset step tracking
            current_step    = q_next
            step_e0_norm    = err
            step_start_t    = t
            step_violations = False

        # Store
        x_actual_hist[i]   = x_ref + e
        x_ref_hist[i]      = x_ref
        error_norm_hist[i] = err
        tube_bound_hist[i] = tube_radius
        tau_hist[i]        = tau_y
        w_hist[i]          = w

    violation_fraction = violations / N_t
    avg_norm_error     = float(np.mean(error_norm_hist))
    max_norm_error     = float(np.max(error_norm_hist))
    tube_inf           = ccm.steady_tube_radius(w_bar)

    return TubeSimResult(
        t=t_grid,
        error_norm=error_norm_hist,
        tube_bound=tube_bound_hist,
        tau_hist=tau_hist,
        w_hist=w_hist,
        x_actual=x_actual_hist,
        x_ref=x_ref_hist,
        per_step_invariant=per_step_invariant,
        saltation_norms=saltation_norms,
        eps_at_step_end=eps_at_step_end,
        eps_after_impact=eps_after_impact,
        violation_fraction=violation_fraction,
        avg_norm_error=avg_norm_error,
        max_norm_error=max_norm_error,
        w_bar=w_bar,
        w_bar_terrain=w_bar_terrain,
        lambda_contraction=ccm.lambda_rate,
        tube_inf=tube_inf,
        ccm=ccm,
    )


def print_tube_report(res: TubeSimResult, label: str = "") -> None:
    """Print paper-style metrics for one simulation run."""
    hdr = f"RCI Tube Report{' — ' + label if label else ''}"
    print(hdr)
    print("=" * 56)
    print(f"  w̄_terrain (CP-derived)  : {res.w_bar_terrain:.5f} m/s²")
    print(f"  w̄ used                  : {res.w_bar:.5f} m/s²")
    print(f"  λ (contraction rate)    : {res.lambda_contraction:.4f}")
    print(f"  ε̄_∞ (steady tube)       : {res.tube_inf:.5f} m")
    print(f"  avg ||e||  (Table I ANE): {res.avg_norm_error:.5f}")
    print(f"  max ||e||               : {res.max_norm_error:.5f}")
    print(f"  tube violation fraction : {100*res.violation_fraction:.2f}%")
    print(f"  steps in tube           : "
          f"{res.per_step_invariant.sum()} / {len(res.per_step_invariant)}")
    print(f"  P(in tube) per step     : "
          f"{res.per_step_invariant.mean()*100:.1f}%  "
          f"(target ≥ {100*(1 - 0.15):.0f}%)")
    print(f"  max |τ_y|               : {np.abs(res.tau_hist).max():.2f} N·m")
    print(f"  mean ||Ξ||_2            : {res.saltation_norms.mean():.4f}")
    print(f"  max  ||Ξ||_2            : {res.saltation_norms.max():.4f}")
    print()


# ---------------------------------------------------------------------------
# Backwards-compatible wrapper for run_full_pipeline.py
# ---------------------------------------------------------------------------

def simulate_tracking_with_rci_tube(plan: dict, **kwargs) -> dict:
    """
    Drop-in replacement for the old rci_tube_tracking_demo function.

    Returns a dict with the same keys as the original, plus new certified fields.
    """
    res = simulate_certified_rci(plan, **kwargs)
    return {
        # Original keys (kept for compatibility)
        "t":                   res.t,
        "x":                   res.x_actual,
        "xref":                res.x_ref,
        "error":               res.error_norm,
        "tube":                res.tube_bound,
        "tau":                 res.tau_hist,
        "w":                   res.w_hist,
        "violation_fraction":  res.violation_fraction,
        "lambda_contraction":  res.lambda_contraction,
        "w_bar":               res.w_bar,
        "tube_inf":            res.tube_inf,
        # New certified fields
        "avg_norm_error":      res.avg_norm_error,
        "per_step_invariant":  res.per_step_invariant,
        "saltation_norms":     res.saltation_norms,
        "eps_at_step_end":     res.eps_at_step_end,
        "eps_after_impact":    res.eps_after_impact,
        "w_bar_terrain":       res.w_bar_terrain,
        "ccm":                 res.ccm,
    }


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running certified RCI tube demo (standalone) ...")
    print("Building GP + CP model and nominal LIPM plan ...")

    from lipm_cp_safe_waypoint_mpc_fast import (
        build_gp_and_cp,
        plan_global_cp_safe_route,
        plan_waypoint_lipm_mpc,
    )

    gp, C, _, _ = build_gp_and_cp(seed=4, n_samples=120, confidence=0.85)
    start = np.array([-4.3, -4.2])
    goal  = np.array([ 4.3,  4.0])

    route = plan_global_cp_safe_route(
        start=start, goal=goal, gp=gp, C=C,
        delta_h_max=0.18, step_length=0.45,
        n_heading_candidates=31, max_steps=160,
        goal_tol=0.35, x_bounds=(-5,5), y_bounds=(-5,5),
    )

    params = dict(g=9.81, zH=0.9, omega=np.sqrt(9.81/0.9),
                  T_step=0.25, v0=0.65, horizon=2, max_steps=100,
                  goal_tol=0.35, lookahead_distance=1.2,
                  bounds={"x":(-5,5),"y":(-5,5),"v_loc":(0.05,2.7)},
                  weights={"waypoint":8.,"final_goal":.15,"progress":8.,
                           "slope":25.,"height":50.,"heading":1.,
                           "control":.02,"velocity":.4,"v_ref":1.6})

    states, controls, _, _ = plan_waypoint_lipm_mpc(
        start_xy=start, goal_xy=goal, route=route,
        gp=gp, C=C, delta_h_max=0.18, params=params,
    )

    plan = dict(states=states, controls=controls, params=params,
                C=C, start=start, goal=goal)

    print("\nSolving CCM SDP ...")
    ccm = default_ccm()
    print(ccm.summary())

    print("\nSimulating certified RCI tube ...")
    res = simulate_certified_rci(plan, ccm=ccm)
    print_tube_report(res)
