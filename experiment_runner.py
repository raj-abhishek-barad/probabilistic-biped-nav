"""
experiment_runner.py
--------------------
Multi-trial experimental validation harness matching the paper's protocol.

Paper: Muenprasitivej et al., arXiv:2510.07725, 2025.

Reproduces the evaluation methodology of Tables I–III:
  - N_TRIALS independent trajectories per terrain / framework
  - Per-step RCI tube invariance fraction
  - Average norm error (ANE) ||x_loc(t) − x_loc*(t)||_AVG
  - CP coverage fraction
  - Comparison: Ours vs Ours-without-τ_y vs Baseline (no CP constraint)
  - Three terrain difficulty levels (T1-style / T2-style / T3-style)
  - Results saved to CSV and printed as LaTeX-ready table

Usage
-----
    python experiment_runner.py
    python experiment_runner.py --n_trials 5 --confidence 0.85 --output results/

Output files
------------
    results_table.csv       — raw per-trial numbers
    summary_table.txt       — LaTeX-ready Table I / II equivalents
    trajectories/           — per-trial plan + tube result (numpy .npz)
"""

from __future__ import annotations

import argparse
import csv
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import numpy as np

from ccm_sdp_solver import default_ccm, CCMResult
from rci_tube_certified import simulate_certified_rci, TubeSimResult, print_tube_report
from lipm_cp_safe_waypoint_mpc_fast import (
    build_gp_and_cp,
    plan_global_cp_safe_route,
    plan_waypoint_lipm_mpc,
    true_terrain,
)


# ---------------------------------------------------------------------------
# Terrain profiles  (analogous to T1 / T2 / T3 in the paper)
# ---------------------------------------------------------------------------

TERRAIN_PROFILES = {
    "T1_bumpy": dict(
        description="Bumpy, high-frequency rough terrain",
        confidence=0.85,
        n_samples=120,
        seed_base=10,
        delta_h_max=0.18,
        w_bar_scale=1.0,
    ),
    "T2_wavy": dict(
        description="Wavy, medium-frequency terrain with slopes",
        confidence=0.85,
        n_samples=120,
        seed_base=20,
        delta_h_max=0.15,
        w_bar_scale=1.2,      # harder terrain → larger disturbance
    ),
    "T3_hilly": dict(
        description="Hilly, smoother surface with elevation gradients",
        confidence=0.85,
        n_samples=120,
        seed_base=30,
        delta_h_max=0.20,
        w_bar_scale=0.9,
    ),
}

# Fixed start / goal (same across all trials for comparability)
START_XY = np.array([-4.3, -4.2])
GOAL_XY  = np.array([ 4.3,  4.0])

# Default MPC parameters
DEFAULT_PARAMS = dict(
    g=9.81, zH=0.9, omega=np.sqrt(9.81 / 0.9),
    T_step=0.25, v0=0.65, horizon=2, max_steps=100,
    goal_tol=0.35, lookahead_distance=1.2,
    bounds={"x": (-5.0, 5.0), "y": (-5.0, 5.0), "v_loc": (0.05, 2.70)},
    weights=dict(waypoint=8., final_goal=.15, progress=8.,
                 slope=25., height=50., heading=1.,
                 control=.02, velocity=.4, v_ref=1.6),
)


# ---------------------------------------------------------------------------
# Framework modes
# ---------------------------------------------------------------------------

class FrameworkMode:
    """Controls which components are active."""
    OURS        = "Ours"              # CP + CCM torque
    NO_TORQUE   = "Ours w/o τ_y"     # CP, no CCM torque
    BASELINE    = "Baseline MPC"      # no CP safety constraint


# ---------------------------------------------------------------------------
# Single-trial runner
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    framework:       str
    terrain:         str
    trial_idx:       int
    goal_reached:    bool
    n_steps:         int
    avg_norm_error:  float
    rci_fraction:    float           # fraction of steps where tube was invariant
    cp_coverage:     float           # fraction of steps where true z ∈ CP interval
    sim_time_s:      float
    n_violations:    int
    max_error:       float
    w_bar_terrain:   float
    mean_saltation:  float


def run_single_trial(
    ccm: CCMResult,
    terrain_cfg: dict,
    trial_idx: int,
    framework: str,
    output_dir: Optional[str] = None,
) -> TrialResult:
    """Run one complete trial and return metrics."""
    t0 = time.time()
    seed = terrain_cfg["seed_base"] + trial_idx
    confidence = terrain_cfg["confidence"]
    delta_h_max = terrain_cfg["delta_h_max"]

    # Terrain + CP model
    gp, C, X_train, X_cal = build_gp_and_cp(
        seed=seed,
        n_samples=terrain_cfg["n_samples"],
        confidence=confidence,
    )

    # Effective CP margin: baseline ignores CP (sets C → 0)
    C_plan = C if framework != FrameworkMode.BASELINE else 0.0

    # Global route
    try:
        route = plan_global_cp_safe_route(
            start=START_XY, goal=GOAL_XY,
            gp=gp, C=C_plan,
            delta_h_max=delta_h_max,
            step_length=0.45,
            n_heading_candidates=31,
            max_steps=160,
            goal_tol=0.35,
            x_bounds=(-5., 5.), y_bounds=(-5., 5.),
        )
    except Exception as exc:
        warnings.warn(f"Route planning failed trial {trial_idx}: {exc}")
        return _failed_trial(framework, terrain_cfg.get("name","?"), trial_idx)

    # LIPM-MPC
    params = {**DEFAULT_PARAMS}
    params["omega"] = np.sqrt(params["g"] / params["zH"])
    try:
        states, controls, _, _ = plan_waypoint_lipm_mpc(
            start_xy=START_XY, goal_xy=GOAL_XY,
            route=route, gp=gp, C=C_plan,
            delta_h_max=delta_h_max, params=params,
        )
    except Exception as exc:
        warnings.warn(f"LIPM-MPC failed trial {trial_idx}: {exc}")
        return _failed_trial(framework, terrain_cfg.get("name","?"), trial_idx)

    final_dist = np.linalg.norm(states[-1, :2] - GOAL_XY)
    goal_reached = final_dist < params["goal_tol"]
    if len(states) < 2 or len(controls) < 1:
        warnings.warn(
            f"Planner produced no executable trajectory in trial {trial_idx} "
            f"for framework {framework}."
        )
        return _failed_trial(framework, terrain_cfg.get("name", "?"), trial_idx)
    plan = dict(states=states, controls=controls, params=params,
                C=C, start=START_XY, goal=GOAL_XY)

    # CCM torque: disabled for NO_TORQUE / BASELINE by zeroing gain
    if framework in (FrameworkMode.NO_TORQUE, FrameworkMode.BASELINE):
        from ccm_sdp_solver import CCMResult as _CCMResult
        import copy
        ccm_trial = copy.copy(ccm)
        ccm_trial.__dict__["K"] = np.zeros_like(ccm.K)
    else:
        ccm_trial = ccm

    # RCI tube simulation
    try:
        res: TubeSimResult = simulate_certified_rci(
            plan=plan,
            ccm=ccm_trial,
            w_bar_scale=terrain_cfg["w_bar_scale"],
            disturbance_seed=float(seed),
        )
    except Exception as exc:
        warnings.warn(f"Tube sim failed trial {trial_idx}: {exc}")
        return _failed_trial(framework, terrain_cfg.get("name","?"), trial_idx)

    # CP coverage fraction: P(z_true ∈ [μ−C, μ+C]) per step
    cp_covered = 0
    for q in range(len(states)):
        x_q, y_q = states[q, :2]
        z_true = true_terrain(x_q, y_q)

        mu_q, var_q = gp.predict(np.array([[x_q, y_q]]))

        if abs(float(z_true) - float(mu_q[0])) <= C:
            cp_covered += 1

    cp_coverage = cp_covered / max(len(states), 1)

    sim_time = time.time() - t0

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        safe_framework = (
            framework
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
            .replace("τ", "tau")
            .replace("ᵧ", "y")
        )

        fname = os.path.join(
            output_dir,
            f"{safe_framework}_{terrain_cfg.get('name','T')}_{trial_idx:02d}.npz"
        )
        np.savez_compressed(
            fname,
            t=res.t, error=res.error_norm, tube=res.tube_bound,
            tau=res.tau_hist, per_step=res.per_step_invariant,
            saltation=res.saltation_norms,
        )

    return TrialResult(
        framework=framework,
        terrain=terrain_cfg.get("name", "?"),
        trial_idx=trial_idx,
        goal_reached=goal_reached,
        n_steps=len(states) - 1,
        avg_norm_error=res.avg_norm_error,
        rci_fraction=float(res.per_step_invariant.mean()),
        cp_coverage=cp_coverage,
        sim_time_s=sim_time,
        n_violations=int(res.violation_fraction * len(res.t)),
        max_error=res.max_norm_error,
        w_bar_terrain=res.w_bar_terrain,
        mean_saltation=float(res.saltation_norms.mean()),
    )


def _failed_trial(framework, terrain, idx) -> TrialResult:
    return TrialResult(framework=framework, terrain=terrain,
                       trial_idx=idx, goal_reached=False, n_steps=0,
                       avg_norm_error=np.nan, rci_fraction=0.,
                       cp_coverage=np.nan, sim_time_s=np.nan,
                       n_violations=0, max_error=np.nan,
                       w_bar_terrain=np.nan, mean_saltation=np.nan)


# ---------------------------------------------------------------------------
# Full experiment
# ---------------------------------------------------------------------------

def run_experiment(
    n_trials: int = 5,
    terrains: Optional[List[str]] = None,
    frameworks: Optional[List[str]] = None,
    output_dir: str = "results",
    verbose: bool = True,
) -> Dict[str, List[TrialResult]]:
    """
    Run the full experiment grid: terrains × frameworks × n_trials.

    Returns a nested dict: results[terrain][framework] = List[TrialResult].
    """
    if terrains is None:
        terrains = list(TERRAIN_PROFILES.keys())
    if frameworks is None:
        frameworks = [FrameworkMode.OURS, FrameworkMode.NO_TORQUE, FrameworkMode.BASELINE]

    # Solve CCM once — shared across all trials
    print("Solving CCM SDP (shared across all trials) ...")
    ccm = default_ccm(lambda_rate=2.5)
    print(f"  status={ccm.sdp_status}  residual={ccm.sdp_residual:+.4f}  "
          f"verified={ccm.verify()}")
    print()

    all_results: Dict[str, Dict[str, List[TrialResult]]] = {
        t: {f: [] for f in frameworks} for t in terrains
    }

    total = len(terrains) * len(frameworks) * n_trials
    done  = 0

    for terrain_key in terrains:
        cfg = {**TERRAIN_PROFILES[terrain_key], "name": terrain_key}
        for fw in frameworks:
            for k in range(n_trials):
                done += 1
                if verbose:
                    print(f"[{done:3d}/{total}] terrain={terrain_key}  "
                          f"framework={fw:<20s}  trial={k} ...", end=" ", flush=True)
                trial = run_single_trial(ccm, cfg, k, fw, output_dir)
                all_results[terrain_key][fw].append(trial)
                if verbose:
                    goal_str = "✓" if trial.goal_reached else "✗"
                    print(f"{goal_str}  ANE={trial.avg_norm_error:.4f}  "
                          f"RCI={trial.rci_fraction*100:.1f}%  "
                          f"t={trial.sim_time_s:.1f}s")

    return all_results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summarise(all_results: dict, confidence: float = 0.85) -> str:
    """Build LaTeX-ready Table I / II equivalent."""
    lines = []
    lines.append("=" * 80)
    lines.append("EXPERIMENT SUMMARY  (analogous to paper Tables I–III)")
    lines.append(f"Confidence level: {confidence*100:.1f}%")
    lines.append("=" * 80)

    for terrain, fw_dict in all_results.items():
        lines.append(f"\nTerrain: {terrain}")
        lines.append("-" * 72)
        lines.append(f"{'Framework':<22} {'GoalRate':>9} {'ANE':>9} {'P(RCI)':>9} "
                     f"{'CP_cov':>9} {'Steps':>7} {'SaltNorm':>9}")
        lines.append("-" * 72)

        for fw, trials in fw_dict.items():
            valid = [r for r in trials if not np.isnan(r.avg_norm_error)]
            goal_rate  = np.mean([r.goal_reached for r in trials])
            ane        = np.mean([r.avg_norm_error for r in valid]) if valid else float("nan")
            rci        = np.mean([r.rci_fraction   for r in valid]) if valid else float("nan")
            cp_cov     = np.mean([r.cp_coverage    for r in valid]) if valid else float("nan")
            steps      = np.mean([r.n_steps        for r in valid]) if valid else float("nan")
            salt       = np.mean([r.mean_saltation for r in valid]) if valid else float("nan")

            lines.append(
                f"{fw:<22} {goal_rate:>9.2f} {ane:>9.4f} "
                f"{rci*100:>8.1f}% {cp_cov*100:>8.1f}% "
                f"{steps:>7.0f} {salt:>9.4f}"
            )
        lines.append("")

    return "\n".join(lines)


def save_csv(all_results: dict, path: str) -> None:
    """Save all trial results to CSV."""
    rows = []
    for terrain, fw_dict in all_results.items():
        for fw, trials in fw_dict.items():
            for r in trials:
                rows.append({
                    "terrain": terrain,
                    "framework": fw,
                    "trial": r.trial_idx,
                    "goal_reached": int(r.goal_reached),
                    "n_steps": r.n_steps,
                    "avg_norm_error": f"{r.avg_norm_error:.6f}",
                    "rci_fraction": f"{r.rci_fraction:.6f}",
                    "cp_coverage": f"{r.cp_coverage:.6f}",
                    "sim_time_s": f"{r.sim_time_s:.2f}",
                    "max_error": f"{r.max_error:.6f}",
                    "w_bar_terrain": f"{r.w_bar_terrain:.6f}",
                    "mean_saltation_norm": f"{r.mean_saltation:.6f}",
                })
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if len(rows) == 0:
        print(f"No valid rows to save: {path}")
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Multi-trial experiment runner")
    parser.add_argument("--n_trials",   type=int,   default=5)
    parser.add_argument("--confidence", type=float, default=0.85)
    parser.add_argument("--output",     type=str,   default="results")
    parser.add_argument("--terrains",   nargs="+",  default=None,
                        choices=list(TERRAIN_PROFILES.keys()))
    parser.add_argument("--quiet",      action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"\nExperiment: n_trials={args.n_trials}  "
          f"confidence={args.confidence*100:.1f}%  output={args.output}\n")

    results = run_experiment(
        n_trials=args.n_trials,
        terrains=args.terrains,
        output_dir=os.path.join(args.output, "trajectories"),
        verbose=not args.quiet,
    )

    summary = summarise(results, args.confidence)
    print("\n" + summary)

    txt_path = os.path.join(args.output, "summary_table.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"Summary saved: {txt_path}")

    csv_path = os.path.join(args.output, "results_table.csv")
    save_csv(results, csv_path)


if __name__ == "__main__":
    main()
