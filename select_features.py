# stepe_feature_selection.py
"""
Step E: Derive the final fixed feature set.

Strategy:
  1. Compute Spearman correlation of each feature with UPDRS for Left and Right independently
  2. Select features that are significant (p < threshold) in BOTH sides
  3. Remove redundant features (inter-feature |r| > 0.85)
  4. Save the final feature list -- this becomes the fixed input to Step F

This is a *design decision*, not part of the evaluation loop.
validated their bilateral robustness, and retained K features for prediction."

Reads:  {OUTPUT_DIR}/stepd_features_subject_{activity}.csv  (for both L and R)
Writes: {OUTPUT_DIR}/stepe_final_features.txt
        {OUTPUT_DIR}/stepe_correlation_table_{L/R}.csv
        {OUTPUT_DIR}/stepe_*.png (visualizations)
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from collections import OrderedDict

import config as C
from stepd_feature_extraction import FEATURE_NAMES


# ============================================================
# Load
# ============================================================
def load_csv(path):
    rows = []
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            parsed = {}
            for k, v in row.items():
                try: parsed[k] = float(v) if v not in ("NaN", "") else float("nan")
                except ValueError: parsed[k] = v
            rows.append(parsed)
    return rows


# ============================================================
# Per-side correlation
# ============================================================
def correlate_features(rows, method="spearman"):
    updrs = np.array([r["updrs_score"] for r in rows])
    corr_fn = spearmanr
    results = {}
    for fn in FEATURE_NAMES:
        vals = np.array([r.get(fn, float("nan")) for r in rows])
        valid = np.isfinite(vals) & np.isfinite(updrs)
        if valid.sum() < 5:
            results[fn] = {"rho": float("nan"), "p": float("nan"), "n": int(valid.sum())}
            continue
        rho, pval = corr_fn(updrs[valid], vals[valid])
        results[fn] = {"rho": float(rho), "p": float(pval), "n": int(valid.sum())}
    return results


# ============================================================
# Inter-feature correlation matrix
# ============================================================
def inter_feature_corr(rows, features):
    n = len(features)
    mat = np.full((n, n), float("nan"))
    for i in range(n):
        vi = np.array([r.get(features[i], float("nan")) for r in rows])
        for j in range(i, n):
            vj = np.array([r.get(features[j], float("nan")) for r in rows])
            valid = np.isfinite(vi) & np.isfinite(vj)
            if valid.sum() >= 5:
                rho, _ = spearmanr(vi[valid], vj[valid])
                mat[i, j] = mat[j, i] = rho
    return mat


# ============================================================
# Core selection logic
# ============================================================
def select_bilateral_features(corr_L, corr_R, p_threshold=0.05, redundancy_threshold=0.85,
                               all_rows=None, max_features=20):
    """
    1. Features significant in BOTH L and R (p < threshold)
    2. Rank by mean |rho| across sides
    3. Remove redundant (greedy, keep higher-ranked)
    4. Cap at max_features
    """
    # Step 1: bilateral significance
    bilateral = []
    for fn in FEATURE_NAMES:
        L = corr_L.get(fn, {}); R = corr_R.get(fn, {})
        pL = L.get("p", 1.0); pR = R.get("p", 1.0)
        rhoL = L.get("rho", 0); rhoR = R.get("rho", 0)

        if pL < p_threshold and pR < p_threshold:
            # Both significant
            mean_abs_rho = (abs(rhoL) + abs(rhoR)) / 2
            bilateral.append({"feature": fn, "rho_L": rhoL, "rho_R": rhoR,
                              "p_L": pL, "p_R": pR, "mean_abs_rho": mean_abs_rho})

    # If too few pass bilateral, relax: require significant in at least one side
    # with marginal significance (p < 0.1) in the other
    if len(bilateral) < 5:
        for fn in FEATURE_NAMES:
            if any(b["feature"] == fn for b in bilateral): continue
            L = corr_L.get(fn, {}); R = corr_R.get(fn, {})
            pL = L.get("p", 1.0); pR = R.get("p", 1.0)
            rhoL = L.get("rho", 0); rhoR = R.get("rho", 0)
            if (pL < p_threshold and pR < 0.1) or (pR < p_threshold and pL < 0.1):
                mean_abs_rho = (abs(rhoL) + abs(rhoR)) / 2
                bilateral.append({"feature": fn, "rho_L": rhoL, "rho_R": rhoR,
                                  "p_L": pL, "p_R": pR, "mean_abs_rho": mean_abs_rho,
                                  "relaxed": True})

    # Step 2: rank by mean |rho|
    bilateral.sort(key=lambda x: x["mean_abs_rho"], reverse=True)

    # Step 3: remove redundant
    if all_rows and len(bilateral) > 1:
        feat_names = [b["feature"] for b in bilateral]
        mat = inter_feature_corr(all_rows, feat_names)

        kept = set(range(len(bilateral)))
        for i in range(len(bilateral)):
            if i not in kept: continue
            for j in range(i + 1, len(bilateral)):
                if j not in kept: continue
                if np.isfinite(mat[i, j]) and abs(mat[i, j]) > redundancy_threshold:
                    kept.discard(j)  # remove lower-ranked

        bilateral = [bilateral[i] for i in sorted(kept)]

    # Step 4: cap
    bilateral = bilateral[:max_features]
    return bilateral


# ============================================================
# Visualizations
# ============================================================
def plot_bilateral_comparison(corr_L, corr_R, selected, output_dir):
    """Scatter: rho_L vs rho_R for all features, highlight selected."""
    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)

    sel_names = {s["feature"] for s in selected}
    for fn in FEATURE_NAMES:
        rL = corr_L.get(fn, {}).get("rho", float("nan"))
        rR = corr_R.get(fn, {}).get("rho", float("nan"))
        if not (np.isfinite(rL) and np.isfinite(rR)): continue
        if fn in sel_names:
            ax.scatter(rL, rR, s=50, c="#4CAF50", edgecolors="black", linewidth=0.5, zorder=5)
            ax.annotate(fn, (rL, rR), fontsize=5, xytext=(3, 3), textcoords="offset points")
        else:
            ax.scatter(rL, rR, s=20, c="#BDBDBD", alpha=0.5, zorder=3)

    ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
    ax.plot([-1, 1], [-1, 1], "--", color="gray", alpha=0.3)
    ax.set_xlabel("Spearman ρ (Left)", fontsize=10)
    ax.set_ylabel("Spearman ρ (Right)", fontsize=10)
    ax.set_title(f"Bilateral Feature Consistency\n"
                 f"({len(selected)} selected features in green)", fontsize=11)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    path = os.path.join(output_dir, "stepe_bilateral_scatter.png")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  Bilateral scatter: {path}")


def plot_selected_bar(selected, output_dir):
    """Horizontal bar: selected features ranked by mean |rho|, showing L and R."""
    if not selected: return

    fig, ax = plt.subplots(figsize=(10, max(4, len(selected)*0.4)), dpi=120)
    names = [s["feature"] for s in selected]
    rhoL = [s["rho_L"] for s in selected]
    rhoR = [s["rho_R"] for s in selected]
    y = np.arange(len(names))

    ax.barh(y - 0.15, rhoL, height=0.3, color="#2196F3", alpha=0.7, label="Left")
    ax.barh(y + 0.15, rhoR, height=0.3, color="#FF9800", alpha=0.7, label="Right")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Spearman ρ with UPDRS")
    ax.set_title(f"Selected Features — Bilateral Correlation\n"
                 f"(ranked by mean |ρ|, {len(selected)} features)", fontsize=11)
    ax.legend(fontsize=9); ax.axvline(0, color="black", lw=0.5)
    ax.grid(True, axis="x", alpha=0.2)
    plt.tight_layout()
    path = os.path.join(output_dir, "stepe_selected_features_bar.png")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  Selected bar: {path}")


def plot_inter_feature_heatmap(selected, all_rows, output_dir):
    """Heatmap of inter-feature correlations among selected features."""
    if len(selected) < 3: return
    names = [s["feature"] for s in selected]
    mat = inter_feature_corr(all_rows, names)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=120)
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
    for i in range(len(names)):
        for j in range(len(names)):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5,
                        color="white" if abs(v) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Spearman ρ")
    ax.set_title("Inter-Feature Correlation (selected features)", fontsize=11)
    plt.tight_layout()
    path = os.path.join(output_dir, "stepe_inter_feature_heatmap.png")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  Inter-feature heatmap: {path}")


# ============================================================
# Report
# ============================================================
def generate_report(corr_L, corr_R, selected, output_dir):
    # Per-side correlation tables
    for side, corr in [("Left", corr_L), ("Right", corr_R)]:
        path = os.path.join(output_dir, f"stepe_correlation_{side}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["feature", "rho", "p_value", "n_valid"])
            for fn in FEATURE_NAMES:
                c = corr.get(fn, {})
                w.writerow([fn, f"{c.get('rho',0):.4f}", f"{c.get('p',1):.6f}", c.get("n", 0)])
        print(f"  Correlation table ({side}): {path}")

    # Final feature list
    path = os.path.join(output_dir, "stepe_final_features.txt")
    with open(path, "w") as f:
        for s in selected:
            f.write(f"{s['feature']}\n")
    print(f"  Final features: {path}")

    # Detailed CSV
    path2 = os.path.join(output_dir, "stepe_final_features_detail.csv")
    with open(path2, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "feature", "rho_L", "p_L", "rho_R", "p_R", "mean_abs_rho", "relaxed"])
        for i, s in enumerate(selected):
            w.writerow([i+1, s["feature"], f"{s['rho_L']:.4f}", f"{s['p_L']:.6f}",
                        f"{s['rho_R']:.4f}", f"{s['p_R']:.6f}", f"{s['mean_abs_rho']:.4f}",
                        s.get("relaxed", False)])
    print(f"  Detail CSV: {path2}")

    # Console
    print(f"\n{'='*80}")
    print(f"  Step E: Feature Selection Summary")
    print(f"{'='*80}")
    n_sig_L = sum(1 for fn in FEATURE_NAMES if corr_L.get(fn, {}).get("p", 1) < C.P_VALUE_THRESHOLD)
    n_sig_R = sum(1 for fn in FEATURE_NAMES if corr_R.get(fn, {}).get("p", 1) < C.P_VALUE_THRESHOLD)
    print(f"  Total features:        43")
    print(f"  Significant (Left):    {n_sig_L}")
    print(f"  Significant (Right):   {n_sig_R}")
    print(f"  Bilateral overlap:     {len(selected)} (after redundancy removal)")
    print(f"\n  {'Rank':<5s} {'Feature':<40s} {'ρ_L':>8s} {'p_L':>8s} {'ρ_R':>8s} {'p_R':>8s} {'|ρ|':>6s}")
    print(f"  {'-'*78}")
    for i, s in enumerate(selected):
        print(f"  {i+1:<5d} {s['feature']:<40s} {s['rho_L']:>8.3f} {s['p_L']:>8.4f} "
              f"{s['rho_R']:>8.3f} {s['p_R']:>8.4f} {s['mean_abs_rho']:>6.3f}")

    # Category breakdown
    cats = {"amp_":"Amplitude","speed_":"Speed","rhythm_":"Rhythm","fatigue_":"Fatigue",
            "smooth_":"Smoothness","spec_":"Spectral","body_":"Body"}
    print(f"\n  Category breakdown:")
    for prefix, name in cats.items():
        n = sum(1 for s in selected if s["feature"].startswith(prefix))
        if n: print(f"    {name}: {n}")
    print(f"{'='*80}\n")


# ============================================================
if __name__ == "__main__":
    os.makedirs(C.OUTPUT_DIR, exist_ok=True)

    print(f"\n{'#'*60}\n  Step E: Feature Selection\n{'#'*60}")

    # Load both sides
    rows_L = load_csv(C.features_subject_csv("making_a_fist_Left"))
    rows_R = load_csv(C.features_subject_csv("making_a_fist_Right"))
    all_rows = rows_L + rows_R
    print(f"  Left: {len(rows_L)} subjects, Right: {len(rows_R)} subjects")

    # Correlations per side
    corr_L = correlate_features(rows_L, method=C.CORRELATION_METHOD)
    corr_R = correlate_features(rows_R, method=C.CORRELATION_METHOD)

    # Select bilateral features
    selected = select_bilateral_features(
        corr_L, corr_R,
        p_threshold=C.P_VALUE_THRESHOLD,
        redundancy_threshold=0.85,
        all_rows=all_rows,
        max_features=C.PRUNE_TOP_K,
    )

    # Report + viz
    generate_report(corr_L, corr_R, selected, C.OUTPUT_DIR)
    plot_bilateral_comparison(corr_L, corr_R, selected, C.OUTPUT_DIR)
    plot_selected_bar(selected, C.OUTPUT_DIR)
    plot_inter_feature_heatmap(selected, all_rows, C.OUTPUT_DIR)
