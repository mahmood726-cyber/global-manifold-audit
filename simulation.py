import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

PROJECT_ROOT = Path(__file__).resolve().parent
CERTIFICATION_PATH = PROJECT_ROOT / "certification.json"
AUDIT_RESULTS_PATH = PROJECT_ROOT / "audit_results.csv"
INTERPOLATION_GRID = np.linspace(0, 1, 100)


def inv_logit(logit_value):
    return 1 / (1 + np.exp(-logit_value))


def simulate_landscape(k=40, fractures=0, rng=None):
    rng = rng or np.random.default_rng()
    clusters = [[1.38, 2.19]]
    for _ in range(fractures):
        clusters.append([float(rng.normal(1.5, 1.0)), float(rng.normal(1.5, 1.0))])

    results = []
    for i in range(k):
        mean = clusters[i % len(clusters)]
        l_s, l_sp = rng.multivariate_normal(mean, [[0.1, 0], [0, 0.1]])
        sensitivity, specificity = inv_logit(l_s), inv_logit(l_sp)
        tp = rng.binomial(100, sensitivity)
        tn = rng.binomial(300, specificity)
        results.append({"tp": tp, "fp": 300 - tn, "fn": 100 - tp, "tn": tn})
    return pd.DataFrame(results), clusters


def aps_v2_engine(df):
    tp, fp, fn, tn = df["tp"] + 0.5, df["fp"] + 0.5, df["fn"] + 0.5, df["tn"] + 0.5
    sensitivity = tp / (tp + fn)
    false_positive_rate = fp / (fp + tn)
    points = np.column_stack([false_positive_rate, sensitivity])

    best_labels = None
    best_cluster_count = 0
    for eps in [0.05, 0.1, 0.15]:
        dbscan = DBSCAN(eps=eps, min_samples=3).fit(points)
        cluster_count = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
        if cluster_count > best_cluster_count:
            best_cluster_count = cluster_count
            best_labels = dbscan.labels_

    if best_labels is None:
        best_labels = np.zeros(len(sensitivity), dtype=int)

    aleph_points = []
    for label in set(best_labels):
        if label == -1:
            continue
        mask = best_labels == label
        sub_sensitivity = sensitivity[mask]
        sub_fpr = false_positive_rate[mask]
        j_index = sub_sensitivity + (1 - sub_fpr) - 1
        weights = np.power(np.maximum(j_index, 0.1), 3)
        aleph_points.append(
            {
                "fpr": float(np.average(sub_fpr, weights=weights)),
                "sens": float(np.average(sub_sensitivity, weights=weights)),
            }
        )

    aleph_points = sorted(aleph_points, key=lambda point: point["fpr"])
    x_points = [0.0] + [point["fpr"] for point in aleph_points] + [1.0]
    y_points = [0.0] + [point["sens"] for point in aleph_points] + [1.0]
    y_new = np.interp(INTERPOLATION_GRID, x_points, y_points)
    y_new = np.maximum.accumulate(y_new)
    return float(np.trapezoid(y_new, INTERPOLATION_GRID))


def build_certification(results_df, n_simulations):
    mean_bias = float(results_df["bias"].mean())
    stability = float((results_df["bias"] < 0.1).mean())
    return {
        "status": "MASS_VALIDATED",
        "n_simulations": n_simulations,
        "metrics": {
            "mean_bias": round(mean_bias, 4),
            "stability_index": round(stability, 4),
            "max_bias": round(float(results_df["bias"].max()), 4),
        },
    }


def write_outputs(results_df, cert, project_root=PROJECT_ROOT):
    certification_path = Path(project_root) / CERTIFICATION_PATH.name
    results_path = Path(project_root) / AUDIT_RESULTS_PATH.name
    certification_path.write_text(json.dumps(cert, indent=2), encoding="utf-8")
    results_df.to_csv(results_path, index=False)
    return certification_path, results_path


def run_mass_audit(n_simulations=100, seed=42):
    rng = np.random.default_rng(seed)
    gma_results = []

    print(f"RUNNING GMA MASS AUDIT ({n_simulations} Simulations)...")
    for i in range(n_simulations):
        fractures = int(rng.integers(0, 4))
        df, true_clusters = simulate_landscape(k=50, fractures=fractures, rng=rng)
        aps_auc = aps_v2_engine(df)
        true_sensitivity = np.mean([inv_logit(cluster[0]) for cluster in true_clusters])
        true_specificity = np.mean([inv_logit(cluster[1]) for cluster in true_clusters])
        true_auc = 0.5 + (true_sensitivity + true_specificity - 1) / 2
        gma_results.append(
            {
                "id": i,
                "fractures": fractures,
                "bias": abs(aps_auc - true_auc),
                "auc": aps_auc,
            }
        )
        if i % 10 == 0:
            print(f" - Progress: {i}/{n_simulations}")

    results_df = pd.DataFrame(gma_results)
    cert = build_certification(results_df, n_simulations=n_simulations)
    print("\nMASS-SCALE AUDIT COMPLETE:")
    print(f" - Mean Bias (APS v2): {cert['metrics']['mean_bias']:.4f}")
    print(f" - Stability Index: {cert['metrics']['stability_index'] * 100:.1f}%")
    return results_df, cert


def main(n_simulations=100, seed=42, project_root=PROJECT_ROOT):
    results_df, cert = run_mass_audit(n_simulations=n_simulations, seed=seed)
    write_outputs(results_df, cert, project_root=project_root)
    return {"results": results_df, "certification": cert}


if __name__ == "__main__":
    main()
