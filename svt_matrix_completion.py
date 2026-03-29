"""
Question 5: Singular Value Thresholding (SVT) for Low-Rank Matrix Completion
CS 754 — Advanced Image Processing, HW4

Implements the Inexact Augmented Lagrangian Method (IALM) variant of SVT,
which converges significantly faster than vanilla SVT.

Reference:
  - Cai, Candes, Shen — "A Singular Value Thresholding Algorithm for
    Matrix Completion", SIAM J. Optim. 2010.
  - Lin, Chen, Ma — "The Augmented Lagrange Multiplier Method for Exact
    Recovery of Corrupted Low-Rank Matrices", Tech. Report 2010.
"""

import numpy as np
from sklearn.utils.extmath import randomized_svd
import time
import json
import warnings
warnings.filterwarnings("ignore")


def create_low_rank_matrix(n, r, seed=42):
    """Rank-r matrix via Eckart-Young: X = A @ B^T, A and B are n x r."""
    rng = np.random.RandomState(seed)
    return rng.randn(n, r) @ rng.randn(r, n)


def sample_and_noise(X, f, noise_factor=0.02, seed=42):
    """Observe f*n^2 entries and add iid Gaussian noise with
    sigma = noise_factor * mean(|observed values|)."""
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    m = int(f * n * n)
    idx = rng.choice(n * n, m, replace=False)
    rows, cols = idx // n, idx % n
    vals = X[rows, cols].copy()
    sigma = noise_factor * np.mean(np.abs(vals))
    vals += rng.randn(m) * sigma
    return rows, cols, vals, sigma


def svt_ialm(rows, cols, b, n, lam, max_iter=200, tol=1e-4, k_svd=50):
    """Singular Value Thresholding via Inexact ALM.

    Solves:  min  lam * ||X||_*  s.t.  P_Omega(X) = b
    via augmented Lagrangian with increasing penalty mu.

    Returns (X_hat, iterations, relative_residual).
    """
    m = len(rows)
    norm_b = np.linalg.norm(b)
    X = np.zeros((n, n))
    Y_omega = np.zeros(m)
    mu = 1.0 / np.std(b)
    rho = 1.15

    for it in range(max_iter):
        temp = X.copy()
        temp[rows, cols] = b + Y_omega / mu

        kk = min(k_svd, n - 1)
        U, s, Vt = randomized_svd(temp, n_components=kk, random_state=0)

        s_t = np.maximum(s - lam / mu, 0)
        r_est = int((s_t > 0).sum())
        if r_est > 0:
            X = (U[:, :r_est] * s_t[:r_est]) @ Vt[:r_est, :]
        else:
            X = np.zeros((n, n))

        res = b - X[rows, cols]
        rel = np.linalg.norm(res) / max(norm_b, 1e-12)
        if rel < tol and it > 5:
            break

        Y_omega += mu * res
        mu = min(rho * mu, 1e7)

    return X, it + 1, rel


def cv_select_lambda(rows, cols, b, n, lam_list, n_folds=2,
                     seed=42, k_svd=50):
    """2-fold cross-validation on observed entries to select lambda."""
    rng = np.random.RandomState(seed)
    m = len(rows)
    perm = rng.permutation(m)
    fs = m // n_folds

    best_lam, best_err = lam_list[0], np.inf
    for lam in lam_list:
        err = 0.0
        for fold in range(n_folds):
            start = fold * fs
            end = (fold + 1) * fs if fold < n_folds - 1 else m
            val_idx = perm[start:end]
            train_mask = np.ones(m, dtype=bool)
            train_mask[val_idx] = False

            Xh, _, _ = svt_ialm(rows[train_mask], cols[train_mask],
                                 b[train_mask], n, lam,
                                 max_iter=80, tol=1e-3, k_svd=k_svd)
            pred = Xh[rows[val_idx], cols[val_idx]]
            err += np.mean((b[val_idx] - pred) ** 2)

        err /= n_folds
        tag = " <-- best" if err < best_err else ""
        print(f"      lam={lam:>10.4f}  CV-MSE={err:.6f}{tag}")
        if err < best_err:
            best_err, best_lam = err, lam
    return best_lam


def run_single(n, r, f):
    """Run one (n, r, f) experiment with cross-validated lambda."""
    print(f"\n  --- n={n}, r={r}, f={f} ---")
    sb = n + r * 100 + int(f * 10000)
    X = create_low_rank_matrix(n, r, seed=sb)
    nX = np.linalg.norm(X, 'fro')
    rows, cols, b, sigma = sample_and_noise(X, f, seed=sb + 1)
    m = len(rows)
    k_svd = min(r + 20, 150)
    print(f"    #obs={m:,}  sigma={sigma:.4f}  k_svd={k_svd}")

    lam_list = [0.01, 0.1, 1.0, 10.0, 50.0]
    print(f"    Cross-validating lambda ({len(lam_list)} values, 2-fold) ...")
    best_lam = cv_select_lambda(rows, cols, b, n, lam_list, n_folds=2,
                                seed=42, k_svd=k_svd)
    print(f"    Selected lambda = {best_lam:.4f}")

    print(f"    Running final SVT (max 200 iters) ...", end=" ", flush=True)
    tic = time.time()
    Xh, its, rel = svt_ialm(rows, cols, b, n, best_lam,
                             max_iter=200, tol=1e-4, k_svd=k_svd)
    elapsed = time.time() - tic

    rmse = np.linalg.norm(Xh - X, 'fro') / nX
    print(f"RMSE={rmse:.6f}  iters={its}  time={elapsed:.1f}s")
    return dict(rmse=float(rmse), time=float(elapsed),
                lam=float(best_lam), iters=int(its), sigma=float(sigma))


def run_all():
    """Run both n=1000 and n=3000 experiment grids."""
    configs = [
        (1000, [3, 10, 40, 80], [0.05, 0.15, 0.3, 0.4]),
        (3000, [3, 10, 40, 80], [0.02, 0.05, 0.1, 0.2]),
    ]
    results = {}
    for n, ranks, fracs in configs:
        print(f"\n{'='*60}")
        print(f"  EXPERIMENT  n = {n}")
        print(f"{'='*60}")
        d = {}
        for r in ranks:
            for f in fracs:
                d[f"r{r}_f{f}"] = run_single(n, r, f)
        results[f"n{n}"] = dict(ranks=ranks, fracs=fracs, data=d)

    with open("svt_results.json", "w") as fp:
        json.dump(results, fp, indent=2)

    for tag in sorted(results):
        info = results[tag]
        R, F, D = info["ranks"], info["fracs"], info["data"]

        print(f"\n{'='*60}")
        print(f"  RMSE TABLE  ({tag})")
        print(f"{'='*60}")
        hdr = f"  {'r \\\\ f':>6}" + "".join(f"{f:>12}" for f in F)
        print(hdr)
        print("  " + "-" * (8 + 12 * len(F)))
        for r in R:
            print(f"  {r:>6}" + "".join(
                f"{D[f'r{r}_f{f}']['rmse']:>12.6f}" for f in F))

        print(f"\n  TIME TABLE (seconds)  ({tag})")
        print("  " + "-" * (8 + 12 * len(F)))
        for r in R:
            print(f"  {r:>6}" + "".join(
                f"{D[f'r{r}_f{f}']['time']:>12.1f}" for f in F))

    print("\n\nDone. Results saved to svt_results.json")
    return results


if __name__ == "__main__":
    run_all()
