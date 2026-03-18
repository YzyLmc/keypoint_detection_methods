"""
Test script for Sticky HDP-HMM implementation.

Generates synthetic multi-dimensional time-series data with known
segment structure, runs HDP-HMM inference, and checks whether the
model recovers the ground-truth segmentation.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core import HDPHMM


def generate_synthetic_data(n_demos=10, d=4, ar_order=1, n_states=3,
                            min_seg_len=30, max_seg_len=80, noise_std=0.3,
                            seed=42):
    """
    Generate synthetic AR time-series with known segmentation.

    Each demo is a sequence of segments, where each segment is generated
    by a different AR model (linear dynamics + noise).

    Returns:
        obs_list: list of (d, T_i) arrays
        true_labels: list of (T_i,) label arrays
        true_A: (d, d*ar_order, n_states) true AR coefficients
    """
    rng = np.random.RandomState(seed)
    m = d * ar_order

    # Generate stable AR matrices for each state
    # Each state has a distinct but stable AR(1) matrix (spectral radius < 1)
    true_A = np.zeros((d, m, n_states))
    base_matrices = [
        np.diag([0.8, -0.5, 0.3, -0.2][:d]),   # state 0: smooth, positive x
        np.diag([-0.3, 0.7, -0.6, 0.4][:d]),    # state 1: oscillatory
        np.diag([0.5, 0.5, -0.4, 0.6][:d]),     # state 2: moderate
    ]
    for k in range(n_states):
        A_base = base_matrices[k % len(base_matrices)][:d, :d]
        # Add small off-diagonal coupling
        A_base += rng.randn(d, d) * 0.05
        # Ensure stability: spectral radius < 0.95
        eigvals = np.abs(np.linalg.eigvals(A_base))
        max_eig = np.max(eigvals)
        if max_eig > 0.9:
            A_base *= 0.85 / max_eig
        true_A[:, :d, k] = A_base
        # For higher lags, use small coefficients
        for lag in range(1, ar_order):
            true_A[:, lag * d:(lag + 1) * d, k] = rng.randn(d, d) * 0.05

    obs_list = []
    true_labels = []

    # Fixed transition sequence for reproducibility
    state_sequences = [
        rng.choice(n_states, size=rng.randint(3, 6))
        for _ in range(n_demos)
    ]

    for demo_idx in range(n_demos):
        states_seq = state_sequences[demo_idx]
        # Ensure adjacent segments have different states
        filtered = [states_seq[0]]
        for s in states_seq[1:]:
            if s != filtered[-1]:
                filtered.append(s)
        states_seq = filtered

        segments = []
        labels = []
        for state in states_seq:
            seg_len = rng.randint(min_seg_len, max_seg_len + 1)
            segments.append((state, seg_len))
            labels.extend([state] * seg_len)

        T = len(labels)
        obs = np.zeros((d, T))

        # Generate AR process
        for t in range(T):
            k = labels[t]
            A_k = true_A[:, :, k]
            x_t = np.zeros(m)
            for lag in range(1, ar_order + 1):
                if t >= lag:
                    x_t[(lag - 1) * d:lag * d] = obs[:, t - lag]
            obs[:, t] = A_k @ x_t + rng.randn(d) * noise_std

        obs_list.append(obs)
        true_labels.append(np.array(labels))

    return obs_list, true_labels, true_A


def compute_accuracy(true_labels, pred_labels):
    """
    Compute segmentation accuracy using best permutation matching.
    (Simple version: try all permutations up to n_states=6)
    """
    from itertools import permutations

    true_all = np.concatenate(true_labels)
    pred_all = np.concatenate(pred_labels)

    true_states = np.unique(true_all)
    pred_states = np.unique(pred_all)

    n_true = len(true_states)
    n_pred = len(pred_states)

    if n_pred > 10 or n_true > 10:
        # Too many states for brute-force permutation; use greedy matching
        return _greedy_accuracy(true_all, pred_all, true_states, pred_states)

    best_acc = 0.0
    # Try all mappings from pred -> true
    for perm in permutations(range(max(n_true, n_pred)), min(n_true, n_pred)):
        mapped = np.full_like(pred_all, -1)
        for i, p in enumerate(perm[:n_pred]):
            if i < len(pred_states) and p < len(true_states):
                mapped[pred_all == pred_states[i]] = true_states[p]
        acc = np.mean(mapped == true_all)
        best_acc = max(best_acc, acc)

    return best_acc


def _greedy_accuracy(true_all, pred_all, true_states, pred_states):
    """Greedy permutation matching for many states."""
    remaining_true = set(true_states)
    mapping = {}

    for ps in pred_states:
        best_overlap = -1
        best_ts = None
        mask_p = pred_all == ps
        for ts in remaining_true:
            overlap = np.sum(mask_p & (true_all == ts))
            if overlap > best_overlap:
                best_overlap = overlap
                best_ts = ts
        if best_ts is not None:
            mapping[ps] = best_ts
            remaining_true.discard(best_ts)

    mapped = np.full_like(pred_all, -1)
    for ps, ts in mapping.items():
        mapped[pred_all == ps] = ts

    return np.mean(mapped == true_all)


def plot_segmentation_comparison(obs_list, true_labels, pred_labels,
                                  save_path="test_hdphmm_results.png",
                                  n_demos=4):
    """Plot a few demos comparing true vs predicted segmentation."""
    n_show = min(n_demos, len(obs_list))
    fig, axes = plt.subplots(n_show, 2, figsize=(16, 3 * n_show))
    if n_show == 1:
        axes = axes.reshape(1, -1)

    cmap = plt.cm.Set2
    all_true = set()
    all_pred = set()
    for tl in true_labels:
        all_true.update(tl)
    for pl in pred_labels:
        all_pred.update(pl)

    true_colors = {s: cmap(i / max(len(all_true), 1)) for i, s in enumerate(sorted(all_true))}
    pred_colors = {s: cmap(i / max(len(all_pred), 1)) for i, s in enumerate(sorted(all_pred))}

    for row in range(n_show):
        obs = obs_list[row]
        d, T = obs.shape
        t = np.arange(T)

        # True
        ax = axes[row, 0]
        for dim in range(min(3, d)):
            ax.plot(t, obs[dim, :], linewidth=0.6, alpha=0.8)
        tl = true_labels[row]
        seg_start = 0
        for ti in range(1, T):
            if tl[ti] != tl[seg_start]:
                ax.axvspan(seg_start, ti, alpha=0.2, color=true_colors[tl[seg_start]])
                seg_start = ti
        ax.axvspan(seg_start, T, alpha=0.2, color=true_colors[tl[seg_start]])
        ax.set_title(f"Demo {row} - True", fontsize=9)
        ax.tick_params(labelsize=7)

        # Predicted
        ax = axes[row, 1]
        for dim in range(min(3, d)):
            ax.plot(t, obs[dim, :], linewidth=0.6, alpha=0.8)
        pl = pred_labels[row]
        seg_start = 0
        for ti in range(1, T):
            if pl[ti] != pl[seg_start]:
                ax.axvspan(seg_start, ti, alpha=0.2, color=pred_colors[pl[seg_start]])
                seg_start = ti
        ax.axvspan(seg_start, T, alpha=0.2, color=pred_colors[pl[seg_start]])
        ax.set_title(f"Demo {row} - Predicted", fontsize=9)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison plot to {save_path}")


def main():
    print("=" * 60)
    print("Sticky HDP-HMM Test: Synthetic Data")
    print("=" * 60)

    # Generate synthetic data
    n_demos = 10
    n_states = 3
    d = 4
    ar_order = 1

    print(f"\nGenerating {n_demos} demos with {n_states} states, d={d}, AR({ar_order})...")
    obs_list, true_labels, true_A = generate_synthetic_data(
        n_demos=n_demos, d=d, ar_order=ar_order, n_states=n_states
    )

    for i in range(min(3, n_demos)):
        T = obs_list[i].shape[1]
        n_segs = len(set(
            tuple(np.where(np.diff(true_labels[i]) != 0)[0])
        )) + 1
        print(f"  Demo {i}: T={T}, segments={n_segs}")

    # Run HDP-HMM
    print(f"\nRunning Sticky HDP-HMM...")
    model = HDPHMM(
        ar_order=ar_order,
        K_max=10,
        gamma=5.0,
        alpha=5.0,
        kappa=50.0,
        sF=1.0,
        n_iter=150,
        n_chains=2,
        n_viterbi=3,
        verbose=True,
    )
    model.fit(obs_list)
    pred_labels = model.segment()

    # Evaluate
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    n_discovered = len(np.unique(np.concatenate(pred_labels)))
    print(f"Ground truth states: {n_states}")
    print(f"Discovered states: {n_discovered}")

    acc = compute_accuracy(true_labels, pred_labels)
    print(f"Segmentation accuracy (best permutation): {acc:.1%}")

    # Print per-demo details
    for i in range(min(5, n_demos)):
        true_segs = np.sum(np.diff(true_labels[i]) != 0) + 1
        pred_segs = np.sum(np.diff(pred_labels[i]) != 0) + 1
        print(f"  Demo {i}: true_segs={true_segs}, pred_segs={pred_segs}, "
              f"true_states={sorted(set(true_labels[i]))}, "
              f"pred_states={sorted(set(pred_labels[i]))}")

    # Plot
    plot_segmentation_comparison(obs_list, true_labels, pred_labels)

    print("\nDone!")


if __name__ == "__main__":
    main()
