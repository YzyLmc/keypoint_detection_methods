"""
Test script for BP-AR-HMM.

Generates synthetic multi-dimensional time-series data with known segment
structure, runs BP-AR-HMM inference, and evaluates the results.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core import BPARHMM


def generate_synthetic_data(
    n_demos=5,
    d=3,
    T=200,
    n_true_skills=4,
    ar_order=1,
    noise_std=0.1,
    seed=42,
):
    """
    Generate synthetic AR time-series with known segmentation.

    Each skill has a distinct AR(1) matrix A_k. Demonstrations share the
    same skill set but may use them in different orders.

    Returns:
        obs_list: list of (d, T) observation arrays
        true_labels: list of (T,) ground-truth label arrays
        true_A: (d, d*ar_order, n_true_skills) true AR matrices
    """
    rng = np.random.RandomState(seed)
    m = d * ar_order

    # Generate distinct AR matrices for each skill (stable, eigenvalues < 1)
    true_A = np.zeros((d, m, n_true_skills))
    for k in range(n_true_skills):
        # Random rotation + scaling to ensure stability
        Q, _ = np.linalg.qr(rng.randn(d, d))
        eigenvalues = rng.uniform(0.3, 0.8, d) * rng.choice([-1, 1], d)
        A_base = Q @ np.diag(eigenvalues) @ Q.T
        # For AR(1), A is (d, d)
        if ar_order == 1:
            true_A[:, :, k] = A_base
        else:
            true_A[:, :d, k] = A_base * 0.8
            for lag in range(1, ar_order):
                true_A[:, lag * d : (lag + 1) * d, k] = (
                    rng.randn(d, d) * 0.1 / (lag + 1)
                )

    # Noise covariance
    Sigma = np.eye(d) * noise_std ** 2

    obs_list = []
    true_labels = []

    for demo in range(n_demos):
        obs = np.zeros((d, T))
        labels = np.zeros(T, dtype=int)

        # Create segment boundaries: 3-6 segments per demo
        n_segments = rng.randint(3, 7)
        boundaries = np.sort(rng.choice(range(20, T - 20), n_segments - 1, replace=False))
        boundaries = np.concatenate([[0], boundaries, [T]])

        # Assign skills to segments (each demo uses a random permutation)
        segment_skills = rng.choice(n_true_skills, n_segments, replace=True)
        # Ensure at least 2 distinct skills
        if len(set(segment_skills)) < 2:
            segment_skills[1] = (segment_skills[0] + 1) % n_true_skills

        for seg in range(n_segments):
            t_start = boundaries[seg]
            t_end = boundaries[seg + 1]
            k = segment_skills[seg]
            labels[t_start:t_end] = k

            A_k = true_A[:, :, k]
            for t in range(t_start, t_end):
                # Build AR feature vector
                x_t = np.zeros(m)
                for lag in range(1, ar_order + 1):
                    if t - lag >= 0:
                        x_t[(lag - 1) * d : lag * d] = obs[:, t - lag]
                obs[:, t] = A_k @ x_t + rng.multivariate_normal(np.zeros(d), Sigma)

        obs_list.append(obs)
        true_labels.append(labels)

    return obs_list, true_labels, true_A


def compute_accuracy(true_labels, pred_labels):
    """
    Compute segmentation accuracy using best-match label permutation.

    Uses a greedy matching to handle label permutation ambiguity.
    """
    # Build confusion matrix
    true_set = set()
    pred_set = set()
    for tl, pl in zip(true_labels, pred_labels):
        true_set.update(tl)
        pred_set.update(pl)
    true_ids = sorted(true_set)
    pred_ids = sorted(pred_set)

    confusion = np.zeros((len(true_ids), len(pred_ids)))
    for tl, pl in zip(true_labels, pred_labels):
        for t_val, p_val in zip(tl, pl):
            i = true_ids.index(t_val)
            j = pred_ids.index(p_val)
            confusion[i, j] += 1

    # Greedy matching
    mapping = {}
    used_true = set()
    used_pred = set()
    flat = []
    for i in range(len(true_ids)):
        for j in range(len(pred_ids)):
            flat.append((-confusion[i, j], i, j))
    flat.sort()

    for _, i, j in flat:
        if i not in used_true and j not in used_pred:
            mapping[pred_ids[j]] = true_ids[i]
            used_true.add(i)
            used_pred.add(j)

    # Compute accuracy
    correct = 0
    total = 0
    for tl, pl in zip(true_labels, pred_labels):
        for t_val, p_val in zip(tl, pl):
            mapped = mapping.get(p_val, -1)
            if mapped == t_val:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.0


def plot_results(obs_list, true_labels, pred_labels, save_path="bparhmm_results.png"):
    """Plot observations colored by true and predicted labels."""
    n_demos = len(obs_list)
    fig, axes = plt.subplots(n_demos, 2, figsize=(16, 3 * n_demos), squeeze=False)

    for i in range(n_demos):
        obs = obs_list[i]
        T = obs.shape[1]
        t = np.arange(T)

        # True labels
        ax = axes[i, 0]
        for dim in range(min(obs.shape[0], 3)):
            ax.plot(t, obs[dim, :], alpha=0.5, linewidth=0.8)
        # Color background by true labels
        labels = true_labels[i]
        unique_labels = np.unique(labels)
        colors = plt.cm.Set3(np.linspace(0, 1, max(len(unique_labels), 1)))
        label_to_color = {l: colors[idx % len(colors)] for idx, l in enumerate(unique_labels)}
        for seg_t in range(T):
            ax.axvspan(seg_t, seg_t + 1, alpha=0.15, color=label_to_color[labels[seg_t]])
        ax.set_title(f"Demo {i + 1} - Ground Truth")
        ax.set_ylabel("Value")

        # Predicted labels
        ax = axes[i, 1]
        for dim in range(min(obs.shape[0], 3)):
            ax.plot(t, obs[dim, :], alpha=0.5, linewidth=0.8)
        labels = pred_labels[i]
        unique_labels = np.unique(labels)
        colors = plt.cm.Set3(np.linspace(0, 1, max(len(unique_labels), 1)))
        label_to_color = {l: colors[idx % len(colors)] for idx, l in enumerate(unique_labels)}
        for seg_t in range(T):
            ax.axvspan(seg_t, seg_t + 1, alpha=0.15, color=label_to_color[labels[seg_t]])
        ax.set_title(f"Demo {i + 1} - BP-AR-HMM")
        ax.set_ylabel("Value")

    axes[-1, 0].set_xlabel("Time")
    axes[-1, 1].set_xlabel("Time")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Results saved to {save_path}")


def main():
    print("=" * 60)
    print("BP-AR-HMM Test: Synthetic Time-Series Segmentation")
    print("=" * 60)

    # Generate synthetic data
    n_demos = 5
    d = 3
    T = 200
    n_true_skills = 4
    ar_order = 1

    print(f"\nGenerating synthetic data:")
    print(f"  {n_demos} demonstrations, d={d}, T={T}")
    print(f"  {n_true_skills} true skills, AR order={ar_order}")

    obs_list, true_labels, true_A = generate_synthetic_data(
        n_demos=n_demos, d=d, T=T, n_true_skills=n_true_skills, ar_order=ar_order
    )

    # Print segment info
    for i in range(n_demos):
        unique, counts = np.unique(true_labels[i], return_counts=True)
        segments = []
        for u, c in zip(unique, counts):
            segments.append(f"skill {u}: {c} steps")
        print(f"  Demo {i + 1}: {', '.join(segments)}")

    # Run BP-AR-HMM
    print(f"\nRunning BP-AR-HMM inference...")
    model = BPARHMM(
        ar_order=ar_order,
        K_init=8,
        alpha0=1.0,
        kappa0=50.0,
        gamma0=2.0,
        n_iter=100,
        n_chains=2,
        verbose=True,
    )
    model.fit(obs_list)

    pred_labels = model.segment()

    # Evaluate
    accuracy = compute_accuracy(true_labels, pred_labels)
    print(f"\nResults:")
    print(f"  Discovered {model.F_.shape[1]} skills (true: {n_true_skills})")
    print(f"  Segmentation accuracy: {accuracy:.1%}")

    for i in range(n_demos):
        unique_pred = np.unique(pred_labels[i])
        print(f"  Demo {i + 1}: {len(unique_pred)} unique predicted skills")

    # Plot
    plot_results(obs_list, true_labels, pred_labels)

    print("\nDone.")


if __name__ == "__main__":
    main()