"""
Apply BP-AR-HMM to robomimic Lift dataset to segment demonstrations
into shared behavioral skills based on action sequences.
"""

import os
import json
import pickle
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core import BPARHMM


def normalize_first_differences(trajectories, gripper_std=None):
    """
    Normalize trajectories so that the variance of first differences is 1.0
    per dimension, then center each dimension at zero.

    The gripper (last dimension) is normalized separately: its delta std is
    replaced with a user-specified value so that the gripper's scale is
    comparable to the pose dimensions after normalization.

    Args:
        trajectories: list of (T_i, d) arrays
        gripper_std: override for the gripper dimension's delta std.
            If None, defaults to the median delta std of the pose dimensions,
            so the gripper ends up on the same scale as a typical pose dim.

    Returns:
        list of (T_i, d) normalized arrays
    """
    # Compute first differences per demo (avoids cross-demo jump artifacts)
    all_deltas = []
    for traj in trajectories:
        all_deltas.append(np.diff(traj, axis=0))
    combined_deltas = np.vstack(all_deltas)

    # Std of deltas per dimension
    delta_std = np.std(combined_deltas, axis=0)
    delta_std[delta_std == 0] = 1.0

    # Override gripper (last dim) normalization
    if gripper_std is None:
        # Use median of pose dims so gripper gets comparable weight
        gripper_std = np.median(delta_std[:-1])
    delta_std[-1] = gripper_std

    # Scale original data by delta std, then mean-center
    normalized = []
    for traj in trajectories:
        scaled = traj / delta_std
        centered = scaled - np.mean(scaled, axis=0)
        normalized.append(centered)

    return normalized


def load_actions_from_hdf5(dataset_path):
    """
    Load absolute pose sequences from a robomimic HDF5 dataset.

    Constructs an 8D observation per timestep:
      [x, y, z]  from obs/robot0_eef_pos
      [qx, qy, qz, qw] from obs/robot0_eef_quat
      [gripper]   from actions (last dim)

    Normalizes so that first-difference variance is 1.0 per dimension.

    Returns:
        obs_list: list of (8, T_i) arrays for BP-AR-HMM
        demo_keys: list of demo key strings in sorted order
    """
    f = h5py.File(dataset_path, "r")
    demos = list(f["data"].keys())
    # Sort by demo index
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    raw_list = []
    demo_keys = []
    for ep in demos:
        eef_pos = f["data/{}/obs/robot0_eef_pos".format(ep)][:]    # (T, 3)
        eef_quat = f["data/{}/obs/robot0_eef_quat".format(ep)][:]  # (T, 4)
        actions = f["data/{}/actions".format(ep)][:]                # (T, 7)
        gripper = actions[:, -1:]                                   # (T, 1)
        pose = np.concatenate([eef_pos, eef_quat, gripper], axis=1)  # (T, 8)
        raw_list.append(pose)
        demo_keys.append(ep)
    f.close()

    # Normalize: first-difference variance -> 1.0, mean-center
    norm_list = normalize_first_differences(raw_list, gripper_std = 0.01)

    obs_list = [traj.T for traj in norm_list]  # (8, T) for BP-AR-HMM

    print(f"Loaded {len(obs_list)} demonstrations, d={obs_list[0].shape[0]}")
    print(f"Sequence lengths: min={min(o.shape[1] for o in obs_list)}, "
          f"max={max(o.shape[1] for o in obs_list)}, "
          f"mean={np.mean([o.shape[1] for o in obs_list]):.1f}")
    return obs_list, demo_keys


def labels_to_segments(labels):
    """
    Convert a per-timestep label array into a dict of {(start, end): skill}.

    Args:
        labels: (T,) integer array of skill labels

    Returns:
        segments: dict mapping (start_time, end_time) -> skill_id
    """
    segments = {}
    T = len(labels)
    seg_start = 0
    current_skill = labels[0]

    for t in range(1, T):
        if labels[t] != current_skill:
            segments[(int(seg_start), int(t - 1))] = int(current_skill)
            seg_start = t
            current_skill = labels[t]
    # Last segment
    segments[(int(seg_start), int(T - 1))] = int(current_skill)
    return segments


def plot_demo_segmentation(obs, labels, demo_key, save_path):
    """
    Plot a single demo's action dimensions with segment coloring.

    Args:
        obs: (d, T) action array
        labels: (T,) skill labels
        demo_key: string identifier
        save_path: where to save the figure
    """
    d, T = obs.shape
    t = np.arange(T)

    dim_names = ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"]

    unique_labels = np.unique(labels)
    cmap = plt.cm.Set2
    colors = {l: cmap(i / max(len(unique_labels), 1)) for i, l in enumerate(unique_labels)}

    fig, axes = plt.subplots(d, 1, figsize=(12, 1.6 * d), sharex=True)
    if d == 1:
        axes = [axes]

    for dim in range(d):
        ax = axes[dim]
        ax.plot(t, obs[dim, :], color="black", linewidth=0.8)

        # Color background by skill
        seg_start = 0
        current = labels[0]
        for ti in range(1, T):
            if labels[ti] != current or ti == T - 1:
                end = ti if labels[ti] != current else ti + 1
                ax.axvspan(seg_start, end, alpha=0.3, color=colors[current])
                seg_start = ti
                current = labels[ti]
        # Handle last segment if not caught
        if seg_start < T:
            ax.axvspan(seg_start, T, alpha=0.3, color=colors[current])

        name = dim_names[dim] if dim < len(dim_names) else f"dim_{dim}"
        ax.set_ylabel(name, fontsize=8)
        ax.tick_params(labelsize=7)

    axes[0].set_title(f"{demo_key} — Skills: {list(unique_labels)}", fontsize=10)
    axes[-1].set_xlabel("Timestep", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_overview_grid(obs_list, all_labels, demo_keys, save_path, demos_per_page=20):
    """
    Plot an overview grid: one row per demo, showing the first 3 action dims
    with segment coloring. Saves multiple pages if needed.
    """
    n_demos = len(obs_list)
    n_pages = (n_demos + demos_per_page - 1) // demos_per_page

    # Collect all unique skills for consistent coloring
    all_skills = set()
    for labels in all_labels:
        all_skills.update(labels)
    all_skills = sorted(all_skills)
    cmap = plt.cm.Set2
    colors = {s: cmap(i / max(len(all_skills), 1)) for i, s in enumerate(all_skills)}

    for page in range(n_pages):
        start_idx = page * demos_per_page
        end_idx = min(start_idx + demos_per_page, n_demos)
        n_rows = end_idx - start_idx

        fig, axes = plt.subplots(n_rows, 1, figsize=(14, 1.2 * n_rows), sharex=False)
        if n_rows == 1:
            axes = [axes]

        for row, idx in enumerate(range(start_idx, end_idx)):
            ax = axes[row]
            obs = obs_list[idx]
            labels = all_labels[idx]
            T = obs.shape[1]
            t = np.arange(T)

            # Plot first 3 dims
            for dim in range(min(3, obs.shape[0])):
                ax.plot(t, obs[dim, :], linewidth=0.6, alpha=0.7)

            # Color background
            seg_start = 0
            current = labels[0]
            for ti in range(1, T):
                if labels[ti] != current:
                    ax.axvspan(seg_start, ti, alpha=0.25, color=colors[current])
                    seg_start = ti
                    current = labels[ti]
            ax.axvspan(seg_start, T, alpha=0.25, color=colors[current])

            ax.set_ylabel(demo_keys[idx], fontsize=6, rotation=0, labelpad=45)
            ax.tick_params(labelsize=5)
            ax.set_xlim(0, max(o.shape[1] for o in obs_list))

        axes[-1].set_xlabel("Timestep", fontsize=8)
        fig.suptitle(f"BP-AR-HMM Segmentation Overview (page {page + 1}/{n_pages})",
                     fontsize=11, y=1.0)
        plt.tight_layout()
        page_path = save_path.replace(".png", f"_page{page + 1}.png")
        plt.savefig(page_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved overview: {page_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_demo", action="store_true",
                        help="save individual per-demo segmentation plots")
    args = parser.parse_args()

    dataset_path = "/tmp/core_datasets/coffee/demo_src_coffee_task_D0/demo.hdf5"
    # dataset_path = "/tmp/robomimic_ds_example/low_dim_v141.hdf5"
    output_dir = "/home/ziyi/git/robomimic/bparhmm/results_robomimic"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "overviews"), exist_ok=True)
    if args.per_demo:
        os.makedirs(os.path.join(output_dir, "per_demo"), exist_ok=True)

    # 1. Load data
    print("=" * 60)
    print("Loading robomimic dataset...")
    print("=" * 60)
    obs_list, demo_keys = load_actions_from_hdf5(dataset_path)

    # 2. Run BP-AR-HMM
    print("\n" + "=" * 60)
    print("Running BP-AR-HMM inference...")
    print("=" * 60)
    model = BPARHMM(
        ar_order=2,   # AR lag order: how many past timesteps predict the next.
                      #   1 = velocity-like, 2 = acceleration-like. Higher = richer dynamics.
        K_init=8,     # Initial number of skills. Model adapts via birth/death moves,
                      #   so this is just a starting point, not a hard cap.
        alpha0=2.0,   # HDP concentration: controls sparsity of transition distributions.
                      #   Higher = more uniform transitions across skills.
                      #   Lower = each skill transitions to fewer distinct next-skills.
        kappa0=1.0,   # Sticky self-transition bias: added to diagonal of transition matrix.
                      #   Higher = longer segments, fewer skill switches.
                      #   The ratio kappa0/alpha0 determines stickiness strength.
        gamma0=2.0,   # IBP mass parameter: controls the birth rate of new skills.
                      #   Higher = more new skills proposed per iteration.
                      #   Births per iter ~ gamma0 (across all demos).
        n_iter=300,   # Gibbs sampling iterations per chain. More = better convergence.
        n_chains=3,   # Number of independent MCMC chains. Best chain (by log-likelihood)
                      #   is selected. More chains = better exploration of posterior.
        verbose=True,
    )
    model.fit(obs_list)
    all_labels = model.segment()

    # 3. Build segmentation dictionary
    print("\n" + "=" * 60)
    print("Building segmentation dictionary...")
    print("=" * 60)
    segmentation_dict = {}
    for i, demo_key in enumerate(demo_keys):
        segments = labels_to_segments(all_labels[i])
        segmentation_dict[demo_key] = segments

    # Print summary
    all_n_segments = [len(v) for v in segmentation_dict.values()]
    all_skills_found = set()
    for v in segmentation_dict.values():
        all_skills_found.update(v.values())
    print(f"Total skills discovered: {len(all_skills_found)}")
    print(f"Segments per demo: min={min(all_n_segments)}, max={max(all_n_segments)}, "
          f"mean={np.mean(all_n_segments):.1f}")

    # Print first 5 demos as examples
    for demo_key in demo_keys[:5]:
        print(f"\n  {demo_key}:")
        for (start, end), skill in segmentation_dict[demo_key].items():
            print(f"    t=[{start:3d}, {end:3d}] -> skill {skill}")

    # 4. Save segmentation dictionary
    dict_path = os.path.join(output_dir, "segmentation.pkl")
    with open(dict_path, "wb") as f:
        pickle.dump(segmentation_dict, f)
    print(f"\nSegmentation dict saved to {dict_path}")

    # Also save as JSON (human-readable, with string keys)
    json_dict = {}
    for demo_key, segments in segmentation_dict.items():
        json_dict[demo_key] = {f"({s},{e})": skill for (s, e), skill in segments.items()}
    json_path = os.path.join(output_dir, "segmentation.json")
    with open(json_path, "w") as f:
        json.dump(json_dict, f, indent=2)
    print(f"Segmentation JSON saved to {json_path}")

    # 5. Plot per-demo segmentations (only if --per_demo flag is passed)
    if args.per_demo:
        print("\nRendering per-demo plots...")
        for i, demo_key in enumerate(demo_keys):
            save_path = os.path.join(output_dir, "per_demo", f"{demo_key}.png")
            plot_demo_segmentation(obs_list[i], all_labels[i], demo_key, save_path)
        print(f"  Saved {len(demo_keys)} per-demo plots to {output_dir}/per_demo/")

    # 6. Plot overview grids
    print("\nRendering overview grid...")
    overview_path = os.path.join(output_dir, "overviews", "overview.png")
    plot_overview_grid(obs_list, all_labels, demo_keys, overview_path, demos_per_page=25)

    print("\nDone!")


if __name__ == "__main__":
    main()