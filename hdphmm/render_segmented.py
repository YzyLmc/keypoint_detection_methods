"""
Render dataset demonstrations as videos with HDP-HMM skill labels overlaid.

Takes a robomimic HDF5 dataset and a segmentation.pkl file, renders each demo
via the simulator, and draws the current skill index + a colored bar on each frame.

Usage:
    python bparhmm/render_segmented.py \
        --dataset /tmp/core_datasets/coffee/demo_src_coffee_task_D0/demo.hdf5 \
        --segmentation hdphmm/results_robomimic/segmentation.pkl \
        --render_image_names agentview \
        --output_dir hdphmm/results_robomimic/videos
"""

import os
import pickle
import argparse
import h5py
import numpy as np
import imageio
import cv2

import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils


# Distinct colors (BGR for cv2) for up to 12 skills
SKILL_COLORS = [
    (76, 153, 0),     # green
    (204, 102, 0),    # blue
    (0, 153, 204),    # orange
    (153, 51, 153),   # purple
    (0, 204, 204),    # yellow
    (204, 0, 102),    # magenta
    (102, 204, 0),    # lime
    (0, 102, 204),    # dark orange
    (204, 153, 0),    # teal
    (102, 0, 204),    # red-purple
    (204, 204, 0),    # cyan
    (0, 0, 204),      # red
]


def get_skill_at_timestep(segments, t):
    """Look up which skill is active at timestep t."""
    for (start, end), skill in segments.items():
        if start <= t <= end:
            return skill
    return -1


def overlay_skill_info(frame, timestep, total_steps, skill_id, segments):
    """
    Draw skill label and a timeline bar on the frame.

    Args:
        frame: (H, W, 3) uint8 RGB image
        timestep: current timestep
        total_steps: total timesteps in this demo
        skill_id: current skill index
        segments: dict of {(start, end): skill} for timeline bar
    """
    h, w = frame.shape[:2]
    # Convert RGB to BGR for cv2 drawing, then back
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    color_bgr = SKILL_COLORS[skill_id % len(SKILL_COLORS)]

    # Skill label text (top-left)
    text = f"Skill {skill_id}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 3

    # Draw background rectangle for text
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(img, (10, 10), (20 + tw, 20 + th + baseline), (0, 0, 0), -1)
    cv2.putText(img, text, (15, 15 + th), font, font_scale, color_bgr, thickness)

    # Timestep counter (top-right)
    ts_text = f"t={timestep}/{total_steps - 1}"
    (tsw, tsh), _ = cv2.getTextSize(ts_text, font, 0.7, 2)
    cv2.rectangle(img, (w - tsw - 20, 10), (w - 10, 20 + tsh), (0, 0, 0), -1)
    cv2.putText(img, ts_text, (w - tsw - 15, 15 + tsh), font, 0.7,
                (255, 255, 255), 2)

    # Timeline bar at bottom
    bar_h = 20
    bar_y = h - bar_h - 10
    bar_x = 20
    bar_w = w - 40

    # Black background for bar
    cv2.rectangle(img, (bar_x - 2, bar_y - 2),
                  (bar_x + bar_w + 2, bar_y + bar_h + 2), (0, 0, 0), -1)

    # Draw colored segments
    for (start, end), sk in segments.items():
        x0 = bar_x + int(start / total_steps * bar_w)
        x1 = bar_x + int((end + 1) / total_steps * bar_w)
        c = SKILL_COLORS[sk % len(SKILL_COLORS)]
        cv2.rectangle(img, (x0, bar_y), (x1, bar_y + bar_h), c, -1)

    # Playhead marker
    px = bar_x + int(timestep / total_steps * bar_w)
    cv2.line(img, (px, bar_y - 4), (px, bar_y + bar_h + 4), (255, 255, 255), 2)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def render_demo_with_skills(env, f, demo_key, segments, camera_names, video_path):
    """
    Render a single demo with skill overlays and save to video.
    """
    states = f[f"data/{demo_key}/states"][()]
    initial_state = dict(states=states[0])
    if "model_file" in f[f"data/{demo_key}"].attrs:
        initial_state["model"] = f[f"data/{demo_key}"].attrs["model_file"]

    total_steps = states.shape[0]

    video_writer = imageio.get_writer(video_path, fps=20)

    env.reset()
    env.reset_to(initial_state)

    for t in range(total_steps):
        env.reset_to({"states": states[t]})

        # Render frame from each camera and concatenate
        frames = []
        for cam in camera_names:
            frame = env.render(mode="rgb_array", height=512, width=512, camera_name=cam)
            frames.append(frame)
        frame = np.concatenate(frames, axis=1)

        # Look up current skill
        skill_id = get_skill_at_timestep(segments, t)

        # Overlay skill info
        frame = overlay_skill_info(frame, t, total_steps, skill_id, segments)
        video_writer.append_data(frame)

    video_writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="path to hdf5 dataset")
    parser.add_argument("--segmentation", type=str, required=True,
                        help="path to segmentation.pkl")
    parser.add_argument("--render_image_names", type=str, nargs="+",
                        default=["agentview"],
                        help="camera name(s) for rendering")
    parser.add_argument("--output_dir", type=str,
                        default="bparhmm/results_robomimic/videos",
                        help="directory to save videos")
    parser.add_argument("--n", type=int, default=None,
                        help="only render first n demos")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load segmentation
    with open(args.segmentation, "rb") as pf:
        segmentation_dict = pickle.load(pf)
    print(f"Loaded segmentation with {len(segmentation_dict)} demos")

    # Initialize obs utils with dummy spec
    dummy_spec = dict(obs=dict(low_dim=["robot0_eef_pos"], rgb=[]))
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)

    # Create environment
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, render=False, render_offscreen=True
    )

    f = h5py.File(args.dataset, "r")

    # Get sorted demo keys
    demos = sorted(segmentation_dict.keys(), key=lambda x: int(x.split("_")[1]))
    if args.n is not None:
        demos = demos[:args.n]

    for i, demo_key in enumerate(demos):
        if demo_key not in segmentation_dict:
            print(f"Skipping {demo_key}: not in segmentation")
            continue

        video_path = os.path.join(args.output_dir, f"{demo_key}.mp4")
        print(f"[{i + 1}/{len(demos)}] Rendering {demo_key} -> {video_path}")

        render_demo_with_skills(
            env=env,
            f=f,
            demo_key=demo_key,
            segments=segmentation_dict[demo_key],
            camera_names=args.render_image_names,
            video_path=video_path,
        )

    f.close()
    print(f"\nDone! Videos saved to {args.output_dir}")


if __name__ == "__main__":
    main()
