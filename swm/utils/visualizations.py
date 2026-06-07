import io
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import imageio
from swm.utils.envs import BaseEnv





def _to_pil_image(frame: np.ndarray) -> Image.Image:
    """Ensure we have a PIL Image in RGBA."""
    if isinstance(frame, Image.Image):
        img = frame.convert("RGBA")
    else:
        if frame.dtype != np.uint8:
            # normalize to 0-255 if needed
            f = frame.astype(np.float32)
            f = 255 * (f - f.min()) / (f.max() - f.min() + 1e-8)
            frame = f.astype(np.uint8)
        img = Image.fromarray(frame).convert("RGBA")
    return img


def _viridis_color(val: float) -> tuple:
    """
    Map a scalar in [0,1] to an (R,G,B,A) tuple with A=255 using Viridis.
    Values outside [0,1] are clipped.
    """
    val = float(np.clip(val, 0.0, 1.0))
    r, g, b, a = cm.get_cmap("plasma")(val)  # returns floats in 0..1
    return (int(r * 255), int(g * 255), int(b * 255), 255)


def _draw_colored_polyline(base_img: Image.Image,
                           xs, ys, heats,
                           width=3, alpha=0.5):
    """
    Draw a polyline where each small segment uses the color from heats[i].
    We draw on an RGBA overlay then alpha-composite onto base.
    """
    overlay = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Convert alpha fraction to integer once
    alpha_i = int(np.clip(alpha, 0.0, 1.0) * 255)

    n = len(xs)
    if n < 2:
        return base_img

    for i in range(n - 1):
        c = _viridis_color(heats[i])
        # apply desired alpha while keeping RGB
        c = (c[0], c[1], c[2], alpha_i)
        draw.line([(xs[i], ys[i]), (xs[i + 1], ys[i + 1])],
                  fill=c, width=width)

    # Composite overlay onto base
    base_img.alpha_composite(overlay)
    return base_img


def _draw_dashed_polyline(base_img: Image.Image,
                          xs, ys, heats,
                          width=3, alpha=1.0,
                          dash_px=8, gap_px=6):
    """
    Draw a dashed polyline for the expert trajectory, using the heat color.
    Each short dash uses heats[i] for the segment it lies on.
    """
    overlay = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    alpha_i = int(np.clip(alpha, 0.0, 1.0) * 255)

    def segment_length(x0, y0, x1, y1):
        return float(np.hypot(x1 - x0, y1 - y0))

    def interp(p0, p1, t):
        return p0 + (p1 - p0) * t

    for i in range(len(xs) - 1):
        x0, y0, x1, y1 = xs[i], ys[i], xs[i + 1], ys[i + 1]
        seg_len = segment_length(x0, y0, x1, y1)
        if seg_len == 0:
            continue

        # unit direction
        dx = (x1 - x0) / seg_len
        dy = (y1 - y0) / seg_len

        dist = 0.0
        color_rgba = _viridis_color(heats[i])
        color_rgba = (color_rgba[0], color_rgba[1], color_rgba[2], alpha_i)

        while dist < seg_len:
            dash_end = min(dist + dash_px, seg_len)
            # dash start point
            sx = x0 + dx * dist
            sy = y0 + dy * dist
            # dash end point
            ex = x0 + dx * dash_end
            ey = y0 + dy * dash_end

            draw.line([(sx, sy), (ex, ey)], fill=color_rgba, width=width)

            dist = dash_end + gap_px

    base_img.alpha_composite(overlay)
    return base_img


def get_heat_map(env: BaseEnv,
                 frame: np.ndarray,
                 action_seq,
                 rewards,
                 expert_action_seq=None,
                 expert_rewards=None):
    """
    Renders trajectories directly onto `frame` using Viridis-colored segments
    (color encodes heat/reward) and returns a PIL.Image.

    Args:
        env: The environment object with camera and block state methods
        frame: Background image frame (np.ndarray HxWx{3,4} or PIL.Image)
        action_seq: List[Iterable] of per-trajectory actions
        rewards: List/array of per-trajectory heat values aligned with action_seq.
                 If multiple reward sets are provided, they will be averaged.
        expert_action_seq: Optional expert trajectory actions (single sequence)
        expert_rewards: Optional rewards aligned to expert_action_seq
    Returns:
        A single PIL.Image (averaged case) or list of PIL.Images (if not averaged)
    """
    average = True  # preserves original behavior

    # Prepare base
    base_img = _to_pil_image(frame).copy()
    rewards += 0.4 # shift to be nonnegative for better color mapping

    # Average rewards if requested (to keep original logic)
    if average:
        rewards = [np.mean(rewards, axis=0)]

    images = []
    for question_num in range(len(rewards)):
        # Work on a copy so multiple question layers don't bleed into each other
        img = base_img.copy()

        x_coords, y_coords, heat_values = [], [], []
        reward_cur = rewards[question_num]

        # Project each trajectory to pixel space and prep its heat sequence
        for i, acts in enumerate(action_seq):
            pix_x, pix_y = env.project_actions_to_camera_frame(acts)
            x_coords.append(pix_x)
            y_coords.append(pix_y)

            # Maintain the original "first reward if only one" behavior
            if (len(reward_cur) == 1 and len(reward_cur[0]) == 1):
                hv = [float(reward_cur[0].item())] * len(acts)
            else:
                # reward_cur[i] should align with acts
                hv = reward_cur[i]
                # Flatten scalars and ensure list
                if np.isscalar(hv):
                    hv = [float(hv)] * len(acts)
                else:
                    hv = [float(x) for x in np.array(hv).ravel()]
                    if len(hv) < len(acts):
                        # pad if shorter (defensive)
                        hv = hv + [hv[-1]] * (len(acts) - len(hv))
                    elif len(hv) > len(acts):
                        hv = hv[:len(acts)]
            heat_values.append(hv)

        # Draw each trajectory in-place
        for tx, ty, th in zip(x_coords, y_coords, heat_values):
            img = _draw_colored_polyline(img, tx, ty, th, width=2, alpha=0.9)

        # Optional: draw expert trajectory as dashed, higher emphasis
        if expert_action_seq is not None:
            assert expert_rewards is not None, "expert_rewards must be provided with expert_action_seq"
            special_pix_x, special_pix_y = env.project_actions_to_camera_frame(expert_action_seq)
            exp_rewards = np.mean(expert_rewards, axis=0)
            # exp_rewards expected shape like (1, T) in original; fall back robustly
            if isinstance(exp_rewards, np.ndarray) and exp_rewards.ndim == 2 and exp_rewards.shape[0] == 1:
                exp_h = [float(v) for v in exp_rewards[0]]
            else:
                exp_h = [float(v) for v in np.array(exp_rewards).ravel()]
            if len(exp_h) < len(special_pix_x):
                exp_h = exp_h + [exp_h[-1]] * (len(special_pix_x) - len(exp_h))
            elif len(exp_h) > len(special_pix_x):
                exp_h = exp_h[:len(special_pix_x)]

            img = _draw_dashed_polyline(img, special_pix_x, special_pix_y, exp_h,
                                        width=4, alpha=1.0, dash_px=10, gap_px=6)

        images.append(img)

    return images[0] if average else images


def save_video(frames, file_path, fps=10):
    """
    Writes a list of frames to an MP4 video file using imageio.

    :param frames: List of frames (NumPy arrays in RGB format)
    :param file_path: Output file path ending in .mp4
    :param fps: Frames per second for the output video
    """
    with imageio.get_writer(file_path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)
