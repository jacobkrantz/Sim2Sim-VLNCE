import textwrap
from copy import deepcopy
from typing import Any, Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
from habitat.utils.visualizations import maps as habitat_maps
from matplotlib import colors, patches
from numpy import ndarray

from habitat_extensions import maps


def rgb_to_plt(img):
    total_width_mul = 3

    trim = 240 * (total_width_mul - 1) + 35
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    w = img.shape[1]

    fig = plt.figure()
    fig.set_dpi(100 * total_width_mul)

    plt.title("RGB", fontsize=20 // total_width_mul)
    plt.imshow(img)
    plt.xticks(
        [0, w // 4, w // 2, (3 * w) // 4, w],
        ["-180", "-90", "0", "90", "180"],
        fontsize=12 // total_width_mul,
    )
    plt.yticks([])

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = data[trim:-trim]
    plt.close()
    return data


def rgb_pano(obs: Dict[str, ndarray]) -> ndarray:
    """stitches an RGB pano together, center elevation only."""
    imgs = []
    if "low_camera_0" in obs:
        for i in range(12):
            imgs.append(obs[f"low_camera_{i}"])
    else:
        for k, v in obs.items():
            if k.split("_")[0] != "rgb":
                continue
            idx = int(k.split("_")[1])
            if idx < 12 or idx > 23:
                continue
            imgs.append(v)

    # cut out overlapping regions (assumes 90 deg HFOV)
    trim = imgs[0].shape[1] // 3
    imgs = [img[:, trim:-trim] for img in imgs]
    w = imgs[0].shape[1]

    final_img = np.concatenate(imgs, axis=1)

    shift = int(final_img.shape[1] / 2 - (w / 2))
    shift += int(
        1 / (48 * 2) * final_img.shape[1]
    )  # for comparison with radial occupancy map
    rolled = np.roll(final_img, shift=shift, axis=1)
    return cv2.cvtColor(rolled, cv2.COLOR_BGR2RGB)


def ro_to_plt(occupancy) -> None:
    trim = 70
    fig = plt.figure()
    plt.gca().set_aspect("equal")

    COLOR_OCCUPIED = "black"
    COLOR_UNKNOWN = "seagreen"
    COLOR_FREE = "yellow"

    cmap = colors.ListedColormap([COLOR_FREE, COLOR_UNKNOWN, COLOR_OCCUPIED])
    bounds = [-2, -0.5, 0.5, 2]
    norm = colors.BoundaryNorm(bounds, cmap.N, clip=True)

    plt.title("Laser Scan", fontsize=20)
    plt.ylabel("Range (m)", fontsize=14)

    # the RO map needs to be centered
    ro_shift = occupancy.shape[1] // 2
    ro_shift -= occupancy.shape[1] // 8
    occupancy = np.roll(occupancy, shift=ro_shift, axis=1)

    plt.pcolormesh(occupancy, cmap=cmap, norm=norm)
    plt.legend(
        handles=[
            patches.Patch(color=COLOR_OCCUPIED, label="Occupied"),
            patches.Patch(color=COLOR_UNKNOWN, label="Uknown"),
            patches.Patch(color=COLOR_FREE, label="Free"),
        ],
        bbox_to_anchor=(
            0.5,
            -0.42,
        ),
        loc="lower center",
    )
    plt.xticks(
        [0, 12, 24, 36, 48], ["-180", "-90", "0", "90", "180"], fontsize=12
    )
    plt.yticks(
        [0, 6, 12, 18, 24], ["0.0", "1.2", "2.4", "3.6", "4.8"], fontsize=12
    )
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = data[trim:]
    plt.close()
    return data


def subgoal_candidates_to_radial_map(shape, candidates_relative):
    r_bins = shape[0]
    h_bins = shape[1]
    range_bin_width = round(4.8 / r_bins, 2)
    h_bin_width = round(np.deg2rad(360 / h_bins), 2)
    gt_preds = np.zeros((r_bins, h_bins))  # 24, 48

    for i in range(candidates_relative.shape[0]):
        if np.all(candidates_relative[i] == 0.0):
            break
        r = candidates_relative[i, 0]
        r_bin = round((r - 0.5 * range_bin_width) / range_bin_width) % r_bins
        h = -candidates_relative[i, 1] % (np.pi * 2)
        h_bin = round((h - 0.5 * h_bin_width) / h_bin_width) % h_bins
        h_bin_rolled = ((h_bins // 2) + h_bin) % h_bins
        gt_preds[int(r_bin), int(h_bin_rolled)] = 1.0
    return gt_preds


def subgoals_to_plt(goals, plotname):
    trim = 70
    fig = plt.figure()
    plt.gca().set_aspect("equal")

    COLOR_EMPTY = "black"
    COLOR_SUBGOAL = "yellow"

    cmap = colors.ListedColormap([COLOR_EMPTY, COLOR_SUBGOAL])
    bounds = [-1, 0.0001, 0.1]
    norm = colors.BoundaryNorm(bounds, cmap.N, clip=True)

    plt.title(plotname, fontsize=20)
    plt.ylabel("Range (m)", fontsize=14)
    plt.pcolormesh(goals, cmap=cmap, norm=norm)
    plt.legend(
        handles=[
            patches.Patch(color=COLOR_SUBGOAL, label="Subgoal"),
        ],
        bbox_to_anchor=(
            0.5,
            -0.42,
        ),
        loc="lower center",
    )
    plt.xticks(
        [0, 12, 24, 36, 48], ["-180", "-90", "0", "90", "180"], fontsize=12
    )
    plt.yticks(
        [0, 6, 12, 18, 24], ["0.0", "1.2", "2.4", "3.6", "4.8"], fontsize=12
    )
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = data[trim:]
    plt.close()
    return data


def prepare_map(td_map, t_obs, map_height):
    def preds_to_coords(r, theta):
        pos = t_obs["globalgps"]
        phi = (t_obs["heading"][0] + theta) % (2 * np.pi)
        x = pos[0] - r * np.sin(phi)
        z = pos[-1] - r * np.cos(phi)
        return np.array([x, z])

    top_down_map = deepcopy(td_map["map"])
    meters_per_px = td_map["meters_per_px"]
    bounds = td_map["bounds"]

    # display all oracle waypoint candidates (aligns with MP3D waypoints)
    for i in range(t_obs["vln_candidates_relative"].shape[0]):
        candidate = t_obs["vln_candidates_relative"][i]
        if np.all(candidate == 0.0):
            continue
        coords = preds_to_coords(r=candidate[0], theta=candidate[1])
        maps.draw_oracle_waypoint(top_down_map, coords, meters_per_px, bounds)

    # Subgoal predictions are (r, theta). Skip if the candidates are [x,y,z].
    has_sgm_predictions = t_obs["candidate_coordinates"].shape[-1] == 2
    if has_sgm_predictions:
        # display all subgoal candidate predictions
        for i in range(t_obs["candidate_coordinates"].shape[0]):
            candidate = t_obs["candidate_coordinates"][i]
            if np.all(candidate == 0.0):
                continue
            coords = preds_to_coords(r=candidate[0], theta=candidate[1])
            maps.draw_waypoint_prediction(
                top_down_map, coords, meters_per_px, bounds, pad=0.25
            )

    top_down_map = maps.colorize_topdown_map(
        top_down_map,
        td_map["fog_of_war_mask"],
        fog_of_war_desat_amount=0.75,
    )
    map_agent_pos = td_map["agent_map_coord"]
    top_down_map = habitat_maps.draw_agent(
        image=top_down_map,
        agent_center_coord=map_agent_pos,
        agent_rotation=td_map["agent_angle"],
        agent_radius_px=int(0.45 / td_map["meters_per_px"]),
    )
    if top_down_map.shape[1] < top_down_map.shape[0]:
        top_down_map = np.rot90(top_down_map, 1)

    if top_down_map.shape[0] > top_down_map.shape[1]:
        top_down_map = np.rot90(top_down_map, 1)

    # scale top down map
    old_h, old_w, _ = top_down_map.shape
    top_down_width = int(old_w * map_height / old_h)
    top_down_map = cv2.resize(
        top_down_map,
        (int(top_down_width), map_height),
        interpolation=cv2.INTER_CUBIC,
    )
    return top_down_map


def append_text_to_image(image: np.ndarray, text: str):
    """Appends text underneath an image of size (height, width, channels).
    The returned image has white text on a black background. Uses textwrap to
    split long text into multiple lines.
    Args:
        image: the image to put text underneath
        text: a string to display
    Returns:
        A new image with text inserted underneath the input image
    """
    h, w, c = image.shape
    font_size = 0.8
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    blank_image = 255 * np.ones(image.shape, dtype=np.uint8)

    char_size = cv2.getTextSize(" ", font, font_size, font_thickness)[0]
    wrapped_text = textwrap.wrap(text, width=int(w / char_size[0]))

    y = 0
    for line in wrapped_text:
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        y += textsize[1] + 10
        x = 10
        cv2.putText(
            blank_image,
            line,
            (x, y),
            font,
            font_size,
            (0, 0, 0),
            font_thickness,
            lineType=cv2.LINE_AA,
        )
    text_image = blank_image[0 : y + 10, 0:w]
    final = np.concatenate((image, text_image), axis=0)
    return final


def pad_to_height(img, height):
    img_height = img.shape[0]
    if img_height > height:
        return cv2.resize(img, (img.shape[1], height))

    white_shape = (height - img_height, img.shape[1], 3)
    white = 255 * np.ones(white_shape, dtype=np.uint8)
    return np.concatenate([img, white], axis=0)


def resize_to_macro_block(img: ndarray, macro_block_size: int = 16):
    """Resizes a [h,w,c] image so that h and w are the nearest multiples of
    the given macro_block_size.
    """
    h, w, _ = img.shape
    height = macro_block_size * round(h / macro_block_size)
    width = macro_block_size * round(w / macro_block_size)
    return cv2.resize(
        img, dsize=(width, height), interpolation=cv2.INTER_CUBIC
    )


def video_frame(
    observations: Dict[str, ndarray],
    transformed_observations: Dict[str, ndarray],
    info: Dict[str, Any],
    map_k: str = "top_down_map_vlnce",
    frame_width: int = 1920,
) -> ndarray:
    """Video frame contains an RGB pano, radial occupancy map, subgoal
    candidates, oracle subgoal candidates, and a top down map with subgoals.

    Args:
        observations (Dict[str, ndarray]): straight from env.reset() or .step()
        transformed_observations (Dict[str, ndarray]): numpy arrays after transform_observations
        info (Dict[str, Any]): must contain map_k.
        map_k (str, optional): Map info key. Defaults to "top_down_map_vlnce".
        frame_width (int, optional): The resized frame width. Defaults to 1920.

    Returns:
        ndarray: the video frame of size [h, frame_width, 3]
    """
    frame = rgb_to_plt(rgb_pano(observations))
    if "radial_occupancy" in transformed_observations:
        gt = subgoal_candidates_to_radial_map(
            transformed_observations["radial_pred"].shape,
            transformed_observations["vln_candidates_relative"],
        )
        ro = ro_to_plt(transformed_observations["radial_occupancy"])
        pred_img = subgoals_to_plt(
            transformed_observations["radial_pred"], "Subgoal Prediction"
        )
        gt_img = subgoals_to_plt(gt, "Subgoal Ground Truth")

        goals = np.concatenate([ro, pred_img, gt_img], axis=1)
        frame = np.concatenate([frame, goals], axis=0)

    top_down_map = prepare_map(
        info[map_k],
        transformed_observations,
        map_height=int(4 / 5 * frame.shape[0]),
    )
    right_side = append_text_to_image(
        top_down_map, observations["instruction"]["text"]
    )
    right_side = pad_to_height(right_side, height=frame.shape[0])

    frame = np.concatenate([frame, right_side], axis=1)
    frame = cv2.resize(
        frame,
        (frame_width, int(frame.shape[0] * frame_width / frame.shape[1])),
        interpolation=cv2.INTER_CUBIC,
    )
    return resize_to_macro_block(frame.astype(np.uint8))
