"""Adapted from https://github.com/meera1hahn/NRNS"""

import traceback
from typing import Callable, List, Optional

import cv2
import habitat
import numpy as np
import quaternion
import skimage
import skimage.morphology
from habitat import logger
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.utils.geometry_utils import (
    angle_between_quaternions,
    quaternion_from_coeff,
)

from habitat_extensions.actions import TeleportAndRotateAction
from habitat_extensions.utils import heading_from_quaternion
from sim2sim_vlnce.Sim2Sim.local_policy.fmm_planner import FMMPlanner
from sim2sim_vlnce.Sim2Sim.local_policy.map_builder import build_mapper


def _get_rotation_plan(
    quat_from: np.quaternion, quat_to: np.quaternion
) -> List[int]:
    """Returns a list of turn actions to make an agent facing the current
    rotation (quat_from) match as closely as possible to the goal rotation
    (quat_to).
    """
    heading = heading_from_quaternion(quat_from)

    delta = angle_between_quaternions(quat_from, quat_to)
    num_steps = int(round(delta / np.deg2rad(15)))
    if num_steps == 0:
        return []

    heading_if_left = (heading + num_steps * np.deg2rad(15)) % (2 * np.pi)
    rotation_if_left = quaternion.from_euler_angles(
        [0.0, heading_if_left, 0.0]
    )
    left_error = angle_between_quaternions(rotation_if_left, quat_to)
    if left_error <= (np.deg2rad(15) / 2.0):
        turn_idx = HabitatSimActions.TURN_LEFT
    else:
        turn_idx = HabitatSimActions.TURN_RIGHT

    return [turn_idx for _ in range(num_steps)]


def env_step(env, episode, action):
    obs = env.sim.step(action)
    env._task.measurements.update_measures(
        episode=episode,
        action=action,
        task=env._task,
    )
    return obs


def navigate(
    env: habitat.Env,
    r: float,
    theta: float,
    max_follower_steps: int = 50,
    info_callback: Optional[Callable] = None,
) -> bool:
    """Spawns a local policy that navigates an agent to the desired goal
    specified in polar coordinates.
    """
    episode = env.current_episode
    infos = []

    start_position = env.sim.get_agent_state().position
    local_agent = LocalAgent(
        curr_pos=start_position,
        curr_rot=env.sim.get_agent_state().rotation,
        map_size_cm=1200,
        map_resolution=5,
    )
    terminate_local = 0
    local_agent.update_local_map(
        np.expand_dims(
            env.sim.get_sensor_observations()["mapping_depth"], axis=2
        )
    )
    local_agent.set_goal(r, theta)
    current_position = env.sim.get_agent_state().position
    curr_rotation = env.sim.get_agent_state().rotation
    succeeded = True
    try:
        action, terminate_local = local_agent.navigate_local()
        for _ in range(max_follower_steps):
            obs = env_step(env, episode, action)
            if info_callback is not None:
                infos.append(info_callback())

            curr_depth_img = obs["mapping_depth"]
            current_position = env.sim.get_agent_state().position
            curr_rotation = env.sim.get_agent_state().rotation
            local_agent.new_sim_origin = get_sim_location(
                current_position, env.sim.get_agent_state().rotation
            )
            local_agent.update_local_map(curr_depth_img)
            action, terminate_local = local_agent.navigate_local()
            if terminate_local == 1:
                break
    except Exception:
        succeeded = False
        logger.info("local navigation threw an error.")
        traceback.print_exc()
        print()

    goal_rotation = quaternion_from_coeff(
        TeleportAndRotateAction.compute_snapped_heading_to(
            start_position, current_position
        ),
    )
    for action in _get_rotation_plan(curr_rotation, goal_rotation):
        env_step(env, episode, action)
        if info_callback is not None:
            infos.append(info_callback())

    return succeeded, infos


class LocalAgent:
    def __init__(
        self,
        curr_pos,
        curr_rot,
        map_size_cm,
        map_resolution,
    ):
        self.mapper = build_mapper()
        self.map_size_cm = map_size_cm
        self.map_resolution = map_resolution
        self.sim_origin = get_sim_location(curr_pos, curr_rot)
        self.collision = False

        #  initialize local map pose
        self.mapper.reset_map()
        self.x_gt = self.map_size_cm / 100.0 / 2.0
        self.y_gt = self.map_size_cm / 100.0 / 2.0
        self.o_gt = 0.0

        self.stg_x, self.stg_y = int(self.y_gt / map_resolution), int(
            self.x_gt / map_resolution
        )
        self.new_sim_origin = self.sim_origin
        self.reset_goal = True

    def update_local_map(self, curr_depth_img):
        self.x_gt, self.y_gt, self.o_gt = self.get_mapper_pose_from_sim_pose(
            self.new_sim_origin,
            self.sim_origin,
        )

        x, y, o = self.x_gt, self.y_gt, self.o_gt
        _, self.local_map, _, self.local_exp_map, _ = self.mapper.update_map(
            curr_depth_img[:, :, 0] * 1000.0, (x, y, o)
        )

        if self.collision:
            self.mapper.map[self.stg_x, self.stg_y, 1] = 10.0
            self.collision = False

    def get_mapper_pose_from_sim_pose(self, sim_pose, sim_origin):
        x, y, o = get_rel_pose_change(sim_pose, sim_origin)
        return (
            self.map_size_cm - (x * 100.0 + self.map_size_cm / 2.0),
            self.map_size_cm - (y * 100.0 + self.map_size_cm / 2.0),
            o,
        )

    def set_goal(self, delta_dist, delta_rot):
        start = (
            int(self.y_gt / self.map_resolution),
            int(self.x_gt / self.map_resolution),
        )
        goal = (
            start[0]
            + int(
                delta_dist
                * np.sin(delta_rot + self.o_gt)
                * 100.0
                / self.map_resolution
            ),
            start[1]
            + int(
                delta_dist
                * np.cos(delta_rot + self.o_gt)
                * 100.0
                / self.map_resolution
            ),
        )
        self.goal = goal

    def navigate_local(self):
        traversible = (
            skimage.morphology.binary_dilation(
                self.local_map, skimage.morphology.disk(2)
            )
            != True
        )

        start = (
            int(self.y_gt / self.map_resolution),
            int(self.x_gt / self.map_resolution),
        )
        traversible[
            start[0] - 2 : start[0] + 3, start[1] - 2 : start[1] + 3
        ] = 1

        planner = FMMPlanner(traversible)

        if self.reset_goal:
            planner.set_goal(self.goal, auto_improve=True)
            self.goal = planner.get_goal()
            self.reset_goal = False
        else:
            planner.set_goal(self.goal, auto_improve=True)

        stg_x, stg_y = start
        stg_x, stg_y, replan = planner.get_short_term_goal2((stg_x, stg_y))

        if get_l2_distance(start[0], self.goal[0], start[1], self.goal[1]) < 3:
            terminate = 1
        else:
            terminate = 0

        agent_orientation = np.rad2deg(self.o_gt)
        action = planner.get_next_action(
            start, (stg_x, stg_y), agent_orientation
        )
        self.stg_x, self.stg_y = int(stg_x), int(stg_y)
        return action, terminate

    def get_map(self):
        self.stg_x, self.stg_y = int(self.y_gt / self.map_resolution), int(
            self.x_gt / self.map_resolution
        )
        metric_map = self.local_map + 0.5 * self.local_exp_map * 255
        metric_map[
            int(self.stg_x) - 1 : int(self.stg_x) + 1,
            int(self.stg_y) - 1 : int(self.stg_y) + 1,
        ] = 255
        return cv2.resize(metric_map / 255.0, (80, 80))


def loop_nav(
    sim, local_agent, start_pos, start_rot, delta_dist, delta_rot, max_steps
):
    prev_poses = []
    nav_length = 0.0
    terminate_local = 0
    obs = sim.get_observations_at(
        start_pos, quaternion.from_float_array(start_rot)
    )
    curr_depth_img = obs["mapping_depth"]
    local_agent.update_local_map(curr_depth_img)
    local_agent.set_goal(delta_dist, delta_rot)
    action, terminate_local = local_agent.navigate_local()
    previous_pose = start_pos
    for _ in range(max_steps):
        obs = sim.step(action)

        curr_depth_img = obs["mapping_depth"]
        curr_position = sim.get_agent_state().position
        curr_rotation = quaternion.as_float_array(
            sim.get_agent_state().rotation
        )
        prev_poses.append([curr_position, curr_rotation])

        local_agent.new_sim_origin = get_sim_location(
            curr_position, sim.get_agent_state().rotation
        )
        nav_length += np.linalg.norm(previous_pose - curr_position)
        previous_pose = curr_position
        local_agent.update_local_map(curr_depth_img)
        action, terminate_local = local_agent.navigate_local()
        if terminate_local == 1:
            break

    return (
        sim.get_agent_state().position,
        quaternion.as_float_array(sim.get_agent_state().rotation),
        nav_length,
        prev_poses,
    )


def map_from_actions(sim, local_agent, start_pos, start_rot, action_list):
    maps = []
    obs = sim.get_observations_at(
        start_pos, quaternion.from_float_array(start_rot)
    )
    curr_depth_img = obs["mapping_depth"]
    local_agent.update_local_map(curr_depth_img)
    maps.append(local_agent.get_map())
    for action in action_list:
        obs = sim.step(action)
        curr_depth_img = obs["mapping_depth"]
        curr_position = sim.get_agent_state().position
        curr_rotation = sim.get_agent_state().rotation
        local_agent.new_sim_origin = get_sim_location(
            curr_position, curr_rotation
        )
        local_agent.update_local_map(curr_depth_img)
        maps.append(local_agent.get_map())
    return maps


def get_sim_location(as_pos, as_rot):
    """
    Input:
        as_pos: agent_state position (x,y,z)
        as_rot: agent_state rotation (4D quaternion)
    Output:
        sim_pose: 3-dof sim pose
    """
    x = as_pos[2]
    y = as_pos[0]
    o = quaternion.as_rotation_vector(as_rot)[1]
    sim_pose = (x, y, o)
    return sim_pose


def get_rel_pose_change(pos2, pos1):
    x1, y1, o1 = pos1
    x2, y2, o2 = pos2

    theta = np.arctan2(y2 - y1, x2 - x1) - o1
    dist = get_l2_distance(x1, x2, y1, y2)
    dx = dist * np.cos(theta)
    dy = dist * np.sin(theta)
    do = o2 - o1

    return dx, dy, do


def get_l2_distance(x1, x2, y1, y2):
    """
    Computes the L2 distance between two points.
    """
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
