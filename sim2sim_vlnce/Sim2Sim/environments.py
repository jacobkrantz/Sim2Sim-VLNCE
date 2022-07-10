import random
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import habitat
import numpy as np
import quaternion
from habitat import (
    Config,
    Dataset,
    Env,
    RLEnv,
    VectorEnv,
    logger,
    make_dataset,
)
from habitat.core.dataset import ALL_SCENES_MASK
from habitat.core.simulator import Observations
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import (
    angle_between_quaternions,
    quaternion_from_coeff,
    quaternion_rotate_vector,
)
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.utils.env_utils import make_env_fn
from habitat_sim.errors import GreedyFollowerError

from habitat_extensions.actions import TeleportAndRotateAction
from habitat_extensions.utils import (
    compute_heading_to,
    heading_from_quaternion,
)
from sim2sim_vlnce.Sim2Sim.local_policy.local_policy import navigate


def construct_envs(
    config: Config,
    env_class: Type[Union[Env, RLEnv]],
    workers_ignore_signals: bool = False,
    auto_reset_done: bool = True,
    episodes_allowed: Optional[List[str]] = None,
) -> VectorEnv:
    """Create VectorEnv object with specified config and env class type.
    To allow better performance, dataset are split into small ones for
    each individual env, grouped by scenes.
    :param config: configs that contain num_environments as well as information
    :param necessary to create individual environments.
    :param env_class: class type of the envs to be created.
    :param workers_ignore_signals: Passed to :ref:`habitat.VectorEnv`'s constructor
    :param auto_reset_done: Whether or not to automatically reset the env on done
    :return: VectorEnv object created according to specification.
    """

    num_envs_per_gpu = config.NUM_ENVIRONMENTS
    if isinstance(config.SIMULATOR_GPU_IDS, list):
        gpus = config.SIMULATOR_GPU_IDS
    else:
        gpus = [config.SIMULATOR_GPU_IDS]
    num_gpus = len(gpus)
    num_envs = num_gpus * num_envs_per_gpu

    if episodes_allowed is not None:
        config.defrost()
        config.TASK_CONFIG.DATASET.EPISODES_ALLOWED = episodes_allowed
        config.freeze()

    configs = []
    env_classes = [env_class for _ in range(num_envs)]
    dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE)
    scenes = config.TASK_CONFIG.DATASET.CONTENT_SCENES
    if ALL_SCENES_MASK in config.TASK_CONFIG.DATASET.CONTENT_SCENES:
        scenes = dataset.get_scenes_to_load(config.TASK_CONFIG.DATASET)

    if num_envs > 1:
        if len(scenes) == 0:
            raise RuntimeError(
                "No scenes to load, multi-process logic relies on being able"
                " to split scenes uniquely between processes"
            )

        if len(scenes) < num_envs and len(scenes) != 1:
            raise RuntimeError(
                "reduce the number of GPUs or envs as there"
                " aren't enough number of scenes"
            )

        random.shuffle(scenes)

    if len(scenes) == 1:
        scene_splits = [[scenes[0]] for _ in range(num_envs)]
    else:
        scene_splits = [[] for _ in range(num_envs)]
        for idx, scene in enumerate(scenes):
            scene_splits[idx % len(scene_splits)].append(scene)

        assert sum(map(len, scene_splits)) == len(scenes)

    for i in range(num_gpus):
        for j in range(num_envs_per_gpu):
            proc_config = config.clone()
            proc_config.defrost()
            proc_id = (i * num_envs_per_gpu) + j

            task_config = proc_config.TASK_CONFIG
            task_config.SEED += proc_id
            if len(scenes) > 0:
                task_config.DATASET.CONTENT_SCENES = scene_splits[proc_id]

            task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpus[i]

            task_config.SIMULATOR.AGENT_0.SENSORS = config.SENSORS

            proc_config.freeze()
            configs.append(proc_config)

    envs = habitat.VectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(zip(configs, env_classes)),
        auto_reset_done=auto_reset_done,
        workers_ignore_signals=workers_ignore_signals,
    )
    return envs


def construct_envs_auto_reset_false(
    config: Config, env_class: Type[Union[Env, RLEnv]]
) -> VectorEnv:
    return construct_envs(config, env_class, auto_reset_done=False)


@baseline_registry.register_env(name="BaseEnv")
class BaseEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        super().__init__(config.TASK_CONFIG, dataset)

    def get_reward_range(self) -> Tuple[float, float]:
        return (0.0, 0.0)

    def get_reward(self, observations: Observations) -> float:
        return 0.0

    def get_done(self, observations: Observations) -> bool:
        return self._env.episode_over

    def get_info(self, observations: Observations) -> Dict[Any, Any]:
        return self.habitat_env.get_metrics()


@baseline_registry.register_env(name="SPFEnv")
class SPFEnv(BaseEnv):

    GOAL_RADIUS: float = 0.15
    MAX_FOLLOWER_STEPS: int = 40

    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        super().__init__(config, dataset)
        self.follower = ShortestPathFollower(
            self._env.sim, self.GOAL_RADIUS, return_one_hot=False
        )

    def _get_path(self, goal_position) -> List[int]:
        """Use the habitat_sim path follower to generate a path plan.
        Return the sequence of HabitatSimActions without STOP.

        Raises:
            GreedyFollowerError when no path is found or path is empty
        """
        self.follower._build_follower()
        path = self.follower._follower.find_path(goal_position)

        if len(path) == 0:
            raise GreedyFollowerError("navigation path is empty")

        path = path[:-1]
        if len(path) > self.MAX_FOLLOWER_STEPS:
            path = path[: self.MAX_FOLLOWER_STEPS]
        return path

    def _get_rotation_plan(self, goal_rotation: np.quaternion) -> List[int]:
        cur_rotation = self._env.sim.get_agent_state().rotation
        goal_rotation = quaternion_from_coeff(goal_rotation)

        heading = heading_from_quaternion(cur_rotation)

        delta = angle_between_quaternions(cur_rotation, goal_rotation)
        num_steps = int(round(delta / np.deg2rad(15)))
        if num_steps == 0:
            return []

        heading_if_left = (heading + num_steps * np.deg2rad(15)) % (2 * np.pi)
        rotation_if_left = quaternion.from_euler_angles(
            [0.0, heading_if_left, 0.0]
        )
        left_error = angle_between_quaternions(rotation_if_left, goal_rotation)
        if left_error <= (np.deg2rad(15) / 2.0):
            turn_idx = HabitatSimActions.TURN_LEFT
        else:
            turn_idx = HabitatSimActions.TURN_RIGHT

        return [turn_idx for _ in range(num_steps)]

    def _run_follower(self, action: Dict[str, Any]) -> None:
        """Navigate to the location specified by the TELEPORT_AND_ROTATE
        action using a shortest path follower. Turn the agent to face the
        MP3D-Sim heading. Uses the VLN-CE action space. Does not query for
        intermediate observations for efficiency.
        """

        def step_follower(step_action):
            self._env.sim.get_agent(0).act(step_action)
            self._env._task.measurements.update_measures(
                episode=episode,
                action=step_action,
                task=self._env._task,
            )

        assert (
            self._env._episode_start_time is not None
        ), "Cannot call step before calling reset"
        assert (
            self._env.episode_over is False
        ), "Episode over, call reset before calling step"
        assert (
            action["action"] == "TELEPORT_AND_ROTATE"
        ), "Action for SPFEnv must be TELEPORT_AND_ROTATE"

        start_pos = self._env.sim.get_agent_state().position

        episode = self._env.current_episode
        (
            goal_position,
            goal_rotation,
        ) = TeleportAndRotateAction.filter_waypoint(
            self._env.sim, action["action_args"]["position"]
        )

        try:
            for step_action in self._get_path(goal_position):
                step_follower(step_action)
        except GreedyFollowerError:
            # planning failed. Instead act step-by-step.
            steps_taken = 0
            while steps_taken < self.MAX_FOLLOWER_STEPS:
                step_action = self.follower.get_next_action(goal_position)
                if step_action == HabitatSimActions.STOP:
                    break
                step_follower(step_action)
                steps_taken += 1
            else:
                agent_pos = self._env.sim.get_agent_state().position
                d = self._env.sim.geodesic_distance(goal_position, agent_pos)
                if d > self.GOAL_RADIUS:
                    # about 1% of navigations can't quite make it to the goal
                    #   when using MP3D waypoints and a 0.2m goal radius
                    logger.warn(
                        f"Max nav steps reached. Distance to goal: {d}"
                    )

        # turn the agent to face the discretized MP3D-Sim heading
        for step_action in self._get_rotation_plan(goal_rotation):
            step_follower(step_action)

        agent_pos = self._env.sim.get_agent_state().position
        cur_rotation = self._env.sim.get_agent_state().rotation
        goal_rotation = quaternion_from_coeff(goal_rotation)
        distance_error = self._env.sim.geodesic_distance(
            goal_position, agent_pos
        )
        heading_error = np.rad2deg(
            angle_between_quaternions(cur_rotation, goal_rotation)
        )

        self._env._task._is_episode_active = (
            self._env._task._check_episode_is_active(
                action=None, episode=episode
            )
        )
        return (
            distance_error,
            heading_error,
            start_pos,
            agent_pos,
            goal_position,
        )

    def _get_observations(self) -> Observations:
        observations = self._env.sim.get_observations_at()
        observations.update(
            self._env._task.sensor_suite.get_observations(
                observations=observations,
                episode=self._env.current_episode,
                task=self._env._task,
            )
        )
        return observations

    def step(
        self, action: Union[int, str, Dict[str, Any]], *args, **kwargs
    ) -> Tuple[Observations, Any, bool, dict]:
        if action == "STOP":
            return super().step(action)

        (
            distance_error,
            heading_error,
            start_pos,
            agent_pos,
            goal_position,
        ) = self._run_follower(action)

        # step limits apply to VLN agent, not follower
        self._env._update_step_stats()

        observations = self._get_observations()
        reward = self.get_reward(observations)
        done = self.get_done(observations)
        info = self.get_info(observations)

        info["nav_stats"] = {
            "distance_error": distance_error,
            "heading_error": heading_error,
            "start_pos": start_pos.tolist(),
            "agent_pos": agent_pos.tolist(),
            "goal_position": goal_position.tolist(),
        }
        return observations, reward, done, info


@baseline_registry.register_env(name="VLNCETeleportEnv")
class VLNCETeleportEnv(BaseEnv):
    def step(
        self, action: Union[int, str, Dict[str, Any]], *args, **kwargs
    ) -> Tuple[Observations, Any, bool, dict]:
        if action == "STOP":
            return super().step(action=action, **kwargs)

        start_pos = self._env.sim.get_agent_state().position
        (
            goal_position,
            goal_rotation,
        ) = TeleportAndRotateAction.filter_waypoint(
            self._env.sim, action["action_args"]["position"]
        )

        observations, reward, done, info = super().step(
            action=action, **kwargs
        )

        agent_pos = self._env.sim.get_agent_state().position
        cur_rotation = self._env.sim.get_agent_state().rotation
        distance_error = self._env.sim.geodesic_distance(
            goal_position, agent_pos
        )
        goal_rotation = quaternion_from_coeff(goal_rotation)
        heading_error = np.rad2deg(
            angle_between_quaternions(cur_rotation, goal_rotation)
        )

        info["nav_stats"] = {
            "distance_error": distance_error,
            "heading_error": heading_error,
            "start_pos": start_pos.tolist(),
            "agent_pos": agent_pos.tolist(),
            "goal_position": goal_position.tolist(),
        }
        return observations, reward, done, info


@baseline_registry.register_env(name="LPEnv")
class LPEnv(BaseEnv):

    MAX_FOLLOWER_STEPS: int = 40

    def _run_follower(self, action: Dict[str, Any]) -> None:

        assert (
            self._env._episode_start_time is not None
        ), "Cannot call step before calling reset"
        assert (
            self._env.episode_over is False
        ), "Episode over, call reset before calling step"
        assert (
            action["action"] == "TELEPORT_AND_ROTATE"
        ), "Action for SPFEnv must be TELEPORT_AND_ROTATE"

        episode = self._env.current_episode

        p = self._env.sim.get_agent_state().position
        start_position_xz = np.array([p[0], p[2]])
        goal_position_xz = None

        subgoal = action["action_args"]["position"]
        if subgoal.shape == (2,):
            r = subgoal[0]
            theta = subgoal[1]
            global_goal = TeleportAndRotateAction.filter_waypoint(
                self._env.sim, subgoal
            )[0]
        elif subgoal.shape == (3,):
            # convert global 3D position to r, theta.
            global_goal = subgoal
            goal_position_xz = np.array([subgoal[0], subgoal[2]])
            r = np.linalg.norm(start_position_xz - goal_position_xz)
            global_theta = compute_heading_to(
                start_position_xz, goal_position_xz
            )[1]
            cur_heading = heading_from_quaternion(
                self._env.sim.get_agent_state().rotation
            )
            theta = abs(cur_heading - global_theta)
            if not np.isclose(cur_heading + theta, global_theta):
                theta = -theta % (2 * np.pi)
        else:
            raise ValueError(f"Invalid subgoal shape: {subgoal.shape}")

        succeeded, _ = navigate(
            self._env, r, theta, max_follower_steps=self.MAX_FOLLOWER_STEPS
        )

        self._env._task._is_episode_active = (
            self._env._task._check_episode_is_active(
                action=None, episode=episode
            )
        )

        current_position = self._env.sim.get_agent_state().position
        nav_error = self._env.sim.geodesic_distance(
            global_goal, current_position
        )
        return nav_error, succeeded

    def _get_observations(
        self, include_task_observations: bool = True
    ) -> Observations:
        observations = self._env.sim.get_observations_at()
        if include_task_observations:
            observations.update(
                self._env._task.sensor_suite.get_observations(
                    observations=observations,
                    episode=self._env.current_episode,
                    task=self._env._task,
                )
            )
        return observations

    def step(
        self, action: Union[int, str, Dict[str, Any]], *args, **kwargs
    ) -> Tuple[Observations, Any, bool, dict]:
        if action == "STOP":
            return super().step(action)

        nav_error, succeeded = self._run_follower(action)

        self._env._update_step_stats()

        observations = self._get_observations()
        reward = self.get_reward(observations)
        done = self.get_done(observations)
        info = self.get_info(observations)

        info["nav_stats"] = {
            "distance_error": nav_error,
            "succeeded": succeeded,
        }

        return observations, reward, done, info


@baseline_registry.register_env(name="LPInferenceEnv")
class LPInferenceEnv(LPEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.infos = []

    def _run_follower(self, action: Dict[str, Any]) -> None:

        assert (
            self._env._episode_start_time is not None
        ), "Cannot call step before calling reset"
        assert (
            self._env.episode_over is False
        ), "Episode over, call reset before calling step"
        assert (
            action["action"] == "TELEPORT_AND_ROTATE"
        ), "Action for SPFEnv must be TELEPORT_AND_ROTATE"

        episode = self._env.current_episode

        p = self._env.sim.get_agent_state().position
        start_position_xz = np.array([p[0], p[2]])
        goal_position_xz = None

        subgoal = action["action_args"]["position"]
        if subgoal.shape == (2,):
            r = subgoal[0]
            theta = subgoal[1]
            global_goal = TeleportAndRotateAction.filter_waypoint(
                self._env.sim, subgoal
            )[0]
        elif subgoal.shape == (3,):
            # convert global 3D position to r, theta.
            global_goal = subgoal
            goal_position_xz = np.array([subgoal[0], subgoal[2]])
            r = np.linalg.norm(start_position_xz - goal_position_xz)
            global_theta = compute_heading_to(
                start_position_xz, goal_position_xz
            )[1]
            cur_heading = heading_from_quaternion(
                self._env.sim.get_agent_state().rotation
            )
            theta = abs(cur_heading - global_theta)
            if not np.isclose(cur_heading + theta, global_theta):
                theta = -theta % (2 * np.pi)
        else:
            raise ValueError(f"Invalid subgoal shape: {subgoal.shape}")

        succeeded, infos = navigate(
            self._env,
            r,
            theta,
            max_follower_steps=self.MAX_FOLLOWER_STEPS,
            info_callback=self.get_single_info,
        )
        self.infos.extend(infos)

        self._env._task._is_episode_active = (
            self._env._task._check_episode_is_active(
                action=None, episode=episode
            )
        )

        current_position = self._env.sim.get_agent_state().position
        nav_error = self._env.sim.geodesic_distance(
            global_goal, current_position
        )
        return nav_error, succeeded

    def get_info(self, observations: Observations):
        if len(self.infos) == 0:
            return [self.get_single_info()]
        else:
            return self.infos

    def get_single_info(self):
        agent_state = self._env.sim.get_agent_state()
        heading_vector = quaternion_rotate_vector(
            agent_state.rotation.inverse(), np.array([0, 0, -1])
        )
        heading = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return {
            "position": agent_state.position.tolist(),
            "heading": heading,
            "stop": self._env.task.is_stop_called,
        }

    def step(
        self, action: Union[int, str, Dict[str, Any]], *args, **kwargs
    ) -> Tuple[Observations, Any, bool, dict]:
        if action == "STOP":
            return super().step(action)

        self.infos = []
        self._run_follower(action)

        self._env._update_step_stats()

        observations = self._get_observations()
        reward = self.get_reward(observations)
        done = self.get_done(observations)
        info = self.get_info(observations)
        self.infos = []
        return observations, reward, done, info
