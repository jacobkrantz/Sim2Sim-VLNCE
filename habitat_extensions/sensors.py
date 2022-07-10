import functools
import math
import pickle
from copy import deepcopy
from typing import Any, Callable, Dict, List

import MatterSim
import numpy as np
from gym import Space, spaces
from habitat.config import Config
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes, Simulator
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.tasks.vln.vln import VLNEpisode
from habitat.utils.geometry_utils import (
    angle_between_quaternions,
    quaternion_from_coeff,
)
from networkx.classes.graph import Graph
from numpy import ndarray

from habitat_extensions.shortest_path_follower import (
    ShortestPathFollowerCompat,
)
from habitat_extensions.task import VLNExtendedEpisode
from habitat_extensions.utils import (
    compute_heading_to,
    heading_from_quaternion,
    mp3d_heading_from_habitat_quat,
    snap_heading,
    vln_style_angle_features,
)
from transformers.pytorch_transformers import BertTokenizer


@registry.register_sensor(name="GlobalGPSSensor")
class GlobalGPSSensor(Sensor):
    """Current agent location in global coordinate frame"""

    cls_uuid: str = "globalgps"

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim
        self._dimensionality = config.DIMENSIONALITY
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.POSITION

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float).min,
            high=np.finfo(np.float).max,
            shape=(self._dimensionality,),
            dtype=np.float,
        )

    def get_observation(self, *args: Any, **kwargs: Any):
        agent_position = self._sim.get_agent_state().position
        if self._dimensionality == 2:
            agent_position = np.array([agent_position[0], agent_position[2]])
        return agent_position.astype(np.float32)


@registry.register_sensor
class VLNOracleProgressSensor(Sensor):
    """Relative progress towards goal"""

    cls_uuid: str = "progress"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ) -> None:
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.MEASUREMENT

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        return spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float)

    def get_observation(self, *args: Any, episode, **kwargs: Any) -> float:
        distance_to_target = self._sim.geodesic_distance(
            self._sim.get_agent_state().position.tolist(),
            episode.goals[0].position,
        )

        # just in case the agent ends up somewhere it shouldn't
        if not np.isfinite(distance_to_target):
            return np.array([0.0])

        distance_from_start = episode.info["geodesic_distance"]
        return np.array(
            [(distance_from_start - distance_to_target) / distance_from_start]
        )


@registry.register_sensor
class AngleFeaturesSensor(Sensor):
    """Returns a fixed array of features describing relative camera poses based
    on https://arxiv.org/abs/1806.02724. This encodes heading angles but
    assumes a single elevation angle.
    """

    cls_uuid: str = "angle_features"

    def __init__(self, *args: Any, config: Config, **kwargs: Any) -> None:
        self.cameras = config.CAMERA_NUM
        super().__init__(config)
        orient = [np.pi * 2 / self.cameras * i for i in range(self.cameras)]
        self.angle_features = np.stack(
            [np.array([np.sin(o), np.cos(o), 0.0, 1.0]) for o in orient]
        )

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.HEADING

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.cameras, 4),
            dtype=np.float,
        )

    def get_observation(self, *args: Any, **kwargs: Any) -> ndarray:
        return deepcopy(self.angle_features)


@registry.register_sensor
class ShortestPathSensor(Sensor):
    """Provides the next action to follow the shortest path to the goal."""

    cls_uuid: str = "shortest_path_sensor"

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        super().__init__(config=config)
        cls = ShortestPathFollower
        if config.USE_ORIGINAL_FOLLOWER:
            cls = ShortestPathFollowerCompat
        self.follower = cls(sim, config.GOAL_RADIUS, return_one_hot=False)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TACTILE

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=0.0, high=100, shape=(1,), dtype=np.float)

    def get_observation(self, *args: Any, episode, **kwargs: Any):
        best_action = self.follower.get_next_action(episode.goals[0].position)
        if best_action is None:
            best_action = HabitatSimActions.STOP
        return np.array([best_action])


@registry.register_sensor
class RxRInstructionSensor(Sensor):
    """Loads pre-computed intruction features from disk in the baseline RxR
    BERT file format.
    https://github.com/google-research-datasets/RxR/tree/7a6b87ba07959f5176aa336192a8c5dc85ca1b8e#downloading-bert-text-features
    """

    cls_uuid: str = "rxr_instruction"

    def __init__(self, *args: Any, config: Config, **kwargs: Any):
        self.features_path = config.features_path
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.MEASUREMENT

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float).min,
            high=np.finfo(np.float).max,
            shape=(512, 768),
            dtype=np.float,
        )

    def get_observation(
        self, *args: Any, episode: VLNExtendedEpisode, **kwargs
    ):
        features = np.load(
            self.features_path.format(
                split=episode.instruction.split,
                id=int(episode.instruction.instruction_id),
                lang=episode.instruction.language.split("-")[0],
            )
        )
        feats = np.zeros((512, 768), dtype=np.float32)
        s = features["features"].shape
        feats[: s[0], : s[1]] = features["features"]
        return feats


@registry.register_sensor
class VLNInstructionSensor(Sensor):
    """loads and encodes instructions the VLN way."""

    cls_uuid: str = "vln_instruction"

    def __init__(self, *args: Any, config: Config, **kwargs: Any):
        self.tokenizer = BertTokenizer.from_pretrained(config.MODEL_OR_PATH)
        self.padded_length = config.PADDED_LENGTH
        self.tokenize = self.compose(
            self.tokenizer.tokenize,
            self.pad_instr_tokens,
            self.tokenizer.convert_tokens_to_ids,
            lambda x: np.array(x, dtype=np.int),
        )
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.MEASUREMENT

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=np.iinfo(np.int).max,
            shape=(self.padded_length,),
            dtype=np.int,
        )

    @staticmethod
    def compose(*functions: Callable):
        """Function composition. Will evaluate in args order."""
        return functools.reduce(
            lambda f, g: lambda x: f(g(x)), functions[::-1], lambda x: x
        )

    def pad_instr_tokens(self, tokens: List[str]) -> List[str]:
        if len(tokens) > self.padded_length - 2:  # -2 for [CLS] and [SEP]
            tokens = tokens[: (self.padded_length - 2)]
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        return tokens + ["[PAD]"] * (self.padded_length - len(tokens))

    def get_observation(self, *args: Any, episode: VLNEpisode, **kwargs):
        return self.tokenize(episode.instruction.instruction_text)


@registry.register_sensor
class VLNCandidateSensor(Sensor):
    """Computes a subset of the 36 view frames that have candidate waypoints
    within them. Returns an ndarray of [I, idx, x, y, z, h, e] for:
        I:          1 if a candidate is present at this line
        idx:        relative viewpoint index to map candidate to
        (x,y,z):    habitat global waypoint coordinates
        h:          candidate heading
        e:          candidate elevation
    """

    cls_uuid: str = "vln_candidates"

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim
        self._max_candidates = config.MAX_CANDIDATES
        with open(config.GRAPHS_FILE, "rb") as f:
            self._conn_graphs: Dict[str, Graph] = pickle.load(f)

        self._mp3d_sim = self._init_mp3d_sim()
        super().__init__(config=config)

    def _init_mp3d_sim(self):
        mp3d_sim = MatterSim.Simulator()
        mp3d_sim.setRenderingEnabled(False)
        mp3d_sim.setDepthEnabled(False)
        mp3d_sim.setPreloadingEnabled(False)
        mp3d_sim.setBatchSize(1)
        mp3d_sim.setCacheSize(1)
        mp3d_sim.setDiscretizedViewingAngles(True)
        mp3d_sim.setCameraResolution(640, 480)
        mp3d_sim.setCameraVFOV(math.radians(60))
        mp3d_sim.initialize()
        return mp3d_sim

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.MEASUREMENT

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float).min,
            high=np.finfo(np.float).max,
            shape=(self._max_candidates, 7),
            dtype=np.float,
        )

    @staticmethod
    def _extract_scene_from_path(scene: str) -> str:
        return scene.split("/")[-1].split(".")[0]

    def _nearest_mp3d_node(self, position: List[float], graph: Graph) -> str:
        nearest = ""
        nearest_dist = float("inf")
        for node in graph:
            distance = self._sim.geodesic_distance(
                position, graph.nodes[node]["position"].tolist()
            )
            if distance < nearest_dist:
                nearest = node
                nearest_dist = distance

        assert nearest != "", "no navigable node found."
        return nearest

    @staticmethod
    def _global_to_rel_ix(ix: int, heading: float) -> int:
        """the viewframe index is global out of MP3D-Sim. The observations we
        want to match to are relative to the agents current heading. This
        function maps the global idx to the observation idx.

        Args:
            ix: global MP3DS-Sim viewframe index
            heading: the agents MP3D-Sim heading snapped to 1 of 12 headings.
        """
        ix_angle = (ix % 12) / 12 * np.pi * 2
        relative_ix_angle = (ix_angle - heading) % (2 * np.pi)

        # map to 0-11
        headingIncrement = np.pi * 2 / 12
        new_ix = int(np.around(relative_ix_angle / headingIncrement)) % 12

        # send to the same elevation
        new_ix += (ix // 12) * 12
        return new_ix

    def _prune_candidates(
        self, candidates: List[Dict], viewpointId: str, graph: Graph
    ) -> List[Dict]:
        new_candidates = []
        for c in candidates:
            # prune illegal viewpoints -- not all exist in Habitat
            if c["viewpointId"] not in graph:
                continue

            # prune viewpoints that cannot be navigated in Habitat
            distance = self._sim.geodesic_distance(
                graph.nodes[viewpointId]["position"].tolist(),
                graph.nodes[c["viewpointId"]]["position"].tolist(),
            )
            if np.isfinite(distance):
                new_candidates.append(c)

        return new_candidates

    def make_candidate(
        self, scanId: str, viewpointId: str, heading: float, graph: Graph
    ) -> List[Dict]:
        def _loc_distance(loc):
            return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)

        base_heading = snap_heading(heading, n=12)
        adj_dict = {}
        for ix in range(36):
            if ix == 0:
                self._mp3d_sim.newEpisode(
                    [scanId],
                    [viewpointId],
                    [0],
                    [math.radians(-30)],
                )
            elif ix % 12 == 0:
                # params: index, heading, elevation
                self._mp3d_sim.makeAction([0], [1.0], [1.0])
            else:
                self._mp3d_sim.makeAction([0], [1.0], [0])

            state = self._mp3d_sim.getState()[0]
            assert state.viewIndex == ix

            # Heading and elevation for the viewpoint center
            heading = state.heading - base_heading
            elevation = state.elevation

            # get adjacent locations
            for loc in state.navigableLocations[1:]:
                # if a loc is visible from multiple view, use the closest
                # view (in angular distance) as its representation
                distance = _loc_distance(loc)

                # Heading and elevation for the loc
                loc_heading = heading + loc.rel_heading
                loc_elevation = elevation + loc.rel_elevation
                if (
                    loc.viewpointId not in adj_dict
                    or distance < adj_dict[loc.viewpointId]["distance"]
                ):
                    adj_dict[loc.viewpointId] = {
                        "heading": loc_heading,
                        "elevation": loc_elevation,
                        "viewpointId": loc.viewpointId,  # Next viewpoint id
                        "pointId": self._global_to_rel_ix(ix, base_heading),
                        "distance": distance,
                    }

        candidates = list(adj_dict.values())
        candidates = self._prune_candidates(candidates, viewpointId, graph)

        # insert Habitat coordinates for each candidate
        for c in candidates:
            c["coords"] = graph.nodes[c["viewpointId"]]["position"].tolist()

        return candidates

    def create_candidates_ndarray(
        self, candidate_list: List[Dict[str, Any]]
    ) -> ndarray:
        candidates = np.zeros((self._max_candidates, 7))

        # limit number of candidates
        if len(candidate_list) > self._max_candidates:
            candidate_list = candidate_list[: self._max_candidates]

        for i, c in enumerate(candidate_list):
            candidates[i][0] = 1
            candidates[i][1] = c["pointId"]
            candidates[i][2:5] = np.array(c["coords"])
            candidates[i][5] = np.array(c["heading"])
            candidates[i][6] = np.array(c["elevation"])

        return candidates

    def get_observation(self, *args: Any, episode: VLNEpisode, **kwargs):
        scene_id = self._extract_scene_from_path(episode.scene_id)
        graph = self._conn_graphs[scene_id]
        state = self._sim.get_agent_state()
        agent_pos = state.position.tolist()

        current_node = self._nearest_mp3d_node(agent_pos, graph)
        mp3d_heading = mp3d_heading_from_habitat_quat(state.rotation)
        candidate_list = self.make_candidate(
            scene_id, current_node, mp3d_heading, graph
        )

        return self.create_candidates_ndarray(candidate_list)


@registry.register_sensor
class VLNCandidateRelativeSensor(VLNCandidateSensor):
    """Takes the output of VLNCandidateSensor and generates aget-relative polar coordinates"""

    cls_uuid: str = "vln_candidates_relative"

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float).min,
            high=np.finfo(np.float).max,
            shape=(self._max_candidates, 2),
            dtype=np.float,
        )

    def get_observation(self, *args: Any, **kwargs):
        obs = super().get_observation(*args, **kwargs)

        state = self._sim.get_agent_state()
        agent_pos = state.position
        agent_rot = state.rotation

        new_coords = np.zeros((obs.shape[0], 2), dtype=obs.dtype)
        for i in range(obs.shape[0]):
            coords = obs[i, 2:5]
            if np.all(coords == 0.0):
                break

            # each candidate: relative (r, theta)
            new_coords[i, 0] = np.linalg.norm(
                np.array([agent_pos[0], agent_pos[2]])
                - np.array([coords[0], coords[2]])
            )
            rel_quat, rel_heading = compute_heading_to(agent_pos, coords)
            agent_heading = heading_from_quaternion(agent_rot)

            angle = angle_between_quaternions(
                agent_rot, quaternion_from_coeff(rel_quat)
            )
            if np.isclose((agent_heading + angle) % (2 * np.pi), rel_heading):
                new_coords[i, 1] = angle
            else:
                new_coords[i, 1] = -angle % (2 * np.pi)

        return new_coords


@registry.register_sensor
class MP3DActionAngleFeature(Sensor):

    cls_uuid: str = "mp3d_action_angle_feature"

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim
        self.snap = config.SNAP_HEADINGS
        self.feature_size = config.FEATURE_SIZE
        assert self.feature_size % 4 == 0
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.MEASUREMENT

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float).min,
            high=np.finfo(np.float).max,
            shape=(1,),
            dtype=np.float,
        )

    def get_observation(self, *args: Any, **kwargs):
        rotation = self._sim.get_agent_state().rotation
        mp3d_heading = mp3d_heading_from_habitat_quat(rotation)
        if self.snap:
            mp3d_heading = snap_heading(mp3d_heading, n=12)
        return vln_style_angle_features(
            mp3d_heading, 0.0, self.feature_size, as_tensor=False
        )
