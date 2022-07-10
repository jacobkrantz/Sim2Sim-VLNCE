from typing import Any, List, Tuple, Union

import numpy as np
import quaternion
from gym import spaces
from habitat.core.registry import registry
from habitat.core.simulator import Observations, Simulator
from habitat.tasks.nav.nav import TeleportAction
from habitat.utils.geometry_utils import quaternion_to_list
from numpy import ndarray

from habitat_extensions.utils import (
    compute_heading_to,
    habitat_quat_from_mp3d_heading,
    mp3d_heading_from_habitat_quat,
    rtheta_to_global_coordinates,
    snap_heading,
)


@registry.register_task_action
class GoTowardPoint(TeleportAction):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """This waypoint action is parameterized by (r, theta) and simulates
        straight-line movement toward a waypoint, stopping upon collision or
        reaching the specified point.
        """
        super().__init__(*args, **kwargs)
        self._rotate_agent = self._config.rotate_agent

    def step(
        self,
        *args: Any,
        r: float,
        theta: float,
        **kwargs: Any,
    ) -> Observations:
        y_delta = kwargs["y_delta"] if "y_delta" in kwargs else 0.0
        pos = rtheta_to_global_coordinates(
            self._sim, r, theta, y_delta=y_delta, dimensionality=3
        )

        agent_pos = self._sim.get_agent_state().position
        new_pos = np.array(self._sim.step_filter(agent_pos, pos))
        new_rot = self._sim.get_agent_state().rotation
        if np.any(np.isnan(new_pos)) or not self._sim.is_navigable(new_pos):
            new_pos = agent_pos
            if self._rotate_agent:
                new_rot, _ = compute_heading_to(agent_pos, pos)
        else:
            new_pos = np.array(self._sim.pathfinder.snap_point(new_pos))
            if np.any(np.isnan(new_pos)) or not self._sim.is_navigable(
                new_pos
            ):
                new_pos = agent_pos
            if self._rotate_agent:
                new_rot, _ = compute_heading_to(agent_pos, pos)

        assert np.all(np.isfinite(new_pos))
        return self._sim.get_observations_at(
            position=new_pos, rotation=new_rot, keep_agent_at_new_pose=True
        )

    @property
    def action_space(self) -> spaces.Dict:
        coord_range = self.COORDINATE_MAX - self.COORDINATE_MIN
        return spaces.Dict(
            {
                "r": spaces.Box(
                    low=np.array([0.0]),
                    high=np.array([np.sqrt(2 * (coord_range ** 2))]),
                    dtype=np.float,
                ),
                "theta": spaces.Box(
                    low=np.array([0.0]),
                    high=np.array([2 * np.pi]),
                    dtype=np.float,
                ),
            }
        )


@registry.register_task_action
class TeleportAndRotateAction(TeleportAction):
    """Teleports the agent to the provided position and rotates the agent
    to face away from the previous location. The heading is snapped to one
    of 12 global discrete heading options. The position is in global
    coordinates.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._snap_heading = getattr(self._config, "SNAP_HEADING", True)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "TELEPORT_AND_ROTATE"

    def step(
        self,
        *args: Any,
        position: ndarray,
        **kwargs: Any,
    ) -> Observations:
        new_pos, new_rot = self.filter_waypoint(
            self._sim, position, self._snap_heading
        )
        return self._sim.get_observations_at(
            position=new_pos, rotation=new_rot, keep_agent_at_new_pose=True
        )

    @classmethod
    def filter_waypoint(
        cls,
        sim: Simulator,
        position: ndarray,
        snap_heading: bool = True,
        verify: bool = True,
        ray_trace: bool = False,
    ) -> Tuple[ndarray, List[float]]:
        """modifies the predicted waypoint position to be navigable. Computes
        the desired terminal heading by snapping to the nearest MP3D-Sim
        heading. If the position is 2D, (r, theta) is assumed. If 3D,
        (X, Y, Z) is assumed.
        """
        if not np.all(np.isfinite(position)):
            raise ValueError("All position elements must be finite.")

        if position.shape[0] == 2:
            new_pos = cls.convert_2d_to_3d(
                sim,
                position[0],
                position[1],
                ray_trace=ray_trace,
            )
        else:
            new_pos = cls.snap_point(sim, position)

        agent_pos = sim.get_agent_state().position

        if verify:
            cls.check_valid_point(sim, new_pos, raise_error=True)

        rotation = cls.compute_snapped_heading_to(
            agent_pos, new_pos, snap=snap_heading
        )
        return new_pos, rotation

    @classmethod
    def convert_2d_to_3d(
        cls, sim: Simulator, r: float, theta: float, ray_trace: bool = False
    ) -> Tuple[ndarray, ndarray]:
        """Expands relative polar coordinates (r, theta) to global 3D (x, y, z)
        coordinates. First, assume the agent's elevation. Snap to the nearest
        navigable location. If the location is invalid, reduce r and try again.
        Snapping through walls is allowed so long as a continugous mesh exists.
        """

        def _ray_trace_point(p):
            agent_pos = sim.get_agent_state().position
            trace_points = [
                cls.snap_point(sim, np.array([p[0], p[1] + delta, p[2]]))
                # +1.5m and -1.5m from agent's height in 0.1m increments
                for delta in list(np.linspace(-1.5, 1.5, 31))
            ]
            trace_points = [
                (p, sim.geodesic_distance(agent_pos, p)) for p in trace_points
            ]
            return min(trace_points, key=lambda p: p[1])[0]

        p = rtheta_to_global_coordinates(sim, r, theta, dimensionality=3)

        if ray_trace:
            p = _ray_trace_point(p)

        while not cls.check_valid_point(sim, cls.snap_point(sim, p)):
            r -= 0.1
            p = rtheta_to_global_coordinates(sim, r, theta, dimensionality=3)
            if ray_trace:
                p = _ray_trace_point(p)
            if r < 0.0:
                cls.check_valid_point(
                    sim, cls.snap_point(sim, p), raise_error=True
                )

        return cls.snap_point(sim, p)

    @staticmethod
    def calc_displacements(p1: ndarray, p2: ndarray) -> Tuple[float, float]:
        xz_disp = np.linalg.norm(
            np.array([p1[0], p1[2]]) - np.array([p2[0], p2[2]])
        )
        y_disp = abs(p1[1] - p2[1])
        return xz_disp, y_disp

    @staticmethod
    def snap_point(sim: Simulator, position: ndarray) -> ndarray:
        return np.array(sim.pathfinder.snap_point(position))

    @staticmethod
    def check_valid_point(
        sim, position: ndarray, raise_error: bool = False
    ) -> Tuple[bool, str]:
        is_valid = True
        msg = ""
        agent_pos = sim.get_agent_state().position

        if not np.all(np.isfinite(position)):
            is_valid = False
            msg = "Goal Location is not finite."
        if not sim.is_navigable(position):
            is_valid = False
            msg = "Goal Location is not navigable."
        if not np.isfinite(sim.geodesic_distance(agent_pos, position)):
            is_valid = False
            msg = "Goal Location cannot be reached from current position."

        if not is_valid and raise_error:
            raise RuntimeError(msg)
        return is_valid

    @staticmethod
    def compute_snapped_heading_to(
        pos_from: Union[List[float], ndarray],
        pos_to: Union[List[float], ndarray],
        snap: bool = True,
    ) -> List[float]:
        delta_x = pos_to[0] - pos_from[0]
        delta_z = pos_to[-1] - pos_from[-1]
        xz_angle = np.arctan2(delta_x, delta_z)
        xz_angle = (xz_angle + np.pi) % (2 * np.pi)
        quat = quaternion.from_euler_angles([0.0, xz_angle, 0.0])

        if snap:
            mp3d_heading = mp3d_heading_from_habitat_quat(quat)
            snapped = snap_heading(mp3d_heading)
            quat = habitat_quat_from_mp3d_heading(snapped)

        return quaternion_to_list(quat)

    @property
    def action_space(self) -> spaces.Dict:
        return spaces.Dict(
            {
                "position": spaces.Box(
                    low=self.COORDINATE_MIN,
                    high=self.COORDINATE_MAX,
                    shape=(3,),
                    dtype=np.float,
                ),
            }
        )
