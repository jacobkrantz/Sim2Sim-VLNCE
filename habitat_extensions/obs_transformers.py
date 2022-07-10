import copy
import numbers
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple, Union

import caffe
import numpy as np
import torch
from gym import Space, spaces
from habitat.config import Config
from habitat.core.logging import logger
from habitat.core.simulator import Observations
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import ObservationTransformer
from habitat_baselines.utils.common import (
    center_crop,
    get_image_height_width,
    overwrite_gym_box_shape,
)
from torch import Tensor

from habitat_extensions.utils import vln_style_angle_features


@baseline_registry.register_obs_transformer()
class CenterCropperPerSensor(ObservationTransformer):
    """Center crop the input on a per-sensor basis"""

    sensor_crops: Dict[str, Union[int, Tuple[int, int]]]
    channels_last: bool

    def __init__(
        self,
        sensor_crops: List[Tuple[str, Union[int, Tuple[int, int]]]],
        channels_last: bool = True,
    ) -> None:
        super().__init__()

        self.sensor_crops = dict(sensor_crops)
        for k in self.sensor_crops:
            size = self.sensor_crops[k]
            if isinstance(size, numbers.Number):
                self.sensor_crops[k] = (int(size), int(size))
            assert len(size) == 2, "forced input size must be len of 2 (h, w)"

        self.channels_last = channels_last

    def transform_observation_space(
        self,
        observation_space: Space,
    ) -> Space:
        observation_space = copy.deepcopy(observation_space)
        for key in observation_space.spaces:
            if (
                key in self.sensor_crops
                and observation_space.spaces[key].shape[-3:-1]
                != self.sensor_crops[key]
            ):
                h, w = get_image_height_width(
                    observation_space.spaces[key], channels_last=True
                )
                logger.info(
                    "Center cropping observation size of %s from %s to %s"
                    % (key, (h, w), self.sensor_crops[key])
                )

                observation_space.spaces[key] = overwrite_gym_box_shape(
                    observation_space.spaces[key], self.sensor_crops[key]
                )
        return observation_space

    @torch.no_grad()
    def forward(self, observations: Dict[str, Tensor]) -> Dict[str, Tensor]:
        observations.update(
            {
                sensor: center_crop(
                    observations[sensor],
                    self.sensor_crops[sensor],
                    channels_last=self.channels_last,
                )
                for sensor in self.sensor_crops
                if sensor in observations
            }
        )
        return observations

    @classmethod
    def from_config(cls, config: Config):
        cc_config = config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR
        return cls(cc_config.SENSOR_CROPS)


@baseline_registry.register_obs_transformer()
class ObsStack(ObservationTransformer):
    """Stack multiple sensors into a single sensor observation."""

    def __init__(
        self, sensor_rewrites: List[Tuple[str, Sequence[str]]]
    ) -> None:
        """Args:
        sensor_rewrites: a tuple of rewrites where a rewrite is a list of
        sensor names to be combined into one sensor.
        """
        self.rewrite_dict: Dict[str, Sequence[str]] = dict(sensor_rewrites)
        super(ObsStack, self).__init__()

    def transform_observation_space(
        self,
        observation_space: Space,
    ) -> Space:
        observation_space = copy.deepcopy(observation_space)
        for target_uuid, sensors in self.rewrite_dict.items():
            orig_space = observation_space.spaces[sensors[0]]
            for k in sensors:
                del observation_space.spaces[k]

            low = (
                orig_space.low
                if np.isscalar(orig_space.low)
                else np.min(orig_space.low)
            )
            high = (
                orig_space.high
                if np.isscalar(orig_space.high)
                else np.max(orig_space.high)
            )
            shape = (len(sensors),) + (orig_space.shape)

            observation_space.spaces[target_uuid] = spaces.Box(
                low=low, high=high, shape=shape, dtype=orig_space.dtype
            )

        return observation_space

    @torch.no_grad()
    def forward(self, observations: Observations) -> Observations:
        for new_obs_keys, old_obs_keys in self.rewrite_dict.items():
            new_obs = torch.stack(
                [observations[k] for k in old_obs_keys], axis=1
            )
            for k in old_obs_keys:
                del observations[k]

            observations[new_obs_keys] = new_obs
        return observations

    @classmethod
    def from_config(cls, config: Config):
        return cls(config.RL.POLICY.OBS_TRANSFORMS.OBS_STACK.SENSOR_REWRITES)


@baseline_registry.register_obs_transformer()
class ResNetCandidateEncoder(ObservationTransformer):
    """Encodes all RGB frames into a 2048-dimensional vector using a
    pretrained ResNet. Assumes the "rgb" observation is a stack of RGB images.
    The model is in Caffe to match exactly the encoder used in R2R VLN.
    """

    OBS_UUID: str = "candidate_rgb_features"

    def __init__(
        self,
        prototxt_f: str,
        weights_f: str,
        rgb_shape: Tuple[int],
        remove_rgb: bool,
        max_batch_size: int,
        gpu_id: int,
    ) -> None:
        """Args:
        prototxt_f (str): file path of the Caffe network definition file
        weights_f (str): file path of the weights for the above network
        """
        super(ResNetCandidateEncoder, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.device("cuda", gpu_id)
            caffe.set_device(gpu_id)
            caffe.set_mode_gpu()
        else:
            self.device = torch.device("cpu")
            caffe.set_mode_cpu()

        self.remove_rgb = remove_rgb
        self.max_batch_size = max_batch_size
        self.net = caffe.Net(prototxt_f, weights_f, caffe.TEST)

        self.height, self.width = rgb_shape
        self.bgr_mean = torch.tensor([103.1, 115.9, 123.2]).to(
            device=self.device
        )

    def transform_observation_space(
        self,
        observation_space: Space,
    ) -> Space:
        spaces_dict: Dict[str, Space] = observation_space.spaces
        max_candidates = spaces_dict["vln_candidates"].shape[0]
        if self.remove_rgb:
            del spaces_dict["rgb"]

        spaces_dict[self.OBS_UUID] = spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(max_candidates, 2048),
            dtype=np.float32,
        )
        return spaces.Dict(spaces_dict)

    def _transform_obs_batch(self, rgb):
        """Prepares an RGB observation for input to the Caffe ResNet."""
        x = rgb.to(dtype=torch.float32)
        x = x[:, :, :, [2, 1, 0]]  # RGB to BGR color channels
        x -= self.bgr_mean
        x = x.permute(0, 3, 1, 2)
        return x.cpu().numpy()  # Caffe loads from CPU unfortunately

    def encode(self, frames: Tensor) -> Tensor:
        """Runs a forward pass on the Caffe ResNet and extracts 2048-dim
        features. Uses a maximum batch size to avoid OOM.

        Args:
            frames (Tensor): [num_frames, width, height, channels]

        Returns:
            frame_features (Tensor): [num_frames, 2048]
        """
        frames_np = self._transform_obs_batch(frames)
        num_frames = frames_np.shape[0]
        num_passes = int(np.ceil(num_frames / self.max_batch_size))

        encoded = np.empty((num_frames, 2048), dtype=np.float32)

        high_idx = 0
        for i in range(num_passes):
            low_idx = high_idx
            high_idx = min(num_frames, (i + 1) * self.max_batch_size)

            input_frames = frames_np[low_idx:high_idx]
            self.net.blobs["data"].reshape(*input_frames.shape)
            self.net.blobs["data"].data[:, :, :, :] = input_frames
            self.net.forward()
            encoded[low_idx:high_idx] = self.net.blobs["pool5"].data[
                :, :, 0, 0
            ]

        return torch.from_numpy(encoded).to(self.device)

    def make_frame_mask(
        self, vln_candidates: Tensor
    ) -> Tuple[Tensor, List[Dict[int, int]]]:
        batch_size = vln_candidates.shape[0]
        max_candidates = vln_candidates.shape[1]

        mask = torch.zeros(
            batch_size,
            max_candidates,
            dtype=torch.bool,
            device=vln_candidates.device,
        )

        # map frame features to candidate indices. can be multiple
        candidate_indices = []

        for batch_idx in range(batch_size):
            candidate_indices.append(defaultdict(list))

            for i in range(max_candidates):
                has_candidate = (
                    vln_candidates[batch_idx, i, 0].to(dtype=torch.bool).item()
                )
                if not has_candidate:
                    break
                frame_idx = int(vln_candidates[batch_idx, i, 1].item())
                mask[batch_idx, frame_idx] = True
                candidate_indices[batch_idx][frame_idx].append(i)

        mask = mask.reshape(batch_size * max_candidates)
        return mask, candidate_indices

    @torch.no_grad()
    def forward(self, observations: Observations) -> Observations:
        batch_size, num_frames, w, h, channels = observations["rgb"].shape
        max_candidates = observations["vln_candidates"].shape[1]

        frame_mask, candidate_indices = self.make_frame_mask(
            observations["vln_candidates"]
        )
        frames = observations["rgb"].reshape(
            batch_size * num_frames, w, h, channels
        )

        frame_features = torch.zeros(
            batch_size * num_frames, 2048, device=frames.device
        )

        # only encode frames that have a candidate
        frame_features[frame_mask] = self.encode(frames[frame_mask])
        frame_features = frame_features.reshape(
            batch_size, max_candidates, 2048
        )
        if torch.isnan(frame_features).any().item():
            print("NaN encountered")

        # convert from frame features to candidate features
        # (frames can have multiple candidates)
        candidate_features = torch.zeros(
            batch_size, max_candidates, 2048, device=frames.device
        )
        for batch_idx in range(len(candidate_indices)):
            for frame_idx, candidate_idx in candidate_indices[
                batch_idx
            ].items():
                candidate_features[batch_idx, candidate_idx] = frame_features[
                    batch_idx, frame_idx
                ]

        observations[self.OBS_UUID] = candidate_features

        if self.remove_rgb:
            del observations["rgb"]
        return observations

    @classmethod
    def from_config(cls, config: Config):
        cfg = config.RL.POLICY.OBS_TRANSFORMS.RESNET_CANDIDATE_ENCODER
        rgb_shape = (
            config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT,
            config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH,
        )
        gpu_id = config.TORCH_GPU_ID
        if cfg.gpu_id >= 0:
            gpu_id = cfg.gpu_id
        return cls(
            cfg.protoxt_file,
            cfg.weights_file,
            rgb_shape,
            cfg.remove_rgb,
            cfg.max_batch_size,
            gpu_id,
        )


@baseline_registry.register_obs_transformer()
class CandidateFeatures(ObservationTransformer):
    """Creates observations for VLN candidates"""

    FEAT_UUID: str = "candidate_features"
    COORD_UUID: str = "candidate_coordinates"
    NUM_UUID: str = "num_candidates"

    def __init__(
        self,
        remove_rgb_feats: bool,
        remove_vln_candidates: bool,
        angle_feature_size: int,
    ) -> None:
        self.remove_rgb_feats = remove_rgb_feats
        self.remove_vln_candidates = remove_vln_candidates
        self.angle_feature_size = angle_feature_size
        super(CandidateFeatures, self).__init__()

    def transform_observation_space(
        self,
        observation_space: Space,
    ) -> Space:
        spaces_dict: Dict[str, Space] = observation_space.spaces
        max_candidates = spaces_dict["vln_candidates"].shape[0]
        rgb_feature_size = spaces_dict["candidate_rgb_features"].shape[1]

        spaces_dict[self.FEAT_UUID] = spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(max_candidates, rgb_feature_size + self.angle_feature_size),
            dtype=np.float32,
        )
        spaces_dict[self.COORD_UUID] = spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(max_candidates, 3),
            dtype=np.float32,
        )
        spaces_dict[self.NUM_UUID] = spaces.Box(
            low=0,
            high=np.iinfo(np.int).max,
            shape=(max_candidates,),
            dtype=np.float32,
        )
        if self.remove_rgb_feats:
            del spaces_dict["candidate_rgb_features"]
        if self.remove_vln_candidates:
            del spaces_dict["vln_candidates"]
        return spaces.Dict(spaces_dict)

    @torch.no_grad()
    def forward(self, observations: Observations) -> Observations:
        candidates = observations["vln_candidates"]
        device = candidates.device
        (batch_size, max_candidates, _) = candidates.shape

        angle_features = vln_style_angle_features(
            heading=candidates[:, :, 5],
            elevation=candidates[:, :, 6],
            feature_size=self.angle_feature_size,
        )

        feature_size = (
            observations["candidate_rgb_features"].shape[2]
            + self.angle_feature_size
        )
        padded_features = torch.zeros(
            batch_size, max_candidates + 1, feature_size, device=device
        )
        padded_features[:, :-1, :2048] = observations["candidate_rgb_features"]

        padded_coordinates = torch.zeros(
            batch_size, max_candidates + 1, 3, device=device
        )

        num_candidates = [1 for _ in range(batch_size)]
        for batch_idx in range(batch_size):
            for i in range(max_candidates):
                has_candidate = (
                    candidates[batch_idx, i, 0].to(dtype=torch.bool).item()
                )
                if not has_candidate:
                    break

                angle_feature = angle_features[batch_idx, i]
                padded_features[batch_idx, i, 2048:] = angle_feature
                padded_coordinates[batch_idx, i] = candidates[
                    batch_idx, i, 2:5
                ]
                num_candidates[batch_idx] = max(
                    num_candidates[batch_idx], i + 2
                )

        observations[self.FEAT_UUID] = padded_features
        observations[self.COORD_UUID] = padded_coordinates
        observations[self.NUM_UUID] = torch.tensor(
            num_candidates, dtype=torch.int, device=device
        )
        if self.remove_rgb_feats:
            del observations["candidate_rgb_features"]
        if self.remove_vln_candidates:
            del observations["vln_candidates"]
        return observations

    @classmethod
    def from_config(cls, config: Config):
        cfg = config.RL.POLICY.OBS_TRANSFORMS.CANDIDATE_FEATURES
        return cls(
            cfg.remove_rgb_feats,
            cfg.remove_vln_candidates,
            cfg.angle_feature_size,
        )


@baseline_registry.register_obs_transformer()
class CurrentAngleFeature(ObservationTransformer):
    """Extracts the agent's current heading from the 36-frame angle features.
    Optionally removes those features.
    """

    OBS_UUID: str = "current_angle_feature"

    def __init__(self, remove_angle_feats: bool) -> None:
        self.remove_angle_feats = remove_angle_feats
        super(CurrentAngleFeature, self).__init__()

    def transform_observation_space(
        self,
        observation_space: Space,
    ) -> Space:
        spaces_dict: Dict[str, Space] = observation_space.spaces
        angle_feature_size = spaces_dict["vln_angle_features"].shape[1]

        spaces_dict[self.OBS_UUID] = spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(angle_feature_size,),
            dtype=np.float32,
        )
        if self.remove_angle_feats:
            del spaces_dict["vln_angle_features"]
        return spaces.Dict(spaces_dict)

    @torch.no_grad()
    def forward(self, observations: Observations) -> Observations:

        # extract heading zero, elevation zero
        # NOTE: the VLN code persists heading elevations. This could cause issues.
        observations[self.OBS_UUID] = observations["vln_angle_features"][:, 12]

        if self.remove_angle_feats:
            del observations["vln_angle_features"]
        return observations

    @classmethod
    def from_config(cls, config: Config):
        cfg = config.RL.POLICY.OBS_TRANSFORMS.CURRENT_ANGLE_FEATURE
        return cls(cfg.remove_angle_feats)


@baseline_registry.register_obs_transformer()
class PreprocessVLNInstruction(ObservationTransformer):
    """Preprocesses the instruction tokens that come from VLNInstructionSensor.
    This adds obesrvation keys:
        `vln_instruction_mask`
        `vln_instruction_token_type_ids`
    """

    def __init__(self, pad_idx: bool) -> None:
        self.pad_idx = pad_idx
        super(PreprocessVLNInstruction, self).__init__()

    def transform_observation_space(
        self,
        observation_space: Space,
    ) -> Space:
        spaces_dict: Dict[str, Space] = observation_space.spaces
        assert "vln_instruction" in spaces_dict

        spaces_dict["vln_instruction_mask"] = spaces.Box(
            low=0,
            high=1,
            shape=(spaces_dict["vln_instruction"].shape[0],),
            dtype=np.uint8,
        )
        spaces_dict["vln_instruction_token_type_ids"] = spaces.Box(
            low=0,
            high=0,
            shape=(spaces_dict["vln_instruction"].shape[0],),
            dtype=np.uint8,
        )
        return spaces.Dict(spaces_dict)

    @torch.no_grad()
    def forward(self, observations: Observations) -> Observations:
        mask = observations["vln_instruction"] != self.pad_idx
        observations["vln_instruction_mask"] = mask
        observations["vln_instruction_token_type_ids"] = torch.zeros_like(mask)
        return observations

    @classmethod
    def from_config(cls, config: Config):
        cfg = config.RL.POLICY.OBS_TRANSFORMS.PREPROCESS_VLN_INSTRUCTION
        return cls(cfg.pad_idx)


@baseline_registry.register_obs_transformer()
class Laser2D(ObservationTransformer):
    """Converts laser_2d observation chunks into a single 2D laser scan.
    Element zero is facing forward and rotation proceeds clockwise.
    """

    cls_uuid: str = "laser_2d"
    frames: int = 4

    def __init__(self, pool_mode: str, frame_width: int) -> None:
        """Args:
        pool_mode: how to reduce the height channel to a single scalar
        frame_width: pixel width of each laser scan chunk
        """
        self.pool_mode: str = pool_mode
        self.frame_width: int = frame_width

        if self.pool_mode not in ["exact", "min", "max", "mean"]:
            raise AssertionError
        super(Laser2D, self).__init__()

    def transform_observation_space(self, observation_space: Space) -> Space:
        observation_space = copy.deepcopy(observation_space)
        orig_space = observation_space.spaces["laser_2d_0"]
        for i in range(self.frames):
            del observation_space.spaces[f"laser_2d_{i}"]

        low = (
            orig_space.low
            if np.isscalar(orig_space.low)
            else np.min(orig_space.low)
        )
        high = (
            orig_space.high
            if np.isscalar(orig_space.high)
            else np.max(orig_space.high)
        )
        shape = (self.frames * self.frame_width,)
        observation_space.spaces[self.cls_uuid] = spaces.Box(
            low=low, high=high, shape=shape, dtype=orig_space.dtype
        )
        return observation_space

    @torch.no_grad()
    def forward(self, observations: Observations) -> Observations:
        keys = [f"laser_2d_{i}" for i in range(self.frames)]
        scan = torch.cat([observations[k] for k in keys], dim=2).squeeze(3)

        # scan: [B, H, W]
        if self.pool_mode == "min":
            scan, _ = scan.min(dim=1)
        elif self.pool_mode == "max":
            scan, _ = scan.max(dim=1)
        elif self.pool_mode == "mean":
            scan = scan.mean(dim=1)
        else:
            scan = scan[:, np.floor(scan.shape[1] / 2).astype(np.int)]

        for k in keys:
            del observations[k]
        observations[self.cls_uuid] = scan

        return observations

    @classmethod
    def from_config(cls, config: Config):
        pool_mode = config.TASK_CONFIG.TASK.LASER_2D.POOL_MODE
        frame_width = config.TASK_CONFIG.TASK.LASER_2D.WIDTH
        return cls(pool_mode, frame_width)


@baseline_registry.register_obs_transformer()
class RadialOccupancy(ObservationTransformer):
    """Convert a 360 degree range scan (1D array) to a 2D array representing
    a radial occupancy map. Values are 1: occupied, -1: free, 0: unknown.
    Adapted from VLN Sim2Real at https://shorturl.at/cqyIJ
    """

    cls_uuid: str = "radial_occupancy"

    def __init__(
        self,
        laser_uuid: str,
        range_bins: int,
        heading_bins: int,
        max_range: float,
        is_normalized: bool,
        agg_percentile: int,
    ) -> None:
        """Args:
        laser_uuid: observation key of the range scan to convert.
        range_bins: number of discretized range bins.
        heading_bins: number of discretized heading bins.
        max_range: maximum range to be sensed. The min is zero.
        is_normalized: if True, the data is normalized to [0,1].
        """
        self.laser_uuid: str = laser_uuid
        self.range_bins: int = range_bins
        self.heading_bins: int = heading_bins
        self.max_range: float = max_range
        self.is_normalized: bool = is_normalized
        self.range_bin_width = round(self.max_range / self.range_bins, 2)
        self.agg_percentile: int = agg_percentile
        super(RadialOccupancy, self).__init__()

    def transform_observation_space(self, observation_space: Space) -> Space:
        observation_space = copy.deepcopy(observation_space)
        del observation_space.spaces[self.laser_uuid]

        shape = (self.range_bins, self.heading_bins)
        observation_space.spaces[self.cls_uuid] = spaces.Box(
            low=-1, high=1, shape=shape, dtype=np.int8
        )
        return observation_space

    def _convert_single(self, scan):
        chunk_size = scan.shape[0] // self.heading_bins
        range_bins = np.arange(
            0,
            self.range_bin_width * (self.range_bins + 1),
            self.range_bin_width,
        )

        output = np.zeros(
            (self.range_bins, self.heading_bins), dtype=np.float32
        )

        i_low = 0
        i_high = chunk_size
        n = 0

        while i_low < scan.shape[0]:
            chunk = scan[i_low:i_high]

            # Remove nan values, negatives will fall outside range_bins
            chunk[np.isnan(chunk)] = -1

            # CUSTOM: specify aggregation percentile
            chunk = np.array(np.percentile(chunk, self.agg_percentile))

            # Add 'inf' as right edge of an extra bin to account for the case
            # if the returned range exceeds the maximum discretized range. In
            # this case we still want to register these cells as free.
            bins = np.array(range_bins.tolist() + [np.Inf])
            hist, _ = np.histogram(chunk, bins=bins)

            output[:, n] = np.clip(hist[:-1], 0, 1)
            free_ix = (
                np.flip(np.cumsum(np.flip(hist, axis=0), axis=0), axis=0)[1:]
                > 0
            )
            output[:, n][free_ix] = -1

            i_low += chunk_size
            i_high += chunk_size
            n += 1

        return output

    @torch.no_grad()
    def forward(self, observations: Observations) -> Observations:
        device = observations[self.laser_uuid].device
        scan = observations[self.laser_uuid]
        del observations[self.laser_uuid]

        if self.is_normalized:
            scan = scan * self.max_range

        scan = scan.cpu().numpy()

        output = np.stack(
            [
                self._convert_single(scan[batch_idx])
                for batch_idx in range(scan.shape[0])
            ]
        )

        observations[self.cls_uuid] = torch.from_numpy(output).to(
            device=device
        )
        return observations

    @classmethod
    def from_config(cls, config: Config):
        cfg = config.RL.POLICY.OBS_TRANSFORMS.RADIAL_OCCUPANCY
        return cls(
            laser_uuid=cfg.laser_uuid,
            range_bins=cfg.range_bins,
            heading_bins=cfg.heading_bins,
            max_range=config.TASK_CONFIG.TASK.LASER_2D.MAX_DEPTH,
            is_normalized=config.TASK_CONFIG.TASK.LASER_2D.NORMALIZE_DEPTH,
            agg_percentile=cfg.agg_percentile,
        )
