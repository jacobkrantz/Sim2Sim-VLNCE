import abc
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import caffe
import numpy as np
import torch
from gym import Space, spaces
from habitat.config import Config
from habitat.core.simulator import Observations
from habitat_baselines.common.baseline_registry import BaselineRegistry
from habitat_baselines.common.obs_transformers import ObservationTransformer
from torch import Tensor, nn
from torch.nn import functional as F

from habitat_extensions.utils import vln_style_angle_features
from sim2sim_vlnce.Sim2Sim.subgoal_module.nonmaximal_suppression import nms
from sim2sim_vlnce.Sim2Sim.subgoal_module.unet_model import UNet
from sim2sim_vlnce.Sim2Sim.vis import subgoal_candidates_to_radial_map


@BaselineRegistry.register_obs_transformer()
class ResNetAllEncoder(ObservationTransformer):
    """Encodes all RGB frames into a 2048-dimensional vector using a
    pretrained ResNet. Assumes the "rgb" observation is a stack of RGB images.
    The model is in Caffe to match exactly the encoder used in R2R VLN.
    """

    OBS_UUID: str = "rgb_features"
    FEATURE_SIZE: int = 2048

    def __init__(
        self,
        prototxt_f: str,
        weights_f: str,
        remove_rgb: bool,
        max_batch_size: int,
        gpu_id: int,
    ) -> None:
        """Args:
        prototxt_f (str): file path of the Caffe network definition file
        weights_f (str): file path of the weights for the above network
        """
        super(ResNetAllEncoder, self).__init__()
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

        self.bgr_mean = torch.tensor([103.1, 115.9, 123.2]).to(
            device=self.device
        )

    def transform_observation_space(
        self,
        observation_space: Space,
    ) -> Space:
        spaces_dict: Dict[str, Space] = observation_space.spaces
        num_frames = spaces_dict["rgb"].shape[0]
        if self.remove_rgb:
            del spaces_dict["rgb"]

        spaces_dict[self.OBS_UUID] = spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(num_frames, self.FEATURE_SIZE),
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

        encoded = np.empty((num_frames, self.FEATURE_SIZE), dtype=np.float32)

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

    @torch.no_grad()
    def forward(self, observations: Observations) -> Observations:
        batch_size, num_frames, h, w, channels = observations["rgb"].shape

        in_shape = (batch_size * num_frames, h, w, channels)
        out_shape = (batch_size, num_frames, self.FEATURE_SIZE)

        frame_features = self.encode(observations["rgb"].reshape(in_shape))
        observations[self.OBS_UUID] = frame_features.reshape(out_shape)

        if self.remove_rgb:
            del observations["rgb"]

        return observations

    @classmethod
    def from_config(cls, config: Config):
        cfg = config.RL.POLICY.OBS_TRANSFORMS.RESNET_CANDIDATE_ENCODER
        gpu_id = config.TORCH_GPU_ID
        if cfg.gpu_id >= 0:
            gpu_id = cfg.gpu_id
        return cls(
            cfg.protoxt_file,
            cfg.weights_file,
            cfg.remove_rgb,
            cfg.max_batch_size,
            gpu_id,
        )


class SubgoalModuleABC(ObservationTransformer, metaclass=abc.ABCMeta):
    """Abstract class of a subgoal prediction module. Given sensor readings,
    compute a subset of the 36 view frames that have candidate waypoints.
    Returns a tensor of [I, idx, x, y, z, h, e] for:
        I:          1 if a candidate is present at this line
        idx:        relative viewpoint index to map candidate to
        (x,y,z):    habitat global waypoint coordinates
        h:          candidate heading
        e:          candidate elevation
    Output format should match the "vln_candidates" sensor, which can be viewed
    as an oracle for this subtask.
    """

    FEAT_UUID: str = "candidate_features"
    COORD_UUID: str = "candidate_coordinates"
    NUM_UUID: str = "num_candidates"

    def transform_observation_space(
        self,
        observation_space: Space,
        max_candidates: int,
        angle_feature_size: int,
        remove_rgb_feats: bool,
    ) -> Space:
        obs_space = deepcopy(observation_space)
        rgb_feature_size = obs_space.spaces["rgb_features"].shape[1]

        obs_space.spaces[self.FEAT_UUID] = spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(
                max_candidates,
                rgb_feature_size + angle_feature_size,
            ),
            dtype=np.float32,
        )
        obs_space.spaces[self.COORD_UUID] = spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(max_candidates, 3),
            dtype=np.float32,
        )
        obs_space.spaces[self.NUM_UUID] = spaces.Box(
            low=0,
            high=max_candidates,
            shape=(1,),
            dtype=np.int,
        )
        if remove_rgb_feats:
            del obs_space.spaces["rgb_features"]
        return obs_space

    @staticmethod
    def feats_from_pose(rgb_feats, rel_heading, elevation, angle_feature_size):
        """Generate features for candidates specified by their heading,
        relative heading, and elevation. Features include RGB (2048) and
        angle features (128).
        """
        rel_heading_lst = rel_heading.cpu().numpy().tolist()
        elevation_lst = elevation.cpu().numpy().tolist()

        feats = torch.zeros(
            rel_heading.shape[0],
            rgb_feats.shape[0] + angle_feature_size,
            device=rgb_feats.device,
        )
        feats[:, rgb_feats.shape[0] :] = vln_style_angle_features(
            rel_heading, elevation, angle_feature_size
        )
        for i in range(len(rel_heading_lst)):
            e = round(elevation_lst[i], 2)
            e_idx = 1
            if e < 0.0:
                e_idx = 0
            elif e > 0.0:
                e_idx = 2

            headingIncrement = np.pi * 2.0 / 12
            h = int(np.around(rel_heading_lst[i] / headingIncrement)) % 12
            feats[i, : rgb_feats.shape[0]] = rgb_feats[:, e_idx, h]

        return feats


@BaselineRegistry.register_obs_transformer()
class SubgoalModule(SubgoalModuleABC):

    PRED_UUID: str = "radial_pred"

    def __init__(
        self,
        max_candidates: int,
        unet_weights_file: str,
        unet_channels: int,
        nms_sigma: float,
        nms_thresh: float,
        angle_feature_size: int,
        remove_rgb_feats: bool,
        ablate_feats: bool,
        use_ground_truth: bool,
        range_correction: float,
        heading_correction: float,
        range_bin_width: float,
        heading_bin_width: float,
        device: torch.device,
    ) -> None:
        """Args:
        max_candidates: max number of subgoal candidates limited via NMS.
            selfmax_candidates includes STOP.
        unet_weights_file: located of re-trained weights for the UNet.
        unet_channels: number of internal channels in the UNet.
        nms_sigma:
        nms_thresh:
        """
        super(SubgoalModule, self).__init__()
        self.max_candidates = max_candidates + 1
        self.sigma = nms_sigma
        self.thresh = nms_thresh
        self.angle_feature_size = angle_feature_size
        self.remove_rgb_feats = remove_rgb_feats
        self.ablate_feats = ablate_feats
        self.use_ground_truth = use_ground_truth
        self.range_bin_width = range_bin_width
        self.heading_bin_width = heading_bin_width
        self.unet = self.load_subgoal_module(
            unet_weights_file, unet_channels, device
        )
        self.range_correction = range_correction
        self.heading_correction = heading_correction

    def transform_observation_space(self, observation_space: Space) -> Space:
        obs_space = super().transform_observation_space(
            observation_space,
            self.max_candidates,
            self.angle_feature_size,
            self.remove_rgb_feats,
        )
        obs_space.spaces[self.PRED_UUID] = spaces.Box(
            low=0.0,
            high=1.0,
            shape=obs_space.spaces["radial_occupancy"].shape,
            dtype=np.float,
        )
        return obs_space

    @torch.no_grad()
    def forward(self, observations: Observations) -> Observations:
        feats = observations["rgb_features"]
        device = feats.device
        batch_size = feats.shape[0]

        feat_shape = (batch_size, 3, 12, feats.shape[2])
        feats = feats.reshape(feat_shape).permute(
            0, 3, 1, 2
        )  # [B, 2048, 3, 12]

        rel_candidates = None
        if "vln_candidates_relative" in observations:
            rel_candidates = observations["vln_candidates_relative"]

        pred, centered_pred = self.predict_subgoal_img(
            feats,
            observations["radial_occupancy"],
            rel_candidates=rel_candidates,
            use_ground_truth=self.use_ground_truth,
        )
        candidates, num_candidates = self.candidates_from_img(pred, device)

        # map candidates to features and relative polar coordinates
        feature_size = feat_shape[-1] + self.angle_feature_size
        candidate_features = torch.zeros(
            batch_size, self.max_candidates, feature_size, device=device
        )
        candidate_coordinates = torch.zeros(
            batch_size, self.max_candidates, 2, device=device
        )
        for i in range(batch_size):
            c = candidates[i].shape[0]

            # for each candidate, insert vision & pose features
            rel_heading, distance, elevation = self._get_pose_attrs(
                candidates[i]
            )
            candidate_features[i, :c] = self.feats_from_pose(
                feats[i], rel_heading, elevation, self.angle_feature_size
            )
            candidate_coordinates[i, :c, 0] = distance
            candidate_coordinates[i, :c, 1] = -rel_heading % (2 * np.pi)

        observations[self.FEAT_UUID] = candidate_features
        observations[self.COORD_UUID] = candidate_coordinates
        observations[self.NUM_UUID] = num_candidates
        observations[self.PRED_UUID] = centered_pred

        if self.remove_rgb_feats:
            del observations["rgb_features"]

        return observations

    def predict_subgoal_img(
        self,
        rgb_feats: Tensor,
        ro_map: Tensor,
        rel_candidates: Optional[Tensor] = None,
        use_ground_truth: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Predicts subgoals from a radial occupancy map and RGB features.
        Args:
            feats: Tensor of size [B, 2048, E, H]
            ro_map: Tensor of size [B, R_bins, H_bins]
        Returns:
            subgoal predictions of size [B, R_bins, H_bins]
        """
        h_bins = ro_map.shape[2]

        # center feats and radial occupancy on zero heading
        centered_feats = rgb_feats.roll(6, dims=3)

        # ablate RGB features
        centered_feats = centered_feats * int(not self.ablate_feats)

        # roll 180 deg so fwd frame is in middle
        ro_shift = h_bins // 2
        # roll back half a frame so center of frame is middle
        ro_shift -= h_bins // 8
        centered_ro_map = ro_map.roll(ro_shift, dims=2)

        if use_ground_truth:
            if rel_candidates is None:
                raise ValueError("relative candidates not populated.")

            # convert candidates to centered ground truth radial labels
            rel_candidates = rel_candidates.cpu().numpy()
            centered_pred = [
                torch.from_numpy(
                    subgoal_candidates_to_radial_map(
                        shape=ro_map.shape[1:],
                        candidates_relative=rel_candidates[idx],
                    )
                ).to(device=centered_feats.device)
                for idx in range(rel_candidates.shape[0])
            ]
            centered_pred = torch.stack(
                [p / p.sum() for p in centered_pred]
            ).to(dtype=torch.float32)
        else:
            logits = self.unet(
                self.prep_scan_for_net(centered_ro_map), centered_feats
            )
            centered_pred = nms(
                F.softmax(logits.flatten(1), dim=1).reshape(logits.shape),
                self.sigma,
                self.thresh,
                self.max_candidates - 1,
            ).squeeze(1)

        # unrolled predictions have heading=0 at index [:, 0].
        unrolled_nms_pred = centered_pred.roll(h_bins // 2, dims=2)
        return unrolled_nms_pred, centered_pred

    def candidates_from_img(self, pred, device) -> Tuple[List[Tensor], Tensor]:
        """Extract a Tensor of subgoal candidates from the 2D prediction map
        for each batch idx. Candidates are represented by their map index.
        Also return a tensor for the number of candidates.
        """
        num_candidates = []
        candidates = []
        batch_size = pred.shape[0]
        for i in range(batch_size):
            waypoint_ix = (pred[i] > 0).nonzero(as_tuple=False)
            num_candidates.append(waypoint_ix.shape[0] + 1)
            candidates.append(waypoint_ix)

        num_candidates = torch.tensor(
            num_candidates, dtype=torch.int, device=device
        )
        return candidates, num_candidates

    def _get_pose_attrs(
        self, candidates: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """candidates: size [n_candidates, 2]
            idx 0: range bin
            idx 1: heading bin
        values are in the MP3D frame
        """
        h_bin = candidates[:, 1] + self.heading_correction
        rel_heading = (h_bin * self.heading_bin_width) % (2 * np.pi)

        d_bin = candidates[:, 0] + self.range_correction
        distance = d_bin * self.range_bin_width

        elevation = torch.zeros_like(distance)  # single-floor assumption
        return rel_heading, distance, elevation

    @staticmethod
    def load_subgoal_module(
        unet_weights_file: str,
        unet_channels: int,
        device: torch.device,
    ) -> nn.Module:
        unet = UNet(n_channels=2, n_classes=1, ch=unet_channels).to(device)
        if unet_weights_file != "":
            state_dict = torch.load(unet_weights_file, map_location="cpu")
            unet.load_state_dict(state_dict)

        unet.eval()
        return unet

    @staticmethod
    def prep_scan_for_net(scan_img: Tensor) -> Tensor:
        shape = (scan_img.shape[0], 2, scan_img.shape[1], scan_img.shape[2])
        imgs = torch.zeros(shape, device=scan_img.device)
        imgs[:, 1, :, :] = scan_img

        ran_ch = np.linspace(-0.5, 0.5, num=imgs.shape[2])
        ran_ch = np.expand_dims(np.expand_dims(ran_ch, axis=0), axis=2)
        imgs[:, 0, :, :] = torch.from_numpy(ran_ch).to(device=scan_img.device)
        return imgs

    @classmethod
    def from_config(cls, config: Config):
        device = (
            torch.device("cuda", config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        rbins = config.RL.POLICY.OBS_TRANSFORMS.RADIAL_OCCUPANCY.range_bins
        hbins = config.RL.POLICY.OBS_TRANSFORMS.RADIAL_OCCUPANCY.heading_bins
        rmax = config.TASK_CONFIG.TASK.LASER_2D.MAX_DEPTH
        range_bin_width = round(rmax / rbins, 3)
        heading_bin_width = (2 * np.pi) / hbins

        cfg = config.RL.POLICY.OBS_TRANSFORMS.SUBGOAL_MODULE
        return cls(
            cfg.max_candidates,
            cfg.unet_weights_file,
            cfg.unet_channels,
            cfg.subgoal_nms_sigma,
            cfg.subgoal_nms_thresh,
            cfg.angle_feature_size,
            cfg.remove_rgb_feats,
            cfg.ablate_feats,
            cfg.use_ground_truth,
            cfg.range_correction,
            cfg.heading_correction,
            range_bin_width,
            heading_bin_width,
            device,
        )
