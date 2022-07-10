from typing import List, Optional, Union

import habitat_baselines.config.default
from habitat.config.default import CONFIG_FILE_SEPARATOR
from habitat.config.default import Config as CN

from habitat_extensions.config.default import (
    get_extended_config as get_task_config,
)

# ----------------------------------------------------------------------------
# EXPERIMENT CONFIG
# ----------------------------------------------------------------------------
_C = CN()
_C.BASE_TASK_CONFIG_PATH = "habitat_extensions/config/vlnce_task.yaml"
_C.TASK_CONFIG = CN()  # task_config will be stored as a config node
_C.CMD_TRAILING_OPTS = []  # store command line options as list of strings
_C.TRAINER_NAME = "sim2sim_trainer"
_C.ENV_NAME = "BaseEnv"
_C.SIMULATOR_GPU_IDS = [0]
_C.VIDEO_OPTION = []  # options: "disk", "tensorboard"
_C.VIDEO_DIR = "data/videos/debug"
_C.TENSORBOARD_DIR = "data/tensorboard_dirs/debug"
_C.RESULTS_DIR = "data/checkpoints/pretrained/evals"
_C.LOCAL_POLICY = False

# ----------------------------------------------------------------------------
# EVAL CONFIG
# ----------------------------------------------------------------------------
_C.EVAL = CN()
_C.EVAL.SPLIT = "val_seen"
_C.EVAL.EPISODE_COUNT = -1
_C.EVAL.SAMPLE = False
_C.EVAL.SAVE_RESULTS = True
_C.EVAL.SAVE_NAV_STATS = False

# ----------------------------------------------------------------------------
# INFERENCE CONFIG
# ----------------------------------------------------------------------------
_C.INFERENCE = CN()
_C.INFERENCE.SPLIT = "test"
_C.INFERENCE.SAMPLE = False
_C.INFERENCE.CKPT_PATH = "data/models/RecVLNBERT-ce_vision-tuned.pth"
_C.INFERENCE.PREDICTIONS_FILE = "predictions.json"

# ----------------------------------------------------------------------------
# IMITATION LEARNING CONFIG
# ----------------------------------------------------------------------------
_C.IL = CN()
_C.IL.lr = 2.5e-4
_C.IL.batch_size = 5
_C.IL.epochs = 4
_C.IL.load_from_ckpt = True
_C.IL.ckpt_to_load = "data/checkpoints/ckpt.0.pth"
_C.IL.lmdb_features_dir = "data/trajectories_dirs/debug/trajectories.lmdb"
_C.IL.min_spl = 0.6  # SGM paths must attain this SPL to be used for training
_C.IL.max_nav_error = 2.0  # SGM paths must get closer to the goal than this
_C.IL.soft_labels = True

# ----------------------------------------------------------------------------
# POLICY CONFIG
# ----------------------------------------------------------------------------
_C.RL = CN()
_C.RL.POLICY = CN()
_C.RL.POLICY.OBS_TRANSFORMS = CN()
_C.RL.POLICY.OBS_TRANSFORMS.ENABLED_TRANSFORMS = [
    "ObsStack",
    "ResNetCandidateEncoder",
    "CandidateFeatures",
    "PreprocessVLNInstruction",
]
_C.RL.POLICY.OBS_TRANSFORMS.OBS_STACK = CN()
_C.RL.POLICY.OBS_TRANSFORMS.OBS_STACK.SENSOR_REWRITES = [
    ["rgb", [f"rgb_{i}" for i in range(36)]],
]
_C.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR = CN()
_C.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS = [
    ("rgb", (224, 224)),
    ("depth", (256, 256)),
]
_C.RL.POLICY.OBS_TRANSFORMS.RADIAL_OCCUPANCY = CN()
_C.RL.POLICY.OBS_TRANSFORMS.RADIAL_OCCUPANCY.laser_uuid = "laser_2d"
_C.RL.POLICY.OBS_TRANSFORMS.RADIAL_OCCUPANCY.range_bins = 24
_C.RL.POLICY.OBS_TRANSFORMS.RADIAL_OCCUPANCY.heading_bins = 48
_C.RL.POLICY.OBS_TRANSFORMS.RADIAL_OCCUPANCY.agg_percentile = 50
_C.RL.POLICY.OBS_TRANSFORMS.RESNET_CANDIDATE_ENCODER = CN()
_C.RL.POLICY.OBS_TRANSFORMS.RESNET_CANDIDATE_ENCODER.protoxt_file = (
    "data/caffe_models/ResNet-152-deploy.prototxt"
)
_C.RL.POLICY.OBS_TRANSFORMS.RESNET_CANDIDATE_ENCODER.weights_file = (
    "data/caffe_models/resnet152_places365.caffemodel"
)
_C.RL.POLICY.OBS_TRANSFORMS.RESNET_CANDIDATE_ENCODER.max_batch_size = 8
_C.RL.POLICY.OBS_TRANSFORMS.RESNET_CANDIDATE_ENCODER.remove_rgb = True
_C.RL.POLICY.OBS_TRANSFORMS.RESNET_CANDIDATE_ENCODER.gpu_id = -1
_C.RL.POLICY.OBS_TRANSFORMS.CANDIDATE_FEATURES = CN()
_C.RL.POLICY.OBS_TRANSFORMS.CANDIDATE_FEATURES.remove_rgb_feats = True
_C.RL.POLICY.OBS_TRANSFORMS.CANDIDATE_FEATURES.remove_vln_candidates = True
_C.RL.POLICY.OBS_TRANSFORMS.CANDIDATE_FEATURES.angle_feature_size = 128
_C.RL.POLICY.OBS_TRANSFORMS.PREPROCESS_VLN_INSTRUCTION = CN()
_C.RL.POLICY.OBS_TRANSFORMS.PREPROCESS_VLN_INSTRUCTION.pad_idx = 0
_C.RL.POLICY.OBS_TRANSFORMS.SUBGOAL_MODULE = CN()
_C.RL.POLICY.OBS_TRANSFORMS.SUBGOAL_MODULE.max_candidates = 5
_C.RL.POLICY.OBS_TRANSFORMS.SUBGOAL_MODULE.unet_weights_file = (
    "data/sgm_models/sgm_sim2sim.pth"
)
_C.RL.POLICY.OBS_TRANSFORMS.SUBGOAL_MODULE.unet_channels = 64
_C.RL.POLICY.OBS_TRANSFORMS.SUBGOAL_MODULE.subgoal_nms_sigma = 2.0
_C.RL.POLICY.OBS_TRANSFORMS.SUBGOAL_MODULE.subgoal_nms_thresh = 0.003
_C.RL.POLICY.OBS_TRANSFORMS.SUBGOAL_MODULE.angle_feature_size = 128
_C.RL.POLICY.OBS_TRANSFORMS.SUBGOAL_MODULE.remove_rgb_feats = True
_C.RL.POLICY.OBS_TRANSFORMS.SUBGOAL_MODULE.ablate_feats = False
_C.RL.POLICY.OBS_TRANSFORMS.SUBGOAL_MODULE.use_ground_truth = False
# number of bins to adjust by (1.0 -> 20cm)
_C.RL.POLICY.OBS_TRANSFORMS.SUBGOAL_MODULE.range_correction = 0.5
# number of bins to adjust by (1.0 -> 7.5deg)
_C.RL.POLICY.OBS_TRANSFORMS.SUBGOAL_MODULE.heading_correction = 0.5

# ----------------------------------------------------------------------------
# MODELING CONFIG
# ----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.policy_name = "VLNBERTPolicy"
_C.MODEL.VLNBERT = CN()
_C.MODEL.VLNBERT.hidden_size = 768
_C.MODEL.VLNBERT.img_feature_dim = 2176
_C.MODEL.VLNBERT.angle_feature_size = 128
_C.MODEL.VLNBERT.dropout_p = 0.4
_C.MODEL.VLNBERT.img_feature_type = ""
_C.MODEL.VLNBERT.vl_layers = 4
_C.MODEL.VLNBERT.la_layers = 9
_C.MODEL.VLNBERT.directions = 4  # a preset random number
_C.MODEL.VLNBERT.pretrained = "data/models/PREVALENT.bin"


def purge_keys(config: CN, keys: List[str]) -> None:
    for k in keys:
        del config[k]
        config.register_deprecated_key(k)


def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    """Create a unified config with default values. Initialized from the
    habitat_baselines default config. Overwritten by values from
    `config_paths` and overwritten by options from `opts`.
    Args:
        config_paths: List of config paths or string that contains comma
        separated list of config paths.
        opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example, `opts = ['FOO.BAR',
        0.5]`. Argument can be used for parameter sweeping or quick tests.
    """
    config = CN()
    config.merge_from_other_cfg(habitat_baselines.config.default._C)
    purge_keys(config, ["SIMULATOR_GPU_ID", "TEST_EPISODE_COUNT"])
    config.merge_from_other_cfg(_C.clone())

    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        prev_task_config = ""
        for config_path in config_paths:
            config.merge_from_file(config_path)
            if config.BASE_TASK_CONFIG_PATH != prev_task_config:
                config.TASK_CONFIG = get_task_config(
                    config.BASE_TASK_CONFIG_PATH
                )
                prev_task_config = config.BASE_TASK_CONFIG_PATH

    if opts:
        config.CMD_TRAILING_OPTS = opts
        config.merge_from_list(opts)

    config.freeze()
    return config
