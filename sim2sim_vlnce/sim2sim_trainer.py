import json
import os
import pickle
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import lmdb
import msgpack_numpy
import numpy as np
import torch
import torch.nn as nn
import tqdm
from gym.spaces.space import Space
from habitat import Config, logger
from habitat_baselines.common.base_trainer import BaseTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    ObservationTransformer,
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import batch_obs
from torch import Tensor

from habitat_extensions.utils import generate_video
from sim2sim_vlnce.base_policy import VLNPolicy
from sim2sim_vlnce.config.modifiers import (
    add_2d_laser_to_config,
    add_depth_for_map,
    add_vln_pano_sensors_to_config,
)
from sim2sim_vlnce.Sim2Sim.environments import construct_envs_auto_reset_false
from sim2sim_vlnce.Sim2Sim.training_utils import (
    VLNDataset,
    collate_fn,
    select_closest_subgoal,
    soft_cross_entropy,
    update_agent,
)
from sim2sim_vlnce.Sim2Sim.vis import video_frame

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf  # noqa: F401


@baseline_registry.register_trainer(name="sim2sim_trainer")
class Vln2ceEvaluator(BaseTrainer):

    config: Config
    flush_secs: int
    device: torch.device  # type: ignore
    policy: VLNPolicy
    obs_transforms: List[ObservationTransformer]

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = add_vln_pano_sensors_to_config(config)
        self.config = add_2d_laser_to_config(self.config)
        self.config = add_depth_for_map(self.config)
        self.flush_secs = 30
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

    def _initialize_policy(
        self,
        config: Config,
        load_from_ckpt: bool,
        observation_space: Space,
        action_space: Space,
        train: bool = True,
    ) -> None:
        policy = baseline_registry.get_policy(self.config.MODEL.policy_name)
        self.policy = policy.from_config(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
        )
        self.policy.to(self.device)

        if load_from_ckpt:
            ckpt_path = config.IL.ckpt_to_load
            ckpt_dict = self.load_checkpoint(ckpt_path, map_location="cpu")
            self.policy.load_state_dict(ckpt_dict)
            logger.info(f"Loaded weights from checkpoint: {ckpt_path}")

        if not train:
            self.policy.eval()

        params = sum(param.numel() for param in self.policy.parameters())
        params_t = sum(
            p.numel() for p in self.policy.parameters() if p.requires_grad
        )
        logger.info(f"Agent parameters: {params}. Trainable: {params_t}")
        logger.info("Finished setting up policy.")

    def _get_spaces(
        self, config: Config, envs: Optional[Any] = None
    ) -> Tuple[Space]:
        """Gets both the observation space and action space.

        Args:
            config (Config): The config specifies the observation transforms.
            envs (Any, optional): An existing Environment. If None, an
                environment is created using the config.

        Returns:
            observation space, action space
        """
        if envs is not None:
            observation_space = envs.observation_spaces[0]
            action_space = envs.action_spaces[0]

        else:
            env = get_env_class(self.config.ENV_NAME)(config=config)
            observation_space = env.observation_space
            action_space = env.action_space

        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )
        return observation_space, action_space

    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        ckpt = torch.load(checkpoint_path, *args, **kwargs)
        if "vln_bert" in ckpt:
            return {
                "net." + k: v
                for k, v in ckpt["vln_bert"]["state_dict"].items()
            }
        return ckpt["state_dict"]

    def save_checkpoint(self, file_name) -> None:
        raise NotImplementedError

    @staticmethod
    def _pause_envs(
        envs_to_pause,
        envs,
        h_t,
        instruction_features,
        not_done_masks,
        batch,
        rgb_frames,
    ):
        if not len(envs_to_pause):
            return (
                envs,
                h_t,
                instruction_features,
                not_done_masks,
                batch,
                rgb_frames,
            )

        state_index = list(range(envs.num_envs))
        for idx in reversed(envs_to_pause):
            state_index.pop(idx)
            rgb_frames.pop(idx)
            envs.pause_at(idx)

        # indexing along the batch dimensions
        h_t = h_t[state_index]
        instruction_features = instruction_features[state_index]
        not_done_masks = not_done_masks[state_index]
        batch = {k: v[state_index] for k, v in batch.items()}
        return (
            envs,
            h_t,
            instruction_features,
            not_done_masks,
            batch,
            rgb_frames,
        )

    def batch_obs(
        self, obs, skip_keys: Optional[List[str]] = None
    ) -> Dict[str, Tensor]:
        if skip_keys is None:
            o_in = obs
        else:
            o_in = [
                {k: v for k, v in o.items() if k not in skip_keys} for o in obs
            ]
        return apply_obs_transforms_batch(
            batch_obs(o_in, self.device), self.obs_transforms
        )

    def train(self, config) -> None:
        """Fine-tunes a VLN agent"""
        os.makedirs(config.CHECKPOINT_FOLDER, exist_ok=True)
        lmdb_dir = config.IL.lmdb_features_dir.format(split="train")

        spaces_fname = os.path.join(lmdb_dir, "vln_spaces.pkl")

        try:
            with open(spaces_fname, "rb") as f:
                spaces = pickle.load(f)
        except FileNotFoundError as e:
            print(
                "\nvln_spaces.pkl needs to be generated from `--run-type collect`.\n"
            )
            raise e

        self._initialize_policy(
            config,
            load_from_ckpt=True,
            observation_space=spaces["observation_space"],
            action_space=spaces["action_space"],
        )
        dataset = VLNDataset(
            lmdb_dir,
            batch_size=config.IL.batch_size,
            soft_labels=config.IL.soft_labels,
            min_spl=config.IL.min_spl,
            max_nav_error=config.IL.max_nav_error,
        )
        diter = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.IL.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=True,
            num_workers=3,
        )
        optim = torch.optim.AdamW(self.policy.parameters(), lr=config.IL.lr)

        if config.IL.soft_labels:
            criterion = soft_cross_entropy
        else:
            criterion = nn.CrossEntropyLoss(reduction="none")

        step_id = 0
        with TensorboardWriter(config.TENSORBOARD_DIR, purge_step=0) as writer:
            for epoch in tqdm.trange(config.IL.epochs, dynamic_ncols=True):
                num_batches = dataset.length // dataset.batch_size
                for obs, step_masks, actions in tqdm.tqdm(
                    diter, total=num_batches, leave=False, dynamic_ncols=True
                ):
                    if config.IL.soft_labels:
                        actions = [a.to(device=self.device) for a in actions]
                    else:
                        actions = actions.to(device=self.device)

                    loss = update_agent(
                        self.policy,
                        obs={
                            k: v.to(device=self.device) for k, v in obs.items()
                        },
                        step_masks=step_masks.to(device=self.device),
                        actions=actions,
                        optimizer=optim,
                        criterion=criterion,
                    )

                    writer.add_scalar("train_loss", loss, step_id)
                    step_id += 1

                torch.save(
                    {"state_dict": self.policy.state_dict(), "config": config},
                    os.path.join(
                        config.CHECKPOINT_FOLDER, f"ckpt.{epoch}.pth"
                    ),
                )

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        logger.info(f"checkpoint_path: {checkpoint_path}")

        config = self.config.clone()
        split = config.EVAL.SPLIT

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = split
        config.TASK_CONFIG.TASK.NDTW.SPLIT = split
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (
            -1
        )
        config.IL.ckpt_to_load = checkpoint_path

        skip_keys = []
        if len(config.VIDEO_OPTION) > 0:
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            config.TASK_CONFIG.TASK.SENSORS.extend(
                [
                    "GLOBAL_GPS_SENSOR",
                    "HEADING_SENSOR",
                    "INSTRUCTION_SENSOR",
                    "VLN_CANDIDATE_RELATIVE_SENSOR",
                ]
            )
            skip_keys = ["instruction"]

        config.freeze()

        if config.EVAL.SAVE_RESULTS:
            os.makedirs(config.RESULTS_DIR, exist_ok=True)
            fname = os.path.join(
                config.RESULTS_DIR,
                f"stats_ckpt_{checkpoint_index}_{split}.json",
            )
            full_results_name = os.path.join(
                config.RESULTS_DIR,
                f"all_stats_ckpt_{checkpoint_index}_{split}.json",
            )
            if os.path.exists(fname):
                logger.info("skipping -- evaluation exists.")
                return

        envs = construct_envs_auto_reset_false(
            config, get_env_class(config.ENV_NAME)
        )
        observation_space, action_space = self._get_spaces(config, envs=envs)

        self._initialize_policy(
            config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=action_space,
            train=False,
        )

        obs = envs.reset()
        batch = self.batch_obs(obs, skip_keys=skip_keys)

        rgb_frames = [[] for _ in range(envs.num_envs)]
        if len(config.VIDEO_OPTION) > 0:
            os.makedirs(config.VIDEO_DIR, exist_ok=True)
            for i in range(envs.num_envs):
                t_obs = {k: v[i].cpu().numpy() for k, v in batch.items()}
                info = envs.call_at(i, "get_info", {"observations": {}})
                rgb_frames[i].append(video_frame(obs[i], t_obs, info))

        h_t = torch.zeros(
            envs.num_envs,
            config.MODEL.VLNBERT.hidden_size,
            device=self.device,
        )

        instruction_features = torch.zeros(
            envs.num_envs,
            observation_space.spaces["vln_instruction"].shape[0],
            config.MODEL.VLNBERT.hidden_size,
            device=self.device,
        )

        not_done_masks = torch.zeros(
            envs.num_envs, 1, dtype=torch.bool, device=self.device
        )

        stats_episodes = {}
        num_eps = sum(envs.number_of_episodes)
        if config.EVAL.EPISODE_COUNT > -1:
            num_eps = min(config.EVAL.EPISODE_COUNT, num_eps)
        pbar = tqdm.tqdm(total=num_eps)

        nav_stats = []

        # STEP LOOP
        while envs.num_envs > 0 and len(stats_episodes) < num_eps:
            current_episodes = envs.current_episodes()

            if (not_done_masks == 0).any().item():
                # for new episodes, update instruction features and set h_t
                with torch.no_grad():
                    mask_select = (not_done_masks == False).squeeze(1)
                    (
                        h_t[mask_select],
                        instruction_features[mask_select],
                    ) = self.policy.encode_instruction(
                        batch["vln_instruction"][mask_select],
                        batch["vln_instruction_mask"][mask_select],
                    )

            with torch.no_grad():
                h_t, actions = self.policy.act(
                    batch,
                    h_t,
                    instruction_features,
                    instruction_mask=batch["vln_instruction_mask"],
                    deterministic=not config.EVAL.SAMPLE,
                )

            outputs = envs.step(actions)
            obs, _, dones, infos = [list(x) for x in zip(*outputs)]

            for i in range(len(infos)):
                # STOP actions dont produce navigation stats
                if "nav_stats" in infos[i]:
                    nav_stats.append(infos[i]["nav_stats"])
                    del infos[i]["nav_stats"]

            not_done_masks = torch.logical_not(
                torch.tensor(dones, device=self.device).unsqueeze(1)
            )

            # reset environments with done episodes
            for i in range(envs.num_envs):
                if not dones[i]:
                    continue

                # for done episodes, save the ending frame
                if len(config.VIDEO_OPTION) > 0:
                    t_obs = {k: v[i].cpu().numpy() for k, v in batch.items()}
                    frame = video_frame(obs[i], t_obs, infos[i])
                    rgb_frames[i].append(frame)

                ep_id = current_episodes[i].episode_id
                stats_episodes[ep_id] = infos[i]
                obs[i] = envs.reset_at(i)[0]
                pbar.update()

                # generate the video and initialize the next info
                if len(config.VIDEO_OPTION) > 0:
                    infos[i] = envs.call_at(
                        i, "get_info", {"observations": obs[i]}
                    )
                    generate_video(
                        video_option=config.VIDEO_OPTION,
                        video_dir=config.VIDEO_DIR,
                        images=rgb_frames[i],
                        episode_id=ep_id,
                        checkpoint_idx=checkpoint_index,
                        metrics={"spl": stats_episodes[ep_id]["spl"]},
                        tb_writer=writer,
                        fps=(1 / 3),
                    )
                    del stats_episodes[ep_id]["top_down_map_vlnce"]
                    rgb_frames[i] = []

            batch = self.batch_obs(obs, skip_keys=skip_keys)

            if len(config.VIDEO_OPTION) > 0:
                for i in range(envs.num_envs):
                    t_obs = {k: v[i].cpu().numpy() for k, v in batch.items()}
                    frame = video_frame(obs[i], t_obs, infos[i])
                    rgb_frames[i].append(frame)

            envs_to_pause = []
            next_episodes = envs.current_episodes()
            for i in range(envs.num_envs):
                if next_episodes[i].episode_id in stats_episodes:
                    envs_to_pause.append(i)

            (
                envs,
                h_t,
                instruction_features,
                not_done_masks,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                envs,
                h_t,
                instruction_features,
                not_done_masks,
                batch,
                rgb_frames,
            )

        envs.close()
        pbar.close()

        if (
            config.EVAL.SAVE_RESULTS
            and len(nav_stats)
            and self.config.EVAL.SAVE_NAV_STATS
        ):
            p = os.path.join(
                config.RESULTS_DIR,
                f"nav_stats_ckpt_{checkpoint_index}_{split}.json",
            )
            with open(p, "w") as f:
                json.dump(nav_stats, f)

        aggregated_stats = {}
        num_episodes = len(stats_episodes)
        for k in next(iter(stats_episodes.values())).keys():
            aggregated_stats[k] = (
                sum(v[k] for v in stats_episodes.values()) / num_episodes
            )

        if config.EVAL.SAVE_RESULTS:
            with open(fname, "w") as f:
                json.dump(aggregated_stats, f, indent=4)
            with open(full_results_name, "w") as f:
                json.dump(stats_episodes, f, indent=4)

        logger.info(f"Episodes evaluated: {num_episodes}")
        for k, v in aggregated_stats.items():
            logger.info(f"{k}: {v:.6f}")
            writer.add_scalar(f"eval_{split}_{k}", v, checkpoint_index)

    def inference(self) -> None:
        """Runs inference on a checkpoint and saves a predictions file."""

        checkpoint_path = self.config.INFERENCE.CKPT_PATH
        logger.info(f"checkpoint_path: {checkpoint_path}")

        config = self.config.clone()

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = self.config.INFERENCE.SPLIT
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (
            -1
        )
        config.IL.ckpt_to_load = config.INFERENCE.CKPT_PATH
        config.TASK_CONFIG.TASK.MEASUREMENTS = []
        config.TASK_CONFIG.TASK.SENSORS = [
            s
            for s in config.TASK_CONFIG.TASK.SENSORS
            if "INSTRUCTION" in s or "ANGLE_SENSOR"
        ]
        config.freeze()

        envs = construct_envs_auto_reset_false(
            config, get_env_class(config.ENV_NAME)
        )
        observation_space, action_space = self._get_spaces(config, envs=envs)

        self._initialize_policy(
            config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=action_space,
            train=False,
        )

        obs = envs.reset()
        batch = self.batch_obs(obs)
        rgb_frames = [[] for _ in range(envs.num_envs)]

        h_t = torch.zeros(
            envs.num_envs,
            config.MODEL.VLNBERT.hidden_size,
            device=self.device,
        )

        instruction_features = torch.zeros(
            envs.num_envs,
            observation_space.spaces["vln_instruction"].shape[0],
            config.MODEL.VLNBERT.hidden_size,
            device=self.device,
        )

        not_done_masks = torch.zeros(
            envs.num_envs, 1, dtype=torch.bool, device=self.device
        )

        episode_predictions = defaultdict(list)

        # populate episode_predictions with the starting state
        current_episodes = envs.current_episodes()
        for i in range(envs.num_envs):
            episode_predictions[current_episodes[i].episode_id].extend(
                envs.call_at(i, "get_info", {"observations": {}})
            )

        with tqdm.tqdm(
            total=sum(envs.count_episodes()),
            desc=f"[inference:{self.config.INFERENCE.SPLIT}]",
        ) as pbar:
            while envs.num_envs > 0:

                current_episodes = envs.current_episodes()

                if (not_done_masks == 0).any().item():
                    # for new episodes, update instruction features and set h_t
                    with torch.no_grad():
                        mask_select = (not_done_masks == False).squeeze(1)
                        (
                            h_t[mask_select],
                            instruction_features[mask_select],
                        ) = self.policy.encode_instruction(
                            batch["vln_instruction"][mask_select],
                            batch["vln_instruction_mask"][mask_select],
                        )

                with torch.no_grad():
                    h_t, actions = self.policy.act(
                        batch,
                        h_t,
                        instruction_features,
                        instruction_mask=batch["vln_instruction_mask"],
                        deterministic=not config.INFERENCE.SAMPLE,
                    )

                outputs = envs.step(actions)
                obs, _, dones, infos = [list(x) for x in zip(*outputs)]

                not_done_masks = torch.logical_not(
                    torch.tensor(dones, device=self.device).unsqueeze(1)
                )

                # reset environments with terminated episodes
                for i in range(envs.num_envs):
                    episode_predictions[current_episodes[i].episode_id].extend(
                        infos[i]
                    )
                    if not dones[i]:
                        continue

                    obs[i] = envs.reset_at(i)[0]
                    pbar.update()

                batch = self.batch_obs(obs)

                envs_to_pause = []
                next_episodes = envs.current_episodes()
                for i in range(envs.num_envs):
                    if not dones[i]:
                        continue

                    neid = next_episodes[i].episode_id
                    if neid in episode_predictions:
                        envs_to_pause.append(i)
                    else:
                        f = envs.call_at(i, "get_info", {"observations": {}})
                        episode_predictions[neid].extend(f)

                (
                    envs,
                    h_t,
                    instruction_features,
                    not_done_masks,
                    batch,
                    rgb_frames,
                ) = self._pause_envs(
                    envs_to_pause,
                    envs,
                    h_t,
                    instruction_features,
                    not_done_masks,
                    batch,
                    rgb_frames,
                )

        envs.close()

        with open(config.INFERENCE.PREDICTIONS_FILE, "w") as f:
            json.dump(episode_predictions, f, indent=2)

        logger.info(
            f"Predictions saved to: {config.INFERENCE.PREDICTIONS_FILE}"
        )

    def collect_sgm_imitation(self) -> None:
        """
        Computes oracle actions over the SGM predictions and saves a dataset
        for imitation learning.
        """
        config = self.config.clone()
        split = config.EVAL.SPLIT

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = split
        config.TASK_CONFIG.TASK.NDTW.SPLIT = split
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (
            -1
        )
        config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = (
            config.SIMULATOR_GPU_IDS[0]
        )
        config.freeze()

        if config.EVAL.SAVE_RESULTS:
            os.makedirs(config.RESULTS_DIR, exist_ok=True)
            fname = os.path.join(config.RESULTS_DIR, f"{split}.json")
            full_results_name = os.path.join(
                config.RESULTS_DIR, f"all_stats_{split}.json"
            )

        skip_keys = ["instruction"]

        env = get_env_class(config.ENV_NAME)(config=config)

        self.obs_transforms = get_active_obs_transforms(config)
        observation_space = apply_obs_transforms_obs_space(
            env.observation_space, self.obs_transforms
        )

        batch = self.batch_obs([env.reset()], skip_keys=skip_keys)

        num_eps = env.number_of_episodes
        pbar = tqdm.tqdm(total=num_eps)
        stats_episodes = {}
        episode = {
            "episode_id": env.current_episode.episode_id,
            "observations": [],
            "actions": [],
            "geodesics": [],
        }

        results_dir = os.path.join(config.RESULTS_DIR, split)
        os.makedirs(results_dir)

        # save spaces for later training without habitat envs
        with open(os.path.join(results_dir, "vln_spaces.pkl"), "wb") as f:
            pickle.dump(
                {
                    "observation_space": observation_space,
                    "action_space": env.action_space,
                },
                f,
            )

        with lmdb.open(results_dir, map_size=int(1e12)) as lmdb_env:
            start_id = lmdb_env.stat()["entries"]
            txn = lmdb_env.begin(write=True)

            while len(stats_episodes) < num_eps:
                ep_id = env.current_episode.episode_id
                assert ep_id not in stats_episodes

                action, a_idx, geodesics = select_closest_subgoal(env, batch)

                episode["observations"].append(
                    {k: v.detach().cpu().numpy() for k, v in batch.items()}
                )
                episode["actions"].append(a_idx)
                episode["geodesics"].append(np.array(geodesics))

                obs, _, done, info = env.step(action)

                if done:
                    stats_episodes[ep_id] = info
                    obs = env.reset()
                    episode["distance_to_goal"] = info["distance_to_goal"]
                    episode["success"] = info["success"]
                    episode["spl"] = info["spl"]

                    # delete unneccessary observation keys
                    skip_sensors = [
                        "mapping_depth",
                        "radial_occupancy",
                        "radial_pred",
                    ]
                    episode["observations"] = [
                        {k: v for k, v in o.items() if k not in skip_sensors}
                        for o in episode["observations"]
                    ]
                    txn.put(
                        str(start_id + len(stats_episodes) - 1).encode(),
                        msgpack_numpy.packb(episode, use_bin_type=True),
                    )
                    txn.commit()
                    txn = lmdb_env.begin(write=True)

                    episode = {
                        "episode_id": env.current_episode.episode_id,
                        "observations": [],
                        "actions": [],
                        "geodesics": [],
                    }
                    pbar.update()

                batch = self.batch_obs([obs], skip_keys=skip_keys)

        aggregated_stats = {}
        num_episodes = len(stats_episodes)
        for k in next(iter(stats_episodes.values())).keys():
            aggregated_stats[k] = (
                sum(v[k] for v in stats_episodes.values()) / num_episodes
            )

        if config.EVAL.SAVE_RESULTS:
            with open(fname, "w") as f:
                json.dump(aggregated_stats, f, indent=4)
            with open(full_results_name, "w") as f:
                json.dump(stats_episodes, f, indent=4)

        logger.info(f"Episodes evaluated: {num_episodes}")
        for k, v in aggregated_stats.items():
            logger.info(f"{k}: {v:.6f}")
