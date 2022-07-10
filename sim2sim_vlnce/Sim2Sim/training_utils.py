import random
from collections import defaultdict
from typing import List

import lmdb
import msgpack_numpy
import numpy as np
import torch
import torch.nn.functional as F

from habitat_extensions.actions import TeleportAndRotateAction


def collate_fn(batch):
    def _pad_helper(t, max_len, fill_val=0):
        pad_amount = max_len - t.size(0)
        if pad_amount == 0:
            return t

        pad = torch.full_like(t[0:1], fill_val).expand(
            pad_amount, *t.size()[1:]
        )
        return torch.cat([t, pad], dim=0)

    def prep_observations(obs, B, max_traj_len):
        new_obs = defaultdict(list)
        for sensor in obs[0]:
            for bid in range(B):
                new_obs[sensor].append(obs[bid][sensor])
        obs = new_obs

        for bid in range(B):
            for sensor in obs:
                obs[sensor][bid] = _pad_helper(
                    obs[sensor][bid], max_traj_len, fill_val=1.0
                )
        obs = {k: torch.stack(v, dim=1) for k, v in obs.items()}
        return obs

    def prep_mask(B, max_traj_len, trajectory_lengths):
        step_mask = torch.ones(max_traj_len, B, dtype=torch.uint8)
        for bid, traj_len in enumerate(trajectory_lengths):
            step_mask[traj_len:, bid] *= 0
        return step_mask

    def prep_actions(actions, B, max_traj_len):
        if isinstance(actions[0], torch.Tensor):
            actions = torch.stack(
                [_pad_helper(actions[bid], max_traj_len) for bid in range(B)],
                dim=1,
            )
        else:
            # soft labels: want [traj_len] [batch, num_candidates]
            assert isinstance(actions[0], List)
            soft_labels = []
            for t in range(max_traj_len):
                # get max num candidates in this step
                max_candidates = []
                for bid in range(B):
                    if t >= len(actions[bid]):  # no step, ep over
                        continue
                    max_candidates.append(actions[bid][t].shape[0])
                max_candidates = max(max_candidates)

                for bid in range(B):
                    if len(actions[bid]) <= t:
                        # pad steps
                        actions[bid].append(torch.zeros(max_candidates))
                    else:
                        # set to zero then fill with existing data
                        num_candidates = actions[bid][t].shape[0]
                        if num_candidates == max_candidates:
                            continue
                        # pad candidates
                        x = torch.zeros(max_candidates)
                        x[:num_candidates] = actions[bid][t]
                        actions[bid][t] = x

                soft_labels.append(
                    torch.stack([a[t] for a in actions]).to(
                        dtype=torch.float32
                    )
                )

            actions = soft_labels

        return actions

    transposed = list(zip(*batch))
    obs = list(transposed[0])
    actions = list(transposed[1])
    B = len(obs)
    k = next(iter(obs[0].keys()))
    trajectory_lengths = [obs[i][k].shape[0] for i in range(B)]
    max_traj_len = max(trajectory_lengths)

    obs = prep_observations(obs, B, max_traj_len)
    step_mask = prep_mask(B, max_traj_len, trajectory_lengths)
    actions = prep_actions(actions, B, max_traj_len)
    return obs, step_mask, actions


def _block_shuffle(lst, block_size):
    blocks = [lst[i : i + block_size] for i in range(0, len(lst), block_size)]
    random.shuffle(blocks)
    return [ele for block in blocks for ele in block]


class VLNDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        lmdb_features_dir,
        lmdb_map_size=1e12,
        batch_size=1,
        skip_failures=True,
        soft_labels=False,
        min_spl=0.0,
        max_nav_error=float("inf"),
    ):
        super().__init__()
        self.lmdb_features_dir = lmdb_features_dir
        self.lmdb_map_size = int(lmdb_map_size)
        self.preload_size = batch_size * 100
        self._preload = []
        self.batch_size = batch_size
        self.skip_failures = skip_failures
        self.soft_labels = soft_labels
        self.min_spl = min_spl
        self.max_nav_error = max_nav_error

        with lmdb.open(
            self.lmdb_features_dir,
            map_size=self.lmdb_map_size,
            readonly=True,
            lock=False,
        ) as lmdb_env:
            self.length = lmdb_env.stat()["entries"]

    def generate_soft_labels(
        self,
        geodesics: List[np.ndarray],
    ) -> List[torch.Tensor]:
        """Determine a target probability distribution over subgoal candidates.
        Use soft labels such that probability is distributed proportionally
        amongst the subgoals that decrease distance to goal.

        Args:
            geodesics (List[np.ndarray]): geodesic distance between each
                subgoal candidate and the goal. geodesics[i][-1] is the
                distance of the stop action (current distance).

        Returns:
            List[torch.Tensor]: list of prediction target distributions.
        """
        soft_labels = []
        for glist in geodesics:
            x = glist[-1] - glist  # neg. delta in geodesic distance to goal
            x[x < 0] = 0.0  # zero probability to subgoals that increase dist
            if np.isclose(x.sum(), 0.0):
                x[-1] = 1.0  # if no subgoal decreases distance, stop.
            x /= x.sum()  # normalize sum to 1
            assert np.isclose(x.sum(), 1.0)
            soft_labels.append(torch.from_numpy(np.copy(x)))

        return soft_labels

    def __next__(self):
        if len(self._preload) == 0:
            if len(self.load_ordering) == 0:
                raise StopIteration

            new_preload = []
            lengths = []
            with lmdb.open(
                self.lmdb_features_dir,
                map_size=self.lmdb_map_size,
                readonly=True,
                lock=False,
            ) as lmdb_env, lmdb_env.begin(buffers=True) as txn:
                for _ in range(self.preload_size):
                    if len(self.load_ordering) == 0:
                        break

                    episode = msgpack_numpy.unpackb(
                        txn.get(str(self.load_ordering.pop()).encode()),
                        raw=False,
                    )

                    # optionally skip failure episodes or inefficient paths
                    if self.skip_failures and not episode["success"]:
                        continue
                    if episode["spl"] < self.min_spl:
                        continue
                    if episode["distance_to_goal"] > self.max_nav_error:
                        continue

                    # transpose observations
                    obs_t = {k: [] for k in episode["observations"][0]}
                    for step_dict in episode["observations"]:
                        for k, v in step_dict.items():
                            obs_t[k].append(v)

                    obs_t = {
                        k: torch.from_numpy(np.copy(v)).squeeze(1)
                        for k, v in obs_t.items()
                    }
                    if self.soft_labels:
                        actions = self.generate_soft_labels(
                            episode["geodesics"]
                        )
                    else:
                        actions = torch.from_numpy(np.copy(episode["actions"]))

                    new_preload.append((obs_t, actions))
                    lengths.append(len(new_preload[-1][0]))

            sort_priority = list(range(len(lengths)))
            random.shuffle(sort_priority)

            sorted_ordering = list(range(len(lengths)))
            sorted_ordering.sort(key=lambda k: (lengths[k], sort_priority[k]))

            for idx in _block_shuffle(sorted_ordering, self.batch_size):
                self._preload.append(new_preload[idx])

        return self._preload.pop()

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start = 0
            end = self.length
        else:
            per_worker = int(np.ceil(self.length / worker_info.num_workers))

            start = per_worker * worker_info.id
            end = min(start + per_worker, self.length)

        # Reverse so we can use .pop()
        self.load_ordering = list(
            reversed(
                _block_shuffle(list(range(start, end)), self.preload_size)
            )
        )
        return self


def soft_cross_entropy(logits, targets):
    """Logits are raw logits. Targets are probablities. Returns the unreduced
    cross entropy loss. If using pytorch 1.10, use the built-in cross entropy
    loss which now supports soft labels.
    """
    if logits.shape != targets.shape:
        raise ValueError("logits and targets have different shapes")
    if len(logits.shape) != 2:
        raise ValueError("inputs must be 2D.")
    return -torch.sum(F.log_softmax(logits, dim=1) * targets, dim=1)


def update_agent(policy, obs, step_masks, actions, optimizer, criterion):
    optimizer.zero_grad()
    vln_instruction_mask = obs["vln_instruction_mask"][0]

    # don't train the instruction encoder
    with torch.no_grad():
        (h_t, instruction_features) = policy.encode_instruction(
            obs["vln_instruction"][0], vln_instruction_mask
        )

    episode_length = obs["vln_instruction"].shape[0]
    loss = 0.0
    for t in range(episode_length):
        h_t, distribution = policy.build_distribution(
            {k: v[t] for k, v in obs.items()},
            h_t,
            instruction_features,
            instruction_mask=vln_instruction_mask,
        )
        step_loss = criterion(distribution.logits, actions[t])

        # mask finished episodes
        loss += step_loss * step_masks[t]

    loss = loss.mean()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(policy.parameters(), 40.0)
    optimizer.step()
    return loss.item()


def select_closest_subgoal(env, batch):
    """Considering all subgoal candidates, select the one whose navigable 3D
    projection has the shortest geodesic distance to the goal.
    """
    sim = env._env.sim
    goal_position = env.current_episode.goals[0].position

    # stop action is considered a candidate
    num_candidates = batch["num_candidates"].item()
    assert num_candidates >= 1, "what happened to stop?"

    # determine current distance-to-goal
    current_geodesic = sim.geodesic_distance(
        sim.get_agent_state().position, goal_position
    )

    # determine expected distance-to-goal for each candidate. This is
    # "expected" because we are projecting the point to 3D, not
    # actually running a navigator.
    subgoal_geodesics = []
    subgoal_candidates = batch["candidate_coordinates"][0].cpu().numpy()
    for candidate_idx in range(num_candidates - 1):
        coordinates = TeleportAndRotateAction.filter_waypoint(
            sim, subgoal_candidates[candidate_idx]
        )[0]
        subgoal_geodesics.append(
            sim.geodesic_distance(coordinates, goal_position)
        )

    if len(subgoal_geodesics) == 0:
        # if there are no subgoals, stop
        action_idx = num_candidates - 1
        action = "STOP"
    elif current_geodesic <= min(subgoal_geodesics):
        # if no candidates are closer than the agent, stop.
        action_idx = num_candidates - 1
        action = "STOP"
    else:
        # there is a closer candidate to goal; select the closest.
        action_idx = np.argmin(subgoal_geodesics)
        action = {
            "action": "TELEPORT_AND_ROTATE",
            "action_args": {"position": subgoal_candidates[action_idx]},
        }
    subgoal_geodesics.append(current_geodesic)
    return action, action_idx, subgoal_geodesics
