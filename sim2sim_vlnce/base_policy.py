import abc
from typing import Any, Tuple

import torch
from habitat_baselines.rl.ppo.policy import Policy
from torch import Size, Tensor

from sim2sim_vlnce.Sim2Sim.vln_action_mapping import (
    create_candidate_features,
    idx_to_action,
)


class CustomFixedCategorical(torch.distributions.Categorical):
    """Same as the CustomFixedCategorical in hab-lab, but renames log_probs
    to log_prob. All the torch distributions use log_prob.
    """

    def sample(
        self, sample_shape: Size = torch.Size()  # noqa: B008
    ) -> Tensor:
        return super().sample(sample_shape).unsqueeze(-1)

    def log_prob(self, actions: Tensor) -> Tensor:
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class VLNPolicy(Policy, metaclass=abc.ABCMeta):
    def __init__(self, net, dim_actions):
        """Defines an imitation learning policy as having functions act() and
        build_distribution().
        """
        super(Policy, self).__init__()
        self.net = net
        self.dim_actions = dim_actions
        self.action_distribution = None
        self.critic = None

    def forward(self, *x):
        raise NotImplementedError

    def encode_instruction(self, tokens, mask) -> Tuple[Tensor, Tensor]:
        """Generates the first hidden state vector h_t and encodes each
        instruction token. Call once for episode initialization.

        Returns:
            h_t (Tensor): [B x hidden_size]
            instruction_features (Tensor): [B x max_len x hidden_size]
        """
        return self.net.vln_bert("language", tokens, lang_mask=mask)

    def act(
        self,
        observations,
        h_t,
        instruction_features,
        instruction_mask,
        deterministic=False,
    ):
        instruction_features = torch.cat(
            (h_t.unsqueeze(1), instruction_features[:, 1:, :]), dim=1
        )
        (
            vis_features,
            coordinates,
            vis_mask,
        ) = create_candidate_features(observations)

        h_t, action_logit = self.net(
            instruction_features=instruction_features,
            attention_mask=torch.cat((instruction_mask, vis_mask), dim=1),
            lang_mask=instruction_mask,
            vis_mask=vis_mask,
            cand_feats=vis_features,
            action_feats=observations["mp3d_action_angle_feature"],
        )

        # Mask candidate logits that have no associated action
        action_logit.masked_fill(vis_mask, -float("inf"))
        distribution = CustomFixedCategorical(logits=action_logit)

        if deterministic:
            action_idx = distribution.mode()
        else:
            action_idx = distribution.sample()

        return h_t, idx_to_action(action_idx, coordinates)

    def get_value(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def evaluate_actions(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def build_distribution(
        self,
        observations,
        h_t,
        instruction_features,
        instruction_mask,
    ) -> CustomFixedCategorical:
        instruction_features = torch.cat(
            (h_t.unsqueeze(1), instruction_features[:, 1:, :]), dim=1
        )
        vis_features, _, vis_mask = create_candidate_features(observations)

        h_t, action_logit = self.net(
            instruction_features=instruction_features,
            attention_mask=torch.cat((instruction_mask, vis_mask), dim=1),
            lang_mask=instruction_mask,
            vis_mask=vis_mask,
            cand_feats=vis_features,
            action_feats=observations["mp3d_action_angle_feature"],
        )

        # Mask candidate logits that have no associated action
        action_logit.masked_fill(vis_mask, -float("inf"))
        return h_t, CustomFixedCategorical(logits=action_logit)
