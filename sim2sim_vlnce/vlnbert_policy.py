"""Adapted from https://github.com/YicongHong/Recurrent-VLN-BERT"""

import torch
from gym.spaces.space import Space
from habitat import Config
from habitat_baselines.common.baseline_registry import BaselineRegistry
from torch import nn

from sim2sim_vlnce.base_policy import VLNPolicy
from sim2sim_vlnce.VLNBERT.vlnbert import build_vlnbert


@BaselineRegistry.register_policy
class VLNBERTPolicy(VLNPolicy):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        model_config: Config,
    ) -> None:
        super().__init__(
            VLNBertNet(
                model_config=model_config,
            ),
            action_space.n,
        )

    @classmethod
    def from_config(
        cls, config: Config, observation_space: Space, action_space: Space
    ):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            model_config=config.MODEL,
        )


class VLNBertNet(nn.Module):
    def __init__(self, model_config):
        super(VLNBertNet, self).__init__()
        self._angle_feature_size = model_config.VLNBERT.angle_feature_size

        self.vln_bert = build_vlnbert(model_config.VLNBERT)

        hidden_size = self.vln_bert.config.hidden_size
        layer_norm_eps = self.vln_bert.config.layer_norm_eps

        self.action_state_project = nn.Sequential(
            nn.Linear(hidden_size + self._angle_feature_size, hidden_size),
            nn.Tanh(),
        )
        self.action_LayerNorm = BertLayerNorm(hidden_size, eps=layer_norm_eps)

        self.drop_env = nn.Dropout(p=model_config.VLNBERT.dropout_p)
        self.img_projection = nn.Linear(
            model_config.VLNBERT.img_feature_dim, hidden_size, bias=True
        )
        self.cand_LayerNorm = BertLayerNorm(hidden_size, eps=layer_norm_eps)

        self.vis_lang_LayerNorm = BertLayerNorm(
            hidden_size, eps=layer_norm_eps
        )
        self.state_proj = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.state_LayerNorm = BertLayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(
        self,
        instruction_features,
        attention_mask,
        lang_mask,
        vis_mask,
        cand_feats,
        action_feats,
    ):
        state_action_embed = torch.cat(
            (instruction_features[:, 0, :], action_feats), 1
        )
        state_with_action = self.action_state_project(state_action_embed)
        state_with_action = self.action_LayerNorm(state_with_action)
        state_feats = torch.cat(
            (state_with_action.unsqueeze(1), instruction_features[:, 1:, :]),
            dim=1,
        )

        # changed dropout application. was getting an in-place error before.
        env_feats = cand_feats[:, :, : -self._angle_feature_size]
        angle_feats = cand_feats[:, :, -self._angle_feature_size :]
        env_feats = self.drop_env(env_feats)
        cand_feats = torch.cat([env_feats, angle_feats], dim=2)

        # logit is the attention scores over the candidate features
        self.vln_bert.config.directions = cand_feats.shape[1]

        h_t, logit, attended_language, attended_visual = self.vln_bert(
            "visual",
            state_feats,
            attention_mask=attention_mask,
            lang_mask=lang_mask,
            vis_mask=vis_mask,
            img_feats=cand_feats,
        )

        # update agent's state, unify history, language and vision by elementwise product
        vis_lang_feat = self.vis_lang_LayerNorm(
            attended_language * attended_visual
        )
        state_output = torch.cat((h_t, vis_lang_feat), dim=-1)
        state_proj = self.state_proj(state_output)
        state_proj = self.state_LayerNorm(state_proj)

        return state_proj, logit


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root)."""
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
