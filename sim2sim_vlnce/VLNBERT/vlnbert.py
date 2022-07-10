"""Adapted from https://github.com/YicongHong/Recurrent-VLN-BERT"""

import torch
from habitat import Config
from torch import nn

from sim2sim_vlnce.VLNBERT.bert_utils import (
    BertEmbeddings,
    BertLayer,
    BertPooler,
    LXRTXLayer,
    VisionEncoder,
)
from transformers.pytorch_transformers.modeling_bert import (
    BertConfig,
    BertPreTrainedModel,
)


class VLNBert(BertPreTrainedModel):
    def __init__(self, config):
        super(VLNBert, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.pooler = BertPooler(config)

        self.img_dim = config.img_feature_dim  # 2176
        self.img_feature_type = config.img_feature_type  # ''
        self.vl_layers = config.vl_layers  # 4
        self.la_layers = config.la_layers  # 9
        self.lalayer = nn.ModuleList(
            [BertLayer(config) for _ in range(self.la_layers)]
        )
        self.addlayer = nn.ModuleList(
            [LXRTXLayer(config) for _ in range(self.vl_layers)]
        )

        self.vision_encoder = VisionEncoder(
            config.img_feature_dim, self.config
        )
        self.apply(self.init_weights)

    def forward(
        self,
        mode,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        lang_mask=None,
        vis_mask=None,
        position_ids=None,
        img_feats=None,
    ):

        attention_mask = lang_mask

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        assert mode in ["language", "visual"]

        if mode == "language":
            """LXMERT language branch (only perform  at initialization)"""
            embedding_output = self.embeddings(
                input_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
            )
            text_embeds = embedding_output

            for layer_module in self.lalayer:
                temp_output = layer_module(
                    text_embeds, extended_attention_mask
                )
                text_embeds = temp_output[0]

            sequence_output = text_embeds
            pooled_output = self.pooler(sequence_output)

            return pooled_output, sequence_output

        """visual mode: LXMERT visual branch"""

        img_embedding_output = self.vision_encoder(img_feats)

        extended_img_mask = vis_mask.unsqueeze(1).unsqueeze(2)
        extended_img_mask = extended_img_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_img_mask = (1.0 - extended_img_mask) * -10000.0
        img_mask = extended_img_mask

        text_embeds = input_ids
        text_mask = extended_attention_mask
        lang_output = text_embeds
        visn_output = img_embedding_output

        for tdx, layer_module in enumerate(self.addlayer):
            (
                lang_output,
                visn_output,
                language_attention_scores,
                visual_attention_scores,
            ) = layer_module(
                lang_output, text_mask, visn_output, img_mask, tdx
            )

        sequence_output = lang_output
        pooled_output = self.pooler(sequence_output)

        language_state_scores = language_attention_scores.mean(dim=1)
        visual_action_scores = visual_attention_scores.mean(dim=1)

        # weighted_feat
        language_attention_probs = nn.Softmax(dim=-1)(
            language_state_scores.clone()
        ).unsqueeze(-1)
        visual_attention_probs = nn.Softmax(dim=-1)(
            visual_action_scores.clone()
        ).unsqueeze(-1)

        attended_language = (
            language_attention_probs * text_embeds[:, 1:, :]
        ).sum(1)
        attended_visual = (visual_attention_probs * img_embedding_output).sum(
            1
        )

        return (
            pooled_output,
            visual_action_scores,
            attended_language,
            attended_visual,
        )


def build_vlnbert(model_config: Config) -> VLNBert:
    bert_config = BertConfig.from_pretrained("bert-base-uncased")
    bert_config.img_feature_dim = model_config.img_feature_dim
    bert_config.img_feature_type = model_config.img_feature_type
    bert_config.vl_layers = model_config.vl_layers
    bert_config.la_layers = model_config.la_layers
    bert_config.directions = model_config.directions

    return VLNBert.from_pretrained(model_config.pretrained, config=bert_config)
