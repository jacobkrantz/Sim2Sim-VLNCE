from typing import Any, Dict, Tuple

import torch
from torch import Tensor


def create_candidate_features(observations) -> Tuple[Tensor, Tensor]:
    """Extracts candidate features and coordinates. Creates a mask for
    ignoring padded candidates. Observations must have `num_candidates`,
    `candidate_features`, and `candidate_coordinates`.

    Returns:
        candidate_features (Tensor): viewpoints with candidates. Size:
            [B, max(num_candidates), feature_size]
        candidate_coordinates (Tensor): (x,y,z) habitat coordinates. Stop
            is denoted [0,0,0]. Size: [B, max(num_candidates), 3]
        visual_temp_mask (Tensor): masks padded viewpointIds. Size:
            [B, max(num_candidates)]
    """

    def to_mask(int_array: Tensor) -> Tensor:
        batch_size = int_array.shape[0]
        mask_size = int_array.max().item()
        mask = torch.ones(
            batch_size, mask_size, dtype=torch.bool, device=int_array.device
        )
        for i in range(batch_size):
            mask[i, int_array[i] :] = False
        return mask

    prune_idx = observations["num_candidates"].max().item()
    features = observations["candidate_features"][:, :prune_idx]
    coordinates = observations["candidate_coordinates"][:, :prune_idx]
    mask = to_mask(observations["num_candidates"])

    return features, coordinates, mask


def idx_to_action(
    action_idx: Tensor, candidate_coordinates: Tensor
) -> Dict[str, Any]:
    actions = []
    batch_size = action_idx.shape[0]
    for i in range(batch_size):
        position = candidate_coordinates[i][action_idx[i]].squeeze(0)

        # map coordinates [0., 0., 0.] to STOP. This sn't great for global
        # coordinates, but will be fine for relative.
        if not position.to(dtype=torch.bool).any().item():
            action = "STOP"
        else:
            action = {
                "action": "TELEPORT_AND_ROTATE",
                "action_args": {"position": position.cpu().numpy()},
            }
        actions.append({"action": action})
    return actions
