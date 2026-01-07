import torch
import torch.nn as nn


class PillarAttention(nn.Module):
    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_point_features = input_channels

        num_heads = self.model_cfg.NUM_HEADS
        dropout = self.model_cfg.get('DROPOUT', 0.0)
        ffn_channels = self.model_cfg.get('FFN_CHANNELS', input_channels)
        use_layer_norm = self.model_cfg.get('USE_LAYER_NORM', True)

        self.attn = nn.MultiheadAttention(
            embed_dim=input_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(input_channels) if use_layer_norm else None
        self.ffn = nn.Sequential(
            nn.Linear(input_channels, ffn_channels),
            nn.GELU(),
            nn.Linear(ffn_channels, input_channels),
        )
        self.norm2 = nn.LayerNorm(input_channels) if use_layer_norm else None

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                pillar_features: (num_pillars, C)
                voxel_coords: (num_pillars, 4)
        Returns:
            batch_dict:
                pillar_features: (num_pillars, C)
        """
        pillar_features = batch_dict['pillar_features']
        coords = batch_dict['voxel_coords']
        batch_size = coords[:, 0].max().int().item() + 1

        updated_features = torch.zeros_like(pillar_features)
        for batch_idx in range(batch_size):
            batch_mask = coords[:, 0] == batch_idx
            if not batch_mask.any():
                continue

            features = pillar_features[batch_mask].unsqueeze(0)
            attn_out, _ = self.attn(features, features, features, need_weights=False)
            if self.norm1 is not None:
                features = self.norm1(features + attn_out)
            else:
                features = features + attn_out

            ffn_out = self.ffn(features)
            if self.norm2 is not None:
                features = self.norm2(features + ffn_out)
            else:
                features = features + ffn_out

            updated_features[batch_mask] = features.squeeze(0)

        batch_dict['pillar_features'] = updated_features
        return batch_dict
