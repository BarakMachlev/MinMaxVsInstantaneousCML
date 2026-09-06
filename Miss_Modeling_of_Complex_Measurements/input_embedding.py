import torch
import torch.nn as nn
from pynncml import neural_networks
from pynncml.neural_networks.normalization import InputNormalization

###############################################################################################
######################### Universal Transformer - Instantaneous only #########################
###############################################################################################
'''
class InputEmbedding(nn.Module):
    def __init__(self, normalization_cfg: neural_networks.InputNormalizationConfig,
                 dynamic_input_size,
                 metadata_input_size,
                 d_model,
                 protocol_n_features,
                 metadata_n_features,
                 num_protocols
                 ):
        super().__init__()
        self.normalization = InputNormalization(normalization_cfg)
        self.protocol_n_features = protocol_n_features
        self.metadata_n_features = metadata_n_features
        self.dynamic_n_features = d_model - protocol_n_features - metadata_n_features
        assert self.dynamic_n_features > 0, "d_model too small for protocol+metadata features"

        # dynamic projections
        self.dynamic_linear = nn.Linear(dynamic_input_size, self.dynamic_n_features)  
        
        # metadata projection
        self.metadata_linear = nn.Linear(metadata_input_size, self.metadata_n_features)

        # protocol embedding
        self.protocol_emb = nn.Embedding(num_protocols, self.protocol_n_features)

    def forward(self, dynamic_data, metadata, protocol_id):
        """
        dynamic_data: [B, T, 180]
        metadata:     [B, 2]
        protocol_id:  [B]
        Returns:      [B, T, d_model]
        """
        B, T, _ = dynamic_data.shape

        input_tensor, input_meta_tensor = self.normalization(dynamic_data, metadata)
        
        # ---- dynamic data projection ----
        dyn_emb = self.dynamic_linear(input_tensor)   # [B, T, dynamic_n_features]

        # ---- metadata projection ----
        meta_emb = self.metadata_linear(input_meta_tensor)  # [B, metadata_n_features]
        meta_emb = meta_emb.unsqueeze(1).expand(-1, T, -1)  # [B, T, metadata_n_features]

        # ---- protocol embedding ----
        proto_emb = self.protocol_emb(protocol_id.long())  # [B, protocol_n_features]
        proto_emb = proto_emb.unsqueeze(1).expand(-1, T, -1)  # [B, T, protocol_n_features]

        # Concatenate along feature dimension
        x = torch.cat([proto_emb, dyn_emb, meta_emb], dim=-1)  # [B, T, d_model]
        return x
'''
###############################################################################################
###############################################################################################
###############################################################################################

###############################################################################################
###################### Signal levels Generalized Universal Transformer 3-FC ####################
###############################################################################################

class InputEmbedding(nn.Module):
    def __init__(self, normalization_cfg: neural_networks.InputNormalizationConfig,
                dynamic_input_size,
                metadata_input_size,
                d_model,
                protocol_n_features,
                metadata_n_features,
                num_protocols):
        super().__init__()

        self.normalization = InputNormalization(normalization_cfg)

        self.protocol_n_features = protocol_n_features
        self.metadata_n_features = metadata_n_features
        self.dynamic_n_features = d_model - protocol_n_features - metadata_n_features

        assert self.dynamic_n_features > 0, "d_model too small for protocol+metadata features"

        # Three dynamic projections for RSL + TSL input
        self.dynamic_linear_inst = nn.Linear(dynamic_input_size, self.dynamic_n_features)  # 180 -> dyn
        self.dynamic_linear_avg = nn.Linear(2, self.dynamic_n_features)                   # [avg_rsl, avg_tsl]
        self.dynamic_linear_mm = nn.Linear(4, self.dynamic_n_features)                    # [max_rsl, min_rsl, max_tsl, min_tsl]

        # Metadata projection
        self.metadata_linear = nn.Linear(metadata_input_size, self.metadata_n_features)

        # Protocol embedding
        self.protocol_emb = nn.Embedding(num_protocols, self.protocol_n_features)

    def forward(self, dynamic_data, metadata, protocol_id):
        """
        dynamic_data: [B, T, 180]
        metadata:     [B, 2]
        protocol_id:  [B]
        returns:      [B, T, d_model]
        """

        B, T, _ = dynamic_data.shape

        input_tensor, input_meta_tensor = self.normalization(dynamic_data, metadata)

        protocol_id = protocol_id.long()

        dyn_emb = torch.zeros(
            B, T, self.dynamic_n_features,
            device=input_tensor.device,
            dtype=input_tensor.dtype
        )

        # Masks according to PROTOCOL_MAP
        inst_mask = protocol_id <= 11
        mm_mask = protocol_id == 12
        avg_mask = protocol_id == 13

        # Instantaneous protocols: full 180 samples
        if inst_mask.any():
            dyn_emb[inst_mask] = self.dynamic_linear_inst(input_tensor[inst_mask])

        # Average protocol: avg RSL + avg TSL
        if avg_mask.any():
            avg_input = torch.stack(
                [
                    input_tensor[avg_mask, :, 0],    # avg_rsl
                    input_tensor[avg_mask, :, 90],   # avg_tsl
                ],
                dim=-1
            )  # [N_avg, T, 2]

            dyn_emb[avg_mask] = self.dynamic_linear_avg(avg_input)

        # Min-max protocol: max/min RSL + max/min TSL
        if mm_mask.any():
            mm_input = torch.stack(
                [
                    input_tensor[mm_mask, :, 0],      # max_rsl
                    input_tensor[mm_mask, :, 45],     # min_rsl
                    input_tensor[mm_mask, :, 90],     # max_tsl
                    input_tensor[mm_mask, :, 135],    # min_tsl
                ],
                dim=-1
            )  # [N_mm, T, 4]

            dyn_emb[mm_mask] = self.dynamic_linear_mm(mm_input)

        # Metadata embedding
        meta_emb = self.metadata_linear(input_meta_tensor)
        meta_emb = meta_emb.unsqueeze(1).expand(-1, T, -1)

        # Protocol embedding
        proto_emb = self.protocol_emb(protocol_id)
        proto_emb = proto_emb.unsqueeze(1).expand(-1, T, -1)

        # Final embedding
        x = torch.cat([proto_emb, dyn_emb, meta_emb], dim=-1)

        return x

###############################################################################################
###############################################################################################
###############################################################################################