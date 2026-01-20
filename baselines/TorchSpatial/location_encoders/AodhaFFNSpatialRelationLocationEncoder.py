from ._common_imports import *
from ._cal_freq_list import _cal_freq_list

from .AodhaFFTSpatialRelationPositionEncoder import AodhaFFTSpatialRelationPositionEncoder

class AodhaFFNSpatialRelationLocationEncoder(LocationEncoder):
    def __init__(
        self,
        spa_embed_dim,
        extent,
        coord_dim=2,
        do_pos_enc=False,
        do_global_pos_enc=True,
        device="cuda",
        ffn_act="relu",
        ffn_num_hidden_layers=1,
        ffn_dropout_rate=0.5,
        ffn_hidden_dim=256,
        ffn_use_layernormalize=True,
        ffn_skip_connection=True,
        ffn_context_str="AodhaFFTSpatialRelationEncoder",
    ):
        super().__init__(spa_embed_dim, coord_dim, device)
        self.extent = extent
        self.do_pos_enc = do_pos_enc
        self.do_global_pos_enc = do_global_pos_enc
        self.ffn_act = ffn_act
        self.ffn_num_hidden_layers = ffn_num_hidden_layers
        self.ffn_dropout_rate = ffn_dropout_rate
        self.ffn_hidden_dim = ffn_hidden_dim
        self.ffn_use_layernormalize = ffn_use_layernormalize
        self.ffn_skip_connection = ffn_skip_connection

        self.position_encoder = AodhaFFTSpatialRelationPositionEncoder(
            coord_dim=coord_dim,
            extent=extent,
            do_pos_enc=do_pos_enc,
            do_global_pos_enc=do_global_pos_enc,
            device=device,
        )
        self.ffn = MultiLayerFeedForwardNN(
            input_dim=self.position_encoder.pos_enc_output_dim,
            output_dim=self.spa_embed_dim,
            num_hidden_layers=self.ffn_num_hidden_layers,
            dropout_rate=ffn_dropout_rate,
            hidden_dim=self.ffn_hidden_dim,
            activation=self.ffn_act,
            use_layernormalize=self.ffn_use_layernormalize,
            skip_connection=ffn_skip_connection,
            context_str=ffn_context_str,
        )

    def forward(self, coords):
        spr_embeds = self.position_encoder(coords)
        sprenc = self.ffn(spr_embeds)

        return sprenc
