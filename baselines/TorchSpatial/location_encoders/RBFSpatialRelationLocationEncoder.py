from ._common_imports import *
from ._cal_freq_list import _cal_freq_list

from .RBFSpatialRelationPositionEncoder import RBFSpatialRelationPositionEncoder

class RBFSpatialRelationLocationEncoder(LocationEncoder):
    def __init__(
        self,
        spa_embed_dim,
        train_locs,
        model_type,
        coord_dim=2,
        device="cuda",
        num_rbf_anchor_pts=100,
        rbf_kernel_size=10e2,
        rbf_kernel_size_ratio=0.0,
        max_radius=10000,
        rbf_anchor_pt_ids=None,
        ffn_act="relu",
        ffn_num_hidden_layers=1,
        ffn_dropout_rate=0.5,
        ffn_hidden_dim=256,
        ffn_use_layernormalize=True,
        ffn_skip_connection=True,
        ffn_context_str="RBFSpatialRelationEncoder",
    ):
        super().__init__(spa_embed_dim, coord_dim, device)
        self.train_locs = train_locs
        self.model_type = model_type
        self.num_rbf_anchor_pts = num_rbf_anchor_pts
        self.rbf_kernel_size = rbf_kernel_size
        self.rbf_kernel_size_ratio = rbf_kernel_size_ratio
        self.max_radius = max_radius
        self.rbf_anchor_pt_ids = rbf_anchor_pt_ids
        self.ffn_act = ffn_act
        self.ffn_num_hidden_layers = ffn_num_hidden_layers
        self.ffn_dropout_rate = ffn_dropout_rate
        self.ffn_hidden_dim = ffn_hidden_dim
        self.ffn_use_layernormalize = ffn_use_layernormalize
        self.ffn_skip_connection = ffn_skip_connection
        self.ffn_context_str = ffn_context_str

        self.position_encoder = RBFSpatialRelationPositionEncoder(
            model_type=model_type,
            train_locs=train_locs,
            coord_dim=coord_dim,
            num_rbf_anchor_pts=num_rbf_anchor_pts,
            rbf_kernel_size=rbf_kernel_size,
            rbf_kernel_size_ratio=rbf_kernel_size_ratio,
            max_radius=max_radius,
            rbf_anchor_pt_ids=rbf_anchor_pt_ids,
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
            skip_connection=self.ffn_skip_connection,
            context_str=self.ffn_context_str,
        )

    def forward(self, coords):
        spr_embeds = self.position_encoder(coords)
        sprenc = self.ffn(spr_embeds)

        return sprenc
