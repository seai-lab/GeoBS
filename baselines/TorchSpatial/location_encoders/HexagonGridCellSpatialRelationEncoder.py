from ._common_imports import *
from ._cal_freq_list import _cal_freq_list

class HexagonGridCellSpatialRelationEncoder(nn.Module):
    """
    Given a list of (deltaX,deltaY), encode them using the position encoding function

    """

    def __init__(
        self,
        spa_embed_dim,
        coord_dim=2,
        frequency_num=16,
        max_radius=10000,
        dropout=0.5,
        f_act="sigmoid",
        device="cuda",
    ):
        """
        Args:
            spa_embed_dim: the output spatial relation embedding dimention
            coord_dim: the dimention of space, 2D, 3D, or other
            frequency_num: the number of different sinusoidal with different frequencies/wavelengths
            max_radius: the largest context radius this model can handle
        """
        super(HexagonGridCellSpatialRelationEncoder, self).__init__()
        self.frequency_num = frequency_num
        self.coord_dim = coord_dim
        self.max_radius = max_radius
        self.spa_embed_dim = spa_embed_dim

        self.pos_enc_output_dim = self.cal_input_dim()

        self.post_linear = nn.Linear(
            self.pos_enc_output_dim, self.spa_embed_dim)
        nn.init.xavier_uniform(self.post_linear.weight)
        self.dropout = nn.Dropout(p=dropout)

        # self.dropout_ = nn.Dropout(p=dropout)

        # self.post_mat = nn.Parameter(torch.FloatTensor(self.pos_enc_output_dim, self.spa_embed_dim))
        # init.xavier_uniform_(self.post_mat)
        # self.register_parameter("spa_postmat", self.post_mat)

        self.f_act = get_activation_function(
            f_act, "HexagonGridCellSpatialRelationEncoder"
        )

        self.device = device

    def cal_elementwise_angle(self, coord, cur_freq):
        """
        Args:
            coord: the deltaX or deltaY
            cur_freq: the frequency
        """
        return coord / (
            np.power(self.max_radius, cur_freq *
                     1.0 / (self.frequency_num - 1))
        )

    def cal_coord_embed(self, coords_tuple):
        embed = []
        for coord in coords_tuple:
            for cur_freq in range(self.frequency_num):
                embed.append(
                    math.sin(self.cal_elementwise_angle(coord, cur_freq)))
                embed.append(
                    math.sin(
                        self.cal_elementwise_angle(
                            coord, cur_freq) + math.pi * 2.0 / 3
                    )
                )
                embed.append(
                    math.sin(
                        self.cal_elementwise_angle(
                            coord, cur_freq) + math.pi * 4.0 / 3
                    )
                )
        # embed: shape (pos_enc_output_dim)
        return embed

    def cal_input_dim(self):
        # compute the dimention of the encoded spatial relation embedding
        return int(self.coord_dim * self.frequency_num * 3)

    def forward(self, coords):
        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape (batch_size, num_context_pt, coord_dim)
        Return:
            sprenc: Tensor shape (batch_size, num_context_pt, spa_embed_dim)
        """
        if type(coords) == np.ndarray:
            assert self.coord_dim == np.shape(coords)[2]
            coords = list(coords)
        elif type(coords) == list:
            assert self.coord_dim == len(coords[0][0])
        else:
            raise Exception(
                "Unknown coords data type for GridCellSpatialRelationEncoder"
            )

        # loop over all batches
        spr_embeds = []
        for cur_batch in coords:
            # loop over N context points
            cur_embeds = []
            for coords_tuple in cur_batch:
                cur_embeds.append(self.cal_coord_embed(coords_tuple))
            spr_embeds.append(cur_embeds)
        # spr_embeds: shape (batch_size, num_context_pt, pos_enc_output_dim)
        spr_embeds = torch.FloatTensor(spr_embeds).to(self.device)

        # sprenc: shape (batch_size, num_context_pt, spa_embed_dim)
        # sprenc = torch.einsum("bnd,dk->bnk", (spr_embeds, self.post_mat))
        sprenc = self.f_act(self.dropout(self.post_linear(spr_embeds)))

        return sprenc
