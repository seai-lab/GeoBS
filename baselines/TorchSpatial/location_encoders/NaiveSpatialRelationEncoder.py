from ._common_imports import *
from ._cal_freq_list import _cal_freq_list

class NaiveSpatialRelationEncoder(nn.Module):
    """
    Given a list of (deltaX,deltaY), encode them using the position encoding function

    """

    def __init__(self, spa_embed_dim, extent, coord_dim=2, ffn=None, device="cuda"):
        """
        Args:
            spa_embed_dim: the output spatial relation embedding dimention
            coord_dim: the dimention of space, 2D, 3D, or other
            extent: (x_min, x_max, y_min, y_max)
        """
        super(NaiveSpatialRelationEncoder, self).__init__()
        self.spa_embed_dim = spa_embed_dim
        self.coord_dim = coord_dim
        self.extent = extent

        # self.post_linear = nn.Linear(self.coord_dim, self.spa_embed_dim)
        # nn.init.xavier_uniform(self.post_linear.weight)
        # self.dropout = nn.Dropout(p=dropout)

        # self.f_act = get_activation_function(f_act, "NaiveSpatialRelationEncoder")
        self.ffn = ffn

        self.device = device

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

        coords_mat = coord_normalize(coords, self.extent)

        # spr_embeds: shape (batch_size, num_context_pt, coord_dim)
        spr_embeds = torch.FloatTensor(coords_mat).to(self.device)

        # sprenc: shape (batch_size, num_context_pt, spa_embed_dim)
        # sprenc = torch.einsum("bnd,dk->bnk", (spr_embeds, self.post_mat))

        # sprenc = self.f_act(self.dropout(self.post_linear(spr_embeds)))
        if self.ffn is not None:
            return self.ffn(spr_embeds)
        else:
            return spr_embeds

        # return sprenc
