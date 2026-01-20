from ._common_imports import *
from ._cal_freq_list import _cal_freq_list

class AodhaFFTSpatialRelationPositionEncoder(PositionEncoder):
    """
    Given a list of (deltaX,deltaY),
    divide the space into grids, each point is using the grid embedding it falls into

    spa_enc_type == "geo_net_fft"
    """

    def __init__(
        self,
        extent,
        coord_dim=2,
        do_pos_enc=False,
        do_global_pos_enc=True,
        device="cuda",
    ):
        """
        Args:
            spa_embed_dim: the output spatial relation embedding dimention
            coord_dim: the dimention of space, 2D, 3D, or other
                if do_pos_enc == False:
                    coord_dim is the computed loc_feat according to get_model_input_feat_dim()
            extent: (x_min, x_max, y_min, y_max)
            do_pos_enc:     True  - we normalize the lat/lon, and [sin(pi*x), cos(pi*x), sin(pi*y), cos(pi*y)]
                            False - we assume the input is prenormalized coordinate features
            do_global_pos_enc:  if do_pos_enc == True:
                            True - lon/180 and lat/90
                            False - min-max normalize based on extent
            f_act: the final activation function, relu ot none
        """
        super().__init__(coord_dim=coord_dim, device=device)
        self.extent = extent
        self.coord_dim = coord_dim

        self.do_pos_enc = do_pos_enc
        self.do_global_pos_enc = do_global_pos_enc
        self.pos_enc_output_dim = 4

    def make_output_embeds(self, coords):
        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape (batch_size, num_context_pt=1, coord_dim)
        Return:
            sprenc: Tensor shape (batch_size, num_context_pt, pos_enc_output_dim)
        """
        if type(coords) == np.ndarray:
            assert self.coord_dim == np.shape(coords)[2]
            # coords = list(coords)
        elif type(coords) == list:
            assert self.coord_dim == len(coords[0][0])
        else:
            raise Exception(
                "Unknown coords data type for AodhaSpatialRelationEncoder")

        assert coords.shape[-1] == 2
        # coords: shape (batch_size, num_context_pt, 2)
        # coords_mat: shape (batch_size, num_context_pt, 2)
        coords_mat = coord_normalize(
            coords, self.extent, do_global=self.do_global_pos_enc
        )

        batch_size = coords_mat.shape[0]
        num_context_pt = coords_mat.shape[1]

        loc_sin = np.sin(math.pi * coords_mat)
        loc_cos = np.cos(math.pi * coords_mat)
        # spr_embeds: shape (batch_size, num_context_pt, 4)
        spr_embeds = np.concatenate((loc_sin, loc_cos), axis=-1)

        # spr_embeds: shape (batch_size, num_context_pt, 4)
        return spr_embeds

    def forward(self, coords):
        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape (batch_size, num_context_pt, coord_dim)
        Return:
            sprenc: Tensor shape (batch_size, num_context_pt, spa_embed_dim)
        """
        if self.do_pos_enc:
            # spr_embeds: shape (batch_size, num_context_pt, pos_enc_output_dim)
            spr_embeds = self.make_output_embeds(coords)
            # assert self.pos_enc_output_dim == np.shape(spr_embeds)[2]
        else:
            # spr_embeds: shape (batch_size, num_context_pt, pos_enc_output_dim)
            spr_embeds = coords

        # spr_embeds: shape (batch_size, num_context_pt, pos_enc_output_dim)
        spr_embeds = torch.FloatTensor(spr_embeds).to(self.device)

        return spr_embeds
