from ._common_imports import *
from ._cal_freq_list import _cal_freq_list
from ..utils.spherical_harmonics_ylm_numpy import get_positional_encoding

class SphericalHarmonicsSpatialRelationPositionEncoder(PositionEncoder):
    """
    Given a list of (lon,lat), convert them to (x,y,z), and then encode them using the MLP

    """

    def __init__(self, coord_dim=2, legendre_poly_num=8, device="cuda"):
        """
        Args:
            spa_embed_dim: the output spatial relation embedding dimention
            coord_dim: the dimention of space, 2D, 3D, or other
            extent: (x_min, x_max, y_min, y_max)
        """
        super().__init__(coord_dim=coord_dim, device=device)

        self.legendre_poly_num = legendre_poly_num
        self.pos_enc_output_dim = legendre_poly_num**2

    def make_output_embeds(self, coords):
        if type(coords) == np.ndarray:
            assert self.coord_dim == np.shape(coords)[2]
            coords = list(coords)
        elif type(coords) == list:
            assert self.coord_dim == len(coords[0][0])
        else:
            raise Exception(
                "Unknown coords data type for SphericalHarmonicsSpatialRelationEncoder"
            )

        # (batch_size, num_context_pt, coord_dim)
        coords_mat = np.asarray(coords).astype(float)
        batch_size = coords_mat.shape[0]
        num_context_pt = coords_mat.shape[1]

        # lon: (batch_size, num_context_pt, 1), convert from degree to radius
        lon = np.deg2rad(coords_mat[:, :, :1])
        # lat: (batch_size, num_context_pt, 1), convert from degree to radius
        lat = np.deg2rad(coords_mat[:, :, 1:])

        # spr_embeds: (batch_size, num_context_pt, legendre_poly_num**2)
        spr_embeds = get_positional_encoding(lon, lat, self.legendre_poly_num)
        return spr_embeds.reshape((-1, self.pos_enc_output_dim))

    def forward(self, coords):
        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape (batch_size, num_context_pt, coord_dim)
        Return:
            sprenc: Tensor shape (batch_size, num_context_pt, spa_embed_dim)
        """
        # (batch_size, num_context_pt, coord_dim)
        coords_mat = np.asarray(coords).astype(float)
        batch_size = coords_mat.shape[0]
        num_context_pt = coords_mat.shape[1]

        # spr_embeds: (batch_size, num_context_pt, 3)
        spr_embeds = self.make_output_embeds(coords)

        # spr_embeds: shape (batch_size, num_context_pt, pos_enc_output_dim = 3)
        spr_embeds = torch.FloatTensor(spr_embeds).to(self.device)

        return spr_embeds
