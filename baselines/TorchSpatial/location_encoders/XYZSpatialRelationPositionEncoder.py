from ._common_imports import *
from ._cal_freq_list import _cal_freq_list

class XYZSpatialRelationPositionEncoder(PositionEncoder):
    """
    Given a list of (lon,lat), convert them to (x,y,z), and then encode them using the MLP

    """

    def __init__(self, coord_dim=2, device="cuda"):
        """
        Args:
            spa_embed_dim: the output spatial relation embedding dimention
            coord_dim: the dimention of space, 2D, 3D, or other
            extent: (x_min, x_max, y_min, y_max)
        """
        super().__init__(coord_dim=coord_dim, device=device)

        self.pos_enc_output_dim = 3  # self.cal_pos_enc_output_dim()

    def make_output_embeds(self, coords):
        if type(coords) == np.ndarray:
            assert self.coord_dim == np.shape(coords)[2]
            coords = list(coords)
        elif type(coords) == list:
            assert self.coord_dim == len(coords[0][0])
        else:
            raise Exception(
                "Unknown coords data type for GridCellSpatialRelationEncoder"
            )

        # (batch_size, num_context_pt, coord_dim)
        coords_mat = np.asarray(coords).astype(float)
        batch_size = coords_mat.shape[0]
        num_context_pt = coords_mat.shape[1]

        # lon: (batch_size, num_context_pt, 1), convert from degree to radius
        lon = np.deg2rad(coords_mat[:, :, :1])
        # lat: (batch_size, num_context_pt, 1), convert from degree to radius
        lat = np.deg2rad(coords_mat[:, :, 1:])

        # convert (lon, lat) to (x,y,z), assume in unit sphere with r = 1
        x = np.cos(lat) * np.cos(lon)
        y = np.cos(lat) * np.sin(lon)
        z = np.sin(lat)

        # spr_embeds: (batch_size, num_context_pt, 3)
        spr_embeds = np.concatenate((x, y, z), axis=-1)
        return spr_embeds

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
