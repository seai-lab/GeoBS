from ._common_imports import *
from ._cal_freq_list import _cal_freq_list

class NERFSpatialRelationPositionEncoder(PositionEncoder):
    """
    Given a list of (lon,lat), convert them to (x,y,z), and then encode them using the MLP

    """

    def __init__(self, coord_dim=2, frequency_num=16, freq_init="nerf", device="cuda"):
        """
        Args:
            coord_dim: the dimention of space, 2D, 3D, or other
        """
        super().__init__(coord_dim=coord_dim, device=device)
        self.coord_dim = coord_dim
        self.frequency_num = frequency_num
        self.freq_init = freq_init

        self.cal_freq_list()
        self.cal_freq_mat()

        self.pos_enc_output_dim = 6 * frequency_num

    def cal_freq_list(self):
        """
        compute according to NeRF position encoding,
        Equation 4 in https://arxiv.org/pdf/2003.08934.pdf
        2^{0}*pi, ..., 2^{L-1}*pi
        """
        # freq_list: shape (frequency_num)
        self.freq_list = _cal_freq_list(
            self.freq_init, self.frequency_num, None, None)

    def cal_freq_mat(self):
        # freq_mat shape: (1, frequency_num)
        freq_mat = np.expand_dims(self.freq_list, axis=0)
        # self.freq_mat shape: (3, frequency_num)
        self.freq_mat = np.repeat(freq_mat, 3, axis=0)

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

        # coords_mat: (batch_size, num_context_pt, 3)
        coords_mat = np.concatenate((x, y, z), axis=-1)
        # coords_mat: (batch_size, num_context_pt, 3, 1)
        coords_mat = np.expand_dims(coords_mat, axis=-1)
        # coords_mat: (batch_size, num_context_pt, 3, frequency_num)
        coords_mat = np.repeat(coords_mat, self.frequency_num, axis=-1)
        # coords_mat: (batch_size, num_context_pt, 3, frequency_num)
        coords_mat = coords_mat * self.freq_mat

        # coords_mat: (batch_size, num_context_pt, 6, frequency_num)
        spr_embeds = np.concatenate(
            [np.sin(coords_mat), np.cos(coords_mat)], axis=2)

        # spr_embeds: (batch_size, num_context_pt, 6*frequency_num)
        spr_embeds = np.reshape(spr_embeds, (batch_size, num_context_pt, -1))

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

        # spr_embeds: shape (batch_size, num_context_pt, pos_enc_output_dim = 6*frequency_num)
        spr_embeds = torch.FloatTensor(spr_embeds).to(self.device)

        return spr_embeds
