from ._common_imports import *
from ._cal_freq_list import _cal_freq_list

class GridCellSpatialRelationPositionEncoder(PositionEncoder):
    """
    Given a list of(deltaX, deltaY), encode them using the position encoding function
    """

    def __init__(
        self,
        coord_dim=2,
        frequency_num=16,
        max_radius=10000,
        min_radius=10,
        freq_init="geometric",
        device="cuda",
    ):
        """
        Args:
            coord_dim: the dimention of space, 2D, 3D, or other
            frequency_num: the number of different sinusoidal with different frequencies / wavelengths
            max_radius: the largest context radius this model can handle
        """
        super().__init__(coord_dim=coord_dim, device=device)
        self.frequency_num = frequency_num
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.freq_init = freq_init
        # the frequence we use for each block, alpha in ICLR paper
        self.cal_freq_list()
        self.cal_freq_mat()

        self.pos_enc_output_dim = self.cal_pos_enc_output_dim()

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
                    math.cos(self.cal_elementwise_angle(coord, cur_freq)))
        # embed: shape (pos_enc_output_dim)
        return embed

    def cal_pos_enc_output_dim(self):
        # compute the dimention of the encoded spatial relation embedding
        return int(self.coord_dim * self.frequency_num * 2)

    def cal_freq_list(self):
        self.freq_list = _cal_freq_list(
            self.freq_init, self.frequency_num, self.max_radius, self.min_radius
        )

    def cal_freq_mat(self):
        # freq_mat shape: (frequency_num, 1)
        freq_mat = np.expand_dims(self.freq_list, axis=1)
        # self.freq_mat shape: (frequency_num, 2)
        self.freq_mat = np.repeat(freq_mat, 2, axis=1)

    def make_output_embeds(self, coords):
        if type(coords) == np.ndarray:
            assert self.coord_dim == np.shape(coords)[2]
            coords = list(coords)
        elif type(coords) == list:  # noqa: E721
            assert self.coord_dim == len(coords[0][0])
        else:
            raise Exception(
                "Unknown coords data type for GridCellSpatialRelationEncoder"
            )

        # coords_mat: shape (batch_size, num_context_pt, 2)
        coords_mat = np.asarray(coords).astype(float)
        batch_size = coords_mat.shape[0]
        num_context_pt = coords_mat.shape[1]
        # coords_mat: shape (batch_size, num_context_pt, 2, 1)
        coords_mat = np.expand_dims(coords_mat, axis=3)
        # coords_mat: shape (batch_size, num_context_pt, 2, 1, 1)
        coords_mat = np.expand_dims(coords_mat, axis=4)
        # coords_mat: shape (batch_size, num_context_pt, 2, frequency_num, 1)
        coords_mat = np.repeat(coords_mat, self.frequency_num, axis=3)
        # coords_mat: shape (batch_size, num_context_pt, 2, frequency_num, 2)
        coords_mat = np.repeat(coords_mat, 2, axis=4)
        # spr_embeds: shape (batch_size, num_context_pt, 2, frequency_num, 2)
        spr_embeds = coords_mat * self.freq_mat

        # make sinuniod function
        # sin for 2i, cos for 2i+1
        # spr_embeds: (batch_size, num_context_pt, 2*frequency_num*2=pos_enc_output_dim)
        spr_embeds[:, :, :, :, 0::2] = np.sin(
            spr_embeds[:, :, :, :, 0::2])  # dim 2i
        spr_embeds[:, :, :, :, 1::2] = np.cos(
            spr_embeds[:, :, :, :, 1::2])  # dim 2i+1

        # (batch_size, num_context_pt, 2*frequency_num*2)
        spr_embeds = np.reshape(spr_embeds, (batch_size, num_context_pt, -1))

        return spr_embeds

    def forward(self, coords):
        """
        Given a list of coords(deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape(batch_size, num_context_pt, coord_dim)
        Return:
            sprenc: Tensor shape(batch_size, num_context_pt, position_embed_dim)
        """

        spr_embeds = self.make_output_embeds(coords)
        spr_embeds = torch.FloatTensor(spr_embeds).to(self.device)

        # sprenc: shape (batch_size, num_context_pt, spa_embed_dim)

        return spr_embeds
