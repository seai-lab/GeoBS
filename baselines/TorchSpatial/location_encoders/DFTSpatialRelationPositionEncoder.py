from ._common_imports import *
from ._cal_freq_list import _cal_freq_list

class DFTSpatialRelationPositionEncoder(PositionEncoder):
    """
    Given a list of (deltaX,deltaY), encode them using the position encoding function

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
            spa_embed_dim: the output spatial relation embedding dimention
            coord_dim: the dimention of space, 2D, 3D, or other
            frequency_num: the number of different sinusoidal with different frequencies/wavelengths
            max_radius: the largest context radius this model can handle
        """
        super().__init__(coord_dim=coord_dim, device=device)
        self.coord_dim = coord_dim
        self.frequency_num = frequency_num
        self.freq_init = freq_init
        self.max_radius = max_radius
        self.min_radius = min_radius
        # the frequence we use for each block, alpha in ICLR paper
        self.cal_freq_list()
        self.cal_freq_mat()

        self.pos_enc_output_dim = self.cal_input_dim()

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

    def cal_input_dim(self):
        # compute the dimention of the encoded spatial relation embedding
        return (
            self.frequency_num * 4 + 4 * self.frequency_num * self.frequency_num
        )  # int(self.coord_dim * self.frequency_num * 2)

    def cal_freq_list(self):
        if self.freq_init == "random":
            # the frequence we use for each block, alpha in ICLR paper
            # self.freq_list shape: (frequency_num)
            self.freq_list = (
                np.random.random(size=[self.frequency_num]) * self.max_radius
            )
        elif self.freq_init == "geometric":
            self.freq_list = []
            for cur_freq in range(self.frequency_num):
                base = 1.0 / (
                    np.power(self.max_radius, cur_freq *
                             1.0 / (self.frequency_num - 1))
                )
                self.freq_list.append(base)

            self.freq_list = np.asarray(self.freq_list)

    def cal_freq_mat(self):
        # freq_mat shape: (frequency_num, 1)
        freq_mat = np.expand_dims(self.freq_list, axis=1)
        # self.freq_mat shape: (frequency_num, 1)
        self.freq_mat = freq_mat

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

        # convert to radius
        coords_mat = coords_mat * math.pi / 180

        # lon, lat: shape (batch_size, num_context_pt, 1, frequency_num, 1)
        lon_single = np.expand_dims(coords_mat[:, :, 0, :, :], axis=2)
        lat_single = np.expand_dims(coords_mat[:, :, 1, :, :], axis=2)

        # make sinuniod function
        # lon_sin, lon_cos: shape (batch_size, num_context_pt, 1, frequency_num, 1)
        lon_single_sin = np.sin(lon_single)
        lon_single_cos = np.cos(lon_single)

        # lat_sin, lat_cos: shape (batch_size, num_context_pt, 1, frequency_num, 1)
        lat_single_sin = np.sin(lat_single)
        lat_single_cos = np.cos(lat_single)

        # spr_embeds: shape (batch_size, num_context_pt, 2, frequency_num, 1)
        spr_embeds = coords_mat * self.freq_mat

        # lon, lat: shape (batch_size, num_context_pt, 1, frequency_num, 1)
        lon = np.expand_dims(spr_embeds[:, :, 0, :, :], axis=2)
        lat = np.expand_dims(spr_embeds[:, :, 1, :, :], axis=2)

        # make sinuniod function
        # lon_sin, lon_cos: shape (batch_size, num_context_pt, 1, frequency_num, 1)
        lon_sin = np.sin(lon)
        lon_cos = np.cos(lon)

        # lat_sin, lat_cos: shape (batch_size, num_context_pt, 1, frequency_num, 1)
        lat_sin = np.sin(lat)
        lat_cos = np.cos(lat)

        # lon_feq: shape (batch_size, num_context_pt, 1, 2*frequency_num, 1)
        lon_feq = np.concatenate([lon_sin, lon_cos], axis=-2)

        # lat_feq: shape (batch_size, num_context_pt, 1, 2*frequency_num, 1)
        lat_feq = np.concatenate([lat_sin, lat_cos], axis=-2)

        # lat_feq_t: shape (batch_size, num_context_pt, 1, 1, 2*frequency_num)
        lat_feq_t = lat_feq.transpose(0, 1, 2, 4, 3)

        # coord_freq: shape (batch_size, num_context_pt, 1, 2*frequency_num, 2*frequency_num)
        coord_freq = np.einsum("abcde,abcek->abcdk", lon_feq, lat_feq_t)

        # coord_freq_: shape (batch_size, num_context_pt, 2*frequency_num * 2*frequency_num)
        coord_freq_ = np.reshape(coord_freq, (batch_size, num_context_pt, -1))

        # coord_1: shape (batch_size, num_context_pt, 1, frequency_num, 4)
        coord_1 = np.concatenate([lat_sin, lat_cos, lon_sin, lon_cos], axis=-1)

        # coord_1_: shape (batch_size, num_context_pt, frequency_num * 4)
        coord_1_ = np.reshape(coord_1, (batch_size, num_context_pt, -1))

        # spr_embeds_: shape (batch_size, num_context_pt, frequency_num * 4 + 4*frequency_num ^^2)
        spr_embeds_ = np.concatenate([coord_freq_, coord_1_], axis=-1)

        # # make sinuniod function
        # # sin for 2i, cos for 2i+1
        # # spr_embeds: (batch_size, num_context_pt, 2*frequency_num*2=pos_enc_output_dim)
        # spr_embeds[:, :, :, :, 0::2] = np.sin(spr_embeds[:, :, :, :, 0::2])  # dim 2i
        # spr_embeds[:, :, :, :, 1::2] = np.cos(spr_embeds[:, :, :, :, 1::2])  # dim 2i+1

        # (batch_size, num_context_pt, frequency_num*3)
        spr_embeds = np.reshape(spr_embeds_, (batch_size, num_context_pt, -1))

        return spr_embeds

    def forward(self, coords):
        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape (batch_size, num_context_pt, coord_dim)
        Return:
            sprenc: Tensor shape (batch_size, num_context_pt, spa_embed_dim)
        """

        spr_embeds = self.make_output_embeds(coords)

        # spr_embeds: shape (batch_size, num_context_pt, pos_enc_output_dim)
        spr_embeds = torch.FloatTensor(spr_embeds).to(self.device)

        # sprenc: shape (batch_size, num_context_pt, spa_embed_dim)
        # sprenc = torch.einsum("bnd,dk->bnk", (spr_embeds, self.post_mat))
        # sprenc = self.f_act(self.dropout(self.post_linear(spr_embeds)))

        # return sprenc
        return spr_embeds
