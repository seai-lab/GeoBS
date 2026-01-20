from ._common_imports import *
from ._cal_freq_list import _cal_freq_list

class RFFSpatialRelationPositionEncoder(PositionEncoder):
    """
    Random Fourier Feature
    Based on paper - Random Features for Large-Scale Kernel Machines
    https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf

    Given a list of (deltaX,deltaY), encode them using the position encoding function

    """

    def __init__(
        self,
        coord_dim=2,
        frequency_num=16,
        rbf_kernel_size=1.0,
        extent=None,
        device="cuda",
    ):
        """
        Args:
            spa_embed_dim: the output spatial relation embedding dimention
            extent: (x_min, x_max, y_min, y_max)
            coord_dim: the dimention of space, 2D, 3D, or other
            frequency_num: here, we understand it as the RFF embeding dimension before FNN
        """
        super().__init__(coord_dim=coord_dim, device=device)
        self.frequency_num = frequency_num
        self.rbf_kernel_size = rbf_kernel_size
        self.extent = extent

        self.generate_direction_vector()

        self.pos_enc_output_dim = self.frequency_num

    def generate_direction_vector(self):
        """
        Generate K direction vector (omega) and shift vector (b)
        Return:
            dirvec: shape (coord_dim, frequency_num), omega in the paper
            shift: shape (frequency_num), b in the paper
        """
        # mean and covarance matrix of the Gaussian distribution
        self.mean = np.zeros(self.coord_dim)
        self.cov = np.diag(np.ones(self.coord_dim) * self.rbf_kernel_size)
        # dirvec: shape (coord_dim, frequency_num), omega in the paper
        dirvec = np.transpose(
            np.random.multivariate_normal(
                self.mean, self.cov, self.frequency_num)
        )
        self.dirvec = torch.nn.Parameter(
            torch.FloatTensor(dirvec), requires_grad=False)
        self.register_parameter("dirvec", self.dirvec)

        # shift: shape (frequency_num), b in the paper
        shift = np.random.uniform(0, 2 * np.pi, self.frequency_num)
        self.shift = torch.nn.Parameter(
            torch.FloatTensor(shift), requires_grad=False)
        self.register_parameter("shift", self.shift)

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
        # coords_mat: shape (batch_size, num_context_pt, 2)
        coords_mat = coord_normalize(coords_mat, self.extent)
        batch_size = coords_mat.shape[0]
        num_context_pt = coords_mat.shape[1]

        # coords_mat: shape (batch_size, num_context_pt, 2)
        coords_mat = torch.FloatTensor(coords_mat).to(self.device)

        # spr_embeds: shape (batch_size, num_context_pt, pos_enc_output_dim = frequency_num)
        spr_embeds = torch.matmul(coords_mat, self.dirvec)

        spr_embeds = torch.cos(spr_embeds + self.shift) * np.sqrt(
            2.0 / self.pos_enc_output_dim
        )

        # spr_embeds: shape (batch_size, num_context_pt, pos_enc_output_dim = frequency_num)
        return spr_embeds

    def forward(self, coords):
        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape (batch_size, num_context_pt, coord_dim)
        Return:
            sprenc: Tensor shape (batch_size, num_context_pt, spa_embed_dim)
        """
        # spr_embeds: shape (batch_size, num_context_pt, pos_enc_output_dim = frequency_num)
        spr_embeds = self.make_output_embeds(coords)

        return spr_embeds
