from ._common_imports import *
from ._cal_freq_list import _cal_freq_list

class GridLookupSpatialRelationPositionEncoder(PositionEncoder):
    """
    Given a list of (deltaX,deltaY),
    divide the space into grids, each point is using the grid embedding it falls into

    """

    def __init__(
        self,
        spa_embed_dim,
        model_type="global",
        max_radius=None,
        coord_dim=2,
        interval=1000000,
        extent=[-180, 180, -90, 90],
        device="cuda",
    ):
        """
        Args:
            spa_embed_dim: the output spatial relation embedding dimention
            coord_dim: the dimention of space, 2D, 3D, or other
            interval: the cell size in X and Y direction
            extent: (left, right, bottom, top)
                "global": the extent of the study area (-1710000, -1690000, 1610000, 1640000)
                "relative": the extent of the relative context
        """
        super().__init__(coord_dim=coord_dim, device=device)
        self.spa_embed_dim = spa_embed_dim
        self.interval = interval
        self.model_type = model_type
        self.max_radius = max_radius
        assert extent[0] < extent[1]
        assert extent[2] < extent[3]

        self.extent = self.get_spatial_context(model_type, extent, max_radius)
        self.make_grid_embedding(self.interval, self.extent)

        self.pos_enc_output_dim = spa_embed_dim

    def get_spatial_context(self, model_type, extent, max_radius):
        if model_type == "global":
            extent = extent
        elif model_type == "relative":
            extent = (
                -max_radius - 500,
                max_radius + 500,
                -max_radius - 500,
                max_radius + 500,
            )
        return extent

    def make_grid_embedding(self, interval, extent):
        self.num_col = int(math.ceil(float(extent[1] - extent[0]) / interval))
        self.num_row = int(math.ceil(float(extent[3] - extent[2]) / interval))

        self.embedding = torch.nn.Embedding(
            self.num_col * self.num_row, self.spa_embed_dim
        )
        self.embedding.weight.data.normal_(0, 1.0 / self.spa_embed_dim)

    def make_output_embeds(self, coords):
        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape (batch_size, num_context_pt=1, coord_dim)
        Return:
            sprenc: Tensor shape (batch_size, num_context_pt, input_embed_dim)
        """
        if type(coords) == np.ndarray:
            assert self.coord_dim == np.shape(coords)[2]
            coords = list(coords)
        elif type(coords) == list:
            assert self.coord_dim == len(coords[0][0])
        else:
            raise Exception(
                "Unknown coords data type for RBFSpatialRelationEncoder")

        # coords_mat: shape (batch_size, num_context_pt, 2)
        coords_mat = np.asarray(coords).astype(float)
        batch_size = coords_mat.shape[0]
        num_context_pt = coords_mat.shape[1]

        # x or y: shape (batch_size, num_context_pt)
        x = coords_mat[:, :, 0]
        y = coords_mat[:, :, 1]

        col = np.floor((x - self.extent[0]) / self.interval)
        row = np.floor((y - self.extent[2]) / self.interval)

        # make sure each row/col index is within range
        col = np.clip(col, 0, self.num_col - 1)
        row = np.clip(row, 0, self.num_row - 1)

        # index_mat: shape (batch_size, num_context_pt)
        index_mat = (row * self.num_col + col).astype(int)
        # index_mat: shape (batch_size, num_context_pt)
        index_mat = torch.LongTensor(index_mat).to(self.device)

        spr_embeds = self.embedding(torch.autograd.Variable(index_mat))
        # spr_embeds: shape (batch_size, num_context_pt, spa_embed_dim)
        return spr_embeds

    def forward(self, coords):
        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape (batch_size, num_context_pt, coord_dim)
        Return:
            sprenc: Tensor shape (batch_size, num_context_pt, spa_embed_dim)
        """
        # spr_embeds: shape (batch_size, num_context_pt, spa_embed_dim)
        spr_embeds = self.make_output_embeds(coords).to(self.device)

        # sprenc: shape (batch_size, num_context_pt, spa_embed_dim)
        # sprenc = torch.einsum("bnd,dk->bnk", (spr_embeds, self.post_mat))

        return spr_embeds
