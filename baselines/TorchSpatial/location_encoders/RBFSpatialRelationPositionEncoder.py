from ._common_imports import *
from ._cal_freq_list import _cal_freq_list

class RBFSpatialRelationPositionEncoder(PositionEncoder):
    """
    Given a list of (X,Y), compute the distance from each pt to each RBF anchor points
    Feed into a MLP

    This is for global position encoding or relative/spatial context position encoding
    """

    def __init__(
        self,
        model_type,
        train_locs,
        coord_dim=2,
        num_rbf_anchor_pts=100,
        rbf_kernel_size=10e2,
        rbf_kernel_size_ratio=0.0,
        max_radius=10000,
        rbf_anchor_pt_ids=None,
        device="cuda",
    ):
        """
        Args:
            train_locs: np.arrary, [batch_size, 2], location data
            spa_embed_dim: the output spatial relation embedding dimention
            coord_dim: the dimention of space, 2D, 3D, or other
            num_rbf_anchor_pts: the number of RBF anchor points
            rbf_kernel_size: the RBF kernel size
                        The sigma in https://en.wikipedia.org/wiki/Radial_basis_function_kernel
            rbf_kernel_size_ratio: if not None, (only applied on relative model)
                        different anchor pts have different kernel size :
                        dist(anchot_pt, origin) * rbf_kernel_size_ratio + rbf_kernel_size
            max_radius: the relative spatial context size in spatial context model
        """
        super().__init__(coord_dim=coord_dim, device=device)
        self.model_type = model_type
        self.train_locs = train_locs
        self.num_rbf_anchor_pts = num_rbf_anchor_pts
        self.rbf_kernel_size = rbf_kernel_size
        self.rbf_kernel_size_ratio = rbf_kernel_size_ratio
        self.max_radius = max_radius
        self.rbf_anchor_pt_ids = rbf_anchor_pt_ids

        # calculate the coordinate matrix for each RBF anchor points
        self.cal_rbf_anchor_coord_mat()

        self.pos_enc_output_dim = self.num_rbf_anchor_pts

    def _random_sampling(self, item_tuple, num_sample):
        """
        poi_type_tuple: (Type1, Type2,...TypeM)
        """

        type_list = list(item_tuple)
        if len(type_list) > num_sample:
            return list(np.random.choice(type_list, num_sample, replace=False))
        elif len(type_list) == num_sample:
            return item_tuple
        else:
            return list(np.random.choice(type_list, num_sample, replace=True))

    def cal_rbf_anchor_coord_mat(self):
        if self.model_type == "global":
            assert self.rbf_kernel_size_ratio == 0
            # If we do RBF on location/global model,
            # we need to random sample M RBF anchor points from training point dataset
            if self.rbf_anchor_pt_ids == None:
                self.rbf_anchor_pt_ids = self._random_sampling(
                    np.arange(len(self.train_locs)), self.num_rbf_anchor_pts
                )

            self.rbf_coords_mat = self.train_locs[self.rbf_anchor_pt_ids]

        elif self.model_type == "relative":
            # If we do RBF on spatial context/relative model,
            # We just ra ndom sample M-1 RBF anchor point in the relative spatial context defined by max_radius
            # The (0,0) is also an anchor point
            x_list = np.random.uniform(
                -self.max_radius, self.max_radius, self.num_rbf_anchor_pts
            )
            x_list[0] = 0.0
            y_list = np.random.uniform(
                -self.max_radius, self.max_radius, self.num_rbf_anchor_pts
            )
            y_list[0] = 0.0
            # self.rbf_coords: (num_rbf_anchor_pts, 2)
            self.rbf_coords_mat = np.transpose(
                np.stack([x_list, y_list], axis=0))

            if self.rbf_kernel_size_ratio > 0:
                dist_mat = np.sqrt(
                    np.sum(np.power(self.rbf_coords_mat, 2), axis=-1))
                # rbf_kernel_size_mat: (num_rbf_anchor_pts)
                self.rbf_kernel_size_mat = (
                    dist_mat * self.rbf_kernel_size_ratio + self.rbf_kernel_size
                )

    def make_output_embeds(self, coords):
        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape (batch_size, num_context_pt=1, coord_dim)
        Return:
            sprenc: Tensor shape (batch_size, num_context_pt, pos_enc_output_dim)
        """
        if type(coords) == np.ndarray:
            # print("np.shape(coords)",np.shape(coords))
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
        # coords_mat: shape (batch_size, num_context_pt, 1, 2)
        coords_mat = np.expand_dims(coords_mat, axis=2)
        # coords_mat: shape (batch_size, num_context_pt, num_rbf_anchor_pts, 2)
        coords_mat = np.repeat(coords_mat, self.num_rbf_anchor_pts, axis=2)
        # compute (deltaX, deltaY) between each point and each RBF anchor points
        # coords_mat: shape (batch_size, num_context_pt, num_rbf_anchor_pts, 2)
        coords_mat = coords_mat - self.rbf_coords_mat
        # coords_mat: shape (batch_size, num_context_pt, num_rbf_anchor_pts=pos_enc_output_dim)
        coords_mat = np.sum(np.power(coords_mat, 2), axis=3)
        if self.rbf_kernel_size_ratio > 0:
            spr_embeds = np.exp(
                (-1 * coords_mat) / (2.0 * np.power(self.rbf_kernel_size_mat, 2))
            )
        else:
            # spr_embeds: shape (batch_size, num_context_pt, num_rbf_anchor_pts=pos_enc_output_dim)
            spr_embeds = np.exp(
                (-1 * coords_mat) / (2.0 * np.power(self.rbf_kernel_size, 2))
            )
        return spr_embeds

    def forward(self, coords):
        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape (batch_size, num_context_pt=1, coord_dim)
        Return:
            sprenc: Tensor shape (batch_size, num_context_pt, spa_embed_dim)
        """
        spr_embeds = self.make_output_embeds(coords)

        # spr_embeds: shape (batch_size, num_context_pt, pos_enc_output_dim)
        spr_embeds = torch.FloatTensor(spr_embeds).to(self.device)

        return spr_embeds
