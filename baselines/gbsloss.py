import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.normal import Normal

from sklearn.neighbors import KDTree, BallTree
from sklearn.metrics.pairwise import haversine_distances

from scipy.stats import norm
from scipy import sparse

from scipy.spatial.transform import Rotation as R

def _latlon_to_xyz(points):
    """
    :param points: spherical coordinates of points, in radians.
    :return: 3D Cartesian coordinates of points.
    """
    xyzs = []

    for phi, theta in points:
        r = np.cos(phi)
        x = np.cos(theta) * r
        y = np.sign(phi) * np.sin(theta) * r
        z = np.sin(phi)

        xyzs.append((x, y, z))

    return np.array(xyzs)

def _xyz_to_latlon(xyzs):
    lat = np.arcsin(xyzs[:, 2])
    lon = np.arctan2(xyzs[:, 1], xyzs[:, 0])

    return np.array([lat, lon]).T

def _get_euler_angles(center):
    return 0.5 * np.pi - center[0], center[1]

def _get_polar_concentric_grid_points_by_density(radius, density):
    """
    :param radius: the range of circular grid, in radians.
    :param density: the density of grid points, in terms of point counts in unit square radian.
    :return: spherical coordinates of circular grid points with given radius and density around the northern pole.
    """
    dist = 1 / np.sqrt(density)
    n_lag = int(np.ceil(radius / dist))

    points = []
    for i in range(1, n_lag + 1):
        n_points = int(np.ceil(2 * np.pi * i))
        delta_angle = 2 * np.pi / n_points
        for j in range(n_points):
            points.append([np.pi / 2 - dist * i, -np.pi + delta_angle * j])

    return np.array(points)

def _get_polar_concentric_grid_points_by_number(radius, n_points):
    """
    :param radius: the range of circular grid, in radians.
    :param density: the density of grid points, in terms of point counts in unit square radian.
    :return: spherical coordinates of circular grid points with given radius and density around the northern pole.
    """
    dist = np.sqrt(np.pi / n_points) * radius
    n_lag = int(np.ceil(radius / dist))

    points = []
    for i in range(1, n_lag + 1):
        n_points = int(np.ceil(2 * np.pi * i))
        delta_angle = 2 * np.pi / n_points
        for j in range(n_points):
            points.append([np.pi / 2 - dist * i, -np.pi + delta_angle * j])

    return np.array(points)

def _center_grid_points(xyzs, center):
    """
    :param xyzs: 3D Cartesian coordinates of circular grid points around the northern pole.
    :param center: the (latitude, longitude) of the center point, in radians.
    :return: the 3D Cartesian coordinates of circular grid points around the given center.
    """
    rotate_phi, rotate_theta = _get_euler_angles(center)
    r = R.from_euler('yz', [rotate_phi, rotate_theta], degrees=False)

    return r.apply(xyzs)

def _move_points_to_polar(xyzs, center):
    """
    :param xyzs: 3D Cartesian coordinates of points.
    :param center: the (latitude, longitude) of the center point, in radians.
    :return: the 3D Cartesian coordinates of points centered at the northern pole.
    """
    rotate_phi, rotate_theta = _get_euler_angles(center)
    r = R.from_euler('zy', [-rotate_theta, -rotate_phi], degrees=False)

    return r.apply(xyzs)

def _get_arc_angles(points, center):
    """
    :param points: spherical coordinates of points, in radians.
    :param center: the (latitude, longitude) of the center point, in radians.
    :return: the absolute arc angles between each point and the south-pointing arc passing the center.
    """
    xyzs = _latlon_to_xyz(points)
    polar_xyzs = _move_points_to_polar(xyzs, center)
    polar_points = _xyz_to_latlon(polar_xyzs)

    raw_angles = polar_points[:, 1] - center[1]
    idx = np.where(np.abs(raw_angles) > np.pi)

    if (raw_angles[idx] > 0).all():
        raw_angles[idx] = -(2 * np.pi - raw_angles[idx])
    elif (raw_angles[idx] < 0).all():
        raw_angles[idx] = 2 * np.pi + raw_angles[idx]
    else:
        assert False, "Incorrect angle sign!"

    return raw_angles

def auto_density(radius, n_neighbors):
    return int(np.ceil((1 * n_neighbors) / (np.pi * radius**2)))

def construct_weight_matrix(points, k):
    """
    :param points: spherical coordinates of points, in radians.
    :param k: k nearest neighbors to build the weight matrix.
    :return: 0-1 weight matrix.
    """
    kdt = BallTree(points, leaf_size=30, metric='haversine')
    nbrs = kdt.query(points, k=k+1, return_distance=False)

    weights, coord_is, coord_js = [], [], []

    for i, ni in enumerate(nbrs):
        for j in ni:
            if i != j:
                weights.append(1.0)
                coord_is.append(i)
                coord_js.append(j)

    return sparse.coo_matrix((weights, (coord_is, coord_js)), shape=(points.shape[0], points.shape[0])).toarray().astype(np.float32)

def generate_background_points(center, radius, density=None, n_points=None):
    """
    :param center: the (latitude, longitude) of the center point, in radians.
    :param radius: the range of circular grid, in radians. It needs to be in the range of (0, pi/2).
    :param density: the density of grid points, in terms of point counts in unit square radians.
    :return: spherical coordinates of circular grid points with given radius and density around the given center, in radians.
    """

    if density is not None:
        polar_concentric_grid_points = _get_polar_concentric_grid_points_by_density(radius, density)
    elif n_points is not None:
        polar_concentric_grid_points = _get_polar_concentric_grid_points_by_number(radius, n_points)
    else:
        assert False, "Either specify density or number of points!"

    xyzs = _latlon_to_xyz(polar_concentric_grid_points)

    rotated_xyzs = _center_grid_points(xyzs, center)

    return _xyz_to_latlon(rotated_xyzs)

class AnalyticalSurprisal():
    def __init__(self):
        super().__init__()

    def correct(self):
        beta1 = 1 - 1 / self.d
        Mu_corrected = 0.
        for k in self.mean_dict.keys():
            if k[0] == k[1]:
                beta2 = 1 - 2 / self.mean_dict[k] + 1e-32
                Mu_corrected += beta1 * self.mean_coef_dict[k] * self.mean_dict[k] / beta2
            else:
                Mu_corrected += beta1 * self.mean_coef_dict[k] * self.mean_dict[k]

        return Mu_corrected

    def fit(self, cs, ns, w_map, ignores):
        self.d = w_map.shape[0]
        self.N, self.W = np.sum(ns), np.sum(w_map)
        self.xmean = np.sum(cs * ns) / np.sum(ns)
        self.xvar = np.sum((ns * (cs - self.xmean)) ** 2) + 1e-32

        self.mean, self.mean_dict, self.mean_coef_dict = self.compute_mean(cs, ns)
        self.mean_corrected = self.correct()
        self.std, self.std_dict, self.std_coef_dict = self.compute_std(cs, ns, ignores)
        self.scaling_factor = self.N / (self.W * self.xvar)

        return self.mean, self.std

    @staticmethod
    def compute_mean(cs, ns):
        N = np.sum(ns)
        xmean = np.sum(cs * ns) / N
        mu = 0.

        mu_dict = {}
        mu_coef_dict = {}

        for i, (ci, ni) in enumerate(zip(cs, ns)):
            for j, (cj, nj) in enumerate(zip(cs, ns)):
                mu_coef = (ci - xmean) * (cj - xmean)
                if ci != cj:
                    mu_num = min(ni, nj) * 4 * max(ni, nj) / N
                    mu += mu_coef * mu_num
                else:
                    mu_num = ((ni - 1) * 4 * ni / N) - 1
                    mu += mu_coef * mu_num

                mu_dict[(i, j)] = mu_num + 1e-32
                mu_coef_dict[(i, j)] = mu_coef + 1e-32

        return mu, mu_dict, mu_coef_dict

    @staticmethod
    def compute_std(cs, ns, ignores=None):
        N = np.sum(ns)
        r_max = np.where(ignores == 0)[0][0]
        xmean = np.sum(cs * ns) / N
        var = 0.
        std_dict, std_coef_dict = {}, {}

        if ignores is None:
            ignores = np.ones_like(ns)
        for i, (ci, ni, igi) in enumerate(zip(cs, ns, ignores)):
            for j, (cj, nj, igj) in enumerate(zip(cs, ns, ignores)):
                if ci != cj:
                    var_num = min(ni, nj) * (4 * max(ni, nj) / N) * (1 - 4 * max(ni, nj) / N)
                    var_coef = ((ci - xmean) * (cj - xmean) - 2 * (ci - xmean) * (cs[r_max] - xmean) + (
                                cs[r_max] - xmean) ** 2) ** 2
                    var += var_coef * var_num * igi * igj
                else:
                    var_num = 2 * (ni - 1) * 4 * ni / N * (1 - (4 * (2 * ni - 1)) / (3 * N))
                    var_coef = (ci - cs[r_max]) ** 4
                    var += var_coef * var_num * igi * igj

                std_dict[(i, j)] = np.sqrt(max(0., var_num * igi * igj)) + 1e-32
                std_coef_dict[(i, j)] = np.sqrt(max(0., var_coef)) + 1e-32

        return np.sqrt(max(0., var)) + 1.0, std_dict, std_coef_dict

class SSIPartitioner():
    def __init__(self, coords, k=100, radius=0.01, min_dist=0.0):
        self.coords = coords
        self.N = coords.shape[0]
        self.k = k
        self.radius = radius
        self.min_dist = min_dist
        self.neighbors = self._construct_tree()
        self.backgrounds = self._construct_backgrounds()

    def _construct_tree(self):
        kdt = BallTree(self.coords, leaf_size=30, metric='haversine')
        dists, nbrs = kdt.query(self.coords, k=self.k, return_distance=True)

        return dict(zip([i for i in range(self.N)], [nbrs[i, (dists[i] >= self.min_dist) & (dists[i] <= self.radius)] for i in range(self.N)]))

    def _construct_backgrounds(self):
        return dict(zip([i for i in range(self.N)], [generate_background_points(self.coords[i], self.radius, auto_density(self.radius, self.neighbors[i].shape[0])) for i in range(self.N)]))

    def get_neighborhood_idx(self, idx):
        return np.array(self.neighbors[idx])

    def get_neighborhood_points(self, idx):
        return self.coords[self.neighbors[idx]]

    def get_background_points(self, idx):
        return np.array(self.backgrounds[idx])

class SRIPartitioner():
    def __init__(self, coords, k=100, radius=0.01, min_dist=0.0):
        self.coords = coords
        self.N = coords.shape[0]
        self.k = k
        self.radius = radius
        self.min_dist = min_dist
        self.neighbors, self.dists = self._construct_tree()

    def _construct_tree(self):
        kdt = BallTree(self.coords, leaf_size=30, metric='haversine')
        dists, nbrs = kdt.query(self.coords, k=self.k, return_distance=True)

        return dict(zip([i for i in range(self.N)], [nbrs[i, (dists[i] >= self.min_dist) & (dists[i] <= self.radius)] for i in range(self.N)])), dict(zip([i for i in range(self.N)], [dists[i, (dists[i] >= self.min_dist) & (dists[i] <= self.radius)] for i in range(self.N)]))

    def get_neighborhood_idx(self, idx):
        return np.array(self.neighbors[idx]), np.array(self.dists[idx]), self.coords[self.neighbors[idx]]

    def get_scale_grid_idx(self, idx, scale, threshold=10):
        partition_idx_list = []
        neighbor_indices, neighbor_dists, neighbor_coords = self.get_neighborhood_idx(idx)

        k = int(np.ceil(self.radius / scale))
        lat, lon = self.coords[idx]

        for i in range(-k, k, 1):
            for j in range(-k, k, 1):
                mask = np.where((neighbor_coords[:, 0] >= lat + i*scale) & (neighbor_coords[:, 0] < lat + (i+1)*scale) & (neighbor_coords[:, 1] >= lon + j*scale) & (neighbor_coords[:, 1] < lon + (j+1)*scale))
                if len(mask[0]) < threshold:
                    continue

                partition_idx = mask[0]
                partition_idx_list.append(partition_idx)

        return partition_idx_list, neighbor_indices

    def get_distance_lag_idx(self, idx, lag, threshold=10):
        partition_idx_list = []
        neighbor_indices, neighbor_dists, neighbor_coords = self.get_neighborhood_idx(idx)

        n_lags = int(np.ceil(self.radius / lag))
        for i in range(n_lags):
            mask = np.where((neighbor_dists >= lag * i) & (neighbor_dists < lag * (i + 1)))

            if len(mask[0]) < threshold:
                continue
            partition_idx = mask[0]
            partition_idx_list.append(partition_idx)

        return partition_idx_list, neighbor_indices

    def get_direction_sector_idx(self, idx, n_splits, threshold=10):
        partition_idx_list = []
        neighbor_indices, neighbor_dists, neighbor_coords = self.get_neighborhood_idx(idx)

        arc_angles = _get_arc_angles(self.coords[neighbor_indices], self.coords[idx])

        split_angle = 2 * np.pi / n_splits
        for i in range(n_splits):
            mask = np.where((arc_angles >= -np.pi + i * split_angle) & (arc_angles < -np.pi + (i + 1) * split_angle))
            if len(mask[0]) < threshold:
                continue
            partition_idx = mask[0]
            partition_idx_list.append(partition_idx)

        return partition_idx_list, neighbor_indices


class DifferentiableCeil(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Save input for backward pass
        ctx.save_for_backward(input)
        # In the forward pass, apply the actual ceil function
        return torch.ceil(input)

    @staticmethod
    def backward(ctx, grad_output):
        # In the backward pass, the gradient is 1 (identity)
        return grad_output

# Create an instance of the custom function
differentiable_ceil = DifferentiableCeil.apply

class BinaryPerformanceTransformer(nn.Module):
    def __init__(self, thres=0.7):
        super(BinaryPerformanceTransformer, self).__init__()
        self.thres = thres

    def get_probs(self, logits):
        return torch.softmax(logits, dim=1)

    def discretize(self, probs):
        return torch.clamp(probs - self.thres, min=0.) / (probs - self.thres)

    def forward(self, logits, y):
        probs = self.get_probs(logits)
        steps = self.discretize(probs)

        return steps[np.arange(logits.shape[0]),y]

class BinnedPerformanceTransformer(nn.Module):
    def __init__(self, scale=5):
        super(BinnedPerformanceTransformer, self).__init__()
        self.scale = scale

    def get_probs(self, logits):
        return torch.softmax(logits, dim=1)

    def discretize(self, probs):
        return differentiable_ceil(self.scale * probs)

    def forward(self, logits, y):
        probs = self.get_probs(logits)
        steps = self.discretize(probs)

        return steps[np.arange(logits.shape[0]),y]

class LogOddsPerformanceTransformer(nn.Module):
    def __init__(self, cls, scale=10):
        super(LogOddsPerformanceTransformer, self).__init__()
        self.cls = cls
        self.scale = scale

    def get_scores(self, logits):
        probs = torch.softmax(logits, dim=1)

        return torch.log(probs) - torch.log(1 - probs) + math.log(self.cls - 1)

    def discretize(self, scores):
        return differentiable_ceil(self.scale * scores)

    def forward(self, logits, y):
        scores = self.get_scores(logits)
        steps = self.discretize(scores)

        return steps[np.arange(logits.shape[0]),y]

class SoftHistogramPerformanceTransformer(nn.Module):
    def __init__(self, bins, min=0., max=1., sigma=3):
        super(SoftHistogramPerformanceTransformer, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(self.max - self.min) / float(bins)
        self.centers = float(self.min) + self.delta * (torch.arange(bins).float() + 0.5)

    def get_probs(self, logits):
        return torch.softmax(logits, dim=1)

    def forward(self, logits, y):
        probs = self.get_probs(logits)[np.arange(logits.shape[0]),y]

        probs = torch.unsqueeze(probs, 0) - torch.unsqueeze(self.centers, 1)
        steps = torch.sigmoid(self.sigma * (probs + self.delta/2)) - torch.sigmoid(self.sigma * (probs - self.delta/2))
        return steps.sum(dim=1) / steps.sum()

class SSILoss(nn.Module):
    def __init__(self):
        super(SSILoss, self).__init__()
        self.analytical_surprisal = AnalyticalSurprisal()
    def forward(self, points, values):
        """
        :param points: the locations of the center and neighbor points, Bx2.
        :param values: the discretized performance values, B.
        :return: SSI loss.
        """

        cs, ns = np.unique(values.detach().cpu().numpy(), return_counts=True)
        rmax = np.argmax(ns)

        ignores = np.ones_like(cs)
        ignores[rmax] = 0

        # Handling extreme cases
        ignore_ratio = ns[rmax] / np.sum(ns)
        # print(f"Ignore ratio: {ignore_ratio}")
        if ignore_ratio > 0.9 or ignore_ratio < 0.6:
            return None, ignore_ratio

        weight_matrix = construct_weight_matrix(points, 4)

        loc, scale = self.analytical_surprisal.fit(cs, ns, weight_matrix, ignores)

        w_map = torch.from_numpy(weight_matrix).to(device=values.device, dtype=values.dtype)

        X = values.flatten().reshape((1, -1)) - torch.mean(values)

        Y = torch.matmul(w_map, X.T)

        moran_I_upper = torch.matmul(X, Y).flatten()

        low = torch.min(moran_I_upper, 2 * loc - moran_I_upper)

        prob = (1 + torch.erf((low - loc) / (scale * math.sqrt(2))))

        return -torch.log(prob + 1e-32), ignore_ratio

class SRILoss(nn.Module):
    def __init__(self):
        super(SRILoss, self).__init__()
        self.kldiv = nn.KLDivLoss(reduction="batchmean")

    def forward(self, partition_values, neighborhood_values):
        """
        :param points: the locations of the center and neighbor points, Bx2.
        :param values: the discretized performance values, B.
        :return: SSI loss.
        """

        return self.kldiv(partition_values.log(), neighborhood_values)