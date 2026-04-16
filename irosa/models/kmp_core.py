################################################################################
# Copyright (c) 2025. Markus Knauer, Joao Silverio                            #
# Licensed under the MIT License. See LICENSE file for details.                 #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 2025                                                                   #
# Author: Markus Knauer                                                        #
# E-mail: markus.knauer@dlr.de                                                 #
# Website: https://github.com/DLR-RM/IROSA                                    #
################################################################################

"""Kernelized Movement Primitives (KMP) core implementation.

Bundled from the open-source interactive-incremental-learning package:
https://github.com/DLR-RM/interactive-incremental-learning

If you are using this code please cite:
M. Knauer, A. Albu-Schaeffer, F. Stulp and J. Silverio, "Interactive Incremental Learning of Generalizable
Skills With Local Trajectory Modulation," in IEEE Robotics and Automation Letters (RA-L), vol. 10, no. 4,
pp. 3398-3405, April 2025, doi: 10.1109/LRA.2025.3542209

Original authors: Markus Knauer, Joao Silverio
License: MIT
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sp
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

# -- Kernel functions --


def matern_kernel_p2(x1: np.ndarray, x2: np.ndarray, length_scale: np.ndarray | float, h: float = 1.0) -> np.ndarray:
    """Matern kernel with p=2."""
    diff = np.repeat(x1[:, np.newaxis, :], x2.shape[0], axis=1) - np.repeat(x2[np.newaxis, :, :], x1.shape[0], axis=0)
    length_scale = np.array(length_scale)
    if len(length_scale.shape) == 0:
        length_scale = length_scale.reshape((1,))
    squared_dist = 5 * np.einsum("...i,ij,...j->...", diff, np.diag(1 / np.square(length_scale)), diff)
    dist = np.sqrt(squared_dist)
    return h**2 * (1 + dist + squared_dist / 3) * np.exp(-dist)


def kernel_matrix(
    x1: np.ndarray, x2: np.ndarray, length_scale: np.ndarray | float, h: float, kernel_function: str = "matern2"
) -> np.ndarray:
    """Compute kernel matrix for two inputs."""
    if kernel_function == "matern2":
        return matern_kernel_p2(x1, x2, length_scale, h)
    raise ValueError(f"Unknown kernel function: {kernel_function}")


# -- Gaussian Mixture Regression --


class GaussianMixtureModel(GaussianMixture):
    """GMM with Gaussian Mixture Regression support."""

    def gaussian_conditioning(self, index: int, x_in: np.ndarray, d_in: list, d_out: list) -> tuple:
        mean = self.means_[index]
        cov = self.covariances_[index]
        nb_dim_in = np.size(d_in)
        nb_dim_out = np.size(d_out)

        mu_ii = mean[d_in].reshape(nb_dim_in, -1)
        mu_oo = mean[d_out].reshape(nb_dim_out, -1)
        cov_ii = cov[np.ix_(d_in, d_in)]
        prec_ii = np.linalg.inv(cov_ii)
        cov_io = cov[np.ix_(d_in, d_out)]
        cov_oi = cov_io.T
        cov_oo = cov[np.ix_(d_out, d_out)]

        mu_cond = mu_oo + cov_oi @ prec_ii @ (x_in.T - mu_ii)
        mu_cond = mu_cond.T
        cov_cond = cov_oo - cov_oi @ prec_ii @ cov_io
        return mu_cond, cov_cond

    def gaussian_mixture_regression(self, x_in: np.ndarray, d_in: list, d_out: list, N: int | None = None) -> tuple:
        if N is None:
            N = x_in.shape[0] if len(x_in.shape) >= 1 else 1

        nb_dim_in = np.size(d_in)
        nb_dim_out = np.size(d_out)

        mu_cond = np.zeros((N, len(d_out), self.n_components))
        sigma_cond = np.zeros((len(d_out), len(d_out), self.n_components))
        mu = np.zeros((N, len(d_out)))
        sigma = np.zeros((N, len(d_out), len(d_out)))
        h = np.zeros((N, self.n_components))

        for i in range(self.n_components):
            mu_ii = self.means_[i, np.ix_(d_in)].reshape(nb_dim_in, -1)
            cov_ii = self.covariances_[i][np.ix_(d_in, d_in)]
            mu_cond[:, :, i], sigma_cond[:, :, i] = self.gaussian_conditioning(i, x_in, d_in, d_out)

            if len(x_in.shape) > 1 and x_in.shape[1] == 1:
                h[:, i] = self.weights_[i] * multivariate_normal.pdf(x_in[:, 0], mean=mu_ii.flatten(), cov=cov_ii)
            else:
                h[:, i] = self.weights_[i] * multivariate_normal.pdf(x_in, mean=mu_ii.flatten(), cov=cov_ii)

        h = h / np.sum(h, axis=1)[:, None]

        for i in range(N):
            mu[i, :] = mu_cond[i, :, :] @ h[i, :]
            sigma_tmp = np.zeros((nb_dim_out, nb_dim_out))
            for n in range(self.n_components):
                sigma_tmp += h[i, n] * (sigma_cond[:, :, n] + np.outer(mu_cond[i, :, n], mu_cond[i, :, n]))
            sigma[i, :, :] = sigma_tmp - np.outer(mu[i, :], mu[i, :])

        return mu, sigma, 1.0


# -- KMP --


class Kmp:
    """Kernelized Movement Primitives.

    See https://arxiv.org/pdf/1708.08638.pdf

    :param gmm_n_components: Number of Gaussians in the GMM
    :param N: Number of sample points for GMR
    :param length_scale: Length scale of the kernel
    :param h: Kernel scaling factor
    :param lambda1: Regularization for mean prediction
    :param lambda2: Regularization for covariance prediction
    :param alpha: Covariance scaling factor
    :param kernel_function: Kernel function name ("matern2")
    """

    def __init__(
        self,
        gmm_n_components: int = 12,
        N: int = 500,
        length_scale: float = 0.1,
        h: float = 1.0,
        lambda1: float = 0.1,
        lambda2: float = 1,
        alpha: float = 1,
        kernel_function: str = "matern2",
        # Legacy parameter names from llm-tools
        l: float | None = None,  # noqa: E741 — standard notation for kernel length scale
    ):
        self.gmm_n_components = gmm_n_components
        self.N = N
        self.l: float | np.ndarray = l if l is not None else length_scale
        self.h = h
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.alpha = alpha
        self.nb_via = 0
        self.kernel_function = kernel_function
        self.via_points: dict = {"input": [], "output": [], "gamma": []}

    def fit(self, X: np.ndarray, Y: np.ndarray, x_in: np.ndarray | None = None) -> None:
        """Fit KMP to data using GMR.

        :param X: Input data (e.g. time), shape (n_samples, n_dim_in)
        :param Y: Output data (e.g. position), shape (n_samples, n_dim_out)
        :param x_in: Test input points for regression
        """
        data = np.hstack([X, Y])
        d_in = list(range(X.shape[1]))
        d_out = list(range(X.shape[1], X.shape[1] + Y.shape[1]))

        self.nb_dim_in = len(d_in)
        self.nb_dim_out = len(d_out)

        if isinstance(self.l, (np.ndarray, list)):
            self.l = np.array(self.l)
        else:
            self.l = self.l * np.ones(self.nb_dim_in)

        gmm = GaussianMixtureModel(
            n_components=self.gmm_n_components,
            covariance_type="full",
            reg_covar=1e-5,
            init_params="kmeans",
        )
        gmm.fit(data)

        if x_in is None:
            x_in = np.linspace(0, 1, self.N)

        if x_in.ndim == 1:
            x_in_2d = x_in.reshape(-1, self.nb_dim_in)
        else:
            x_in_2d = x_in

        mu, sigma, _ = gmm.gaussian_mixture_regression(x_in_2d, d_in, d_out, self.N)

        mu_block = mu.reshape(-1, 1)
        pntr_sigma = [sigma[i, :, :] for i in range(sigma.shape[0])]
        sigma_block = sp.block_diag(*pntr_sigma)

        self.x_in = x_in_2d
        self.model = gmm
        self.mu = mu
        self.sigma = sigma
        self.mu_block = mu_block
        self.sigma_block = sigma_block
        self.update_K()

    def update_K(self) -> None:
        """Compute kernel matrix for all outputs."""
        I_O = np.eye(self.nb_dim_out)
        K_reduced = kernel_matrix(self.x_in, self.x_in, self.l, self.h, self.kernel_function)
        self.K = np.kron(K_reduced, I_O)

        self.invK = np.linalg.inv(self.K + self.lambda1 * self.sigma_block)
        self.invK2 = np.linalg.inv(self.K + self.lambda2 * self.sigma_block)

    def update_inputs(self, x_test: np.ndarray) -> None:
        """Compute K_s and K_ss for test inputs."""
        self.x_test = x_test.reshape(-1, self.nb_dim_in)
        K_s_reduced = kernel_matrix(self.x_test, self.x_in, self.l, self.h, self.kernel_function)
        K_ss_reduced = kernel_matrix(self.x_test, self.x_test, self.l, self.h, self.kernel_function)

        I_O = np.eye(self.nb_dim_out)
        self.K_s = np.kron(K_s_reduced, I_O)
        self.K_ss = np.kron(K_ss_reduced, I_O)

    def mean(self) -> np.ndarray:
        """KMP mean prediction, shape (N, dim_out)."""
        return (self.K_s @ self.invK @ self.mu_block).reshape((-1, self.nb_dim_out))

    def cov(self) -> np.ndarray:
        """KMP covariance prediction, shape (N*dim_out, N*dim_out)."""
        return self.alpha * (self.K_ss - self.K_s @ self.invK2 @ self.K_s.T)  # type: ignore[return-value]

    def predict(self, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict mean and covariance for test inputs."""
        self.update_inputs(x_test)
        return self.mean(), self.cov()

    def predict_with_uncertainty(self, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict mean and covariance (alias for predict)."""
        return self.predict(x_test)

    @staticmethod
    def check_via_point_structure(input_via_or_output_via):
        """Ensure via-point input is in list-of-lists format."""
        if isinstance(input_via_or_output_via, np.ndarray):
            return input_via_or_output_via.tolist()
        if isinstance(input_via_or_output_via, float):
            return [[input_via_or_output_via]]
        return input_via_or_output_via

    def add_viapoints(self, input_via, output_via, gamma: float = 1e-8) -> None:
        """Add via-points to the KMP.

        :param input_via: Time positions for via-points
        :param output_via: Output values for via-points
        :param gamma: Covariance parameter
        """
        if not isinstance(input_via, np.ndarray):
            input_via = np.array(input_via)
        if not isinstance(output_via, np.ndarray):
            output_via = np.array(output_via)

        I_O = np.eye(self.nb_dim_out)

        for i in range(len(input_via)):
            via_in = np.array(input_via[i]).reshape(1, -1)
            via_out = np.array(output_via[i]).reshape(-1)
            precision = gamma * I_O

            self.x_in = np.vstack([self.x_in, via_in])
            self.mu_block = np.append(self.mu_block, via_out)
            self.sigma = np.append(self.sigma, precision.reshape(1, self.nb_dim_out, -1), axis=0)
            self.nb_via += 1

            self.via_points["input"].append(input_via[i])
            self.via_points["output"].append(output_via[i])
            self.via_points["gamma"].append(gamma)

        self.mu_block = self.mu_block.reshape(-1, 1)
        pntr_sigma = [self.sigma[i, :, :] for i in range(self.sigma.shape[0])]
        self.sigma_block = sp.block_diag(*pntr_sigma)
        self.update_K()
        self.update_inputs(self.x_test)

    def remove_viapoints(self, input_via) -> None:
        """Remove via-points at given input positions."""
        if not isinstance(input_via, list):
            input_via = [input_via]

        for via_input in reversed(input_via):
            try:
                idx = self.via_points["input"].index(via_input)
            except ValueError:
                continue

            # Remove from tracking
            self.via_points["input"].pop(idx)
            self.via_points["output"].pop(idx)
            self.via_points["gamma"].pop(idx)

            # Remove from model arrays (via-points are appended after original N points)
            array_idx = self.N + idx
            self.x_in = np.delete(self.x_in, array_idx, axis=0)
            self.mu_block = np.delete(
                self.mu_block.flatten(), range(array_idx * self.nb_dim_out, (array_idx + 1) * self.nb_dim_out)
            )
            self.mu_block = self.mu_block.reshape(-1, 1)
            self.sigma = np.delete(self.sigma, array_idx, axis=0)
            self.nb_via -= 1

        pntr_sigma = [self.sigma[i, :, :] for i in range(self.sigma.shape[0])]
        self.sigma_block = sp.block_diag(*pntr_sigma)
        self.update_K()

    def edit_viapoints(self, input_via_old, input_via_new, output_via_new, gamma: float = 1e-8, replace: bool = True) -> None:
        """Edit existing via-points."""
        self.remove_viapoints(input_via_old)
        self.add_viapoints(input_via_new, output_via_new, gamma)
