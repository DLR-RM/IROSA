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

"""Test KMP model wrapper."""

import numpy as np


def test_kmp_init(trained_kmp):
    assert trained_kmp.is_trained()
    assert trained_kmp.mean is not None
    assert trained_kmp.trajectory is not None


def test_kmp_trajectory_shape(trained_kmp):
    # trajectory = [time, x, y, z, qw, qx, qy, qz]
    assert trained_kmp.trajectory.shape[1] == 8  # time + 7D output


def test_kmp_add_viapoint(trained_kmp):
    original_mean = trained_kmp.mean.copy()
    mid_idx = len(trained_kmp.x_in) // 2

    via_pos = original_mean[mid_idx, :].copy()
    via_pos[1] += 0.1  # shift y by 10cm

    trained_kmp.add_viapoints(
        input_via=[[0.5]],
        output_via=[via_pos.tolist()],
    )

    # Trajectory should have changed
    assert not np.allclose(trained_kmp.mean, original_mean)


def test_kmp_get_parameters(trained_kmp):
    params = trained_kmp.get_parameters()
    assert "gmm_n_components" in params
    assert "l" in params
    assert "lambda1" in params


def test_kmp_core_predict():
    from irosa.models.kmp_core import Kmp

    kmp = Kmp(gmm_n_components=3, N=20, l=0.4)

    # Create simple training data
    rng = np.random.RandomState(42)
    n_samples = 100
    X = np.linspace(0, 1, n_samples).reshape(-1, 1)
    Y = np.sin(2 * np.pi * X) + rng.randn(n_samples, 1) * 0.1

    x_test = np.linspace(0, 1, 20)
    kmp.fit(X, Y, x_in=x_test)
    mean, cov = kmp.predict(x_test)

    assert mean.shape == (20, 1)
    assert cov.shape == (20, 20)
