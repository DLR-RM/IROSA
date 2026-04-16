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

"""Pytest fixtures for IROSA tests."""

import numpy as np
import pytest


@pytest.fixture
def mock_demonstrations():
    """Create mock demonstration data (3 demos, 50 points, 8 variables)."""
    n_demos = 3
    n_points = 50
    rng = np.random.RandomState(42)

    demos = []
    for _ in range(n_demos):
        time = np.linspace(0, 1, n_points).reshape(-1, 1)
        x = 0.5 + 0.1 * np.sin(2 * np.pi * time) + rng.randn(n_points, 1) * 0.005
        y = 0.1 * np.cos(2 * np.pi * time) + rng.randn(n_points, 1) * 0.005
        z = 0.4 + 0.05 * time + rng.randn(n_points, 1) * 0.005
        qw = np.ones((n_points, 1))
        qx = np.zeros((n_points, 1))
        qy = np.zeros((n_points, 1))
        qz = np.zeros((n_points, 1))
        demo = np.hstack([time, x, y, z, qw, qx, qy, qz])
        demos.append(demo)

    return np.array(demos)


@pytest.fixture
def trained_kmp(mock_demonstrations):
    """Create a trained KMP wrapper with mock data."""
    from irosa.models.kmp import KMPWrapper

    model = KMPWrapper(
        demonstrations=mock_demonstrations,
        test_mode=True,
        force_retrain=True,
        base_kwargs={
            "gmm_n_components": 3,
            "N": 20,
            "l": 0.4,
            "h": 1.0,
            "lambda1": 0.1,
            "lambda2": 1,
            "alpha": 1,
            "kernel_function": "matern2",
        },
    )
    return model
