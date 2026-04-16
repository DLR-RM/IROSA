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

"""Test basic imports."""


def test_import_irosa():
    import irosa

    assert hasattr(irosa, "KMPWrapper")


def test_import_tool():
    from irosa.core.tool import Tool, Toolkit

    assert Tool is not None
    assert Toolkit is not None


def test_import_trajectory_corrector():
    from irosa.core.trajectory_corrector import TrajectoryCorrector

    assert TrajectoryCorrector is not None


def test_import_kmp_core():
    from irosa.models.kmp_core import Kmp

    assert Kmp is not None


def test_import_tool_definitions():
    from irosa.tool_definitions import ALL_AVAILABLE_TOOLS

    assert len(ALL_AVAILABLE_TOOLS) > 0
