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

"""KMP-specific LLM-callable tools for trajectory adaptation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from irosa.core.tool import Tool
from irosa.exceptions import ToolArgsValidationError

if TYPE_CHECKING:
    from irosa.models.kmp import KMPWrapper


class GetViaPoints(Tool):
    """If you want to know all the current via-points, use this function."""

    def execute(self, kmp: KMPWrapper) -> str:
        try:
            return f"IROSA: {kmp.get_viapoints()}"
        except Exception as e:
            raise ToolArgsValidationError(f"Error getting via-points: {e}") from e


class AddViaPointsAtTime(Tool):
    """If you want to add several via-points at specific times, use this function.
    Add multiple via-points at specific times with specific positions.
    :param input_values: List of floats representing the times of the viapoints
    :param output_values: List of lists of floats representing the positions of the viapoints
    """

    input_values: list[float]
    output_values: list[list[float]]

    def execute(self, kmp: KMPWrapper) -> str:
        try:
            input_values = [[value] for value in self.input_values]
            kmp.add_viapoints(input_via=input_values, output_via=self.output_values, gamma=1e-8)
        except Exception as e:
            raise ToolArgsValidationError(
                f"Error adding via-points: {e}"
                " The input should be time: list[float], position: list[list[float]]."
                " Stick to 3 decimal places for the time and positions."
            ) from e
        return f"IROSA: Added via-points at position {self.output_values}."


class RemoveViaPointsAtTime(Tool):
    """If you want to remove several via-points at a specific time, use this function.
    :param input_values: times when via-points should be removed. time starts at 0 and ends at 1.
    """

    input_values: list[float]

    def execute(self, kmp: KMPWrapper) -> str:
        try:
            input_values = [[value] for value in self.input_values]
            kmp.remove_viapoints(input_via=input_values)
        except Exception as e:
            raise ToolArgsValidationError(f"Error removing via points: {e}") from e
        return f"IROSA: Via-points removed at time(s) {self.input_values}"


class RemoveViaPointsAtPosition(Tool):
    """If you want to remove several via-points at a specific position, use this function.
    :param output_values: positions of via-points to be removed.
    """

    output_values: list[list[float]]

    def execute(self, kmp: KMPWrapper) -> str:
        try:
            kmp._ensure_trained_model("remove_viapoint_position")
            assert kmp.kmp is not None
            input_via = []
            for i, point in enumerate(kmp.kmp.via_points["output"]):
                if point in self.output_values:
                    input_via.append(kmp.kmp.via_points["input"][i])
            kmp.remove_viapoints(input_via=input_via)
        except Exception as e:
            raise ToolArgsValidationError(f"Error removing via point: {e}") from e
        return f"IROSA: Via-points removed at position(s) {self.output_values}"


class RemoveAllViaPoints(Tool):
    """If you want to remove all via-points, use this function."""

    def execute(self, kmp: KMPWrapper) -> str:
        try:
            kmp._ensure_trained_model("remove_all_viapoints")
            assert kmp.kmp is not None
            kmp.remove_viapoints(input_via=kmp.kmp.via_points["input"])
        except Exception as e:
            raise ToolArgsValidationError(f"Error removing all via points: {e}") from e
        return "IROSA: All via-points removed"


class EditViaPointsAtTime(Tool):
    """If you want to edit several via points at specific times, use this function."""

    input_values_old: list[float]
    input_values_new: list[float]
    output_values_new: list[list[float]]

    def execute(self, kmp: KMPWrapper) -> str:
        try:
            input_values_old = [[value] for value in self.input_values_old]
            input_values_new = [[value] for value in self.input_values_new]
            kmp.edit_viapoints(
                input_via_old=input_values_old,
                input_via_new=input_values_new,
                output_via_new=self.output_values_new,
                gamma=1e-8,
            )
        except Exception as e:
            raise ToolArgsValidationError(f"Error editing via point: {e}") from e
        return (
            f"IROSA: The via-points at times {self.input_values_old} were moved to"
            f" times {self.input_values_new} and positions {self.output_values_new}"
        )


class AddRepulsionPoint(Tool):
    """If you want to add a repulsion point in order to avoid an obstacle, use this function.
    :param position: Position in x,y,z of the obstacle, where a repulsion point is added
    :param radius: Radius around the repulsion point in which area the trajectory
        points will be pushed away, default is 0.5 (meter)
    """

    position: list[float]
    radius: float

    def execute(self, kmp: KMPWrapper) -> str:
        try:
            kmp.add_repulsion_point(position=np.array(self.position), radius=self.radius)
        except Exception as e:
            raise ToolArgsValidationError(f"Error adding repulsion-point: {e}") from e
        return f"IROSA: Added repulsion-point at position {self.position}."


class GetKMPParameters(Tool):
    """If you want to know the current KMP parameters, use this function."""

    def execute(self, kmp: KMPWrapper) -> str:
        return f"IROSA: The KMP has the following parameters: {kmp.get_parameters()}"


class SetKMPParameters(Tool):
    """If you want to set the parameters of the KMP, use this function.
    So far we only allow to change the kernel length value l.
    :param kernel_length: The length scale of the kernel. Default is 0.2
    """

    kernel_length: float

    def execute(self, kmp: KMPWrapper) -> str:
        try:
            if self.kernel_length <= 0:
                raise ToolArgsValidationError("The length scale must be greater than 0. Default is 0.2.")
            if self.kernel_length > 0.6:
                raise ToolArgsValidationError("Kernel length too high, try between 0.01 and 0.6")
            kmp.update_kernel_length(self.kernel_length)
            return f"IROSA: The kernel length was successfully changed to {self.kernel_length}"
        except Exception as e:
            raise ToolArgsValidationError(f"Error setting KMP parameters: {e}") from e
