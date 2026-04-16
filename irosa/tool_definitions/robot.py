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

"""Robot control tools for speed modulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from irosa.core.tool import Tool
from irosa.exceptions import ToolArgsValidationError

if TYPE_CHECKING:
    from irosa.models.kmp import KMPWrapper


class SpeedUpRobot(Tool):
    """If you want to speed up the robot, use this function. The default is to speed up
    for the whole time period, so use 0.0 for adaption_start and 1.0 for adaption_end.
    :param speed_up_value: percentual speedup of the robot. Default value is 40 for 40 percent.
    :param adaption_start: Start time from when you want to speed up the robot, ranges from 0.0 to 1.0,
    where 0.0 is the start and 1.0 is the end time.
    :param adaption_end: Time when you want to stop the speedup of the robot, ranges from 0.0 to 1.0
    and has to be bigger than the adaption_start.
    """

    speed_up_value: int
    adaption_start: float
    adaption_end: float

    def execute(self, kmp: KMPWrapper) -> str:
        if kmp.robot is None:
            raise ToolArgsValidationError("No robot configured for speed modulation")
        try:
            kmp.robot.change_predicting_frequency(-self.speed_up_value, self.adaption_start, self.adaption_end)
        except Exception as e:
            raise ToolArgsValidationError(f"Error speeding up robot: {e}") from e
        return (
            f"IROSA: Increased robot speed by {self.speed_up_value} percent"
            f" at the range {self.adaption_start} - {self.adaption_end}"
        )


class SlowDownRobot(Tool):
    """If you want to slow down the robot, use this function. The default is to slow down
    for the whole time period, so use 0.0 for adaption_start and 1.0 for adaption_end.
    :param slow_down_value: percentual slowdown of the robot, default value is 80. Only values > 0 are allowed.
    :param adaption_start: Start time from when you want to slow down the robot,
        ranges from 0.0 to 1.0, where 0.0 is the start and 1.0 is the end time.
    :param adaption_end: Time when you want to stop the slowdown of the robot,
        ranges from 0.0 to 1.0 and has to be bigger than the adaption_start.
    """

    slow_down_value: int
    adaption_start: float
    adaption_end: float

    def execute(self, kmp: KMPWrapper) -> str:
        if kmp.robot is None:
            raise ToolArgsValidationError("No robot configured for speed modulation")
        try:
            kmp.robot.change_predicting_frequency(self.slow_down_value, self.adaption_start, self.adaption_end)
        except Exception as e:
            raise ToolArgsValidationError(f"Error slowing down the robot: {e}") from e
        return (
            f"IROSA: Decreased robot speed by {self.slow_down_value} percent"
            f" at the range {self.adaption_start} - {self.adaption_end}"
        )
