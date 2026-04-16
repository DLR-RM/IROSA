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

"""LLM-callable tool definitions for IROSA."""

from irosa.core.tool import Tool
from irosa.tool_definitions.general import NoToolIsAvailable, TellUserAMessage
from irosa.tool_definitions.kmp import (
    AddRepulsionPoint,
    AddViaPointsAtTime,
    EditViaPointsAtTime,
    GetKMPParameters,
    GetViaPoints,
    RemoveAllViaPoints,
    RemoveViaPointsAtPosition,
    RemoveViaPointsAtTime,
    SetKMPParameters,
)
from irosa.tool_definitions.robot import SlowDownRobot, SpeedUpRobot

AVAILABLE_TOOLS_GENERAL: list[type[Tool]] = [
    NoToolIsAvailable,
    TellUserAMessage,
]

AVAILABLE_TOOLS_KMP: list[type[Tool]] = [
    SpeedUpRobot,
    SlowDownRobot,
    AddViaPointsAtTime,
    RemoveViaPointsAtTime,
    RemoveViaPointsAtPosition,
    RemoveAllViaPoints,
    EditViaPointsAtTime,
    GetViaPoints,
    GetKMPParameters,
    SetKMPParameters,
    AddRepulsionPoint,
]

ALL_AVAILABLE_TOOLS: list[type[Tool]] = [
    *AVAILABLE_TOOLS_GENERAL,
    *AVAILABLE_TOOLS_KMP,
]

__all__ = [
    "ALL_AVAILABLE_TOOLS",
    "AVAILABLE_TOOLS_GENERAL",
    "AVAILABLE_TOOLS_KMP",
]
