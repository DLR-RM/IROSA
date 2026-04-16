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

"""General LLM-callable tools."""

from __future__ import annotations

from typing import Annotated

from irosa.core.tool import Tool


class NoToolIsAvailable(Tool):
    """If there are no tools that can solve this problem, use this function."""

    why_cannot_do_anything: Annotated[str, "explain why you cannot call other functions adapting the persona of the robot"]

    def execute(self, *args) -> str:
        return f"IROSA: I am sorry, I am afraid I can't do that.\n{self.why_cannot_do_anything}"


class TellUserAMessage(Tool):
    """If there is something you want to communicate or say to the user use this tool, instead of saying it directly."""

    message: Annotated[str, "Act as the robot and tell the user a message. Use the persona of the robot."]

    def execute(self, *args) -> str:
        return f"IROSA: {self.message}"
