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

"""Test tool definitions."""


from irosa.tool_definitions.general import NoToolIsAvailable, TellUserAMessage
from irosa.tool_definitions.kmp import GetKMPParameters, GetViaPoints
from irosa.tool_definitions.robot import SlowDownRobot, SpeedUpRobot


def test_no_tool_available():
    tool = NoToolIsAvailable(why_cannot_do_anything="test reason")
    result = tool.execute()
    assert "sorry" in result.lower()
    assert "test reason" in result


def test_tell_user_message():
    tool = TellUserAMessage(message="Hello user!")
    result = tool.execute()
    assert "Hello user!" in result


def test_get_kmp_parameters(trained_kmp):
    tool = GetKMPParameters()
    result = tool.execute(trained_kmp)
    assert "parameters" in result.lower()


def test_get_viapoints(trained_kmp):
    tool = GetViaPoints()
    result = tool.execute(trained_kmp)
    assert "IROSA" in result


def test_tool_schema_generation():
    schema = SpeedUpRobot.to_openai_tool()
    params = schema["function"]["parameters"]
    assert "speed_up_value" in params["properties"]
    assert "adaptation_start" in params["properties"]
    assert "adaptation_end" in params["properties"]


def test_slow_down_schema():
    schema = SlowDownRobot.to_openai_tool()
    assert schema["function"]["name"] == "SlowDownRobot"
