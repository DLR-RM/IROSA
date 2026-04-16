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

"""Test the tool framework."""

import json
from typing import Annotated

from irosa.core.tool import Tool, Toolkit


class MockTool(Tool):
    """A mock tool for testing."""

    value: Annotated[float, "A test value"]
    name: Annotated[str, "A test name"]

    def execute(self, *args) -> str:
        return f"executed with value={self.value}, name={self.name}"


class SimpleTool(Tool):
    """No parameters tool."""

    def execute(self, *args) -> str:
        return "simple result"


def test_tool_name():
    assert MockTool.get_name() == "MockTool"


def test_tool_description():
    assert "mock tool" in MockTool.get_description().lower()


def test_tool_parameters_schema():
    schema = MockTool.get_parameters_schema()
    assert schema["type"] == "object"
    assert "value" in schema["properties"]
    assert schema["properties"]["value"]["type"] == "number"
    assert "name" in schema["properties"]
    assert schema["properties"]["name"]["type"] == "string"


def test_tool_openai_format():
    openai_tool = MockTool.to_openai_tool()
    assert openai_tool["type"] == "function"
    assert openai_tool["function"]["name"] == "MockTool"
    assert "parameters" in openai_tool["function"]


def test_tool_execute():
    tool = MockTool(value=3.14, name="test")
    result = tool.execute()
    assert "3.14" in result
    assert "test" in result


def test_toolkit():
    toolkit = Toolkit([MockTool, SimpleTool])
    tools = toolkit.get_openai_tools()
    assert len(tools) == 2


def test_toolkit_parse_and_execute():
    toolkit = Toolkit([MockTool])
    tool_calls = [
        {
            "function": {
                "name": "MockTool",
                "arguments": json.dumps({"value": 1.0, "name": "hello"}),
            }
        }
    ]
    results = toolkit.parse_and_execute(tool_calls, model=None)
    assert len(results) == 1
    assert "executed" in results[0]
