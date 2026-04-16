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

"""Lightweight tool framework for LLM function calling.

Replaces the proprietary rmc_kangaroo Tool/Toolkit with an open-source implementation
that generates OpenAI-compatible JSON schemas for tool definitions.
"""

from __future__ import annotations

import inspect
import json
import logging
from typing import Annotated, Any, get_args, get_origin, get_type_hints

logger = logging.getLogger(__name__)

# Python type to JSON schema type mapping
_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _python_type_to_json_schema(annotation: type) -> dict[str, Any]:
    """Convert a Python type annotation to a JSON schema fragment."""
    origin = get_origin(annotation)

    if annotation in _TYPE_MAP:
        return {"type": _TYPE_MAP[annotation]}

    if origin is list:
        args = get_args(annotation)
        if args:
            return {"type": "array", "items": _python_type_to_json_schema(args[0])}
        return {"type": "array"}

    if origin is dict:
        return {"type": "object"}

    # Fallback
    return {"type": "string"}


class Tool:
    """Base class for LLM-callable tools.

    Subclasses define their parameters as class-level type annotations
    (optionally using ``Annotated[type, description]``). The ``execute``
    method contains the tool logic.

    Example::

        class MyTool(Tool):
            \"\"\"Description shown to the LLM.\"\"\"
            value: Annotated[float, "A value between 0 and 1"]

            def execute(self, model) -> str:
                return f"Value is {self.value}"
    """

    def __init__(self, **kwargs: Any) -> None:
        # Set provided keyword arguments as instance attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

    def validate_args(self) -> None:
        """Validate tool arguments before execution.

        Override in subclasses. Raise ``ToolArgsValidationError`` on failure.
        """

    def execute(self, *args: Any, **kwargs: Any) -> str:
        """Execute the tool. Override in subclasses."""
        raise NotImplementedError

    @classmethod
    def get_name(cls) -> str:
        """Return the tool name (class name)."""
        return cls.__name__

    @classmethod
    def get_description(cls) -> str:
        """Return the tool description from the docstring."""
        return inspect.getdoc(cls) or ""

    @classmethod
    def get_parameters_schema(cls) -> dict[str, Any]:
        """Generate JSON schema for tool parameters from type annotations."""
        hints = get_type_hints(cls, include_extras=True)
        properties: dict[str, Any] = {}
        required: list[str] = []

        for name, annotation in hints.items():
            # Skip private/inherited attributes
            if name.startswith("_"):
                continue

            origin = get_origin(annotation)
            description = ""

            # Handle Annotated types
            if origin is Annotated:
                args = get_args(annotation)
                actual_type = args[0]
                if len(args) > 1 and isinstance(args[1], str):
                    description = args[1]
                schema = _python_type_to_json_schema(actual_type)
            else:
                schema = _python_type_to_json_schema(annotation)

            if description:
                schema["description"] = description

            properties[name] = schema
            required.append(name)

        result: dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            result["required"] = required
        return result

    @classmethod
    def to_openai_tool(cls) -> dict[str, Any]:
        """Convert tool to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": cls.get_name(),
                "description": cls.get_description(),
                "parameters": cls.get_parameters_schema(),
            },
        }

    def __repr__(self) -> str:
        params = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return f"{self.__class__.__name__}({params})"


class Toolkit:
    """Manages a collection of tools and dispatches LLM tool calls."""

    def __init__(self, tools: list[type[Tool]]) -> None:
        self.tools = {tool.get_name(): tool for tool in tools}

    def get_openai_tools(self) -> list[dict[str, Any]]:
        """Get all tools in OpenAI function calling format."""
        return [tool.to_openai_tool() for tool in self.tools.values()]

    def parse_and_execute(self, tool_calls: list[dict[str, Any]], model: Any) -> list[str]:
        """Parse tool calls from LLM response and execute them.

        :param tool_calls: List of tool call dicts with 'function' containing 'name' and 'arguments'
        :param model: The model instance to pass to tool.execute()
        :return: List of result strings
        """
        results = []
        for call in tool_calls:
            func = call.get("function", call)
            name = func["name"]
            args_raw = func.get("arguments", "{}")

            if isinstance(args_raw, str):
                try:
                    args = json.loads(args_raw)
                except json.JSONDecodeError:
                    results.append(f"Error: Could not parse arguments for {name}")
                    continue
            else:
                args = args_raw

            if name not in self.tools:
                results.append(f"Error: Unknown tool '{name}'")
                continue

            tool_cls = self.tools[name]
            tool_instance = tool_cls(**args)

            try:
                tool_instance.validate_args()
                result = tool_instance.execute(model)
                logger.info("Tool %s executed: %s", name, result)
                results.append(result)
            except Exception as e:
                error_msg = f"Error executing {name}: {e}"
                logger.error(error_msg)
                results.append(error_msg)

        return results
