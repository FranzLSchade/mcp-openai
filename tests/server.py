# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mcp",
# ]
# ///
#
# This is a simple calculator server which implements the basic mathematical
# operations: add, subtract, multiply, and divide. It is used for testing the
# implementation of the client.

import asyncio

import mcp.server.stdio
import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions

# Initialize the server
server = Server("calculator")


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available mathematical tools."""
    return [
        types.Tool(
            name="add",
            description="Add two numbers (+)",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "First number",
                    },
                    "b": {
                        "type": "number",
                        "description": "Second number",
                    },
                },
                "required": ["a", "b"],
            },
        ),
        types.Tool(
            name="sub",
            description="Subtract second number from first number (-)",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "First number",
                    },
                    "b": {
                        "type": "number",
                        "description": "Second number",
                    },
                },
                "required": ["a", "b"],
            },
        ),
        types.Tool(
            name="mul",
            description="Multiply two numbers (*)",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "First number",
                    },
                    "b": {
                        "type": "number",
                        "description": "Second number",
                    },
                },
                "required": ["a", "b"],
            },
        ),
        types.Tool(
            name="div",
            description="Divide first number by second number (/)",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "First number (dividend)",
                    },
                    "b": {
                        "type": "number",
                        "description": "Second number (divisor)",
                    },
                },
                "required": ["a", "b"],
            },
        ),
    ]


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Handle mathematical tool execution requests."""
    if not arguments:
        raise ValueError("Missing arguments")

    try:
        a = float(arguments.get("a", 0))
        b = float(arguments.get("b", 0))
    except (TypeError, ValueError):
        return [
            types.TextContent(
                type="text", text="Invalid input. Please provide valid numbers."
            )
        ]

    result = None
    if name == "add":
        result = a + b
    elif name == "sub":
        result = a - b
    elif name == "mul":
        result = a * b
    elif name == "div":
        if b == 0:
            return [
                types.TextContent(
                    type="text", text="Error: Division by zero is not allowed"
                )
            ]
        result = a / b
    else:
        raise ValueError(f"Unknown tool: {name}")

    return [types.TextContent(type="text", text=f"Result: {result}")]


async def main():
    """Run the server using stdin/stdout streams."""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="calculator",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
