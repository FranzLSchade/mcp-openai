import asyncio
import json
from contextlib import AsyncExitStack
from dataclasses import asdict
from typing import Optional, Dict, List, Tuple

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters, Tool
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion_message_tool_call_param import Function
from openai.types.shared_params.function_definition import FunctionDefinition

from .config import LLMClientConfig, LLMRequestConfig, MCPClientConfig

load_dotenv()


class MCPClient:
    def __init__(
        self,
        mpc_client_config: MCPClientConfig = MCPClientConfig(),
        llm_client_config: LLMClientConfig = LLMClientConfig(),
        llm_request_config: LLMRequestConfig = LLMRequestConfig("gpt-4o"),
    ):
        self.mpc_client_config = mpc_client_config
        self.llm_client_config = llm_client_config
        self.llm_request_config = llm_request_config
        self.llm_client = AsyncOpenAI(**asdict(self.llm_client_config))
        self.exit_stack = AsyncExitStack()

        # NEW: Store sessions and tools in dictionaries
        self.sessions: Dict[str, ClientSession] = {}
        self.available_tools: Dict[str, Tool] = {} # Mapping tool_name to Tool object

        print("CLIENT CREATED")

    async def connect_to_server(self, server_name: str):
        """Connect to an MCP server using its configuration name"""

        if server_name not in self.mpc_client_config.mcpServers:
            raise ValueError(
                f"Server '{server_name}' not found in MCP client configuration"
            )

        mcp_server_config = self.mpc_client_config.mcpServers[server_name]
        if not mcp_server_config.enabled:
            raise ValueError(f"Server '{server_name}' is disabled")

        stdio_server_params = StdioServerParameters(**asdict(mcp_server_config))

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(stdio_server_params)
        )
        stdio_reader, stdio_writer = stdio_transport # Renamed for clarity
        session = await self.exit_stack.enter_async_context(
            ClientSession(stdio_reader, stdio_writer)
        )

        await session.initialize()  # type: ignore
        self.sessions[server_name] = session # NEW: Store the session under its name

        # List available tools and store them
        response = await session.list_tools()  # type: ignore
        print(f"CLIENT CONNECTED to {server_name}")
        print("AVAILABLE TOOLS FOR THIS SERVER", [tool.name for tool in response.tools])
        for tool in response.tools:
            # NEW: Aggregate tools from all servers.
            # Here, you might need to handle naming conflicts if tools from different servers have the same name.
            # For now, simply add them.
            self.available_tools[tool.name] = tool

    async def _get_server_for_tool(self, tool_name: str) -> ClientSession:
        """Helper to find which server provides a specific tool"""
        # This is a simplified implementation.
        # In a more complex environment, you might need to know which server offers which tool.
        # For now, we simply iterate through all sessions until we find the tool.
        # A better solution would be to store which server provides each tool when collecting them.

        # Since we aggregate 'self.available_tools', we could use the tool name here
        # to identify the server that registered this tool.
        # This would require extended storage in connect_to_server.
        # But for now, we can simply iterate:
        for server_name, session in self.sessions.items():
            server_tools_response = await session.list_tools()
            if any(tool.name == tool_name for tool in server_tools_response.tools):
                return session
        raise ValueError(f"Tool '{tool_name}' not found on any connected server.")


    async def process_tool_call(self, tool_call) -> ChatCompletionToolMessageParam:
        match tool_call.type:
            case "function":
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                # NEW: Find the correct session for the tool call
                target_session = await self._get_server_for_tool(tool_name)

                call_tool_result = await target_session.call_tool(tool_name, tool_args)  # type: ignore

                if call_tool_result.isError:
                    # Improved error message
                    error_message = f"An error occurred while calling the tool '{tool_name}': {call_tool_result.error if hasattr(call_tool_result, 'error') else call_tool_result} Tool call arguments: {tool_args}"
                    raise ValueError(error_message)

                results = []
                for result in call_tool_result.content:
                    match result.type:
                        case "text":
                            results.append(result.text)
                        case "image":
                            raise NotImplementedError("Image content is not supported")
                        case "resource":
                            raise NotImplementedError(
                                "Embedded resource is not supported"
                            )
                        case _:
                            raise ValueError(f"Unknown content type: {result.type}")

                return ChatCompletionToolMessageParam(
                    role="tool",
                    content=json.dumps({**tool_args, tool_name: results}),
                    tool_call_id=tool_call.id,
                )

            case _:
                raise ValueError(f"Unknown tool call type: {tool_call.type}")

    async def process_messages(
        self,
        messages: list[ChatCompletionMessageParam],
        llm_request_config: LLMRequestConfig | None = None,
    ) -> list[ChatCompletionMessageParam]:
        # Set up tools and LLM request config
        if not self.sessions: # Check if any sessions exist
            raise RuntimeError("Not connected to any server")

        # NEW: Use the aggregated tools
        tools = [
            ChatCompletionToolParam(
                type="function",
                function=FunctionDefinition(
                    name=tool.name,
                    description=tool.description if tool.description else "",
                    parameters=tool.inputSchema,
                ),
            )
            for tool_name, tool in self.available_tools.items()
        ]

        llm_request_config = LLMRequestConfig(
            **{
                **asdict(self.llm_request_config),
                **(asdict(llm_request_config) if llm_request_config else {}),
            }
        )

        last_message_role = messages[-1]["role"]

        match last_message_role:
            case "user":
                response = await self.llm_client.chat.completions.create(
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    **asdict(llm_request_config),
                )
                finish_reason = response.choices[0].finish_reason

                match finish_reason:
                    case "stop":
                        messages.append(
                            ChatCompletionAssistantMessageParam(
                                role="assistant",
                                content=response.choices[0].message.content,
                            )
                        )
                        return messages

                    case "tool_calls":
                        tool_calls = response.choices[0].message.tool_calls
                        assert tool_calls is not None
                        messages.append(
                            ChatCompletionAssistantMessageParam(
                                role="assistant",
                                tool_calls=[
                                    ChatCompletionMessageToolCallParam(
                                        id=tool_call.id,
                                        function=Function(
                                            arguments=tool_call.function.arguments,
                                            name=tool_call.function.name,
                                        ),
                                        type=tool_call.type,
                                    )
                                    for tool_call in tool_calls
                                ],
                            )
                        )
                        tasks = [
                            asyncio.create_task(self.process_tool_call(tool_call))
                            for tool_call in tool_calls
                        ]
                        messages.extend(await asyncio.gather(*tasks))
                        return await self.process_messages(messages, llm_request_config)
                    case "length":
                        raise ValueError("Length limit reached")
                    case "content_filter":
                        raise ValueError("Content filter triggered")
                    case "function_call":
                        raise NotImplementedError("Function call not implemented")
                    case _:
                        raise ValueError(f"Unknown finish reason: {finish_reason}")

            case "assistant":
                # NOTE: the only purpose of this case is to trigger other tool
                # calls based on the results of the previous tool calls
                response = await self.llm_client.chat.completions.create(
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    **asdict(llm_request_config),
                )
                finish_reason = response.choices[0].finish_reason

                match finish_reason:
                    case "stop":
                        # NOTE: we do not add the last response message
                        return messages

                    case "tool_calls":
                        tool_calls = response.choices[0].message.tool_calls
                        assert tool_calls is not None
                        messages.append(
                            ChatCompletionAssistantMessageParam(
                                role="assistant",
                                tool_calls=[
                                    ChatCompletionMessageToolCallParam(
                                        id=tool_call.id,
                                        function=Function(
                                            arguments=tool_call.function.arguments,
                                            name=tool_call.function.name,
                                        ),
                                        type=tool_call.type,
                                    )
                                    for tool_call in tool_calls
                                ],
                            )
                        )
                        results_messages = [
                            await self.process_tool_call(tool_call)
                            for tool_call in tool_calls
                        ]
                        messages.extend(results_messages)
                        return await self.process_messages(messages, llm_request_config)
                    case "length":
                        raise ValueError("Length limit reached")
                    case "content_filter":
                        raise ValueError("Content filter triggered")
                    case "function_call":
                        raise NotImplementedError("Function call not implemented")
                    case _:
                        raise ValueError(f"Unknown finish reason: {finish_reason}")

            case "tool":
                response = await self.llm_client.chat.completions.create(
                    messages=messages,
                    **asdict(llm_request_config),
                )
                finish_reason = response.choices[0].finish_reason

                match finish_reason:
                    case "stop":
                        messages.append(
                            ChatCompletionAssistantMessageParam(
                                role="assistant",
                                content=response.choices[0].message.content,
                            )
                        )

                        return await self.process_messages(messages, llm_request_config)
                    case "tool_calls":
                        raise ValueError(
                            "The message following a tool message cannot be a tool call"
                        )
                    case "length":
                        raise ValueError("Length limit reached")
                    case "content_filter":
                        raise ValueError("Content filter triggered")
                    case "function_call":
                        raise NotImplementedError("Function call not implemented")
                    case _:
                        raise ValueError(f"Unknown finish reason: {finish_reason}")

            case "developer":
                raise NotImplementedError("Developer messages are not supported")
            case "system":
                raise NotImplementedError("System messages are not supported")
            case "function":
                raise NotImplementedError("System messages are not supported")
            case _:
                raise ValueError(f"Invalid message role: {last_message_role}")


    async def cleanup(self):
        """Clean up resources"""
        # Close all sessions in the exit stack
        await self.exit_stack.aclose()