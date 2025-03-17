<div align="center">
  <h1>𝔐&nbsp;&nbsp;mpc-openai&nbsp;&nbsp;✧</h1>
  <p><em>MCP Client with OpenAI compatible API</em></p>
</div>

> Model Context Protocol (MCP) is an open protocol that standardizes how applications provide context to LLMs. Think of MCP like a USB-C port for AI applications.
>
> — [https://modelcontextprotocol.io](https://modelcontextprotocol.io)

______________________________________________________________________

> [!WARNING]
> This is a simple toy project. Support is not planned. Use as a reference for minimal MCP client development.

This is a MCP **client** (not a server). It is meant to be used as a library for building LLMs UI that support MCP through an OpenAI compatible API. This opens the door to locally runnable inference engines ([vLLM](https://github.com/vllm-project/vllm), [Ollama](https://github.com/ollama/ollama), [TGI](https://github.com/huggingface/text-generation-inference), [llama.cpp](https://github.com/ggerganov/llama.cpp), [LMStudio](https://github.com/lmstudio-ai), ...) that support providing support for the OpenAI API ([text generation](https://platform.openai.com/docs/guides/text-generation), [function calling](https://platform.openai.com/docs/guides/function-calling), etc.).

## Usage

It is highly recommended to use [uv](https://docs.astral.sh/uv/) in your project based on mpc-openai:

- It manages python installation and virtual environment.
- It is an executable that can run self-contained python scripts (in our case MCP server)
- It is used for CI workflows.

Add mcp-openai to your project dependencies with:

```
uv add mcp-openai
```

or use classic pip install.

### Create a MCP client

Now you can create a MCP client by specifying your custom configuration.

```python
from mcp_openai import MCPClient
from mcp_openai import config

mcp_client_config = config.MCPClientConfig(
    mcpServers={
        "the-name-of-the-server": config.MCPServerConfig(
            command="uv",
            args=["run", "path/to/server/scripts.py/or/github/raw"],
        )
        # add here other servers ...
    }
)

llm_client_config = config.LLMClientConfig(
    api_key="api-key-for-auth",
    base_url="https://api.openai.com/v1",
)

llm_request_config = config.LLMRequestConfig(model=os.environ["MODEL_NAME"])

client = MCPClient(
    mcp_client_config,
    llm_client_config,
    llm_request_config,
)
```

### Connect and process messages with MCP client

```python
async def main():

    # Establish connection between the client and the server.
    await client.connect_to_server(server_name)

    # messages_in are coming from user interacting with the LLM
    # e.g. UI making use of this MCP client.
    messages_in = ...
    messages_out = await client.process_messages(messages_in)

    # messages_out contains the LLM response. If required, the LLM make use of
    # the available tools offered by the connected servers.
```
