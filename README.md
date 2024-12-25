<div align="center">
  <h1>𝔐&nbsp;&nbsp;mpc-openai&nbsp;&nbsp;✧</h1>
  <p><em>MCP Client with OpenAI compatible API</em></p>
</div>

> Model Context Protocol (MCP) is an open protocol that standardizes how applications provide context to LLMs. Think of MCP like a USB-C port for AI applications.
>
> — [https://modelcontextprotocol.io](https://modelcontextprotocol.io)

______________________________________________________________________

> [!WARNING]
> This project is still in the early stages of development. Support is not planned.

This is a MCP **client** (not a server). It is meant to be used as a library for building LLMs UI that support MCP through an OpenAI compatible API. This opens the door to locally runnable inference engines ([vLLM](https://github.com/vllm-project/vllm), [Ollama](https://github.com/ollama/ollama), [TGI](https://github.com/huggingface/text-generation-inference), [llama.cpp](https://github.com/ggerganov/llama.cpp), [LMStudio](https://github.com/lmstudio-ai), ...) that support providing support for the OpenAI API ([text generation](https://platform.openai.com/docs/guides/text-generation), [function calling](https://platform.openai.com/docs/guides/function-calling), etc.).

<!--TODO: mermaid diagram-->

<!--TODO: usage-->
