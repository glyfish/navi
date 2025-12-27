"""SSE MCP client wrapper for shared use."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Mapping

from mcp.client.sse import sse_client
from mcp.client.session import ClientSession
from mcp.types import CallToolResult, Tool


@dataclass(slots=True)
class MCPClientConfig:
    """Configuration for connecting to an MCP server via SSE."""

    url: str
    headers: Mapping[str, str] | None = None
    timeout: float = 5.0


class MCPClient:
    """Async helper that wraps the official MCP SSE client."""

    def __init__(self, config: MCPClientConfig) -> None:
        self.config = config
        self._session: ClientSession | None = None
        self._transport_cm = None

    async def __aenter__(self) -> "MCPClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.disconnect()

    async def connect(self) -> None:
        if self._session:
            return
        headers = dict(self.config.headers) if self.config.headers is not None else None
        transport_cm = sse_client(self.config.url, headers=headers, timeout=self.config.timeout)
        read_stream, write_stream = await transport_cm.__aenter__()
        self._transport_cm = transport_cm
        self._session = ClientSession(read_stream, write_stream)
        await self._session.__aenter__()
        await self._session.initialize()

    async def disconnect(self) -> None:
        if not self._session:
            return
        await self._session.__aexit__(None, None, None)
        if self._transport_cm:
            await self._transport_cm.__aexit__(None, None, None)
        self._session = None
        self._transport_cm = None

    async def list_tools(self) -> list[Tool]:
        session = await self._ensure_session()
        result = await session.list_tools()
        return result.tools

    async def get_tool_schema(self, name: str) -> Mapping[str, Any] | None:
        tools = await self.list_tools()
        for tool in tools:
            if tool.name == name and tool.inputSchema:
                return tool.inputSchema
        return None

    async def call_tool(self, name: str, arguments: Mapping[str, Any] | None = None) -> CallToolResult:
        session = await self._ensure_session()
        payload = dict(arguments) if arguments is not None else {}
        return await session.call_tool(name, payload)

    async def stream_tool(
        self, name: str, arguments: Mapping[str, Any] | None = None
    ) -> AsyncIterator[CallToolResult]:
        """Iterate over results by repeatedly calling the tool.

        The current MCP client SDK does not expose a streaming API, so this helper
        simply yields the single response from `call_tool` to preserve the async
        iterator contract for callers that expect streaming semantics."""

        yield await self.call_tool(name, arguments)

    async def _ensure_session(self) -> ClientSession:
        if not self._session:
            await self.connect()
        assert self._session
        return self._session
