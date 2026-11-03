"""Run from backend: python -m mcp_tools.server (standard MCP stdio transport)."""
from mcp.server.fastmcp import FastMCP

from tool_registry import invoke_tool

mcp = FastMCP("ZeroCode tools")


@mcp.tool()
def query_database(sql: str) -> dict:
    """Execute a bounded read-only PostgreSQL query."""
    return invoke_tool("query_database", sql=sql)


@mcp.tool()
def call_llm(prompt: str, system: str = "") -> str:
    """Generate guarded text with the configured Gemini model."""
    return invoke_tool("call_llm", prompt=prompt, system=system)


@mcp.tool()
def parse_csv(filepath: str) -> dict:
    """Validate and summarize a CSV inside outputs."""
    return invoke_tool("parse_csv", filepath=filepath)


if __name__ == "__main__":
    from database import init_db
    init_db()
    mcp.run(transport="stdio")
