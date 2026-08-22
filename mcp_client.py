import os
import asyncio

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv(override=True)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
AVIATION_STACK_API_KEY = os.getenv("AVIATION_STACK_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL")


# ============================================================
# MCP CLIENT
# ============================================================

client = MultiServerMCPClient(
    {
        # ----------------------------------------------------
        # Tavily MCP
        # ----------------------------------------------------
        "tavily": {
            "transport": "streamable_http",
            "url": (
                f"https://mcp.tavily.com/mcp/"
                f"?tavilyApiKey={TAVILY_API_KEY}"
            )
        },

        # ----------------------------------------------------
        # AviationStack MCP
        # ----------------------------------------------------
        "aviationstack": {
            "transport": "stdio",

            "command": (
                "/Users/somesh/Desktop/New Development/"
                "TripAgent/aviationstack-mcp/.venv/bin/python"
            ),

            "args": [
                "-m",
                "aviationstack_mcp",
                "mcp",
                "run"
            ],

            "env": {
                "AVIATION_STACK_API_KEY": AVIATION_STACK_API_KEY
            }
        },

        # ----------------------------------------------------
        # Weather MCP
        # ----------------------------------------------------
        "weather": {
            "transport": "stdio",

            "command": (
                "/opt/anaconda3/envs/TripAgent/bin/python"
            ),

            "args": [
                "/Users/somesh/Desktop/New Development/"
                "TripAgent/weather-mcp-server.py"
            ],

            "env": {
                "OPENWEATHER_API_KEY": OPENWEATHER_API_KEY
            }
        }
    }
)


# Cache tools so we don't load them repeatedly
_tools_cache = None


async def get_tools():
    global _tools_cache

    if _tools_cache is None:
        try:
            _tools_cache = await client.get_tools()

        except Exception as e:
            print("\n========== FULL ERROR ==========")
            print(type(e))
            print(repr(e))

            if hasattr(e, "exceptions"):
                print("\nSUB EXCEPTIONS:")
                for i, sub in enumerate(e.exceptions):
                    print(f"\n--- Exception {i+1} ---")
                    print(type(sub))
                    print(repr(sub))

            raise

    return _tools_cache

async def call_tool(tool_name: str, args: dict = None):
    tools = await get_tools()

    tool = next(
        (tool for tool in tools if tool.name == tool_name),
        None,
    )

    if tool is None:
        raise ValueError(f"Tool '{tool_name}' not found")

    return await tool.ainvoke(args or {})


# ------------------------
# Tavily MCP Tools
# ------------------------



async def tavily_search(query: str):
    return await call_tool("tavily_search", {"query": query})


async def list_airports(search: str = "", limit: int = 10):
    return await call_tool("list_airports", {"search": search, "limit": limit, "offset": 0})


async def list_airlines(search: str = "", limit: int = 10):
    return await call_tool("list_airlines", {"search": search, "limit": limit, "offset": 0})


async def current_weather(city: str):
    return await call_tool("get_current_weather", {"city": city})


async def forecast(city: str):
    return await call_tool("get_forecast", {"city": city})