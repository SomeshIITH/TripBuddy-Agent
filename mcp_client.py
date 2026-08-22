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


# ============================================================
# GLOBAL TOOL CACHE
# ============================================================

search_tool = None

aviation_tools = {}

weather_tool = None
forecast_tool = None

_initialized = False


# ============================================================
# INITIALIZE MCP TOOLS
# ============================================================

async def initialize_mcp():

    global search_tool
    global aviation_tools
    global weather_tool
    global forecast_tool
    global _initialized

    if _initialized:
        return

    print("Loading MCP tools...")

    tools = await client.get_tools()

    # --------------------------------------------------------
    # Tavily
    # --------------------------------------------------------

    search_tool = next(
        (
            tool
            for tool in tools
            if tool.name == "tavily_search"
        ),
        None
    )

    # --------------------------------------------------------
    # AviationStack
    # --------------------------------------------------------

    aviation_tool_names = {
        "flights_with_airline",
        "historical_flights_by_date",
        "flight_arrival_departure_schedule",
        "future_flights_arrival_departure_schedule",
        "random_aircraft_type",
        "random_airplanes_detailed_info",
        "random_countries_detailed_info",
        "random_cities_detailed_info",
        "list_airports",
        "list_airlines",
        "list_routes",
        "list_taxes",
    }

    aviation_tools = {
        tool.name: tool
        for tool in tools
        if tool.name in aviation_tool_names
    }

    # --------------------------------------------------------
    # Weather
    # --------------------------------------------------------

    weather_tool = next(
        (
            tool
            for tool in tools
            if tool.name == "get_current_weather"
        ),
        None
    )

    forecast_tool = next(
        (
            tool
            for tool in tools
            if tool.name == "get_forecast"
        ),
        None
    )

    _initialized = True

    print("MCP tools loaded.")


# ============================================================
# TAVILY
# ============================================================

async def tavily_mcp_search(query: str):

    await initialize_mcp()

    if search_tool is None:
        return "Tavily tool unavailable"

    return await search_tool.ainvoke({
        "query": query
    })


# ============================================================
# AVIATIONSTACK
# ============================================================

async def aviation_mcp_call(
    tool_name: str,
    tool_args: dict = None
):

    await initialize_mcp()

    tool = aviation_tools.get(tool_name)

    if tool is None:
        return f"Tool '{tool_name}' unavailable"

    return await tool.ainvoke(
        tool_args or {}
    )


async def get_airports():

    await initialize_mcp()

    tool = aviation_tools.get("list_airports")

    if tool is None:
        return "Airport tool unavailable"

    return await tool.ainvoke({})


async def get_airlines():

    await initialize_mcp()

    tool = aviation_tools.get("list_airlines")

    if tool is None:
        return "Airline tool unavailable"

    return await tool.ainvoke({})


# ============================================================
# WEATHER
# ============================================================

async def weather_mcp_search(city: str):

    await initialize_mcp()

    if weather_tool is None:
        return "Weather tool unavailable"

    return await weather_tool.ainvoke({
        "city": city
    })


async def forecast_mcp_search(city: str):

    await initialize_mcp()

    if forecast_tool is None:
        return "Forecast tool unavailable"

    return await forecast_tool.ainvoke({
        "city": city
    })


# ============================================================
# LLM
# ============================================================

# llm = ChatGroq(
#     model=LLM_MODEL
# )

llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    temperature=0
)


# ============================================================
# DESTINATION EXTRACTION
# ============================================================

def extract_destination(query: str):

    prompt = f"""
Extract only the destination city or country from the
following travel request.

Travel request:
{query}

Return ONLY the destination name.
Do not add explanation.
"""

    response = llm.invoke(prompt)

    return response.content.strip()


# ============================================================
# TEST
# ============================================================

async def main():

    await initialize_mcp()

    print("\nMCP initialization successful.")


if __name__ == "__main__":
    asyncio.run(main())