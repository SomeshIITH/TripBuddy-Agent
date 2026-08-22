# LangGraph Multi-Agent Travel Planning System

import os
import asyncio
import operator

import psycopg

from typing import TypedDict, Annotated

from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from langchain_groq import ChatGroq

from mcp_client import (
    tavily_mcp_search,
    aviation_mcp_call,
    extract_destination,
    forecast_mcp_search,
    weather_mcp_search,
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv(override=True)

DATABASE_URL = os.getenv("DATABASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL")


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
# STATE
# ============================================================

class TravelState(TypedDict):

    messages: Annotated[list[AnyMessage], operator.add]

    user_query: str

    flight_results: str

    hotel_results: str

    weather_results: str

    itinerary: str

    llm_calls: int


# ============================================================
# FLIGHT AGENT
# ============================================================

FLIGHT_PROMPT = """
You are a travel flight planning assistant.

User request:
{query}

Airport data:
{airport_data}

Airline data:
{airline_data}

Provide concise flight guidance.

Return ONLY:

Departure airport:
Arrival airport:
Airlines:
Typical duration:
Estimated airfare:
Booking advice:

Do not invent exact flight prices or schedules.
Clearly label estimates.
"""


def flight_agent(state: TravelState):

    print("\n========== FLIGHT AGENT ==========")

    query = state["user_query"]

    try:

        # Get aviation information through MCP
        airports = asyncio.run(
            aviation_mcp_call("list_airports")
        )

        airlines = asyncio.run(
            aviation_mcp_call("list_airlines")
        )

        # IMPORTANT:
        # Do not send huge MCP responses to the LLM.
        airport_data = str(airports)[:1500]
        airline_data = str(airlines)[:1500]

        prompt = FLIGHT_PROMPT.format(
            query=query,
            airport_data=airport_data,
            airline_data=airline_data,
        )

        print("Flight prompt length:", len(prompt))

        response = llm.invoke([
            SystemMessage(
                content="You are a concise flight planning assistant."
            ),
            HumanMessage(
                content=prompt
            )
        ])

        flight_data = response.content

    except Exception as e:

        flight_data = (
            f"Flight information unavailable: {str(e)}"
        )

    return {

        "flight_results": flight_data,

        "messages": [
            AIMessage(
                content="Flight recommendations generated."
            )
        ],

        "llm_calls":
            state.get("llm_calls", 0) + 1
    }


# ============================================================
# HOTEL AGENT
# ============================================================

HOTEL_PROMPT = """
Extract the best 3 hotel options from the search results.

Return ONLY this format:

Hotel 1:
Name:
Location:
Price:
Rating:
Why recommended:

Hotel 2:
Name:
Location:
Price:
Rating:
Why recommended:

Hotel 3:
Name:
Location:
Price:
Rating:
Why recommended:

Rules:

- Keep each hotel very short.
- Do not repeat search results.
- Do not invent missing information.
- If price or rating is unavailable, write "Not available".
"""


def hotel_agent(state: TravelState):

    print("\n========== HOTEL AGENT ==========")

    query = (
        f"Best hotels for {state['user_query']}"
    )

    try:

        # Search using Tavily MCP
        raw_results = asyncio.run(
            tavily_mcp_search(query)
        )

        # IMPORTANT:
        # Limit the amount of search data sent to LLM.
        search_data = str(raw_results)[:2500]

        prompt = f"""
{HOTEL_PROMPT}

Search Results:
{search_data}
"""

        print("Hotel prompt length:", len(prompt))

        response = llm.invoke([
            SystemMessage(
                content=(
                    "You are a hotel information "
                    "extraction assistant."
                )
            ),
            HumanMessage(
                content=prompt
            )
        ])

        hotel_data = response.content

    except Exception as e:

        hotel_data = (
            f"Hotel information unavailable: {str(e)}"
        )

    return {

        "hotel_results": hotel_data,

        "messages": [
            AIMessage(
                content="Hotel information fetched."
            )
        ],

        "llm_calls":
            state.get("llm_calls", 0) + 1
    }


# ============================================================
# WEATHER AGENT
# ============================================================

def weather_agent(state: TravelState):

    print("\n========== WEATHER AGENT ==========")

    try:

        # Extract destination
        city = extract_destination(
            state["user_query"]
        )

        print("Destination:", city)

        # Current weather
        weather_data = asyncio.run(
            weather_mcp_search(city)
        )

        # Forecast
        forecast_data = asyncio.run(
            forecast_mcp_search(city)
        )

        # Keep weather information compact
        weather_result = f"""
Current Weather:
{str(weather_data)[:1000]}

Forecast:
{str(forecast_data)[:1500]}
"""

    except Exception as e:

        weather_result = (
            f"Weather information unavailable: {str(e)}"
        )

    return {

        "weather_results": weather_result,

        "messages": [
            AIMessage(
                content="Weather information fetched."
            )
        ],

        "llm_calls":
            state.get("llm_calls", 0)
    }


# ============================================================
# ITINERARY AGENT
# ============================================================

ITINERARY_PROMPT = """
You are an expert travel planner.

Create a practical travel itinerary based on the
information provided below.

USER REQUEST:
{query}

FLIGHT:
{flight}

HOTELS:
{hotels}

WEATHER:
{weather}

Requirements:

1. Follow the user's duration and budget.
2. Create a day-by-day itinerary.
3. Include important attractions.
4. Suggest transportation.
5. Give approximate costs.
6. Keep the total trip within the stated budget
   when reasonably possible.
7. Clearly mark estimated costs.
8. Do not invent exact flight or hotel prices.
9. Do not repeat the raw data.
10. Keep the final answer concise.

Format:

Trip Summary
-----------

Day 1:
- Activities:
- Transport:
- Estimated cost:

Day 2:
- Activities:
- Transport:
- Estimated cost:

Day 3:
- Activities:
- Transport:
- Estimated cost:

Day 4:
- Activities:
- Transport:
- Estimated cost:

Budget Summary
--------------
Flights:
Hotels:
Food:
Transport:
Activities:
Estimated Total:

Important Notes:
- ...
"""


def itinerary_agent(state: TravelState):

    print("\n========== ITINERARY AGENT ==========")

    # Compact data passed to final LLM
    flight_data = str(
        state["flight_results"]
    )[:1800]

    hotel_data = str(
        state["hotel_results"]
    )[:2200]

    weather_data = str(
        state["weather_results"]
    )[:1500]

    prompt = ITINERARY_PROMPT.format(

        query=state["user_query"],

        flight=flight_data,

        hotels=hotel_data,

        weather=weather_data,
    )

    print(
        "Itinerary prompt length:",
        len(prompt)
    )

    response = llm.invoke([

        SystemMessage(
            content=(
                "You are a concise expert travel "
                "planner."
            )
        ),

        HumanMessage(
            content=prompt
        )
    ])

    return {

        "itinerary": response.content,

        "messages": [
            response
        ],

        "llm_calls":
            state.get("llm_calls", 0) + 1
    }


# ============================================================
# LANGGRAPH
# ============================================================

graph = StateGraph(TravelState)


graph.add_node(
    "flight_agent",
    flight_agent
)

graph.add_node(
    "hotel_agent",
    hotel_agent
)

graph.add_node(
    "weather_agent",
    weather_agent
)

graph.add_node(
    "itinerary_agent",
    itinerary_agent
)


# Sequential workflow

graph.add_edge(
    START,
    "flight_agent"
)

graph.add_edge(
    "flight_agent",
    "hotel_agent"
)

graph.add_edge(
    "hotel_agent",
    "weather_agent"
)

graph.add_edge(
    "weather_agent",
    "itinerary_agent"
)

graph.add_edge(
    "itinerary_agent",
    END
)


# ============================================================
# POSTGRES CHECKPOINTER
# ============================================================

_conn = psycopg.connect(
    DATABASE_URL
)

checkpointer = PostgresSaver(
    _conn
)

checkpointer.setup()

app = graph.compile(
    checkpointer=checkpointer
)


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    import uuid

    user_input = input(
        "Enter travel request: "
    )

    config = {

        "configurable": {

            "thread_id":
                str(uuid.uuid4())

        }

    }

    initial_state = {

        "messages": [
            HumanMessage(
                content=user_input
            )
        ],

        "user_query":
            user_input,

        "flight_results":
            "",

        "hotel_results":
            "",

        "weather_results":
            "",

        "itinerary":
            "",

        "llm_calls":
            0,
    }

    result = app.invoke(
        initial_state,
        config=config
    )

    print(
        "\n\n========== FINAL RESPONSE ==========\n"
    )

    print(
        result["itinerary"]
    )