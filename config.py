import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
AVIATION_STACK_API_KEY = os.getenv("AVIATION_STACK_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

def get_llm():
    # return ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    return ChatGoogleGenerativeAI(model="gemini-3.5-flash",temperature=0)