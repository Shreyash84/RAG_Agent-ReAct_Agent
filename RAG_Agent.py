# --- agent_function_calling.py ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent

import os
from datetime import datetime
import pytz
from dotenv import load_dotenv

load_dotenv()

# --- 1️⃣ Define custom functions ---
def check_weather(location: str) -> str:
    """Return a mock weather forecast for the specified location."""
    return f"The weather in {location} is currently sunny with a temperature of 28°C."

def get_time_in_city(city: str) -> str:
    """Return the current local time in the specified city (timezone format)."""
    try:
        timezone = pytz.timezone(city)
        current_time = datetime.now(timezone).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {city} is {current_time}."
    except Exception:
        return f"Sorry, I couldn't find timezone info for {city}. Please use format like 'Asia/Kolkata'."

# --- 2️⃣ Define Gemini LLM ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("API_KEY"),
    temperature=0.2,
)

# --- 3️⃣ Define the DuckDuckGo search tool ---
search_tool = DuckDuckGoSearchRun()

# --- 4️⃣ Create the Agent ---
agent = create_agent(
    model=llm,
    tools=[search_tool, check_weather, get_time_in_city],  # 🔥 You can pass functions directly
    system_prompt="You are a helpful assistant who uses tools when necessary to provide accurate answers.",
)

# --- 5️⃣ Run Example Queries ---
queries = [
    "What's the current time in Asia/Kolkata?",
    "How's the weather in Mumbai?",
    "Who is the current Prime Minister of Japan?",
]

for q in queries:
    inputs = {"messages": [{"role": "user", "content": q}]}
    result = agent.invoke(inputs)

    # --- Cleanly print only text output ---
    final_message = result["messages"][-1]
    if isinstance(final_message.content, list):
        text_output = "".join([c["text"] for c in final_message.content if c["type"] == "text"])
    else:
        text_output = final_message.content

    print(f"\n🧠 Q: {q}")
    print(f"✅ A: {text_output.strip()}")
