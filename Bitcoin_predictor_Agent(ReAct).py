# --- react_agent.py (updated for LangChain v1+) ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage

import os
from dotenv import load_dotenv

load_dotenv()

# 1) Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("API_KEY"),
    temperature=0.2,
)

# 2) Define tools
search_tool = DuckDuckGoSearchRun()  # BaseTool from community package

@tool
def btc_to_inr(usd_price: float) -> str:
    """Convert Bitcoin price in USD to INR (approx conversion)."""
    conversion_rate = 83.1  # could be dynamic from forex API
    return f"₹{usd_price * conversion_rate:,.2f} INR"

# 3) Build ReAct-style agent via create_agent
agent = create_agent(
    model=llm,
    tools=[search_tool, btc_to_inr],
    system_prompt=(
        "You are a careful, ReAct-style assistant: plan, use tools when needed, "
        "and return a concise final answer with brief justification."
    ),
)

# 4) Run an example query
query = "Find the current price of Bitcoin and summarize its 3-day trend and convert it's value to inr."
result = agent.invoke({"messages": [{"role": "user", "content": query}]})

# 5) Print only final answer
#print(result["messages"][-1])

final_answer = None

print("\n🧠 Reasoning Trace:\n" + "-" * 50)

for chunk in agent.stream({"messages": [{"role": "user", "content": query}]}, stream_mode="updates"):
    # --- Detect tool usage ---
    if "function_call" in str(chunk):
        try:
            fn = chunk["model"]["messages"][0].additional_kwargs["function_call"]["name"]
            args = chunk["model"]["messages"][0].additional_kwargs["function_call"]["arguments"]
            print(f"🔧 Action: {fn}({args})")
        except Exception:
            pass

    # --- Detect tool results ---
    elif "tools" in chunk:
        try:
            tool_msg = chunk["tools"]["messages"][0]
            print(f"📊 Observation: {tool_msg.content[:200]}{'...' if len(tool_msg.content) > 200 else ''}")
        except Exception:
            pass

    # --- Detect final model response ---
    elif "model" in chunk and isinstance(chunk["model"]["messages"][0], AIMessage):
        ai_msg = chunk["model"]["messages"][0]
        if ai_msg.content:
            # Handle both list or plain string content
            if isinstance(ai_msg.content, list) and "text" in ai_msg.content[0]:
                final_answer = ai_msg.content[0]["text"]
            else:
                final_answer = ai_msg.content

            # 🧹 Clean response to stop after INR conversion
            if "INR" in final_answer:
                final_answer = final_answer.split("INR")[0] + "INR"
            
            print(f"\n✅ Final Answer: {final_answer}")
            print("-" * 50)
            break  # Stop after INR conversion


