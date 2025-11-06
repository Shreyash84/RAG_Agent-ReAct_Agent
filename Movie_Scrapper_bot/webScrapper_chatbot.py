# --- movie_agent_react_api.py (Gemini + TMDb + Google Places) ---
import os
import json
import time
import requests
from typing import List, Dict
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import AIMessage

# ----------------------------------------------------------
# ENVIRONMENT
# ----------------------------------------------------------
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "6391415fc15934ab1bd4304bf9249d85")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
REGION = "IN"
LANGUAGE = "en-US"
REQUEST_TIMEOUT = 10

# ----------------------------------------------------------
# TMDB HELPER FUNCTIONS
# ----------------------------------------------------------
def get_now_playing_tmdb(region: str = REGION) -> List[Dict]:
    """Fetch now playing movies via TMDb API."""
    url = "https://api.themoviedb.org/3/movie/now_playing"
    results = []
    for page in range(1, 3):  # fetch first 2 pages
        params = {
            "api_key": TMDB_API_KEY,
            "language": LANGUAGE,
            "region": region,
            "page": page,
        }
        r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            break
        data = r.json()
        for m in data.get("results", []):
            results.append({
                "id": m.get("id"),
                "title": m.get("title"),
                "release_date": m.get("release_date"),
                "language": m.get("original_language"),
                "overview": m.get("overview"),
                "poster": f"https://image.tmdb.org/t/p/w500{m.get('poster_path')}" if m.get("poster_path") else None
            })
        if page >= data.get("total_pages", 1):
            break
        time.sleep(0.2)
    return results

def search_movie_tmdb(title: str) -> Dict:
    """Search for a movie by title."""
    url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": title, "language": LANGUAGE}
    r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    if r.status_code != 200:
        return {"error": "Failed to fetch TMDb data"}
    data = r.json().get("results", [])
    return data[0] if data else {"message": "No results found"}

# ----------------------------------------------------------
# GOOGLE PLACES HELPER FUNCTIONS
# ----------------------------------------------------------
def get_theaters_open(city: str, limit: int = 15) -> Dict:
    """
    Find movie theaters near a city using OpenStreetMap (Nominatim + Overpass API).
    No API key required. 100% free and open-source.
    """
    try:
        # Step 1: Get city coordinates using Nominatim (OpenStreetMap)
        geo_url = "https://nominatim.openstreetmap.org/search"
        geo_params = {"q": city, "format": "json", "limit": 1}
        geo_res = requests.get(geo_url, params=geo_params, timeout=10, headers={"User-Agent": "MovieAI/1.0"})
        geo_data = geo_res.json()

        if not geo_data:
            return {"error": f"Could not geocode city: {city}"}

        lat, lon = geo_data[0]["lat"], geo_data[0]["lon"]

        # Step 2: Query Overpass API for cinemas nearby
        overpass_url = "https://overpass-api.de/api/interpreter"
        query = f"""
        [out:json][timeout:25];
        (
          node["amenity"="cinema"](around:20000,{lat},{lon});
          way["amenity"="cinema"](around:20000,{lat},{lon});
          relation["amenity"="cinema"](around:20000,{lat},{lon});
        );
        out center;
        """
        res = requests.post(overpass_url, data={"data": query}, timeout=25)
        data = res.json()

        if "elements" not in data or not data["elements"]:
            return {"message": f"No theaters found near {city}"}

        theaters = []
        for el in data["elements"][:limit]:
            tags = el.get("tags", {})
            theaters.append({
                "name": tags.get("name", "Unnamed Cinema"),
                "address": tags.get("addr:street", "N/A"),
                "operator": tags.get("operator", "Unknown"),
                "brand": tags.get("brand", None),
                "wikidata": tags.get("wikidata", None),
                "latitude": el.get("lat") or el.get("center", {}).get("lat"),
                "longitude": el.get("lon") or el.get("center", {}).get("lon"),
            })

        return {"city": city, "source": "OpenStreetMap", "theaters": theaters}

    except Exception as e:
        return {"error": str(e)}




# ----------------------------------------------------------
# TOOLS
# ----------------------------------------------------------
@tool
def get_now_playing(region: str = "IN") -> str:
    """List now playing movies in a region via TMDb."""
    return json.dumps(get_now_playing_tmdb(region)[:10], indent=2)

@tool
def search_movie(title: str) -> str:
    """Search a specific movie by title."""
    return json.dumps(search_movie_tmdb(title), indent=2)

@tool
def get_theaters(city: str) -> str:
    """
    Find movie theaters near a given city in India using Google Places.
    Example cities: "Mumbai", "Delhi", "Bangalore", "Pune".
    Returns a list of theaters with name, address, and rating.
    """
    return json.dumps(get_theaters_open(city), indent=2)

@tool
def find_movie_in_city(movie_title: str, city: str) -> str:
    """
    Try to determine if a given movie is currently playing in a region,
    and if yes, list possible theaters in that city where it may be running.
    """
    # Step 1: Fetch currently playing movies
    now_playing = get_now_playing_tmdb("IN")
    movie = None
    for m in now_playing:
        if movie_title.lower() in m["title"].lower():
            movie = m
            break

    if not movie:
        return json.dumps({
            "message": f"{movie_title} is not currently showing in India or not found."
        }, indent=2)

    # Step 2: Get theaters in the specified city (using OSM)
    theaters_data = get_theaters_open(city)
    theaters = theaters_data.get("theaters", [])

    # Step 3: Combine response
    result = {
        "movie": movie["title"],
        "release_date": movie["release_date"],
        "overview": movie["overview"][:180] + "...",
        "city": city,
        "likely_theaters": [t["name"] for t in theaters[:8]] if theaters else "No data found"
    }
    return json.dumps(result, indent=2)

# ----------------------------------------------------------
# GEMINI + AGENT
# ----------------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("API_KEY"),
    temperature=0.2,
)

agent = create_agent(
    model=llm,
    tools=[get_now_playing, search_movie, get_theaters, find_movie_in_city],
    system_prompt=(
        "You are a ReAct-style AI Movie Agent. "
        "You can use tools to:\n"
        "1. List now playing movies (get_now_playing).\n"
        "2. Search movie details (search_movie).\n"
        "3. Find theaters near any city (get_theaters).\n"
        "4. Find where a movie is playing in a city (find_movie_in_city).\n"
        "Always respond in JSON format without markdown."
    ),
)

# ----------------------------------------------------------
# INTERACTIVE LOOP (FIXED)
# ----------------------------------------------------------
if __name__ == "__main__":
    print("🎬 Movie AI Agent (API version) — type 'exit' to quit.\n")

    while True:
        query = input("🎤 You: ")
        if query.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break

        final_answer = None
        print("\n🧠 Reasoning Trace:\n" + "-" * 50)

        # ✅ Moved inside the while-loop so it runs for each query
        for chunk in agent.stream({"messages": [{"role": "user", "content": query}]}, stream_mode="updates"):

            # --- Detect tool usage ---
            if isinstance(chunk, dict) and "model" in chunk:
                msg = chunk["model"]["messages"][0]

                # ⚙️ Tool call detected
                if hasattr(msg, "additional_kwargs") and "function_call" in msg.additional_kwargs:
                    fn = msg.additional_kwargs["function_call"].get("name")
                    args = msg.additional_kwargs["function_call"].get("arguments")
                    print(f"🔧 Action: {fn}({args})")

                # ⚙️ AI text response detected
                elif isinstance(msg, AIMessage):
                    content = msg.content
                    text_output = ""

                    # Gemini sometimes returns a list of dicts [{"type": "text", "text": "..."}]
                    if isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                text_output += part.get("text", "")
                    elif isinstance(content, str):
                        text_output = content
                    else:
                        text_output = str(content)

                    # ✂️ Clean any metadata
                    text_output = (
                        text_output.replace("```json", "")
                        .replace("```", "")
                        .split("signature")[0]
                        .split("safetyRatings")[0]
                        .strip()
                    )

                    # 🧠 Try parsing to JSON
                    try:
                        parsed = json.loads(text_output)
                        final_answer = json.dumps(parsed, indent=2, ensure_ascii=False)
                    except Exception:
                        final_answer = text_output

                    if final_answer.strip():
                        print(f"\n✅ Final Answer:\n{final_answer}")
                        print("-" * 50)
                        break

            # --- Detect tool results ---
            elif isinstance(chunk, dict) and "tools" in chunk:
                try:
                    tool_msg = chunk["tools"]["messages"][0]
                    print(f"📊 Observation: {tool_msg.content[:200]}{'...' if len(tool_msg.content) > 200 else ''}")
                except Exception:
                    pass
