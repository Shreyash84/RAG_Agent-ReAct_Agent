import os
import requests
import time
import sys
from typing import List, Dict
from dotenv import load_dotenv
load_dotenv()

TMDB_API_KEY = "6391415fc15934ab1bd4304bf9249d85"  # required
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")  # optional

# Defaults
REGION = "IN"           # TMDb region code (India)
CITY_NAME = "Pune"      # city for theater search
TMDB_LANGUAGE = "en-US" # language for TMDb results
REQUEST_TIMEOUT = 10


def get_now_playing_tmdb(api_key: str, region: str = REGION, language: str = TMDB_LANGUAGE) -> List[Dict]:
    """
    Fetch now-playing movies from TMDb for a given region.
    Returns a list of movie dicts with keys: id, title, original_title, release_date, language.
    """
    if not api_key:
        raise ValueError("TMDB_API_KEY is required. Set TMDB_API_KEY environment variable.")

    base = "https://api.themoviedb.org/3/movie/now_playing"
    page = 1
    results = []
    while True:
        params = {
            "api_key": api_key,
            "language": language,
            "region": region,
            "page": page,
        }
        r = requests.get(base, params=params, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            raise RuntimeError(f"TMDb API error {r.status_code}: {r.text[:400]}")
        data = r.json()
        for m in data.get("results", []):
            results.append({
                "id": m.get("id"),
                "title": m.get("title"),
                "original_title": m.get("original_title"),
                "release_date": m.get("release_date"),
                "language": m.get("original_language"),
                "overview": m.get("overview"),
                "tmdb_popularity": m.get("popularity")
            })
        if page >= data.get("total_pages", 1):
            break
        page += 1
        # polite pause
        time.sleep(0.2)
    return results


def get_theaters_google(city: str, api_key: str, limit: int = 20) -> List[Dict]:
    """
    Use Google Places Text Search to find 'movie theaters in <city>'.
    Returns list of places with name, address, location.
    """
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is required for theater lookup.")
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        "query": f"movie theaters in {city}",
        "key": api_key,
    }
    r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    if r.status_code != 200:
        raise RuntimeError(f"Google Places error {r.status_code}: {r.text[:400]}")
    data = r.json()
    places = []
    for p in data.get("results", [])[:limit]:
        places.append({
            "name": p.get("name"),
            "address": p.get("formatted_address"),
            "rating": p.get("rating"),
            "user_ratings_total": p.get("user_ratings_total"),
            "location": p.get("geometry", {}).get("location"),
            "place_id": p.get("place_id")
        })
    return places


def print_movie_list(movies: List[Dict]):
    if not movies:
        print("No movies found.")
        return
    print(f"\nFound {len(movies)} movies now playing (TMDb):\n")
    for i, m in enumerate(movies, start=1):
        title = m.get("title")
        od = m.get("original_title")
        rel = m.get("release_date") or "N/A"
        lang = m.get("language") or "N/A"
        print(f"{i}. {title} ({od}) — release: {rel} — lang: {lang}")


def print_theaters(places: List[Dict]):
    if not places:
        print("No theaters found.")
        return
    print(f"\nFound {len(places)} theaters near {CITY_NAME}:\n")
    for i, p in enumerate(places, start=1):
        print(f"{i}. {p['name']} — {p['address']} — rating: {p.get('rating')} ({p.get('user_ratings_total')})")


def main():
    # Validate TMDb key
    if not TMDB_API_KEY:
        print("ERROR: TMDB_API_KEY environment variable not set.")
        print("Get a key from https://www.themoviedb.org/settings/api and set TMDB_API_KEY.")
        sys.exit(1)

    print("Fetching now-playing movies from TMDb (region=", REGION, ") ...")
    try:
        movies = get_now_playing_tmdb(TMDB_API_KEY, region=REGION)
    except Exception as e:
        print("Failed to fetch TMDb data:", e)
        sys.exit(1)

    print_movie_list(movies)

    # Optional: get nearby theaters using Google Places
    if GOOGLE_API_KEY:
        print(f"\nFetching theaters in {CITY_NAME} using Google Places...")
        try:
            theaters = get_theaters_google(CITY_NAME, GOOGLE_API_KEY)
            print_theaters(theaters)
        except Exception as e:
            print("Failed to fetch Google Places data:", e)
    else:
        print("\n(Google Places API key not provided — skipping theater lookup.)")

    # Return combined result (for programmatic use)
    return {
        "movies": movies,
        "theaters": theaters if GOOGLE_API_KEY else []
    }


if __name__ == "__main__":
    main()