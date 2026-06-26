"""
Steps:
1. Load MovieLens 1M data
2. Fetch director name + production countries from TMDb API
3. Annotate director gender using gender-guesser
4. Classify production country as western / non-western
5. Save enriched CSV for graph building (Day 4+)

Requirements:
    pip install requests gender-guesser pandas tqdm

Directory structure expected:
    /data/ml-1m/
        ratings.dat
        movies.dat
        users.dat

Download MovieLens 1M from:
    https://grouplens.org/datasets/movielens/1m/
"""

import os
import time
import requests
import pandas as pd
import gender_guesser.detector as gender
from tqdm import tqdm

# ─── CONFIG ───────────────────────────────────────────────────────────────────

TMDB_API_KEY = "5782260b08afc35762a551c188457464"   
DATA_DIR = "data/ml-25m"
OUTPUT_PATH  = "data/movies_enriched.csv"
CACHE_PATH   = "data/tmdb_cache.csv"  

# TMDb rate limit: ~40 requests/10 seconds. Sleep keeps it safe.
SLEEP_BETWEEN_REQUESTS = 0.26

# ─── WESTERN COUNTRIES ────────────────────────────────────────────────────────
# Production countries classified as "western"
WESTERN_COUNTRIES = {
    "US", "GB", "FR", "DE", "IT", "ES", "CA", "AU", "NL", "SE",
    "NO", "DK", "FI", "BE", "AT", "CH", "NZ", "IE", "PT", "LU"
}

# ─── LOAD MOVIELENS 1M ────────────────────────────────────────────────────────

def load_movielens(data_dir):
    movies = pd.read_csv(
        os.path.join(data_dir, "movies.csv"),   # .csv not .dat
        # no sep, encoding args needed
    ).rename(columns={"movieId": "movie_id"})   # normalise column name

    ratings = pd.read_csv(
        os.path.join(data_dir, "ratings.csv"),
    ).rename(columns={"movieId": "movie_id", "userId": "user_id"})

    print(f"Loaded: {len(movies)} movies, {len(ratings)} ratings")
    # No users.dat in 25M — return None for users
    return movies, ratings, None

# ─── TMDB SEARCH ──────────────────────────────────────────────────────────────

def extract_year(title):
    """Extract year from MovieLens title format: 'Movie Name (1999)'"""
    if "(" in title and ")" in title:
        try:
            return int(title[title.rfind("(")+1:title.rfind(")")])
        except:
            pass
    return None


def clean_title(title):
    """Remove year from title for TMDb search."""
    if "(" in title:
        return title[:title.rfind("(")].strip()
    return title.strip()


def fetch_tmdb_data(title, year, api_key):
    """
    Search TMDb for a movie and return:
    - director name (first credited director)
    - list of production country codes (ISO 3166-1)
    Returns (None, []) on failure.
    """
    search_url = "https://api.themoviedb.org/3/search/movie"
    params = {
        "api_key": api_key,
        "query": title,
        "year": year,
        "language": "en-US"
    }

    try:
        r = requests.get(search_url, params=params, timeout=10)
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return None, []

        movie_id = results[0]["id"]

        # Fetch credits + details together
        detail_url = f"https://api.themoviedb.org/3/movie/{movie_id}"
        detail_params = {
            "api_key": api_key,
            "append_to_response": "credits",
            "language": "en-US"
        }
        d = requests.get(detail_url, params=detail_params, timeout=10)
        d.raise_for_status()
        data = d.json()

        # Director
        crew = data.get("credits", {}).get("crew", [])
        directors = [c["name"] for c in crew if c.get("job") == "Director"]
        director = directors[0] if directors else None

        # Production countries (ISO codes)
        countries = [c["iso_3166_1"] for c in data.get("production_countries", [])]

        return director, countries

    except Exception as e:
        return None, []


# ─── GENDER ANNOTATION ────────────────────────────────────────────────────────

def annotate_gender(name, detector):
    """
    Use gender-guesser on first name.
    Returns: 'female', 'male', or 'unknown'
    """
    if not name:
        return "unknown"
    first_name = name.strip().split()[0]
    result = detector.get_gender(first_name)
    if result in ("female", "mostly_female"):
        return "female"
    elif result in ("male", "mostly_male"):
        return "male"
    else:
        return "unknown"


# ─── REGION CLASSIFICATION ────────────────────────────────────────────────────

def classify_region(country_codes):
    """
    Given a list of ISO country codes, classify the movie as:
    - 'western' if any production country is western
    - 'non-western' if all are non-western
    - 'unknown' if no country data
    """
    if not country_codes:
        return "unknown"
    for code in country_codes:
        if code in WESTERN_COUNTRIES:
            return "western"
    return "non-western"


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("data", exist_ok=True)

    # Load MovieLens
    movies, ratings, users = load_movielens(DATA_DIR)

    # Load cache if it exists (lets you resume after interruption)
    if os.path.exists(CACHE_PATH):
        cache = pd.read_csv(CACHE_PATH)
        cache["movie_id"] = cache["movie_id"].astype(int)
        done_ids = set(cache["movie_id"].tolist())
        print(f"Resuming from cache: {len(done_ids)} movies already fetched")
    else:
        cache = pd.DataFrame(columns=["movie_id", "director", "countries"])
        done_ids = set()

    detector = gender.Detector()
    rows = []

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed 
    cache_lock = threading.Lock()

    # First, load already-cached movies into rows directly
    for _, cached_row in cache.iterrows():
        mid = cached_row["movie_id"]
        if mid not in movies["movie_id"].values:
            continue
        movie_row = movies[movies["movie_id"] == mid].iloc[0]
        director = cached_row["director"] if pd.notna(cached_row["director"]) else None
        country_codes = cached_row["countries"].split("|") if pd.notna(cached_row["countries"]) and cached_row["countries"] else []
        rows.append({
            "movie_id":        mid,
            "title":           movie_row["title"],
            "genres":          movie_row["genres"],
            "director":        director,
            "director_gender": annotate_gender(director, detector),
            "countries":       "|".join(country_codes),
            "region":          classify_region(country_codes)
        })

    def process_movie(row):
        mid   = row["movie_id"]
        title = clean_title(row["title"])
        year  = extract_year(row["title"])
        director, country_codes = fetch_tmdb_data(title, year, TMDB_API_KEY)
        time.sleep(SLEEP_BETWEEN_REQUESTS)
        return mid, row["title"], row["genres"], director, country_codes
    
    movies["movie_id"] = movies["movie_id"].astype(int)
    pending = movies[~movies["movie_id"].isin(done_ids)]
    print(f"\nFetching TMDb data for {len(pending)} remaining movies (8 threads)...")

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_movie, row): row for _, row in pending.iterrows()}
        for future in tqdm(as_completed(futures), total=len(futures)):
            mid, title, genres, director, country_codes = future.result()

            director_gender = annotate_gender(director, detector)
            region = classify_region(country_codes)
            rows.append({
                "movie_id":        mid,
                "title":           title,
                "genres":          genres,
                "director":        director,
                "director_gender": director_gender,
                "countries":       "|".join(country_codes),
                "region":          region
            })

            with cache_lock:
                new_row = pd.DataFrame([{"movie_id": mid, "director": director,
                                         "countries": "|".join(country_codes)}])
                cache = pd.concat([cache, new_row], ignore_index=True)
                if len(cache) % 50 == 0:
                    cache.to_csv(CACHE_PATH, index=False)

    # Final cache save
    cache.to_csv(CACHE_PATH, index=False)

    # Build enriched dataframe
    enriched = pd.DataFrame(rows)

    # Print stats
    print("\n--- Enrichment Summary ---")
    print(f"Total movies:       {len(enriched)}")
    print(f"Director found:     {enriched['director'].notna().sum()}")
    print(f"Gender female:      {(enriched['director_gender'] == 'female').sum()}")
    print(f"Gender male:        {(enriched['director_gender'] == 'male').sum()}")
    print(f"Gender unknown:     {(enriched['director_gender'] == 'unknown').sum()}")
    print(f"Region western:     {(enriched['region'] == 'western').sum()}")
    print(f"Region non-western: {(enriched['region'] == 'non-western').sum()}")
    print(f"Region unknown:     {(enriched['region'] == 'unknown').sum()}")

    enriched.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved enriched data to: {OUTPUT_PATH}")

    # Also save ratings and users as-is for later steps
    ratings.to_csv("data/ratings.csv", index=False)
    #users.to_csv("data/users.csv", index=False)
    print("Saved ratings.csv and users.csv")


if __name__ == "__main__":
    main()
