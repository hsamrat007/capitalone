# context_retriever.py
import os
import glob
import pickle
import pdfplumber
import requests
import time

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import torch

# ======= Config =======
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = "pdf_index.pkl"
CHUNK_SIZE = 200     # words per chunk
CHUNK_OVERLAP = 50   # words overlap

# ======= Globals =======
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
EMBED_MODEL = SentenceTransformer(EMBED_MODEL_NAME, device=device)
_INDEX_DATA = None

# ======= PDF chunking & indexing =======
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

def build_pdf_index(pdf_folder="all_pdfs_dummy", index_path=INDEX_PATH):
    """
    Walk all PDFs in pdf_folder, extract text, chunk, embed and save an index file.
    Run this once (or when your PDFs change).
    """
    chunks = []
    metadatas = []

    # Sort PDFs by size (largest first)
    pdf_paths = sorted(glob.glob(os.path.join(pdf_folder, "*.pdf")), key=os.path.getsize, reverse=True)
    print(pdf_folder)
    if not pdf_paths:
        print(f"No PDFs found in {pdf_folder}")
        return False

    print(f"Found {len(pdf_paths)} PDFs. Extracting text and chunking...")
    for idx, pdf_path in enumerate(pdf_paths, start=1):
        src = os.path.basename(pdf_path)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for pnum, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    page_chunks = chunk_text(text)
                    for c in page_chunks:
                        chunks.append(c)
                        metadatas.append({"source": src, "page": pnum})
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")

        # Show progress every 50 PDFs or at the very end
        if idx % 50 == 0 or idx == len(pdf_paths):
            print(f"  Processed {idx}/{len(pdf_paths)} PDFs, chunks so far: {len(chunks)}")

    print("\nComputing embeddings (this can take a while)...")
    print(f"Total chunks to embed: {len(chunks)}")
    batch_size = 256 if device == "cuda" else 32  # adjust if GPU OOM

    embeddings = EMBED_MODEL.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=batch_size
    )

    # Fit a NearestNeighbors index
    nn = NearestNeighbors(metric="cosine")
    nn.fit(embeddings)

    data = {"chunks": chunks, "embeddings": embeddings, "metadatas": metadatas}
    with open(index_path, "wb") as f:
        pickle.dump(data, f)

    size_mb = os.path.getsize(index_path) / (1024 * 1024)
    print(f"\nIndex saved to {index_path} (chunks: {len(chunks)}, size: {size_mb:.2f} MB)")

    return True


def _load_index(index_path=INDEX_PATH):
    global _INDEX_DATA
    if _INDEX_DATA is not None:
        return _INDEX_DATA
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index file not found: {index_path}. Run build_pdf_index() first.")
    with open(index_path, "rb") as f:
        data = pickle.load(f)
    # attach NearestNeighbors fitted model
    nn = NearestNeighbors(metric="cosine")
    nn.fit(data["embeddings"])
    data["nn"] = nn
    _INDEX_DATA = data
    return _INDEX_DATA

import re

def get_pdf_context(query, top_k=3, index_path=INDEX_PATH):
    """Return top_k relevant PDF chunks (cleaned, no source refs) for the query."""
    data = _load_index(index_path)
    q_emb = EMBED_MODEL.encode([query], convert_to_numpy=True)

    # ensure top_k <= n_samples
    n_samples = data["embeddings"].shape[0]
    k = min(top_k, n_samples)

    distances, indices = data["nn"].kneighbors(q_emb, n_neighbors=k)
    pieces = []

    for idx in indices[0]:
        chunk_text = data["chunks"][idx]

        # 🔹 Remove extra whitespace/newlines
        text = re.sub(r"\s+", " ", chunk_text).strip()

        if text:
            pieces.append(f"• {text}")

    return "\n".join(pieces) if pieces else None

# ======= Geocoding & Tomorrow.io weather =======
def nominatim_geocode(location_query, sleep_between=1.0):
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": location_query, "format": "json", "limit": 1}
    headers = {"User-Agent": "AgroAdvisorApp/1.0"}
    resp = requests.get(url, params=params, headers=headers)
    if resp.status_code != 200:
        return None
    data = resp.json()
    if not data:
        return None
    lat = float(data[0]["lat"])
    lon = float(data[0]["lon"])
    name = data[0]["display_name"]
    time.sleep(sleep_between)
    return lat, lon, name

# def get_weather_context_from_tomorrow(latitude, longitude, api_key, days=3):
#     url = f"https://api.tomorrow.io/v4/timelines"
#     params = {
#         "location": f"{latitude},{longitude}",
#         "timesteps": "1d",
#         "units":"metric",
#         "fields": ["temperatureAvg","precipitationSum","windSpeed","humidity"],
#         "apikey": api_key
#     }
#     # tomorrow.io may expect a JSON body; some accounts use different endpoints. Check the API docs if you get 400.
#     resp = requests.get(url, params=params)
#     if resp.status_code != 200:
#         # fallback: return raw status
#         return None
#     data = resp.json()
#     # Drip a concise summary (next `days`)
#     timelines = data.get("data", {}).get("timelines", [])
#     if not timelines:
#         return None
#     # Find daily timeline (if available)
#     daily = None
#     for t in timelines:
#         if t.get("timestep","").lower().startswith("1d"):
#             daily = t
#             break
#     if not daily:
#         daily = timelines[0]
#     entries = daily.get("entries", [])[:days]
#     summary_lines = []
#     for e in entries:
#         dt = e.get("time","")[:10]
#         vals = e.get("values", {})
#         summary_lines.append(f"{dt}: temp_avg={vals.get('temperatureAvg','?')}°C, rain={vals.get('precipitationSum','?')}mm, wind={vals.get('windSpeed','?')} m/s")
#     return "Weather forecast summary:\n" + "\n".join(summary_lines)

# def generate_weather_context(location_query, tomorrow_api_key):
#     geo = nominatim_geocode(location_query)
#     if not geo:
#         return None
#     lat, lon, resolved_name = geo
#     weather = get_weather_context_from_tomorrow(lat, lon, tomorrow_api_key)
#     if not weather:
#         return f"Resolved location: {resolved_name} ({lat},{lon}). Weather lookup failed or returned no data."
#     return f"Resolved location: {resolved_name} ({lat},{lon}).\n\n{weather}"

def generate_weather_context(location_query, _api_key=None):
    geo = nominatim_geocode(location_query)
    if not geo:
        return None
    lat, lon, resolved_name = geo
    weather = get_weather_context_open_meteo(lat, lon)
    if not weather:
        return f"Resolved location: {resolved_name} ({lat},{lon}). Weather lookup failed or returned no data."
    return f"Resolved location: {resolved_name} ({lat},{lon}).\n\n{weather}"


def get_weather_context_open_meteo(latitude, longitude, days=3):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode",
        "timezone": "auto",
        "forecast_days": days
    }
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        return None
    data = resp.json()

    daily = data.get("daily", {})
    dates = daily.get("time", [])
    temps_max = daily.get("temperature_2m_max", [])
    temps_min = daily.get("temperature_2m_min", [])
    precipitation = daily.get("precipitation_sum", [])
    weather_codes = daily.get("weathercode", [])

    summary_lines = []
    for i in range(len(dates)):
        summary_lines.append(
            f"{dates[i]}: Max {temps_max[i]}°C, Min {temps_min[i]}°C, Rain {precipitation[i]}mm, Weather Code {weather_codes[i]}"
        )

    return "Weather forecast summary (Open-Meteo):\n" + "\n".join(summary_lines)
