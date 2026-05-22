import zipfile
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import RAW_DATA_DIR, DATASET_URL


def download_dataset(url: str = DATASET_URL, data_dir: Path = RAW_DATA_DIR) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "movielens.zip"

    if not zip_path.exists():
        print(f"Downloading dataset from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))

        with open(zip_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
        print("Download complete.")
    else:
        print("Dataset zip already exists, skipping download.")

    extract_dir = data_dir / "extracted"
    if not extract_dir.exists():
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        print("Extraction complete.")

    subdirs = [d for d in extract_dir.iterdir() if d.is_dir()]
    return subdirs[0] if subdirs else extract_dir


def load_ratings(data_path: Path) -> pd.DataFrame:
    ratings_file = data_path / "ratings.csv"
    print(f"Loading ratings from {ratings_file}...")
    df = pd.read_csv(ratings_file, dtype={"userId": "int32", "movieId": "int32", "rating": "float32"})
    print(f"Loaded {len(df):,} ratings from {df['userId'].nunique():,} users and {df['movieId'].nunique():,} movies")
    return df


def load_movies(data_path: Path) -> pd.DataFrame:
    movies_file = data_path / "movies.csv"
    print(f"Loading movies from {movies_file}...")
    df = pd.read_csv(movies_file, dtype={"movieId": "int32"})
    print(f"Loaded {len(df):,} movies")
    return df
