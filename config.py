from pathlib import Path

# Directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "saved_models"

# Dataset URLs
MOVIELENS_25M_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
MOVIELENS_SMALL_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

# Use small dataset by default for quick start; set to MOVIELENS_25M_URL for full run
DATASET_URL = MOVIELENS_SMALL_URL

# Model hyperparameters
SVD_N_COMPONENTS = 100
CF_N_NEIGHBORS = 20
MIN_RATINGS_PER_USER = 5
MIN_RATINGS_PER_MOVIE = 2  # lowered: keeps niche movies that have at least 2 ratings

# Max seed movies used in Item CF recommend() — caps evaluation time
ITEM_CF_MAX_SEEDS = 30

# Recommendation settings
TOP_N = 10
TEST_SIZE = 0.2
RANDOM_STATE = 42
