import pandas as pd
import numpy as np
import re
from datetime import datetime
from sklearn.preprocessing import MultiLabelBinarizer
import nltk
from nltk.corpus import stopwords

# Ensure NLTK stopwords are available
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))


def clean_text(text: str) -> str:
    """
    Lowercase, strip punctuation/extra whitespace, remove English stopwords.
    """
    if pd.isna(text):
        return ""
    s = text.lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    words = s.split()
    return " ".join(w for w in words if w not in stop_words)


def fill_missing_date(date_str: str) -> str:
    """
    Normalize release_date formats:
    - YYYY     -> YYYY-01-01
    - YYYY-MM  -> YYYY-MM-01
    """
    if pd.isna(date_str):
        return np.nan
    ds = str(date_str).strip()
    if re.fullmatch(r"\d{4}", ds):
        return ds + "-01-01"
    if re.fullmatch(r"\d{4}-\d{2}", ds):
        return ds + "-01"
    return ds


def convert_to_minutes(duration_str: str) -> float:
    """
    Parse durations like '2 h 30 min' to total minutes.
    """
    if pd.isna(duration_str):
        return np.nan
    s = duration_str.strip().lower()
    h = re.search(r"(\d+)\s*h", s)
    m = re.search(r"(\d+)\s*min", s)
    hours = int(h.group(1)) if h else 0
    mins = int(m.group(1)) if m else 0
    return hours * 60 + mins


def preprocess(movie_details: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and merge movie_details & reviews; returns merged DataFrame.
    """
    # Text cleaning
    for col in ['plot_summary', 'plot_synopsis']:
        movie_details[col] = movie_details[col].map(clean_text)
    for col in ['review_text', 'review_summary']:
        reviews[col] = reviews[col].map(clean_text)

    # Date normalization
    movie_details['release_date'] = (
        movie_details['release_date']
            .map(fill_missing_date)
            .pipe(pd.to_datetime, format="%Y-%m-%d", errors='coerce')
    )
    reviews['review_date'] = pd.to_datetime(
        reviews['review_date'], format="%d %B %Y", errors='coerce'
    )

    # Days since events
    today = pd.to_datetime(datetime.today().date())
    movie_details['days_since_release'] = (today - movie_details['release_date']).dt.days
    reviews['days_since_review'] = (today - reviews['review_date']).dt.days

    # Genre one-hot encoding
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(movie_details['genre'])
    genre_df = pd.DataFrame(genre_matrix, columns=mlb.classes_, index=movie_details.index)
    movie_details = pd.concat([movie_details, genre_df], axis=1)

    # Duration minutes
    movie_details['duration_min'] = movie_details['duration'].map(convert_to_minutes)

    # Merge datasets
    merged = reviews.merge(
        movie_details,
        on='movie_id',
        how='left',
        suffixes=('_review', '_movie')
    )

    # Flags for summaries/synopses
    merged['has_summary'] = merged['review_summary'].str.strip().ne("").astype(int)
    merged['has_synopsis'] = merged['plot_synopsis'].str.strip().ne("").astype(int)

    return merged
