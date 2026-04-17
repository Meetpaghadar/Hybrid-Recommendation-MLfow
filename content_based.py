import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from category_encoders.count import CountEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import save_npz


# ----------------------------
# CONFIG
# ----------------------------
CLEANED_DATA_PATH = "data/cleaned_data.csv"

frequency_encode_cols = ['year']
ohe_cols = ["artist", "time_signature", "key"]
tfidf_col = 'tags'
standard_scale_cols = ["duration_ms", "loudness", "tempo"]
min_max_scale_cols = [
    "danceability", "energy", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence"
]


# ----------------------------
# DATA CLEANING FOR MODEL
# ----------------------------
def prepare_data(df):
    df = df.copy()

    # ensure correct dtypes (VERY IMPORTANT)
    for col in ohe_cols + frequency_encode_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    if tfidf_col in df.columns:
        df[tfidf_col] = df[tfidf_col].astype(str)

    return df


# ----------------------------
# TRANSFORMER TRAINING
# ----------------------------
def train_transformer(data):
    transformer = ColumnTransformer(
        transformers=[
            ("frequency_encode", CountEncoder(normalize=True), frequency_encode_cols),
            ("ohe", OneHotEncoder(handle_unknown="ignore"), ohe_cols),
            ("tfidf", TfidfVectorizer(max_features=85), tfidf_col),
            ("standard_scale", StandardScaler(), standard_scale_cols),
            ("min_max_scale", MinMaxScaler(), min_max_scale_cols)
        ],
        remainder='drop'
    )

    transformer.fit(data)

    joblib.dump(transformer, "transformer.joblib")
    print("✅ Transformer trained and saved")


# ----------------------------
# TRANSFORM DATA
# ----------------------------
def transform_data(data):
    transformer = joblib.load("transformer.joblib")
    return transformer.transform(data)


# ----------------------------
# SAVE TRANSFORMED MATRIX
# ----------------------------
def save_transformed_data(transformed_data, path):
    save_npz(path, transformed_data)
    print("Transformed data saved")


# ----------------------------
# SIMILARITY
# ----------------------------
def calculate_similarity(input_vector, data):
    return cosine_similarity(input_vector, data)


# ----------------------------
# RECOMMENDER
# ----------------------------
def content_recommendation(song_name, artist_name, songs_data, transformed_data, k=10):

    song_name = song_name.lower()
    artist_name = artist_name.lower()

    mask = (
        (songs_data["name"] == song_name) &
        (songs_data["artist"] == artist_name)
    )

    song_row = songs_data.loc[mask]

    if song_row.empty:
        return "Song not found"

    song_index = song_row.index[0]

    input_vector = transformed_data[song_index].reshape(1, -1)

    similarity_scores = calculate_similarity(input_vector, transformed_data)

    top_k_indexes = np.argsort(similarity_scores.ravel())[-(k+1):-1][::-1]

    results = songs_data.iloc[top_k_indexes][
        ["name", "artist", "spotify_preview_url"]
    ].reset_index(drop=True)

    return results


# ----------------------------
# PIPELINE TEST
# ----------------------------
def test_pipeline(data_path, song_name, artist_name):

    # load data
    df = pd.read_csv(data_path)

    # clean for ML
    df = prepare_data(df)

    # train transformer
    train_transformer(df)

    # transform data
    transformed = transform_data(df)

    # save matrix
    save_transformed_data(transformed, "data/transformed_data.npz")

    # recommend
    recs = content_recommendation(song_name, artist_name, df, transformed)

    print(recs)


# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    test_pipeline(
        CLEANED_DATA_PATH,
        "hips don't lie",
        "shakira"
    )