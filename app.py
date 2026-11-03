"""
Aplikasi Rekomendasi Lagu Berbasis Genre
Alur: 1) pilih lagu -> 2) prediksi genre tambahan -> 3) rekomendasi lagu serupa
File yang dibutuhkan di folder yang sama: df_rec.csv (hasil cleaning dari notebook)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).parent / "df_rec.csv"

st.set_page_config(page_title="Rekomendasi Lagu", page_icon="🎵")


# ---------- Data dan model: dimuat sekali, lalu disimpan di cache ----------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, usecols=["title", "genres", "year"])
    df = df.dropna(subset=["genres"]).reset_index(drop=True)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    tahun = df["year"].astype("string").fillna("tahun ?")
    df["label"] = df["title"] + " (" + tahun + ") – " + df["genres"].str.replace("|", ", ", regex=False)
    return df


@st.cache_resource
def build_tfidf():
    # Sama dengan notebook: satu genre utuh = satu token
    vectorizer = TfidfVectorizer(token_pattern=r"[^|]+")
    matrix = vectorizer.fit_transform(load_data()["genres"])
    return vectorizer, matrix


@st.cache_resource
def train_genre_models():
    # Sama dengan notebook: one-hot genre, split 80/20 (random_state=42), satu model per genre
    G = load_data()["genres"].str.get_dummies(sep="|")
    G_train, _ = train_test_split(G, test_size=0.2, random_state=42)
    models = {
        g: LogisticRegression(max_iter=1000).fit(G_train.drop(columns=g), G_train[g])
        for g in G.columns
    }
    return G.columns.tolist(), models


def predict_genres(genre_list, top_n=3):
    cols, models = train_genre_models()
    x = pd.DataFrame([[int(g in genre_list) for g in cols]], columns=cols)
    probs = {
        g: models[g].predict_proba(x.drop(columns=g))[0, 1]
        for g in cols
        if g not in genre_list
    }
    return pd.Series(probs).sort_values(ascending=False).head(top_n)


def recommend(idx, query_genres, top_n=10):
    df = load_data()
    vectorizer, matrix = build_tfidf()
    # Kemiripan hanya dihitung antara profil pencarian dan semua lagu (bukan matriks penuh)
    query = vectorizer.transform(["|".join(query_genres)])
    sims = cosine_similarity(query, matrix).ravel()
    out = df[["title", "genres", "year"]].assign(similarity=sims).drop(index=idx)
    y0 = df.at[idx, "year"]
    # Tie-breaker: skor sama -> tahun terdekat dulu; lagu tanpa tahun di urutan terakhir
    out["year_gap"] = (out["year"] - y0).abs().astype("float") if pd.notna(y0) else np.nan
    out = out.sort_values(["similarity", "year_gap"], ascending=[False, True], na_position="last")
    n_identical = int((out["similarity"] > 0.9999).sum())
    return out.head(top_n).drop(columns="year_gap"), n_identical


# ---------- Tampilan ----------
df = load_data()

st.title("🎵 Rekomendasi Lagu Berbasis Genre")
st.caption(
    "Pilih sebuah lagu, lihat genre lain yang mungkin cocok, lalu dapatkan rekomendasi lagu serupa. "
    "Catatan: label genre pada dataset latihan ini tidak mencerminkan genre musik yang sebenarnya."
)

# 1. Pilih lagu
st.subheader("1. Pilih lagu")
is_default = df["title"] == "Soul Deep"
default_idx = int(df.index[is_default][0]) if is_default.any() else 0
idx = st.selectbox(
    "Ketik judul untuk mencari",
    options=df.index,
    index=default_idx,
    format_func=lambda i: df.at[i, "label"],
)
song = df.loc[idx]
song_genres = song["genres"].split("|")
tahun_lagu = "tidak diketahui" if pd.isna(song["year"]) else str(song["year"])
st.markdown(f"**Genre:** {', '.join(song_genres)}  \n**Tahun rilis:** {tahun_lagu}")

# 2. Prediksi genre tambahan
st.subheader("2. Prediksi genre tambahan")
pred = predict_genres(song_genres, top_n=3)
st.dataframe(
    pred.rename("Probabilitas").rename_axis("Genre").reset_index(),
    column_config={
        "Probabilitas": st.column_config.ProgressColumn(format="%.2f", min_value=0.0, max_value=1.0)
    },
    hide_index=True,
)
st.caption("Probabilitas dihitung oleh model terpisah untuk setiap genre, sehingga totalnya tidak harus 100%.")

# 3. Rekomendasi lagu serupa
st.subheader("3. Rekomendasi lagu serupa")
top_n = st.slider("Jumlah rekomendasi", min_value=5, max_value=20, value=10)
expand = st.checkbox(f"Perluas pencarian dengan genre hasil prediksi teratas ({pred.index[0]})")
query_genres = song_genres + ([pred.index[0]] if expand else [])
recs, n_identical = recommend(idx, query_genres, top_n)

urutan = "tahun terdekat" if pd.notna(song["year"]) else "urutan data, karena tahun lagu ini tidak diketahui"
st.info(
    f"{n_identical} lagu memiliki kombinasi genre yang identik dengan profil pencarian. "
    f"Lagu dengan skor sama diurutkan berdasarkan {urutan}."
)
st.dataframe(
    recs.rename(columns={"title": "Judul", "genres": "Genre", "year": "Tahun", "similarity": "Skor kemiripan"}),
    column_config={"Skor kemiripan": st.column_config.NumberColumn(format="%.2f")},
    hide_index=True,
)

with st.expander("Tentang metode dan evaluasi"):
    st.markdown(
        "- **Data:** 9.610 lagu bergenre setelah cleaning; tahun 1997 diperlakukan sebagai missing (placeholder).\n"
        "- **Prediksi genre:** satu Logistic Regression per genre (19 model), dengan fitur genre lain pada lagu "
        "yang sama, dilatih pada 80% data (random_state=42).\n"
        "- **Evaluasi pada 20% data uji:** median AUC 0,85; Hit@3 72,48%, dibandingkan 51,70% untuk baseline "
        "genre terpopuler.\n"
        "- **Rekomendasi:** TF-IDF atas genre dan cosine similarity; skor yang sama diurutkan berdasarkan tahun terdekat."
    )
