# streamlit_app.py

import streamlit as st

# ✅ Page config must be first Streamlit command
st.set_page_config(page_title="Resume Classifier", layout="wide")

import pandas as pd
import numpy as np
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# ----------------------------
# Text Preprocessing Function
# ----------------------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in ENGLISH_STOP_WORDS and len(word) > 2]
    return ' '.join(tokens)

# ----------------------------
# Load Model & Vectorizer
# ----------------------------
@st.cache_resource
def load_model_and_vectorizer():
    df = pd.read_csv("UpdatedResumeDataSet.csv")
    df = df[['Category', 'Resume']].dropna()
    df['cleaned_resume'] = df['Resume'].apply(preprocess)

    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df['cleaned_resume'])
    y = df['Category']

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return df, tfidf, model

df, tfidf, model = load_model_and_vectorizer()

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🧠 Resume Classification & Job Matching App")

tab1, tab2 = st.tabs(["📂 Resume Classification", "🔍 Job Description Matching"])

with tab1:
    st.subheader("🔎 Predict Resume Category")
    resume_text = st.text_area("Paste your Resume text below:", height=250)

    if st.button("Predict Category"):
        if resume_text.strip() == "":
            st.warning("Please paste a resume to classify.")
        else:
            cleaned = preprocess(resume_text)
            vector = tfidf.transform([cleaned])
            prediction = model.predict(vector)[0]
            st.success(f"🎯 Predicted Category: **{prediction}**")

with tab2:
    st.subheader("🔍 Find Matching Resumes for Job Description")
    job_desc = st.text_area("Paste a Job Description here:", height=250)

    if st.button("Find Top Matches"):
        if job_desc.strip() == "":
            st.warning("Please paste a job description to match resumes.")
        else:
            job_cleaned = preprocess(job_desc)
            corpus_vectors = tfidf.transform(df['cleaned_resume'].tolist())  # ✅ keep the same vectorizer
            job_vector = tfidf.transform([job_cleaned])
            cosine_sim = cosine_similarity(job_vector, corpus_vectors)
            df['similarity_score'] = cosine_sim[0]
            top_matches = df.sort_values(by='similarity_score', ascending=False).head(10)

            st.write("### 🏆 Top 10 Matching Resumes")
            st.dataframe(top_matches[['Category', 'similarity_score']])

            st.write("### 📊 Similarity Scores")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.barh(top_matches['Category'], top_matches['similarity_score'], color='skyblue')
            ax.set_xlabel("Similarity Score")
            ax.set_title("Top 10 Matching Resume Categories")
            ax.invert_yaxis()
            st.pyplot(fig)
