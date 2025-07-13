# Resume_Screening

<h1 align="center">🧠 Resume Screening & Job Matching using Machine Learning</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Streamlit-Deployed-green?style=for-the-badge&logo=streamlit" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Logistic%20Regression-blueviolet?style=for-the-badge&logo=scikit-learn" />
  <img src="https://img.shields.io/badge/NLP-TF--IDF-orange?style=for-the-badge&logo=nlp" />
  <img src="https://img.shields.io/badge/Python-3.11+-yellow?style=for-the-badge&logo=python" />
</p>

> 🎯 AI-powered resume classifier to predict job categories  
> 🧑‍💼 Instantly match resumes to job descriptions with similarity scoring  
> ⚡ Built using Machine Learning, TF-IDF vectorization, and Streamlit

---

## 🚀 Features

- 📑 **Resume Category Prediction** — Classifies resumes into job domains (e.g., Web Dev, Data Scientist)
- 🧠 **Job Description Matching** — Matches job descriptions to most relevant resumes
- 📊 **Top Matches Visualized** — Similarity scores with bar charts
- 📤 **User Input** — Supports raw text input for resumes & JD
- 🔎 **Insightful Matching** — See how well a resume fits a given job
- ⚙️ **Trained ML Model** — Uses Logistic Regression + TF-IDF
- 🌐 Powered by **Python**, **scikit-learn**, **Streamlit**, **Pandas**, **NumPy**

---

## 🎯 Use Cases

- 🧑‍💼 Recruiters — Speed up candidate screening
- 💼 HR Tech Products — Add smart matching logic
- 🧪 Students — Use as a portfolio project in ML/NLP
- 👨‍💻 Developers — Extend it with PDF parsing, login, database, etc.

---

## 🧠 ML Pipeline

| Step | Component |
|------|----------|
| 📄 Dataset | Resume dataset with text & category |
| ✨ Preprocessing | Cleaned, tokenized, and lemmatized resume/JD text |
| 🧪 Feature Extraction | TF-IDF Vectorization |
| 🔍 Model | Logistic Regression |
| 📦 Model Saved | `.pkl` files with `joblib` |
| 🌐 Frontend | Streamlit Web App with two modes: Category & Matching |

---

## 🖥️ Screenshots

### 📑 Resume Classification Mode  
<img src="screenshots/resume_categoryimg.png" width="700">

<img src="screenshots/resume_categoryimg1.png" width="700">

### 💼 Job Matching Mode  
<img src="screenshots/job_matchingimg2.1.png" width="700">
<img src="screenshots/job_matchingimg2.2.png" width="700">
---

## 📁 Folder Structure

📦 resume-matching-app/
│
├── app.py # Streamlit main application
├── UpdatedResumeDataSet.csv # Input resume dataset
├── tfidf_vectorizer.pkl # Saved TF-IDF vectorizer
├── resume_model.pkl # Trained ML model
├── requirements.txt # Required dependencies
└── README.md # You're here!