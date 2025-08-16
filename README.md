# 🎬 CineMatch — Movie Recommendation System
CineMatch is a machine learning–powered movie recommendation engine that suggests movies based on user preferences.
It combines content-based filtering and collaborative filtering to deliver personalized recommendations.
It supports **content-based** recommendations out of the box and **item-item collaborative filtering** when a `ratings.csv` is present.

## 🚀 Features
Search for any movie and get similar movie recommendations
Hybrid system:
         Content-based filtering (using movie metadata like genres, overview, keywords)
          Item-Item Collaborative Filtering (based on user ratings)
Interactive Streamlit web app
Clean, modular Python code

## 📁 Project Structure
```text
CineMatch/
├── data/
│   ├── movies.csv
│   └── ratings.csv   # optional
├── src/
│   └── recommender.py
├── app/
│   └── streamlit_app.py
├── requirements.txt
└── README.md
```

## 🧰 Setup
```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```
1. **Goal**: Recommend similar movies to a given title.
2. **Data**: Use titles, genres, and overviews (and optional user ratings).
3. **Model**: TF-IDF → cosine similarity (content). Item-item similarity (collab).
4. **Result**: Top-K similar movies with a simple web UI.

## 🛠️ Tech Stack

Python (pandas, numpy, scikit-learn)
Streamlit (for web app deployment)
Surprise / scikit-learn (for collaborative filtering)
Pickle (for model persistence) 

## 🔧 Notes
- If your dataset uses different column names, `recommender.py` tries to auto-normalize common variants (e.g., `original_title`, `description`, etc.).
- Collaborative mode requires a ratings file with columns like `userId`, `movieId`/`title`, and `rating`.