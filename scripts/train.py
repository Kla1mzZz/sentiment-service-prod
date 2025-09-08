import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Загружаем данные
df = pd.read_csv("datasets/IMDB_Dataset.csv")
df["sentiment"] = df["sentiment"].replace({"positive": 1, "negative": 0})

X = df["review"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = Pipeline(
    [
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        (
            "logreg",
            LogisticRegression(
                C=0.1,
                random_state=42,
                max_iter=500,
                verbose=1,
                n_jobs=-1,
                solver="saga",
            ),
        ),
    ]
)

pipeline.fit(X_train, y_train)

joblib.dump(pipeline, "models/pipeline.joblib")
