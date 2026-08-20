"""Deterministic retrieval for the explanation layer. See CLAUDE.md
Section 6E and Section 9. TF-IDF, not embedding similarity -- the
retrievable corpus (calendar rows, one EDA report) is too small to
justify sentence-transformers (see Section 20).
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def retrieve_calendar_window(calendar: pd.DataFrame, start_date: str, horizon: int) -> pd.DataFrame:
    start = pd.to_datetime(start_date)
    end = start + pd.Timedelta(days=horizon - 1)
    mask = (calendar["date"] >= start) & (calendar["date"] <= end)
    return calendar.loc[mask].copy()


def retrieve_eda_passages(query: str, eda_text: str, k: int = 2) -> list[str]:
    passages = [p.strip() for p in eda_text.split("\n\n") if p.strip()]
    if not passages:
        return []

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(passages + [query])
    passage_vecs, query_vec = tfidf[:-1], tfidf[-1]
    similarities = (passage_vecs @ query_vec.T).toarray().ravel()

    ranked = np.argsort(-similarities)[:k]
    return [passages[i] for i in ranked]
