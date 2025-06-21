from __future__ import annotations
import json, pickle
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from sklearn.cluster import KMeans

BASE_DIR = Path(__file__).resolve().parent
ART_DIR = BASE_DIR / "artifacts"
ART_DIR.mkdir(exist_ok=True)
MODEL_PKL = ART_DIR / "kmeans.pkl"
VOCAB_PKL = ART_DIR / "vocab.pkl"
CLUSTER_SUMMARY_PKL = ART_DIR / "cluster_summary.pkl"

def load_sessions(path: Path) -> List[List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_vocab(sessions: List[List[str]]) -> Dict[str, int]:
    vocab = {}
    for session in sessions:
        for tag in session:
            if tag not in vocab:
                vocab[tag] = len(vocab)
    return vocab

def sessions_to_matrix(sessions: List[List[str]], vocab: Dict[str,int]) -> np.ndarray:
    X = np.zeros((len(sessions), len(vocab)), dtype=np.float32)
    for i, session in enumerate(sessions):
        for tag in session:
            X[i, vocab[tag]] = 1.0
    return X

def train_kmeans(X: np.ndarray, n_clusters: int = 4, random_state: int = 42) -> KMeans:
    # n_clusters small for demo; in real use, tune via silhouette score
    model = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
    model.fit(X)
    return model

def summarize_clusters(X: np.ndarray, model: KMeans, idx2tag: List[str], top_k: int = 5) -> Dict[int, List[Tuple[str, float]]]:
    summary = {}
    labels = model.labels_
    for c in range(model.n_clusters):
        members = X[labels == c]
        if len(members) == 0:
            summary[c] = []
            continue
        freq = members.mean(axis=0)  # probability of tag in this cluster
        top_idx = np.argsort(freq)[::-1][:top_k]
        summary[c] = [(idx2tag[i], float(freq[i])) for i in top_idx]
    return summary

def save_artifacts(model: KMeans, vocab: Dict[str,int], cluster_summary: Dict[int, List[Tuple[str,float]]]):
    with open(MODEL_PKL, "wb") as f:
        pickle.dump(model, f)
    with open(VOCAB_PKL, "wb") as f:
        pickle.dump(vocab, f)
    with open(CLUSTER_SUMMARY_PKL, "wb") as f:
        pickle.dump(cluster_summary, f)

def load_artifacts():
    with open(MODEL_PKL, "rb") as f:
        model = pickle.load(f)
    with open(VOCAB_PKL, "rb") as f:
        vocab = pickle.load(f)
    with open(CLUSTER_SUMMARY_PKL, "rb") as f:
        cluster_summary = pickle.load(f)
    return model, vocab, cluster_summary

def vectorize_picks(picks: List[str], vocab: Dict[str,int]) -> np.ndarray:
    x = np.zeros((1, len(vocab)), dtype=np.float32)
    for t in picks:
        if t in vocab:
            x[0, vocab[t]] = 1.0
    return x

def recommend_next(picks: List[str], top_n: int = 3) -> Dict[str, object]:
    model, vocab, cluster_summary = load_artifacts()
    x = vectorize_picks(picks, vocab)
    cluster = int(model.predict(x)[0])
    # Rank tags by cluster frequency excluding already-picked
    freq_list = cluster_summary.get(cluster, [])
    suggestions = [(tag, p) for tag, p in freq_list if tag not in picks]
    # If cluster_summary too small, fall back to centroid weights
    if not suggestions:
        centroid = model.cluster_centers_[cluster]
        idx2tag = [t for t,_ in sorted(vocab.items(), key=lambda kv: kv[1])]
        order = np.argsort(centroid)[::-1]
        suggestions = [(idx2tag[i], float(centroid[i])) for i in order if idx2tag[i] not in picks]
    return {
        "cluster": cluster,
        "suggestions": suggestions[:top_n],
    }
