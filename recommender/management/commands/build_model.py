from django.core.management.base import BaseCommand
from pathlib import Path
from recommender.ml import (
    load_sessions, build_vocab, sessions_to_matrix,
    train_kmeans, summarize_clusters, save_artifacts
)


class Command(BaseCommand):
    help = "Build K-Means model from historical sessions"

    def handle(self, *args, **kwargs):
        data_path = Path(__file__).resolve().parents[2] / "data" / "sessions.json"
        sessions = load_sessions(data_path)
        vocab = build_vocab(sessions)
        X = sessions_to_matrix(sessions, vocab)
        kmeans = train_kmeans(X, n_clusters=min(4, max(2, len(sessions)//2)))
        idx2tag = [t for t,_ in sorted(vocab.items(), key=lambda kv: kv[1])]
        summary = summarize_clusters(X, kmeans, idx2tag, top_k=7)
        save_artifacts(kmeans, vocab, summary)
        self.stdout.write(self.style.SUCCESS(f"Model trained. Clusters: {kmeans.n_clusters}. Tags: {len(vocab)}"))
