from django.shortcuts import render
from django.http import HttpRequest
from pathlib import Path
import json
from .ml import recommend_next, load_artifacts

def home(request: HttpRequest):
    # Load available tags from vocab
    try:
        model, vocab, _ = load_artifacts()
        tags = sorted(vocab.keys())
    except Exception:
        # If not built yet, fall back to tags present in sessions.json
        data_path = Path(__file__).resolve().parent / "data" / "sessions.json"
        tags = sorted({t for row in json.load(open(data_path)) for t in row})

    picks = request.GET.getlist("tag")
    result = None
    if picks:
        result = recommend_next(picks, top_n=3)

    return render(request, "recommend.html", {
        "tags": tags,
        "picks": picks,
        "result": result
    })
