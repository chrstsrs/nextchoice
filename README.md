# AI based NextChoice: A Machine Learning, unsupervised clustering project

Goal: show machine learning (unsupervised clustering) in a Django app that recommends the next most relevant choice given a few user-selected tags.

ML technique: K-Means on user–tag binary vectors

### Demo Walkthrough

- Open the app

- Tick a few tags (e.g., Action, Sci-Fi, Nolan)

- Click Recommend next choice

The backend assigns you to a cluster and suggests the most frequent tags in that cluster that you haven’t chosen yet.

## Project Structure

```
nextchoice/
├─ manage.py
├─ requirements.txt
├─ nextchoice/
│ ├─ init.py
│ ├─ settings.py
│ ├─ urls.py
│ └─ wsgi.py
└─ recommender/
├─ init.py
├─ apps.py
├─ data/
│ └─ sessions.json
├─ ml.py
├─ views.py
├─ urls.py
├─ templates/
│ └─ recommend.html
└─ management/
└─ commands/
└─ build_model.py
```

## Quickstart

Works on macOS/Linux/Windows. No DB setup required.

1) Create and activate a virtual environment

Windows (PowerShell):
```
python -m venv venv
.\venv\Scripts\Activate.ps1
```

macOS/Linux (bash):
```
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies

```
pip install -r requirements.txt
```

3) This demo doesn’t require a database, but if you want to run the standard Django migration steps:

```
python manage.py migrate
```

4) Build the clustering model

```
python manage.py build_model
```

5) Run the server

```
python manage.py runserver
```

Open the app at:
```
http://127.0.0.1:8000/
```


### How the ML Works

We ship a tiny historical sessions file (recommender/data/sessions.json): each session is a set of tags a past user picked.
We build a vocabulary of tags and turn each session into a binary vector (1 if the tag appears, else 0). 
We run K-Means on these vectors (unsupervised; no labels). 
For each cluster, we compute tag frequencies (how often a tag appears among members of that cluster).

### For a new request:

- vectorize your selected tags

- predict your cluster

- recommend the top tags by frequency in that cluster that you haven’t selected yet

- if frequencies are thin, we back off to centroid weights. This yields a simple, transparent “people like you also pick …” recommendation.

### Key Files

- recommender/ml.py: all ML utilities (vectorization, training, inference, artifact IO)

- recommender/management/commands/build_model.py: CLI command to train + persist artifacts

- recommender/views.py: GET handler that renders the form and shows recommendations

- recommender/templates/recommend.html: minimal UI

### Requirements

```
Django==5.0.6
scikit-learn==1.5.1
pandas==2.2.2
numpy==1.26.4
```

## FAQ

Q. Why clustering (unsupervised) instead of a classifier (supervised)? 

A. We don’t have labels or outcomes. Just co-occurring choices. Clustering naturally groups similar users and yields explainable “next choice” suggestions.

---

Q. What if a user picks a tag not in the vocabulary?

A. Unknown tags are ignored for vectorization; 

---

Q. Is the model deterministic?

A. We fix random_state for repeatability. K-Means still may vary if you change data or n_clusters.

## License
MIT
