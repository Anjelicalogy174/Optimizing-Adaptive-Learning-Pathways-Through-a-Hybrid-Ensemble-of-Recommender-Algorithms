# ALPS — Adaptive Learning Pathway System

Optimizing adaptive learning pathways through a hybrid ensemble of recommender algorithms — built for the Philippine Alternative Learning System (ALS).

One FastAPI backend serves live ML predictions to three separate frontends: a student survey, a per-learner analyst dashboard, and an instructor cohort console.

```
scaler → pca → kmeans → ensemble → adaptive
```

No ML inference runs in the browser. Every score is computed server-side from trained artifacts (`scaler.joblib`, `pca.joblib`, `kmeans.joblib`) at request time, so all three frontends always agree.

## How it fits together

| Page | Role | Calls |
|---|---|---|
| `index.html` | Landing page — explains the system, links to the other three | — |
| `survey.html` | 11-item bilingual (EN/FIL) learner survey | `POST /recommend` |
| `dashboard.html` | Per-learner profile, clustering, pathway, adaptive outcomes | `GET /learners`, `GET /clusters`, `POST /recommend` |
| `instructor.html` | Whole-cohort triage view | `POST /cohort` |

Finishing the survey hands that exact respondent (answers + live prediction) to the dashboard via `localStorage`, so "Open full dashboard" opens on *you*, not a random learner — it's one continuous walkthrough, not three separate demos.

## Model summary

- **PCA** — 4 components, 61.6% variance on PC1, 83.8% cumulative
- **K-Means** — k = 4, silhouette ≈ 0.26, mapped to readiness tiers: `High Readiness`, `Advanced`, `Intermediate`, `Basic`
- **Hybrid ensemble** — ranks modules by a weighted blend of cluster similarity, content similarity, and an adaptive signal (`score = w₁·cluster_sim + w₂·content_sim + w₃·adaptive`), weights adjustable live from the dashboard
- **Adaptive layer** — per-module assessment is a deterministic, seeded *simulation* standing in for a live formative-assessment system (see `simulate_assessment()` in `app.py`); it's the one part of the pipeline that isn't backed by a trained model

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Artifact load status |
| `GET` | `/clusters` | Tier definitions and centroids |
| `GET` | `/learners` | Full roster, scored live |
| `POST` | `/recommend` | Full prediction for one learner |
| `POST` | `/cohort` | Whole cohort scored in one batch |
| `GET` | `/metrics` | Live PCA variance, silhouette, confidence — never fabricated |

## Getting started

**Requirements:** Python 3.10+, `fastapi`, `uvicorn`, `pandas`, `numpy`, `joblib`, `scikit-learn`, `pydantic`

```bash
pip install fastapi uvicorn pandas numpy joblib scikit-learn pydantic
```

**Expected files** alongside `app.py` (or in the directory set by `ALPS_ARTIFACTS`):

```
scaler.joblib
pca.joblib
kmeans.joblib
recommender_config.joblib
module_data.csv
learner_data.csv        # optional — dashboard falls back to a built-in roster if absent
```

**Run the backend:**

```bash
uvicorn app:app --reload --port 8000
```

**Open the demo:** open `index.html` in a browser and start from there, or jump straight to `survey.html`, `dashboard.html`, or `instructor.html`.

If the API is unreachable, `dashboard.html` falls back to a small built-in roster automatically so the UI still runs.

## Project structure

```
app.py            FastAPI inference backend
index.html         Landing page / system overview
survey.html         Learner-facing profile survey
dashboard.html       Per-learner analyst dashboard
instructor.html       Instructor cohort console
```

## Thesis

*Optimizing Adaptive Learning Pathways Through a Hybrid Ensemble of Recommender Algorithms*
Anjelica M. Castillo · M.S. Computer Science, Technological Institute of the Philippines, Manila
Adviser: Dr. Melvin Ballera
