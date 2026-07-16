"""
================================================================================
 Adaptive Learning Pathway System (ALPS) — FastAPI Inference Backend
================================================================================
 Author : (thesis) Anjelica M. Castillo
 Purpose: Serve clustering + recommendation + adaptive-pathway predictions that
          are produced ENTIRELY by the trained ML artifacts:

              scaler.joblib  →  pca.joblib  →  kmeans.joblib
              + recommender_config.joblib + module_data.csv

          The dashboard calls this API; NO ML math runs in the browser.

 Inference pipeline (per learner):
     11 raw features
        → scaler.transform()            (StandardScaler)
        → pca.transform()               (PCA, 4 components)
        → kmeans.predict()              (KMeans, k=4)  → cluster label
        → map label → readiness tier
        → hybrid ensemble scoring        (cluster + content + adaptive signals)
        → top-K initial pathway
        → simulated assessment + adaptive actions (skip / continue / remedial)
        → optimized pathway

 Endpoints:
     GET  /health      → artifact load status
     GET  /clusters    → tier definitions + centroids (original feature space)
     POST /recommend   → full prediction for one learner

 Run:
     pip install -r requirements.txt
     uvicorn app:app --reload --port 8000
================================================================================
"""

from __future__ import annotations

import os
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import joblib

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

# Directory holding the .joblib + .csv artifacts (override with ALPS_ARTIFACTS env)
ARTIFACT_DIR = Path(os.getenv("ALPS_ARTIFACTS", "."))

# Canonical feature order. MUST match the column order used when training the
# scaler / PCA. If your training used a different order, change this list only.
FEATURE_ORDER: List[str] = [
    "capability", "academic_fit", "engagement", "motivation", "study_habits",
    "prior_learning", "preferences", "constraints", "commitment", "strategies",
    "interest",
]

# Readiness tier names, highest readiness first. The cluster with the highest
# mean centroid (in original feature space) is mapped to TIER_NAMES[0], etc.
TIER_NAMES: List[str] = ["High Readiness", "Advanced", "Intermediate", "Basic"]

# "constraints" is reverse-coded (a higher value is a *barrier*), so it is
# inverted when computing a learner/cluster readiness score.
REVERSE_CODED = {"constraints"}


# ------------------------------------------------------------------------------
# Artifact container — everything is loaded ONCE at startup
# ------------------------------------------------------------------------------

class Artifacts:
    scaler: Any = None
    pca: Any = None
    kmeans: Any = None
    config: Dict[str, Any] = {}

    modules: pd.DataFrame = None          # raw module table
    module_feats: np.ndarray = None       # (n_modules, 11) feature matrix
    module_req: np.ndarray = None         # (n_modules,) readiness requirement 0..1
    module_dscore: np.ndarray = None      # (n_modules,) numeric difficulty score

    centroids_orig: np.ndarray = None     # (k, 11) centroids in ORIGINAL space
    cluster_to_tier: Dict[int, Dict] = {} # label → {rank, name, score}

    learners: pd.DataFrame = None         # learner roster (optional)
    learner_feats: np.ndarray = None      # (n_learners, 11)
    learner_meta: List[Dict] = []         # id/age/sex/type per learner

    load_status: Dict[str, Any] = {}


A = Artifacts()


def _readiness_score(vec: np.ndarray) -> float:
    """Mean of feature values with reverse-coded items inverted (scale 0..5)."""
    v = vec.copy().astype(float)
    for i, name in enumerate(FEATURE_ORDER):
        if name in REVERSE_CODED:
            v[i] = 5.0 - v[i]
    return float(np.clip(v.mean(), 0, 5))


def _difficulty_to_score(label: str) -> float:
    return {"beginner": 1.0, "intermediate": 2.5, "advanced": 4.0}.get(
        str(label).strip().lower(), 2.0
    )


def load_artifacts() -> None:
    """Load all trained artifacts and derive everything the API needs."""
    status: Dict[str, Any] = {}

    # ---- 1. Trained models -----------------------------------------------------
    A.scaler = joblib.load(ARTIFACT_DIR / "scaler.joblib")
    status["scaler"] = True
    A.pca = joblib.load(ARTIFACT_DIR / "pca.joblib")
    status["pca"] = True
    A.kmeans = joblib.load(ARTIFACT_DIR / "kmeans.joblib")
    status["kmeans"] = True

    # ---- 2. Recommender config (optional keys, all have fallbacks) -------------
    try:
        cfg = joblib.load(ARTIFACT_DIR / "recommender_config.joblib")
        A.config = cfg if isinstance(cfg, dict) else {"_raw": cfg}
        status["recommender_config"] = True
    except Exception as exc:  # noqa: BLE001
        A.config = {}
        status["recommender_config"] = f"missing ({exc})"

    # Allow config to override the global feature order / tier names / weights.
    global FEATURE_ORDER, TIER_NAMES
    FEATURE_ORDER = A.config.get("feature_order", FEATURE_ORDER)
    TIER_NAMES = A.config.get("tier_names", TIER_NAMES)

    # ---- 3. Module catalogue ---------------------------------------------------
    A.modules = pd.read_csv(ARTIFACT_DIR / "module_data.csv")
    status["modules"] = int(len(A.modules))

    # Feature columns: prefer exact names; otherwise the config may carry a
    # precomputed module feature matrix.
    cols = [c for c in FEATURE_ORDER if c in A.modules.columns]
    if len(cols) == len(FEATURE_ORDER):
        A.module_feats = A.modules[FEATURE_ORDER].to_numpy(dtype=float)
    elif "module_features" in A.config:
        A.module_feats = np.asarray(A.config["module_features"], dtype=float)
    else:
        raise RuntimeError(
            "module_data.csv has no recognizable feature columns and "
            "recommender_config has no 'module_features'. "
            f"Expected columns: {FEATURE_ORDER}"
        )

    # Difficulty score (numeric). Prefer an explicit column, else derive.
    if "difficulty_score" in A.modules.columns:
        A.module_dscore = A.modules["difficulty_score"].to_numpy(dtype=float)
    else:
        diff_col = next(
            (c for c in ("difficulty", "diff", "level") if c in A.modules.columns),
            None,
        )
        labels = A.modules[diff_col] if diff_col else pd.Series(["Intermediate"] * len(A.modules))
        A.module_dscore = np.array([_difficulty_to_score(x) for x in labels])

    # Readiness requirement (0..1). Prefer explicit columns, else derive from difficulty.
    req_cols = [c for c in ("mastery_req", "engagement_req") if c in A.modules.columns]
    if len(req_cols) == 2:
        A.module_req = A.modules[req_cols].mean(axis=1).to_numpy(dtype=float)
    else:
        dmin, dmax = A.module_dscore.min(), A.module_dscore.max()
        A.module_req = (A.module_dscore - dmin) / ((dmax - dmin) or 1.0)

    # ---- 4. Cluster centroids back-projected to ORIGINAL feature space --------
    # KMeans was trained on PCA output, so its centroids live in PCA space.
    # Inverse-transform them so we can score modules in the same space as learners.
    centroids_pca = np.asarray(A.kmeans.cluster_centers_, dtype=float)
    centroids_scaled = A.pca.inverse_transform(centroids_pca)
    A.centroids_orig = A.scaler.inverse_transform(centroids_scaled)

    # ---- 5. Map arbitrary KMeans labels → ordered readiness tiers -------------
    if "cluster_to_tier" in A.config:
        # Config explicitly provides the mapping {label: tier_name}
        provided = {int(k): str(v) for k, v in A.config["cluster_to_tier"].items()}
        A.cluster_to_tier = {}
        for label, centroid in enumerate(A.centroids_orig):
            name = provided.get(label, TIER_NAMES[min(label, len(TIER_NAMES) - 1)])
            A.cluster_to_tier[label] = {
                "name": name,
                "score": round(_readiness_score(centroid), 3),
            }
    else:
        # Derive ordering: highest readiness centroid → TIER_NAMES[0]
        scores = [_readiness_score(c) for c in A.centroids_orig]
        order = list(np.argsort(scores)[::-1])  # labels, best first
        A.cluster_to_tier = {
            int(label): {
                "name": TIER_NAMES[min(rank, len(TIER_NAMES) - 1)],
                "score": round(scores[label], 3),
            }
            for rank, label in enumerate(order)
        }

    A.load_status = status

    # ---- 6. Learner roster (optional) — enables a fully LIVE scatter ----------
    # If learner_data.csv is present, every learner is projected live through
    # scaler → pca and labelled by kmeans, so the scatter shows real model
    # output rather than precomputed coordinates.
    #
    # This whole step is wrapped in try/except: a malformed learner file must
    # NEVER take down /health, /clusters, or /recommend. If it fails, the
    # dashboard simply falls back to its built-in roster.
    try:
        learner_file = next(
            (ARTIFACT_DIR / n for n in ("learner_data.csv", "learners.csv")
             if (ARTIFACT_DIR / n).exists()),
            None,
        )
        if learner_file is None:
            status["learners"] = "no learner_data.csv (dashboard will use its own roster)"
        else:
            df = pd.read_csv(learner_file)
            lcols = [c for c in FEATURE_ORDER if c in df.columns]
            if len(lcols) != len(FEATURE_ORDER):
                missing = [c for c in FEATURE_ORDER if c not in df.columns]
                status["learners"] = f"missing feature columns: {missing}"
            else:
                # Feature matrix — coerce to numeric and impute NaN with the
                # column mean so a few blank survey cells don't crash inference.
                feats = df[FEATURE_ORDER].apply(pd.to_numeric, errors="coerce")
                n_nan = int(feats.isna().sum().sum())
                feats = feats.fillna(feats.mean()).fillna(0.0)
                A.learner_feats = feats.to_numpy(dtype=float)
                A.learners = df

                def _col(*names):
                    return next((c for c in names if c in df.columns), None)

                id_c = _col("id", "learner_id", "respondent_id", "learnerid")
                age_c = _col("age")
                sex_c = _col("sex", "gender")
                type_c = _col("type", "learner_type", "category", "program")

                # NaN-safe scalar coercion helpers
                def _safe_int(series, i, fallback):
                    if series is None:
                        return fallback
                    v = series.iloc[i]
                    try:
                        if pd.isna(v):
                            return fallback
                        return int(float(v))
                    except (ValueError, TypeError):
                        return fallback

                def _safe_str(series, i):
                    if series is None:
                        return ""
                    v = series.iloc[i]
                    return "" if pd.isna(v) else str(v).strip()

                id_s = df[id_c] if id_c else None
                age_s = df[age_c] if age_c else None
                sex_s = df[sex_c] if sex_c else None
                type_s = df[type_c] if type_c else None

                A.learner_meta = [
                    {
                        "id": _safe_int(id_s, i, i + 1),
                        "age": _safe_int(age_s, i, None),
                        "sex": _safe_str(sex_s, i),
                        "type": _safe_str(type_s, i),
                    }
                    for i in range(len(df))
                ]
                status["learners"] = int(len(df))
                if n_nan:
                    status["learners_note"] = f"imputed {n_nan} NaN feature value(s) with column mean"
    except Exception as exc:  # noqa: BLE001
        # Degrade gracefully — keep the API fully usable without the roster.
        A.learners = None
        A.learner_feats = None
        A.learner_meta = []
        status["learners"] = f"failed to load roster ({exc}) — using dashboard fallback"

    A.load_status = status


# Load immediately so failures surface at boot, not on first request.
try:
    load_artifacts()
    _BOOT_ERROR = None
except Exception as exc:  # noqa: BLE001
    _BOOT_ERROR = str(exc)


# ------------------------------------------------------------------------------
# Math — the actual ML inference logic (server-side only)
# ------------------------------------------------------------------------------

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def predict_cluster(features: np.ndarray) -> Dict[str, Any]:
    """scaler → pca → kmeans. Returns label, tier, PCA components, confidence."""
    scaled = A.scaler.transform(features.reshape(1, -1))
    pcs = A.pca.transform(scaled)
    label = int(A.kmeans.predict(pcs)[0])

    # Confidence = softmax over the negative distances to each centroid.
    # The assigned (nearest) cluster gets the highest probability; a learner
    # sitting between two centroids yields a lower, honest confidence.
    d = A.kmeans.transform(pcs).ravel()              # distance to each centroid
    temp = max(float(d.mean()), 1e-6)
    e = np.exp(-(d - d.min()) / temp)
    probs = e / e.sum()
    confidence = float(probs[label])

    tier = A.cluster_to_tier.get(label, {"name": "Unknown", "score": 0.0})
    return {
        "cluster": label,
        "tier": {"id": label, "name": tier["name"], "score": tier["score"]},
        "pca_components": [round(float(x), 4) for x in pcs.ravel().tolist()],
        "confidence": round(confidence, 4),
    }


def ensemble_scores(features: np.ndarray, cluster_label: int,
                    weights: List[float]) -> np.ndarray:
    """
    Hybrid ensemble — three signals, weighted:
       1. cluster similarity : cosine(cluster centroid, module)
       2. content similarity : cosine(learner features, module)
       3. adaptive signal    : 1 - |learner readiness - module requirement|
    """
    w = np.asarray(weights, dtype=float)
    w = w / (w.sum() or 1.0)  # normalize so the three weights sum to 1

    centroid = A.centroids_orig[cluster_label]
    learner_readiness = _readiness_score(features) / 5.0

    n = len(A.module_feats)
    out = np.zeros(n)
    for i in range(n):
        cl = _cosine(centroid, A.module_feats[i])           # cluster signal
        co = _cosine(features, A.module_feats[i])            # content signal
        ad = 1.0 - abs(learner_readiness - A.module_req[i])  # adaptive signal
        out[i] = w[0] * cl + w[1] * co + w[2] * ad
    return out


def simulate_assessment(features: np.ndarray, module_idx: int) -> int:
    """
    Deterministic, reproducible assessment-score simulation (35..99).
    Higher capability raises the score; higher module difficulty lowers it.
    Seeded by the learner's feature vector + module index so the same learner
    always yields the same demo result during the defense.
    """
    cap_norm = features[FEATURE_ORDER.index("capability")] / 5.0
    dmin, dmax = A.module_dscore.min(), A.module_dscore.max()
    diff_norm = (A.module_dscore[module_idx] - dmin) / ((dmax - dmin) or 1.0)

    seed_src = features.tobytes() + str(module_idx).encode()
    seed = int(hashlib.md5(seed_src).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed)

    score = 50 + 45 * cap_norm - 22 * diff_norm + (rng.rand() * 2 - 1) * 11
    return int(np.clip(round(score), 35, 99))


def classify_action(score: int, pass_thr: int, mastery_thr: int) -> str:
    if score >= mastery_thr:
        return "Skip (Mastered)"
    if score >= pass_thr:
        return "Continue"
    return "Remedial Required"


def _module_row(i: int) -> Dict[str, Any]:
    """Pull human-readable module metadata from the CSV by row index.

    Uses case-insensitive, alias-aware column matching so the function works
    regardless of how the original spreadsheet was exported.
    """
    row = A.modules.iloc[i]
    # Build a lowercase alias map once per call (cheap for a single row)
    low = {c.lower().replace(" ", "_").replace("-", "_"): c for c in row.index}

    def pick(*aliases, default=""):
        for alias in aliases:
            # 1. exact match
            if alias in row.index:
                v = row[alias]
                return "" if pd.isna(v) else str(v).strip()
            # 2. case-insensitive / normalised match
            norm = alias.lower().replace(" ", "_").replace("-", "_")
            if norm in low:
                v = row[low[norm]]
                return "" if pd.isna(v) else str(v).strip()
        return default

    return {
        "id": pick(
            "id", "module_id", "code", "module_code",
            "mod_id", "modid", "ID",
            default=f"MOD-{i:03d}",
        ),
        "name": pick(
            # common export names from Google Sheets / Excel / WEKA / SPSS
            "name", "module", "title", "module_name", "module_title",
            "modulename", "moduletitle", "subject", "topic", "lesson",
            "course", "course_name", "learning_module", "content",
            "description", "label",
            default=f"Module {i + 1}",
        ),
        "strand": pick(
            "strand", "subject", "track", "area", "field",
            "learning_area", "domain", "category", "subject_area",
            default="",
        ),
        "difficulty": pick(
            "difficulty", "diff", "level", "difficulty_level",
            "complexity", "proficiency", "grade",
            default="",
        ),
    }


# ------------------------------------------------------------------------------
# Pydantic request / response schemas
# ------------------------------------------------------------------------------

class RecommendRequest(BaseModel):
    features: Dict[str, float] = Field(..., description="11 learner feature values")
    top_k: int = Field(12, ge=1, le=50)
    weights: List[float] = Field([0.33, 0.33, 0.34], min_length=3, max_length=3)
    pass_thr: int = Field(75, ge=0, le=100)
    mastery_thr: int = Field(90, ge=0, le=100)

    @field_validator("features")
    @classmethod
    def _check_features(cls, v: Dict[str, float]) -> Dict[str, float]:
        missing = [f for f in FEATURE_ORDER if f not in v]
        if missing:
            raise ValueError(f"missing feature(s): {missing}")
        return v


class CohortRequest(BaseModel):
    """Same knobs as /recommend, applied class-wide for the instructor view."""
    top_k: int = Field(12, ge=1, le=50)
    weights: List[float] = Field([0.33, 0.33, 0.34], min_length=3, max_length=3)
    pass_thr: int = Field(75, ge=0, le=100)
    mastery_thr: int = Field(90, ge=0, le=100)


# ------------------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------------------

app = FastAPI(title="ALPS Inference API", version="1.0")

# CORS so dashboard.html (opened from file:// or another origin) can call us.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten to your demo host in production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, Any]:
    """Artifact load status — lets the dashboard show a 'connected' badge."""
    return {
        "status": "ok" if _BOOT_ERROR is None else "error",
        "error": _BOOT_ERROR,
        "models_loaded": A.load_status,
        "feature_order": FEATURE_ORDER,
        "n_clusters": int(getattr(A.kmeans, "n_clusters", len(A.centroids_orig)))
        if A.kmeans is not None else 0,
        "version": app.version,
    }


@app.get("/clusters")
def clusters() -> Dict[str, Any]:
    """Tier definitions + centroids (original feature space) for the UI."""
    if _BOOT_ERROR:
        raise HTTPException(status_code=503, detail=f"models not loaded: {_BOOT_ERROR}")

    sizes = A.config.get("cluster_sizes", {})  # optional {label: size}
    out = []
    for label, meta in sorted(A.cluster_to_tier.items(),
                              key=lambda kv: kv[1]["score"], reverse=True):
        out.append({
            "id": label,
            "name": meta["name"],
            "score": meta["score"],
            "size": int(sizes.get(str(label), sizes.get(label, 0))) or None,
            "centroid": {f: round(float(A.centroids_orig[label][j]), 3)
                         for j, f in enumerate(FEATURE_ORDER)},
        })
    return {"n_clusters": len(out), "feature_order": FEATURE_ORDER, "clusters": out}


@app.get("/learners")
def learners() -> Dict[str, Any]:
    """
    All learners projected LIVE through scaler → pca, labelled by kmeans.
    Powers the interactive scatter: every dot is real model output, and each
    row carries the 11 features the dashboard re-sends to POST /recommend.
    """
    if _BOOT_ERROR:
        raise HTTPException(status_code=503, detail=f"models not loaded: {_BOOT_ERROR}")
    if A.learner_feats is None:
        raise HTTPException(status_code=404, detail="no learner roster loaded on server")

    # scaler → pca → kmeans, for ALL learners at once
    scaled = A.scaler.transform(A.learner_feats)
    pcs = A.pca.transform(scaled)                 # (n, n_components)
    labels = A.kmeans.predict(pcs).astype(int)    # (n,)

    def _f(x):  # JSON-safe float (no NaN/Inf)
        x = float(x)
        return round(x, 4) if np.isfinite(x) else 0.0

    out = []
    for i, meta in enumerate(A.learner_meta):
        tier = A.cluster_to_tier.get(int(labels[i]), {"name": "Unknown", "score": 0.0})
        out.append({
            **meta,
            "features": {f: _f(A.learner_feats[i][j]) for j, f in enumerate(FEATURE_ORDER)},
            "pca": [_f(pcs[i][0]), _f(pcs[i][1]) if pcs.shape[1] > 1 else 0.0],
            "cluster": int(labels[i]),
            "tier": tier["name"],
        })
    return {"n": len(out), "feature_order": FEATURE_ORDER, "learners": out}


@app.get("/debug/modules")
def debug_modules() -> Dict[str, Any]:
    """
    Shows the first 5 modules exactly as _module_row() resolves them,
    plus the raw CSV column names. Open this in your browser to diagnose
    'Module 1 / Module 2' display issues — you'll see which column
    the backend is (or isn't) finding for the name field.
    """
    if _BOOT_ERROR:
        raise HTTPException(status_code=503, detail=f"models not loaded: {_BOOT_ERROR}")
    return {
        "csv_columns": list(A.modules.columns),
        "n_modules": len(A.modules),
        "sample_rows_raw": A.modules.head(5).to_dict(orient="records"),
        "sample_rows_resolved": [_module_row(i) for i in range(min(5, len(A.modules)))],
    }


@app.get("/metrics")
def metrics() -> Dict[str, Any]:
    """
    Model evaluation metrics for the academic 'Model Evaluation' panel.

    Real, computed-from-artifacts values:
      • PCA variance retained / PC1 / PC2  (pca.explained_variance_ratio_)
      • Number of clusters                  (kmeans.n_clusters)
      • Silhouette score                     (silhouette_score on the cohort)
      • Average cluster-assignment confidence (softmax over centroid distances)

    Supervised metrics (accuracy / precision / recall / F1) are NOT computable
    at inference — they come from the training/validation run. Provide them in
    recommender_config['metrics']; otherwise they are reported as null with a
    clear 'source' note (never fabricated).
    """
    if _BOOT_ERROR:
        raise HTTPException(status_code=503, detail=f"models not loaded: {_BOOT_ERROR}")

    out: Dict[str, Any] = {"generated_at": pd.Timestamp.now().isoformat(timespec="seconds")}

    # ---- PCA (real) ----
    evr = getattr(A.pca, "explained_variance_ratio_", None)
    if evr is not None:
        out["pca"] = {
            "pc1": round(float(evr[0]), 4),
            "pc2": round(float(evr[1]), 4) if len(evr) > 1 else 0.0,
            "variance_retained": round(float(np.sum(evr)), 4),
            "n_components": int(len(evr)),
        }

    # ---- Clustering (real) ----
    clustering: Dict[str, Any] = {"n_clusters": int(getattr(A.kmeans, "n_clusters", 0))}
    if A.learner_feats is not None:
        pcs = A.pca.transform(A.scaler.transform(A.learner_feats))
        labels = A.kmeans.predict(pcs)
        try:
            from sklearn.metrics import silhouette_score
            if len(set(labels)) > 1:
                clustering["silhouette"] = round(float(silhouette_score(pcs, labels)), 4)
        except Exception:  # noqa: BLE001
            pass
        # average max-softmax confidence over the cohort
        d = A.kmeans.transform(pcs)
        dmin = d.min(axis=1, keepdims=True)
        temp = np.maximum(d.mean(axis=1, keepdims=True), 1e-6)
        e = np.exp(-(d - dmin) / temp)
        probs = e / e.sum(axis=1, keepdims=True)
        clustering["avg_confidence"] = round(float(probs.max(axis=1).mean()), 4)
        clustering["n_evaluated"] = int(len(labels))
    elif "silhouette" in A.config:
        clustering["silhouette"] = A.config["silhouette"]
    out["clustering"] = clustering

    # ---- Supervised eval (from config, else null) ----
    cfg_m = A.config.get("metrics", {}) if isinstance(A.config.get("metrics", {}), dict) else {}
    has = bool(cfg_m)
    out["evaluation"] = {
        "accuracy": cfg_m.get("accuracy"),
        "precision": cfg_m.get("precision"),
        "recall": cfg_m.get("recall"),
        "f1": cfg_m.get("f1"),
        "source": ("recommender_config['metrics']" if has
                   else "not provided — set recommender_config['metrics'] = {accuracy, precision, recall, f1}"),
    }
    return out


@app.post("/recommend")
def recommend(req: RecommendRequest) -> Dict[str, Any]:
    """
    FULL prediction for one learner — this is the endpoint the dashboard calls
    whenever a learner is selected or a slider changes.
    """
    if _BOOT_ERROR:
        raise HTTPException(status_code=503, detail=f"models not loaded: {_BOOT_ERROR}")

    # Order the incoming dict into the canonical feature vector.
    try:
        features = np.array([float(req.features[f]) for f in FEATURE_ORDER])
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"bad features: {exc}")

    # 1) CLUSTERING  (scaler → pca → kmeans)
    clustering = predict_cluster(features)
    label = clustering["cluster"]

    # 2) HYBRID ENSEMBLE SCORING
    scores = ensemble_scores(features, label, req.weights)
    ranked_idx = list(np.argsort(scores)[::-1])

    recommendations = []
    centroid = A.centroids_orig[label]
    lr = _readiness_score(features) / 5.0
    w = np.asarray(req.weights, float)
    w = w / (w.sum() or 1.0)
    for rank, i in enumerate(ranked_idx[: req.top_k * 2]):  # a few extra for display
        recommendations.append({
            **_module_row(i),
            "score": round(float(scores[i]), 4),
            "cluster_sim": round(_cosine(centroid, A.module_feats[i]), 4),
            "content_sim": round(_cosine(features, A.module_feats[i]), 4),
            "adaptive": round(1.0 - abs(lr - A.module_req[i]), 4),
        })

    # 3) INITIAL PATHWAY = top-K
    init_idx = ranked_idx[: req.top_k]
    initial_pathway = [
        {**_module_row(i), "position": p + 1, "score": round(float(scores[i]), 4)}
        for p, i in enumerate(init_idx)
    ]

    # 4) SIMULATED ASSESSMENT + ADAPTIVE ACTIONS
    assessment = []
    for p, i in enumerate(init_idx):
        s = simulate_assessment(features, i)
        assessment.append({
            **_module_row(i),
            "position": p + 1,
            "score": s,
            "action": classify_action(s, req.pass_thr, req.mastery_thr),
        })

    # 5) OPTIMIZED PATHWAY = re-ranked by assessment score
    optimized = sorted(assessment, key=lambda r: r["score"], reverse=True)
    optimized_pathway = [{**r, "new_position": j + 1} for j, r in enumerate(optimized)]

    # 6) SUMMARY
    counts = {"Continue": 0, "Skip (Mastered)": 0, "Remedial Required": 0}
    for r in assessment:
        counts[r["action"]] += 1
    total = len(assessment) or 1
    n_pass = sum(1 for r in assessment if r["score"] >= req.pass_thr)
    summary = {
        "total": total,
        "pass_rate": round(n_pass / total, 4),
        "adaptation_rate": round(
            (counts["Skip (Mastered)"] + counts["Remedial Required"]) / total, 4),
        "mean_ensemble": round(float(np.mean([m["score"] for m in initial_pathway])), 4),
        "counts": counts,
    }

    return {
        **clustering,             # cluster, tier, pca_components
        "weights_used": [round(float(x), 4) for x in w.tolist()],
        "recommendations": recommendations,
        "initial_pathway": initial_pathway,
        "assessment": assessment,
        "optimized_pathway": optimized_pathway,
        "summary": summary,
    }


@app.post("/cohort")
def cohort(req: CohortRequest) -> Dict[str, Any]:
    """
    INSTRUCTOR VIEW — runs the full pipeline for EVERY learner server-side and
    returns class-level aggregates plus one compact row per learner. One call
    instead of 199, so the instructor dashboard loads fast.
    """
    if _BOOT_ERROR:
        raise HTTPException(status_code=503, detail=f"models not loaded: {_BOOT_ERROR}")
    if A.learner_feats is None:
        raise HTTPException(status_code=404, detail="no learner roster loaded on server")

    rows: List[Dict[str, Any]] = []
    tier_counts: Dict[str, int] = {}
    module_remedial: Dict[int, int] = {}        # module idx → times remedial
    pass_rates: List[float] = []

    def _status(remedial: int) -> str:
        if remedial >= 5:
            return "Needs support"
        if remedial >= 2:
            return "Monitor"
        return "On track"

    for li, meta in enumerate(A.learner_meta):
        features = A.learner_feats[li]

        # scaler → pca → kmeans
        scaled = A.scaler.transform(features.reshape(1, -1))
        pcs = A.pca.transform(scaled)
        label = int(A.kmeans.predict(pcs)[0])
        tier = A.cluster_to_tier.get(label, {"name": "Unknown", "score": 0.0})

        # ensemble → top-K
        scores = ensemble_scores(features, label, req.weights)
        init_idx = list(np.argsort(scores)[::-1])[: req.top_k]

        # simulate → classify
        counts = {"Continue": 0, "Skip (Mastered)": 0, "Remedial Required": 0}
        n_pass = 0
        for mi in init_idx:
            s = simulate_assessment(features, mi)
            act = classify_action(s, req.pass_thr, req.mastery_thr)
            counts[act] += 1
            if s >= req.pass_thr:
                n_pass += 1
            if act == "Remedial Required":
                module_remedial[mi] = module_remedial.get(mi, 0) + 1

        total = len(init_idx) or 1
        pr = round(n_pass / total, 4)
        pass_rates.append(pr)
        tier_counts[tier["name"]] = tier_counts.get(tier["name"], 0) + 1

        rows.append({
            **meta,
            "cluster": label,
            "tier": tier["name"],
            "readiness": tier["score"],
            "mean_ensemble": round(float(np.mean(scores[init_idx])), 4),
            "pass_rate": pr,
            "continue": counts["Continue"],
            "skip": counts["Skip (Mastered)"],
            "remedial": counts["Remedial Required"],
            "status": _status(counts["Remedial Required"]),
        })

    # cohort aggregates
    n = len(rows)
    needs = sum(1 for r in rows if r["status"] == "Needs support")
    monitor = sum(1 for r in rows if r["status"] == "Monitor")
    cohort_summary = {
        "n_learners": n,
        "avg_pass_rate": round(float(np.mean(pass_rates)), 4) if pass_rates else 0.0,
        "needs_support": needs,
        "monitor": monitor,
        "on_track": n - needs - monitor,
        "total_remedial": sum(r["remedial"] for r in rows),
    }

    # module hotspots — most frequently remediated across the cohort
    hotspots = sorted(module_remedial.items(), key=lambda kv: kv[1], reverse=True)[:8]
    module_hotspots = [
        {**_module_row(mi), "remedial_count": cnt, "remedial_rate": round(cnt / n, 4)}
        for mi, cnt in hotspots
    ]

    # tier distribution ordered by readiness (highest first)
    tier_distribution = [
        {"name": meta["name"], "score": meta["score"],
         "count": tier_counts.get(meta["name"], 0)}
        for _, meta in sorted(A.cluster_to_tier.items(),
                              key=lambda kv: kv[1]["score"], reverse=True)
    ]

    return {
        "n": n,
        "thresholds": {"pass": req.pass_thr, "mastery": req.mastery_thr},
        "weights_used": [round(float(x), 4)
                         for x in (np.asarray(req.weights) / (sum(req.weights) or 1)).tolist()],
        "cohort": cohort_summary,
        "tier_distribution": tier_distribution,
        "module_hotspots": module_hotspots,
        "learners": rows,
    }


# Optional: serve dashboard.html from the same origin (avoids CORS entirely).
# Place dashboard.html + dashboard_data.json next to app.py and uncomment:
#
# from fastapi.staticfiles import StaticFiles
# app.mount("/", StaticFiles(directory=".", html=True), name="static")