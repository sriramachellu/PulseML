"""PulseML inference API.

Two models:
  cnn     waveform-only CNN, needs just the 280-sample beat
  hybrid  the recommended ensemble (beat + deviation from the patient's template + RR features),
          used whenever the request includes rr_context

Research demonstration only, not for clinical use.
"""

import json
import logging
import math
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from src import config
from src.models import apply_calibration, count_parameters, load_cnn, prediction_sets
from src.preprocess import rr_features
from src.utils import array_hash, file_sha256, load_json, utc_now

MODEL_PATH = Path(os.environ.get("PULSEML_MODEL_PATH", config.CNN_CHECKPOINT))
HYBRID_MODEL_PATH = Path(os.environ.get("PULSEML_HYBRID_MODEL_PATH", config.HYBRID_CHECKPOINT))
CALIBRATION_PATH = Path(os.environ.get("PULSEML_CALIBRATION_PATH", config.MODELS_DIR / "calibration.json"))
MAX_ABS_VALUE = 50.0
EVAL_KEYS = {"cnn": "CNN", "hybrid": "Hybrid"}


class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "timestamp": utc_now(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        return json.dumps(payload | getattr(record, "extra_fields", {}))


logger = logging.getLogger("pulseml.api")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def log(level, msg, **fields):
    logger.log(level, msg, extra={"extra_fields": fields})


# name -> {"model", "meta", "hash"}
models = {}
load_errors = {}
calibration = None
started_at = time.time()


def load_one(name, path, required):
    try:
        model, meta = load_cnn(path)
        models[name] = {"model": model, "meta": meta, "hash": file_sha256(path)}
        load_errors.pop(name, None)
        log(logging.INFO, "model loaded", model=name, path=str(path), version_hash=models[name]["hash"])
    except Exception as exc:  # keep /health up even without a model
        models.pop(name, None)
        load_errors[name] = f"{type(exc).__name__}: {exc}"
        log(logging.ERROR if required else logging.WARNING, "model load failed", model=name, error=str(exc))


def load_calibration(path):
    # only use calibration fitted for the hybrid model type actually loaded
    hybrid = models.get("hybrid")
    if hybrid is None or not path.exists():
        return None
    try:
        cal = load_json(path)
    except Exception as exc:
        log(logging.WARNING, "calibration load failed", error=str(exc))
        return None
    if cal.get("model") != hybrid["meta"].get("model_type"):
        log(logging.WARNING, "calibration ignored: fitted for a different model type")
        return None
    log(logging.INFO, "calibration loaded", method=cal.get("method"), alpha=cal.get("alpha"))
    return cal


@asynccontextmanager
async def lifespan(app):
    global calibration, started_at
    started_at = time.time()
    load_one("cnn", MODEL_PATH, required=True)
    load_one("hybrid", HYBRID_MODEL_PATH, required=False)
    calibration = load_calibration(CALIBRATION_PATH)
    yield


app = FastAPI(
    title="PulseML ECG Beat Classifier",
    description="AAMI EC57 beat classification (N/S/V/F) from a single 280-sample MLII window, optionally with "
    "RR-interval context. **Research demonstration only. Not for clinical use.**",
    version=config.MODEL_VERSION,
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    expose_headers=["X-Disclaimer"],
)


@app.middleware("http")
async def disclaimer_header(request: Request, call_next):
    response = await call_next(request)  # also covers validation errors
    response.headers["X-Disclaimer"] = config.DISCLAIMER
    return response


class RRContext(BaseModel):
    """RR intervals around the beat in ms. Turns on the hybrid model."""

    rr_prev_ms: float = Field(gt=150, lt=10000, description="Previous R-peak to this one")
    rr_next_ms: float = Field(gt=150, lt=10000, description="This R-peak to the next one")
    rr_local_avg_ms: float | None = Field(
        None, gt=150, lt=10000, description="Mean RR over the patient's last ~32 beats (defaults to rr_prev_ms)"
    )


class PredictRequest(BaseModel):
    waveform: list[float] = Field(
        min_length=config.WINDOW_SIZE,
        max_length=config.WINDOW_SIZE,
        description="280 samples at 360 Hz centred on the R-peak, band-passed 0.5–40 Hz. Z-scored by the service.",
    )
    rr_context: RRContext | None = None
    patient_template: list[float] | None = Field(
        None,
        min_length=config.WINDOW_SIZE,
        max_length=config.WINDOW_SIZE,
        description="Median of the patient's previous ~16 z-scored beats (src.preprocess.build_patient_template). "
        "Used as-is. Without it the hybrid gets no deviation signal.",
    )

    @field_validator("waveform", "patient_template")
    @classmethod
    def check_values(cls, v):
        if v is not None:
            if not all(math.isfinite(x) for x in v):
                raise ValueError("waveform contains NaN or infinite values")
            if any(abs(x) > MAX_ABS_VALUE for x in v):
                raise ValueError(f"waveform values must lie within ±{MAX_ABS_VALUE}")
        return v


class PredictResponse(BaseModel):
    predicted_class: str
    predicted_class_description: str
    confidence: float
    probabilities: dict[str, float]
    raw_probabilities: dict[str, float]
    calibrated: bool
    prediction_set: list[str] | None
    prediction_set_coverage_target: float | None
    model_used: str
    patient_template_used: bool
    model_version: str
    model_hash: str | None
    disclaimer: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    models_available: list[str]
    uptime_seconds: float
    disclaimer: str


def require_model():
    if "cnn" not in models:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {load_errors.get('cnn')}")


@app.get("/health", response_model=HealthResponse)
def health():
    loaded = "cnn" in models
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        models_available=sorted(models),
        uptime_seconds=round(time.time() - started_at, 2),
        disclaimer=config.DISCLAIMER,
    )


@app.get("/model-info")
def model_info():
    require_model()
    ev = load_json(config.EVAL_METRICS) if config.EVAL_METRICS.exists() else {}
    info = {}
    for name, m in models.items():
        meta, model = m["meta"], m["model"]
        ds2 = ev.get("models", {}).get(EVAL_KEYS[name])
        inputs = ["waveform", "rr_context"] if name == "hybrid" else ["waveform"]
        if getattr(model, "in_channels", 1) == 2:
            inputs.append("patient_template (optional)")
        info[name] = {
            "version_hash": m["hash"],
            "architecture_summary": meta.get("architecture"),
            "architecture": str(model),
            "n_parameters": count_parameters(model),
            "inputs": inputs,
            "ensemble_members": meta.get("n_members", 1),
            "augmentation": meta.get("augmentation"),
            "training_date": meta.get("trained_at"),
            "best_epoch": meta.get("best_epoch"),
            "best_val_macro_f1": meta.get("best_val_macro_f1"),
            "ds2_metrics": {"evaluated_at": ev.get("evaluated_at"), "n_test_beats": ev.get("n_test_beats"), **ds2}
            if ds2
            else None,
        }
    cal_keys = [
        "method",
        "temperature",
        "bias",
        "alpha",
        "qhat",
        "conformal",
        "use_calibrated_argmax",
        "fit_data",
        "fitted_at",
    ]
    cnn = info["cnn"]
    return {
        "name": "PulseML ECG beat classifier",
        "model_version": config.MODEL_VERSION,
        # top-level fields describe the cnn, as in the first API version
        "version_hash": cnn["version_hash"],
        "architecture": cnn["architecture"],
        "n_parameters": cnn["n_parameters"],
        "training_date": cnn["training_date"],
        "ds2_metrics": cnn["ds2_metrics"],
        "models": info,
        "routing": "requests with rr_context use 'hybrid' when available; otherwise 'cnn'",
        "calibration": {k: calibration.get(k) for k in cal_keys} if calibration else None,
        "input": {"length": config.WINDOW_SIZE, "sampling_rate_hz": config.FS, "lead": config.PREFERRED_LEAD},
        "classes": {c: config.CLASS_DESCRIPTIONS[c] for c in config.CLASS_NAMES},
        "split": "de Chazal inter-patient: trained on DS1, evaluated on held-out DS2",
        "disclaimer": config.DISCLAIMER,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    require_model()
    x = np.asarray(req.waveform, dtype=np.float32)
    x = (x - x.mean()) / (x.std() + 1e-8)
    wf, feats, name, template_used = x[None, None], None, "cnn", False

    if req.rr_context and "hybrid" in models:
        name = "hybrid"
        rc = req.rr_context
        to_samples = config.FS / 1000
        f = rr_features(
            rc.rr_prev_ms * to_samples, rc.rr_next_ms * to_samples, (rc.rr_local_avg_ms or rc.rr_prev_ms) * to_samples
        )
        feats = torch.tensor([[f[c] for c in config.RR_FEATURES]], dtype=torch.float32)
        if getattr(models[name]["model"], "in_channels", 1) == 2:
            # the template is already a median of z-scored beats (as in training), so no re-normalising
            template_used = req.patient_template is not None
            tmpl = np.asarray(req.patient_template, dtype=np.float32) if template_used else x
            wf = np.stack([x, x - tmpl])[None]

    m = models[name]
    with torch.no_grad():
        raw = m["model"](torch.from_numpy(wf), feats).softmax(1)[0].double().numpy()

    cal = calibration if name == "hybrid" else None
    probs, pset, k = raw, None, int(raw.argmax())
    if cal:
        probs = apply_calibration(np.log(np.clip(raw, 1e-12, None)), cal["temperature"], cal.get("bias"))
        if cal.get("use_calibrated_argmax"):
            k = int(probs.argmax())
        pset = [c for c, keep in zip(config.CLASS_NAMES, prediction_sets(probs, cal["qhat"])[0]) if keep]
    cls = config.CLASS_NAMES[k]

    # never log the waveform itself, only a hash of it
    log(
        logging.INFO, "prediction", input_hash=array_hash(req.waveform), model=name, patient_template_used=template_used,
        predicted_class=cls, max_confidence=round(float(probs[k]), 4), prediction_set=pset, model_hash=m["hash"],
    )  # fmt: skip
    return PredictResponse(
        predicted_class=cls,
        predicted_class_description=config.CLASS_DESCRIPTIONS[cls],
        confidence=float(probs[k]),
        probabilities=dict(zip(config.CLASS_NAMES, probs.tolist())),
        raw_probabilities=dict(zip(config.CLASS_NAMES, raw.tolist())),
        calibrated=cal is not None,
        prediction_set=pset,
        prediction_set_coverage_target=1 - cal["alpha"] if cal else None,
        model_used=name,
        patient_template_used=template_used,
        model_version=config.MODEL_VERSION,
        model_hash=m["hash"],
        disclaimer=config.DISCLAIMER,
    )
