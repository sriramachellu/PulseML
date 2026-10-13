# PulseML API Reference

FastAPI service in [`app/api.py`](../app/api.py). Interactive OpenAPI docs are at `http://localhost:8000/docs` while it's running.

> Every response carries `X-Disclaimer: Research demonstration only. Not for clinical use.` and every
> JSON body from `/predict` includes the same `disclaimer` field.

## Running

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000          # local
docker compose up api                                # container (port 8000, or $PORT if set)
```

| Environment variable | Default | Purpose |
|---|---|---|
| `PULSEML_MODEL_PATH` | `outputs/models/cnn_best.pt` | Waveform-only CNN checkpoint (**required**) |
| `PULSEML_HYBRID_MODEL_PATH` | `outputs/models/hybrid_best.pt` | Hybrid checkpoint (optional; enables `rr_context`) |
| `PORT` | `8000` | Port used by the Docker image |

## Model routing

| Request contains | Model used | Why |
|---|---|---|
| `waveform` only | `cnn` | Needs nothing beyond the beat window |
| `waveform` + `rr_context` (+ optional `patient_template`) | `hybrid` (falls back to `cnn` if not loaded) | The recommended model: a 3-seed ensemble of HybridCNN-rel using rhythm context and deviation from the patient's typical beat |

The response's `model_used` field says which model answered.

## `GET /health`

```json
{"status": "ok", "model_loaded": true, "models_available": ["cnn", "hybrid"],
 "uptime_seconds": 12.3, "disclaimer": "Research demonstration only. Not for clinical use."}
```

`status` is `degraded` and `model_loaded` is `false` if the primary checkpoint failed to load; the
service stays up so orchestrators can report the failure.

## `GET /model-info`

Returns, per loaded model: number of ensemble members, SHA-256 version hash (first 12 hex chars), architecture summary and full
module printout, parameter count, required inputs, augmentation flag, training date, best epoch,
validation macro-F1, and the DS2 test metrics from `outputs/reports/evaluation_metrics.json`.
Top-level `version_hash`, `architecture`, `training_date` and `ds2_metrics` describe the primary `cnn`
model, for backwards compatibility.

## `POST /predict`

### Request

```json
{
  "waveform": [/* exactly 280 floats */],
  "rr_context": {                      // optional
    "rr_prev_ms": 620,                 // previous R-peak -> this R-peak
    "rr_next_ms": 1050,                // this R-peak -> next R-peak
    "rr_local_avg_ms": 830             // optional: mean RR over the patient's preceding ~32 beats
  },
  "patient_template": [/* optional: 280 floats */]
}
```

| Field | Constraint |
|---|---|
| `waveform` | Exactly 280 finite floats, each within ±50. 360 Hz, R-peak at index 140, ideally band-passed 0.5–40 Hz. The service z-scores the beat itself, so any amplitude unit works |
| `rr_prev_ms`, `rr_next_ms` | 150–10,000 ms (MIT-BIH contains genuine sinus pauses up to ~6 s, e.g. record 232) |
| `rr_local_avg_ms` | 150–10,000 ms; defaults to `rr_prev_ms` (then `rr_prev_norm` = 1, which removes the prematurity signal, so send it whenever possible) |
| `patient_template` | Optional; exactly 280 finite floats: the sample-wise **median of the patient's previous ~16 beat windows, each z-scored first**. Compute it with `src.preprocess.build_patient_template(previous_beats)`. It is used **as-is** (not re-normalised), exactly as in training. Without it the deviation channel is zero and the model relies on morphology + rhythm only |

Validation failures return **422** with Pydantic error details (still with the disclaimer header).
If no model is loaded, **503**.

### Response

```json
{
  "predicted_class": "S",
  "predicted_class_description": "Supraventricular ectopic",
  "confidence": 0.71,
  "probabilities": {"N": 0.22, "S": 0.71, "V": 0.04, "F": 0.03},
  "raw_probabilities": {"N": 0.05, "S": 0.90, "V": 0.03, "F": 0.02},
  "calibrated": true,
  "prediction_set": ["N", "S"],
  "prediction_set_coverage_target": 0.9,
  "model_used": "hybrid",
  "patient_template_used": true,
  "model_version": "1.0.0",
  "model_hash": "3f1c0d2a9b7e",
  "disclaimer": "Research demonstration only. Not for clinical use."
}
```

| Field | Meaning |
|---|---|
| `probabilities` | Calibrated probabilities when `calibrated` is true, otherwise the raw softmax |
| `raw_probabilities` | The ensemble's softmax before calibration (inflated for S/F by class-weighted training) |
| `calibrated` | True only for the hybrid model when `calibration.json` was fitted for the same model type |
| `predicted_class` | Argmax of the **raw** probabilities by default (BCTS argmax almost never predicts S/F); argmax of the calibrated ones if `use_calibrated_argmax` is set in `calibration.json` |
| `confidence` | `probabilities[predicted_class]` |
| `prediction_set` | Mondrian conformal set: every class *c* with 1 − p_c ≤ qhat_c. Targets 90% coverage per class on patients like DS1 (DS2 achieved 86% overall, 84.5% for N); `null` for the CNN |

Calibration parameters come from `outputs/models/calibration.json` (written by `python -m src.calibrate`,
fitted on DS1 out-of-fold predictions only). Override the path with `PULSEML_CALIBRATION_PATH`; if the file is
missing, responses are uncalibrated and `prediction_set` is `null`. `GET /model-info` reports the loaded
calibration under `calibration`.

### Example

```bash
python - <<'EOF'
import json, urllib.request, pandas as pd
beat = pd.read_parquet("outputs/parquet/demo_beats.parquet").query("label == 1").iloc[0]
body = {"waveform": [float(v) for v in beat.waveform],
        "patient_template": [float(v) for v in beat.template],
        "rr_context": {"rr_prev_ms": beat.rr_prev / 0.36, "rr_next_ms": beat.rr_next / 0.36,
                       "rr_local_avg_ms": beat.rr_local_avg / 0.36}}
req = urllib.request.Request("http://localhost:8000/predict", json.dumps(body).encode(),
                             {"Content-Type": "application/json"})
print(json.load(urllib.request.urlopen(req)))
EOF
```

(Parquet RR values are in samples at 360 Hz; dividing by 0.36 converts samples to ms.)

**Parity check:** sending all 400 demo beats through the running API (with calibration loaded) reproduces the
offline evaluation's hybrid labels and raw probabilities exactly (400/400, |Δp| < 1e-3), at about 9 ms per request;
the prediction sets contain the true label for 95.8% of these (minority-enriched) demo beats.

## Logging and privacy

Every prediction is logged to stdout as one JSON line:

```json
{"timestamp": "...", "level": "INFO", "logger": "pulseml.api", "message": "prediction",
 "input_hash": "298cb7db4f07ea42", "model": "hybrid", "predicted_class": "S",
 "max_confidence": 0.71, "model_hash": "3f1c0d2a9b7e"}
```

Raw waveforms and RR values are **never** logged. Only a SHA-256 hash of the waveform is kept, so a
specific request can be matched without storing the signal. A test checks this
(`tests/test_api.py::test_prediction_log_contains_hash_not_waveform`).
