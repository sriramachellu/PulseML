import io
import json
import logging

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import api
from src import config


@pytest.fixture
def client(cnn_checkpoint, hybrid_checkpoint, tmp_path, monkeypatch):
    monkeypatch.setattr(api, "MODEL_PATH", cnn_checkpoint)
    monkeypatch.setattr(api, "HYBRID_MODEL_PATH", hybrid_checkpoint)
    monkeypatch.setattr(api, "CALIBRATION_PATH", tmp_path / "no_calibration.json")
    with TestClient(api.app) as c:
        yield c


def beat(n=280):
    t = np.arange(n)
    return (np.exp(-0.5 * ((t - 140) / 5) ** 2) + 0.01 * np.sin(t)).tolist()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["models_available"] == ["cnn", "hybrid"]
    assert body["uptime_seconds"] >= 0


def test_predict_valid_input(client):
    r = client.post("/predict", json={"waveform": beat()})
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_class"] in config.CLASS_NAMES
    assert set(body["probabilities"]) == set(config.CLASS_NAMES)
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-4
    assert body["confidence"] == max(body["probabilities"].values())
    assert body["model_version"] == config.MODEL_VERSION
    assert body["model_used"] == "cnn"


def test_predict_with_rr_context_uses_hybrid(client):
    rr = {"rr_prev_ms": 620, "rr_next_ms": 1050, "rr_local_avg_ms": 830}
    r = client.post("/predict", json={"waveform": beat(), "rr_context": rr})
    assert r.status_code == 200
    body = r.json()
    assert body["model_used"] == "hybrid"
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-4
    # rr_local_avg_ms is optional
    r = client.post("/predict", json={"waveform": beat(), "rr_context": {"rr_prev_ms": 800, "rr_next_ms": 800}})
    assert r.json()["model_used"] == "hybrid"


@pytest.mark.parametrize("rr", [{"rr_prev_ms": 10, "rr_next_ms": 800}, {"rr_prev_ms": 800}, {"rr_next_ms": -5}])
def test_predict_rejects_invalid_rr_context(client, rr):
    assert client.post("/predict", json={"waveform": beat(), "rr_context": rr}).status_code == 422


def test_predict_response_includes_disclaimer(client):
    r = client.post("/predict", json={"waveform": beat()})
    assert r.headers["X-Disclaimer"] == "Research demonstration only. Not for clinical use."
    assert "Not for clinical use" in r.json()["disclaimer"]


@pytest.mark.parametrize("n", [0, 279, 281, 500])
def test_predict_rejects_wrong_length(client, n):
    r = client.post("/predict", json={"waveform": [0.1] * n})
    assert r.status_code == 422
    assert r.headers["X-Disclaimer"]


def test_predict_rejects_out_of_range_and_bad_types(client):
    wf = beat()
    wf[10] = 1e6
    assert client.post("/predict", json={"waveform": wf}).status_code == 422
    assert client.post("/predict", json={"waveform": ["a"] * 280}).status_code == 422
    assert client.post("/predict", json={}).status_code == 422


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["version_hash"]
    assert body["input"]["length"] == 280
    assert "Conv1d" in body["architecture"]
    assert set(body["models"]) == {"cnn", "hybrid"}
    assert body["models"]["hybrid"]["inputs"] == ["waveform", "rr_context"]


def test_prediction_log_contains_hash_not_waveform(client):
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(api.JsonFormatter())
    api.logger.addHandler(handler)
    wf = beat()
    try:
        client.post("/predict", json={"waveform": wf})
    finally:
        api.logger.removeHandler(handler)
    out = buf.getvalue()
    lines = [json.loads(line) for line in out.splitlines() if '"prediction"' in line]
    assert lines, "expected a structured prediction log line"
    rec = lines[-1]
    assert {"timestamp", "input_hash", "predicted_class", "max_confidence"} <= set(rec)
    assert "waveform" not in rec
    assert str(round(wf[140], 6))[:6] not in json.dumps(rec)


def test_health_degraded_without_model(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "MODEL_PATH", tmp_path / "missing.pt")
    monkeypatch.setattr(api, "HYBRID_MODEL_PATH", tmp_path / "missing_hybrid.pt")
    logging.getLogger("pulseml.api").disabled = True
    try:
        with TestClient(api.app) as c:
            assert c.get("/health").json()["model_loaded"] is False
            assert c.post("/predict", json={"waveform": beat()}).status_code == 503
    finally:
        logging.getLogger("pulseml.api").disabled = False


@pytest.fixture
def rel_client(cnn_checkpoint, hybrid_rel_ensemble_checkpoint, tmp_path, monkeypatch):
    monkeypatch.setattr(api, "MODEL_PATH", cnn_checkpoint)
    monkeypatch.setattr(api, "HYBRID_MODEL_PATH", hybrid_rel_ensemble_checkpoint)
    monkeypatch.setattr(api, "CALIBRATION_PATH", tmp_path / "no_calibration.json")
    with TestClient(api.app) as c:
        yield c


def test_patient_template_with_relative_ensemble(rel_client):
    rr = {"rr_prev_ms": 620, "rr_next_ms": 1050, "rr_local_avg_ms": 830}
    with_t = rel_client.post("/predict", json={"waveform": beat(), "rr_context": rr, "patient_template": beat()})
    without = rel_client.post("/predict", json={"waveform": beat(), "rr_context": rr})
    assert with_t.status_code == without.status_code == 200
    assert with_t.json()["model_used"] == "hybrid" and with_t.json()["patient_template_used"] is True
    assert without.json()["patient_template_used"] is False
    info = rel_client.get("/model-info").json()["models"]["hybrid"]
    assert info["ensemble_members"] == 2
    assert "patient_template (optional)" in info["inputs"]


def test_patient_template_validated(rel_client):
    rr = {"rr_prev_ms": 800, "rr_next_ms": 800}
    bad = rel_client.post("/predict", json={"waveform": beat(), "rr_context": rr, "patient_template": [0.0] * 10})
    assert bad.status_code == 422


def test_long_pause_rr_accepted(client):
    rr = {"rr_prev_ms": 5800, "rr_next_ms": 900, "rr_local_avg_ms": 1200}  # sinus pause seen in DS2
    assert client.post("/predict", json={"waveform": beat(), "rr_context": rr}).status_code == 200


def _calibration_file(tmp_path, model="hybrid_rel"):
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "model": model,
                "method": "bcts",
                "temperature": 2.0,
                "bias": [0.5, -0.2, -0.1, -0.2],
                "alpha": 0.1,
                "qhat": [0.9, 0.9, 0.9, 0.9],
                "use_calibrated_argmax": True,
            }
        )
    )
    return path


def test_calibrated_probabilities_and_prediction_set(
    cnn_checkpoint, hybrid_rel_ensemble_checkpoint, tmp_path, monkeypatch
):
    monkeypatch.setattr(api, "MODEL_PATH", cnn_checkpoint)
    monkeypatch.setattr(api, "HYBRID_MODEL_PATH", hybrid_rel_ensemble_checkpoint)
    monkeypatch.setattr(api, "CALIBRATION_PATH", _calibration_file(tmp_path))
    rr = {"rr_prev_ms": 620, "rr_next_ms": 1050, "rr_local_avg_ms": 830}
    with TestClient(api.app) as c:
        body = c.post("/predict", json={"waveform": beat(), "rr_context": rr}).json()
        assert body["calibrated"] is True
        assert body["prediction_set_coverage_target"] == pytest.approx(0.9)
        assert body["predicted_class"] == max(body["probabilities"], key=body["probabilities"].get)
        assert set(body["prediction_set"]) <= set(config.CLASS_NAMES)
        expected = [k for k, p in body["probabilities"].items() if 1 - p <= 0.9]
        assert body["prediction_set"] == expected
        assert body["raw_probabilities"] != body["probabilities"]
        # the waveform-only CNN is never calibrated with the hybrid's parameters
        cnn = c.post("/predict", json={"waveform": beat()}).json()
        assert cnn["calibrated"] is False and cnn["prediction_set"] is None
        assert c.get("/model-info").json()["calibration"]["method"] == "bcts"


def test_calibration_for_other_model_type_is_ignored(cnn_checkpoint, hybrid_checkpoint, tmp_path, monkeypatch):
    monkeypatch.setattr(api, "MODEL_PATH", cnn_checkpoint)
    monkeypatch.setattr(api, "HYBRID_MODEL_PATH", hybrid_checkpoint)  # plain "hybrid"
    monkeypatch.setattr(api, "CALIBRATION_PATH", _calibration_file(tmp_path, model="hybrid_rel"))
    rr = {"rr_prev_ms": 620, "rr_next_ms": 1050}
    with TestClient(api.app) as c:
        body = c.post("/predict", json={"waveform": beat(), "rr_context": rr}).json()
        assert body["calibrated"] is False and body["raw_probabilities"] == body["probabilities"]
