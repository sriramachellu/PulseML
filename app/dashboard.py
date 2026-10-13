"""PulseML research dashboard: `streamlit run app/dashboard.py`. Not for clinical diagnosis."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # `streamlit run` only puts app/ on the path

from src import config  # noqa: E402
from src.dataset import template_matrix  # noqa: E402
from src.models import apply_calibration, load_cnn, prediction_sets  # noqa: E402
from src.preprocess import rr_features  # noqa: E402

BLUE = "#1f5fa8"
MODEL_COLORS = {"LogReg": "#7f8c9b", "XGBoost": "#e8710a", "CNN": "#8fb3e0", "Hybrid": BLUE}
CLASS_COLORS = {"N": BLUE, "S": "#e8710a", "V": "#c5221f", "F": "#7b3fa0"}
LIVE_COLORS = {"CNN (waveform)": "#8fb3e0", "Hybrid (waveform + RR)": "#7f8c9b", "Hybrid (calibrated)": BLUE}
T_MS = (np.arange(config.WINDOW_SIZE) - config.WINDOW_BEFORE) / config.FS * 1000

st.set_page_config(page_title="PulseML: ECG Arrhythmia Research Dashboard", page_icon="🫀", layout="wide")
st.markdown(
    f"""<style>
      .stApp {{ background: #ffffff; }}
      h1, h2, h3 {{ color: {BLUE}; }}
      .pm-banner {{ background:#fdecea; border:2px solid #d93025; color:#a50e0e; padding:12px 16px;
                   border-radius:8px; font-weight:700; font-size:1.05rem; text-align:center; margin-bottom:12px; }}
      .pm-note {{ color:#5f6b7a; font-size:0.85rem; }}
    </style>""",
    unsafe_allow_html=True,
)


def banner():
    st.markdown(
        "<div class='pm-banner'>⚠️ RESEARCH DEMONSTRATION, NOT FOR CLINICAL DIAGNOSIS</div>", unsafe_allow_html=True
    )


def note(text):
    st.markdown(f"<span class='pm-note'>{text}</span>", unsafe_allow_html=True)


def chart(fig, height=360, where=st):
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=10, r=10, t=40, b=10),
        font=dict(family="Segoe UI, Helvetica, Arial, sans-serif"),
    )  # fmt: skip
    where.plotly_chart(fig, width="stretch")


def table(df):
    st.dataframe(df.style.format(precision=3), hide_index=True, width="stretch")


@st.cache_data(show_spinner="Loading feature store…")
def load_split(name):
    path = {"DS1": config.DS1_PARQUET, "DS2": config.DS2_PARQUET}[name]
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["aami"] = df["label"].map(dict(enumerate(config.CLASS_NAMES)))
    if name == "DS2":
        df["template"] = list(template_matrix(df))  # needed by the patient-relative hybrid
    return df


@st.cache_data
def load_demo():
    if not config.DEMO_PARQUET.exists():
        return None
    df = pd.read_parquet(config.DEMO_PARQUET)
    df["aami"] = df["label"].map(dict(enumerate(config.CLASS_NAMES)))
    return df


@st.cache_data
def read_json(path):
    try:
        return json.loads(open(path, encoding="utf-8").read())
    except FileNotFoundError:
        return None


@st.cache_resource
def load_model(path):
    return load_cnn(path)[0] if path.exists() else None


@st.cache_resource
def load_xgb():
    return joblib.load(config.XGB_MODEL) if config.XGB_MODEL.exists() else None


def page_data_explorer():
    st.title("Data Explorer")
    st.caption("MIT-BIH Arrhythmia Database · 44 non-paced records · AAMI EC57 classes · de Chazal inter-patient split")
    banner()

    dq = read_json(config.REPORTS_DIR / "data_quality.json")
    ds1, ds2, demo = load_split("DS1"), load_split("DS2"), load_demo()
    if ds1 is None and demo is None:
        st.error("No processed data found. Run `python -m src.download && python -m src.preprocess`.")
        return
    if ds1 is None:
        st.info("Full feature store not found, showing the bundled DS2 demo sample (100 beats per class).")

    if dq:
        c = st.columns(5)
        c[0].metric("Records", dq["total_records"])
        c[1].metric("Beats", f"{dq['total_beats']:,}")
        c[2].metric("DS1 beats", f"{dq['checks']['DS1']['n_beats']:,}")
        c[3].metric("DS2 beats", f"{dq['checks']['DS2']['n_beats']:,}")
        c[4].metric("Quality checks", "PASS" if dq["passed"] else "FAIL")

        st.subheader("Class distribution")
        fig = px.bar(
            pd.DataFrame(dq["class_distribution"]), x="aami", y="n", color="split", barmode="group", text="n",
            log_y=st.toggle("Log scale", value=True), color_discrete_map={"DS1": BLUE, "DS2": "#8fb3e0"},
            labels={"aami": "AAMI class", "n": "beats"}, category_orders={"aami": config.CLASS_NAMES},
        )  # fmt: skip
        chart(fig)
        note("~90% of beats are Normal, which is why we report macro-F1 and per-class metrics rather than accuracy.")

    st.subheader("Sample waveforms by class")
    source = ds2 if ds2 is not None else demo
    n_show = st.slider("Beats per class", 1, 10, 5)
    seed = int(st.number_input("Random seed", value=config.SEED, step=1))
    for col, (k, name) in zip(st.columns(4), enumerate(config.CLASS_NAMES)):
        sub = source[source["label"] == k]
        fig = go.Figure()
        for _, row in sub.sample(min(n_show, len(sub)), random_state=seed).iterrows():
            line = dict(width=1.2, color=CLASS_COLORS[name])
            fig.add_scatter(x=T_MS, y=row["waveform"], mode="lines", line=line, opacity=0.7, showlegend=False)
        fig.update_layout(
            title=f"{name}: {config.CLASS_DESCRIPTIONS[name]}", xaxis_title="ms from R-peak", yaxis_title="z-score"
        )
        chart(fig, 300, col)

    st.subheader("Feature distributions")
    feat = st.selectbox("Feature", config.FEATURE_COLUMNS, index=config.FEATURE_COLUMNS.index("rr_ratio"))
    fsrc = ds1 if ds1 is not None else demo
    lo, hi = fsrc[feat].quantile([0.005, 0.995])
    fig = px.box(
        fsrc[fsrc[feat].between(lo, hi)], x="aami", y=feat, color="aami", points=False,
        color_discrete_map=CLASS_COLORS, category_orders={"aami": config.CLASS_NAMES},
    )  # fmt: skip
    chart(fig)

    if not dq:
        return
    st.subheader("Data-quality summary")
    rows = []
    for split in ("DS1", "DS2"):
        ch = dq["checks"][split]
        rows.append(
            {
                "split": split,
                "beats": ch["n_beats"],
                "records": ch["n_records"],
                "amplitude flags (>±5 mV)": ch["amplitude"]["n_flagged"],
                "flat beats": ch["flat"]["n_flat"],
                "duplicate keys": ch["duplicates"]["duplicate_keys"],
                "wrong length": ch["waveform_length"]["n_wrong_length"],
                "passed": ch["passed"],
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if leak := dq["checks"]["patient_leakage"]["overlapping_records"]:
        st.error(f"Leakage: {leak}")
    else:
        st.success("No patient leakage: DS1 and DS2 share no records.")
    report = config.REPORTS_DIR / "data_quality.html"
    if report.exists():
        st.download_button(
            "Download full HTML data-quality report", report.read_bytes(), "data_quality.html", mime="text/html"
        )
    st.subheader("Pipeline lineage")
    st.dataframe(pd.DataFrame(dq["lineage"]), hide_index=True, width="stretch")


def page_model_performance():
    st.title("Model Performance on DS2")
    st.caption("All metrics are on DS2: 22 patients never seen during training or model selection.")
    banner()
    ev = read_json(config.EVAL_METRICS)
    if ev is None:
        st.error("No evaluation results. Run `python -m src.train && python -m src.evaluate`.")
        return

    st.subheader("Model comparison")
    comp = pd.DataFrame(ev["comparison"])
    best = comp.loc[comp["macro_f1"].idxmax()]
    c = st.columns(4)
    c[0].metric(f"Best macro-F1 ({best['model']})", f"{best['macro_f1']:.3f}")
    c[1].metric("Test beats", f"{ev['n_test_beats']:,}")
    for col, cls in ((c[2], "V"), (c[3], "S")):
        top = comp.loc[comp[f"{cls}_recall"].idxmax()]
        col.metric(f"Best {cls} recall ({top['model']})", f"{top[f'{cls}_recall']:.3f}")
    st.dataframe(
        comp.style.format(precision=3).highlight_max(subset=comp.columns[1:], color="#dbe8f7"),
        hide_index=True,
        width="stretch",
    )
    f1 = comp.melt(
        id_vars="model", value_vars=[f"{c}_f1" for c in config.CLASS_NAMES], var_name="class", value_name="F1"
    )
    f1["class"] = f1["class"].str[0]
    chart(
        px.bar(
            f1,
            x="class",
            y="F1",
            color="model",
            barmode="group",
            range_y=[0, 1],
            color_discrete_map=MODEL_COLORS,
            title="Per-class F1",
        )
    )

    model = st.selectbox("Model for detailed view", list(ev["models"]))
    m = ev["models"][model]
    left, right = st.columns(2)
    with left:
        st.markdown("**Per-class metrics**")
        table(pd.DataFrame(m["per_class"]))
        st.markdown(f"One-vs-rest AUC: `{ {k: v and round(v, 3) for k, v in m['overall']['auc'].items()} }`")
    if config.PREDICTIONS_PARQUET.exists():
        p = pd.read_parquet(config.PREDICTIONS_PARQUET, columns=["label", f"{model}_pred"])
        cm = pd.crosstab(p["label"], p[f"{model}_pred"]).reindex(index=range(4), columns=range(4), fill_value=0)
        norm = right.toggle("Row-normalise", value=True)
        z = cm.div(cm.sum(1).replace(0, 1), axis=0) if norm else cm
        fig = px.imshow(
            z.to_numpy(), x=config.CLASS_NAMES, y=config.CLASS_NAMES, text_auto=".2f" if norm else "d",
            color_continuous_scale="Blues", labels=dict(x="Predicted", y="True", color=""), title=f"{model} confusion matrix",
        )  # fmt: skip
        chart(fig, 380, right)

    st.subheader("Curves")
    figs = {
        "ROC": "roc_curves.png",
        "Precision–recall": "pr_curves.png",
        "Calibration": "calibration.png",
        "Hybrid calibration & conformal": "hybrid_calibration.png",
        "Confusion matrices": "confusion_matrix.png",
        "CNN training": "cnn_training_curves.png",
    }
    for tab, fname in zip(st.tabs(list(figs)), figs.values()):
        path = config.FIGURES_DIR / fname
        if path.exists():
            tab.image(str(path), width="stretch")
        else:
            tab.info(f"{fname} not generated yet.")

    if exp := read_json(config.REPORTS_DIR / "experiments.json"):
        st.subheader("Ablation: what actually helps on unseen patients")
        st.caption(
            f"Each deep variant trained with seeds {exp['seeds']}; chosen on validation macro-F1 only "
            f"(selected: **{exp['selected_torch_variant']}**). DS2 shown for transparency, not for selection."
        )
        table(pd.DataFrame(exp["torch_summary"]))
        table(pd.DataFrame(exp["feature_ablation"])[["variant", "cv_macro_f1", "ds2_macro_f1", "ds2_S_recall"]])
        if cv := read_json(config.REPORTS_DIR / "experiments_cv.json"):
            st.markdown(
                f"**Final selection by 5-fold patient-grouped CV on all 22 DS1 patients:** **{cv['selected_variant']}** "
                "(the 4-patient validation split could not separate the top variants)."
            )
            table(pd.DataFrame(cv["results"]))
        if (config.FIGURES_DIR / "ablation.png").exists():
            st.image(str(config.FIGURES_DIR / "ablation.png"), width="stretch")

    st.subheader("Per-record analysis")
    path = config.REPORTS_DIR / "per_record_metrics.csv"
    if not path.exists():
        return
    rec = pd.read_csv(path, dtype={"record_id": str})
    acc = rec.melt(
        id_vars="record_id", value_vars=[f"{n}_acc" for n in ev["models"]], var_name="model", value_name="accuracy"
    )
    acc["model"] = acc["model"].str.removesuffix("_acc")
    fig = px.bar(
        acc,
        x="record_id",
        y="accuracy",
        color="model",
        barmode="group",
        range_y=[0, 1.02],
        color_discrete_map=MODEL_COLORS,
    )
    fig.update_xaxes(type="category")
    chart(fig, 380)
    flagged = rec.loc[rec[f"{model}_flag"], "record_id"]
    if len(flagged):
        st.warning(f"{model}: {len(flagged)} record(s) with accuracy >10 pp below overall: {', '.join(flagged)}")
    table(rec)


def saliency(model, inp, target, feats=None):
    """|d p_target / d input|, summed over channels and scaled to [0, 1]."""
    t = torch.from_numpy(np.atleast_2d(inp).astype(np.float32))[None].requires_grad_(True)
    model.zero_grad()
    model(t, feats)[0, target].backward()
    g = t.grad.abs().sum(1).numpy().ravel()
    return g / (g.max() + 1e-12)


def parse_waveform(text):
    for sep in "\n; ":
        text = text.replace(sep, ",")
    return np.array([float(v) for v in text.split(",") if v.strip()])


def choose_beat(pool):
    """Input widgets. Returns (waveform, test-set row or None, RR features or None, template or None)."""
    mode = st.radio("Input", ["Random DS2 test beat", "Paste values", "Upload file"], horizontal=True)
    if mode == "Random DS2 test beat":
        if pool is None:
            st.warning("No DS2 data available.")
            return None, None, None, None
        c1, c2 = st.columns([1, 3])
        cls = c1.selectbox("True class filter", ["Any"] + config.CLASS_NAMES)
        sub = pool if cls == "Any" else pool[pool["aami"] == cls]
        if c2.button("🎲 Draw a random beat", type="primary") or st.session_state.get("beat_filter", "") != cls:
            st.session_state.beat_pos = int(np.random.default_rng().integers(len(sub)))
            st.session_state.beat_filter = cls
        row = sub.iloc[st.session_state.beat_pos % len(sub)]
        st.caption(
            f"Record {row['record_id']}, annotation #{row['beat_idx']}, expert label **{row['aami']}** (symbol `{row['symbol']}`)"
        )
        template = np.asarray(row["template"], dtype=np.float64) if "template" in row else None
        return (
            np.asarray(row["waveform"], dtype=np.float64),
            row,
            row[config.RR_FEATURES].to_numpy(np.float32),
            template,
        )

    wf = None
    if mode == "Paste values":
        text = st.text_area(
            f"Paste {config.WINDOW_SIZE} comma/space-separated values (360 Hz, R-peak at sample 140)", height=120
        )
    else:
        up = st.file_uploader("CSV / TXT with 280 values (one row or one column)", type=["csv", "txt"])
        text = up.getvalue().decode() if up else ""
    if text.strip():
        try:
            wf = parse_waveform(text)
        except ValueError:
            st.error("Could not parse the numbers.")

    rr = None
    if st.checkbox("Add RR-interval context (enables the hybrid model)"):
        c = st.columns(3)
        ms = [
            c[0].number_input("RR previous (ms)", 150.0, 3000.0, 800.0, step=10.0),
            c[1].number_input("RR next (ms)", 150.0, 3000.0, 800.0, step=10.0),
            c[2].number_input("Patient's recent mean RR (ms)", 150.0, 3000.0, 800.0, step=10.0),
        ]
        f = rr_features(*(v * config.FS / 1000 for v in ms))
        rr = np.array([f[k] for k in config.RR_FEATURES], dtype=np.float32)
    return wf, None, rr, None


def page_live_prediction():
    st.title("Live Prediction")
    banner()
    cnn, hybrid = load_model(config.CNN_CHECKPOINT), load_model(config.HYBRID_CHECKPOINT)
    if cnn is None:
        st.error("CNN checkpoint not found. Run `python -m src.train`.")
        return
    pool = load_split("DS2")
    wf, row, rr, template = choose_beat(pool if pool is not None else load_demo())

    if wf is None:
        st.info("Provide a waveform to classify.")
        return
    if len(wf) != config.WINDOW_SIZE:
        st.error(f"Expected exactly {config.WINDOW_SIZE} values, got {len(wf)}.")
        return
    if not np.isfinite(wf).all():
        st.error("Waveform contains NaN or infinite values.")
        return

    x = (wf - wf.mean()) / (wf.std() + 1e-8)
    probs = {}
    with torch.no_grad():
        probs["CNN (waveform)"] = cnn(torch.tensor(x[None, None], dtype=torch.float32)).softmax(1)[0].numpy()
    use_hybrid = hybrid is not None and rr is not None
    if use_hybrid:
        feats = torch.from_numpy(np.nan_to_num(rr))[None]
        # the template is already in z-score units, as in training
        inp = np.stack([x, x - (x if template is None else template)]) if hybrid.in_channels == 2 else x[None]
        with torch.no_grad():
            probs["Hybrid (waveform + RR)"] = (
                hybrid(torch.tensor(inp[None], dtype=torch.float32), feats).softmax(1)[0].numpy()
            )
        model_name, model, sal_in, sal_feats = "Hybrid (waveform + RR)", hybrid, inp, feats
    else:
        model_name, model, sal_in, sal_feats = "CNN (waveform)", cnn, x, None

    p = probs[model_name]
    k = int(p.argmax())
    cal = read_json(config.MODELS_DIR / "calibration.json") if use_hybrid else None
    pset = None
    if cal and cal.get("model") == config.HYBRID_MODEL_TYPE:
        p = probs["Hybrid (calibrated)"] = apply_calibration(
            np.log(np.clip(p, 1e-12, None)), cal["temperature"], cal.get("bias")
        )
        if cal.get("use_calibrated_argmax"):
            k = int(p.argmax())
        pset = [c for c, keep in zip(config.CLASS_NAMES, prediction_sets(p, cal["qhat"])[0]) if keep]
    pred = config.CLASS_NAMES[k]
    short = model_name.split(" ")[0]

    left, right = st.columns([3, 2])
    fig = go.Figure()
    fig.add_scatter(x=T_MS, y=x, mode="lines", line=dict(color=BLUE, width=2), name="ECG (z-scored)")
    if template is not None:
        fig.add_scatter(
            x=T_MS,
            y=template,
            mode="lines",
            line=dict(color="#7f8c9b", width=1.5, dash="dash"),
            name="patient's recent median beat",
        )
    fig.add_bar(
        x=T_MS, y=saliency(model, sal_in, k, sal_feats) * (x.max() - x.min()) * 0.4, base=x.min(),
        marker_color="rgba(232,113,10,0.3)", name=f"{short} saliency |∂p/∂x|", hoverinfo="skip",
    )  # fmt: skip
    fig.update_layout(
        title=f"Input beat with {short} gradient saliency", xaxis_title="ms from R-peak", yaxis_title="z-score",
        legend=dict(orientation="h", y=-0.25), bargap=0,
    )  # fmt: skip
    chart(fig, 400, left)

    with right:
        st.metric(
            f"Predicted class ({model_name})",
            f"{pred}: {config.CLASS_DESCRIPTIONS[pred]}",
            f"confidence {p[k]:.1%}",
            delta_color="off",
        )
        pdf = pd.DataFrame(
            [
                {"model": m, "class": c, "probability": float(v)}
                for m, ps in probs.items()
                for c, v in zip(config.CLASS_NAMES, ps)
            ]
        )
        fig = px.bar(
            pdf, x="probability", y="class", color="model", barmode="group", orientation="h", range_x=[0, 1],
            text=pdf["probability"].map("{:.1%}".format), color_discrete_map=LIVE_COLORS,
            category_orders={"class": config.CLASS_NAMES[::-1]},
        )  # fmt: skip
        fig.update_layout(title="Class confidence", legend=dict(orientation="h", y=-0.3))
        chart(fig, 300)
        if pset is not None:
            st.info(
                f"**Prediction set ({1 - cal['alpha']:.0%} target coverage per class):** "
                + (", ".join(pset) or "∅ (no class is plausible enough: treat as out-of-distribution)")
                + ". Larger sets mean more uncertainty; flag these beats for human review."
            )
        if row is not None:
            if pred == row["aami"]:
                st.success(f"Expert label: {row['aami']}. Model agrees.")
            else:
                st.warning(f"Expert label: {row['aami']}. Model disagrees.")

    st.subheader("Top contributing features")
    booster = load_xgb()
    if row is None or booster is None:
        note(
            "Engineered-feature contributions need the full feature vector, which is only available for beats drawn "
            "from the test set. For pasted/uploaded beats, the orange saliency overlay shows which samples drive the "
            "network's prediction; add RR context to use the hybrid model."
        )
    else:
        feats = np.nan_to_num(row[config.FEATURE_COLUMNS].to_numpy(dtype=np.float32))[None]
        xp = booster.predict_proba(feats)[0]
        kx = int(xp.argmax())
        dm = xgb.DMatrix(feats, feature_names=config.FEATURE_COLUMNS)
        contrib = booster.get_booster().predict(dm, pred_contribs=True)[0, kx, :-1]  # last column is the bias
        top = (
            pd.DataFrame({"feature": config.FEATURE_COLUMNS, "contribution": contrib, "value": feats[0]})
            .sort_values("contribution", key=np.abs)
            .tail(8)
        )
        fig = px.bar(
            top, x="contribution", y="feature", orientation="h", color=top["contribution"] > 0,
            color_discrete_map={True: "#c5221f", False: BLUE}, hover_data={"value": ":.3f"},
            title=f"XGBoost (engineered features) predicts {config.CLASS_NAMES[kx]} ({xp[kx]:.1%}); SHAP contributions toward that class",
        )  # fmt: skip
        fig.update_layout(showlegend=False)
        chart(fig, 380)
        note(
            "Feature contributions come from the XGBoost baseline (TreeSHAP), which uses RR-interval context the "
            "single-beat CNN cannot see. Orange shading on the waveform shows which samples most influence the "
            "selected neural model's output."
        )
    st.caption(config.DISCLAIMER)


PAGES = {
    "📊 Data Explorer": page_data_explorer,
    "📈 Model Performance": page_model_performance,
    "🫀 Live Prediction": page_live_prediction,
}
with st.sidebar:
    st.markdown(f"<h2 style='color:{BLUE};margin-bottom:0'>PulseML</h2>", unsafe_allow_html=True)
    st.caption("ECG arrhythmia pipeline · MIT-BIH · AAMI EC57")
    page = st.radio("Navigate", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.error("Research demonstration only. Not a medical device. Not FDA-cleared. Not for clinical diagnosis.")
PAGES[page]()
