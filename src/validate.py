"""Data-quality checks, DuckDB analytics and the HTML/JSON quality report."""

import base64
import html
import io

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config
from src.utils import ensure_dirs, get_logger, save_json, utc_now

plt.switch_backend("Agg")
log = get_logger("validate")


def _by_record(df, mask):
    return df.loc[mask, "record_id"].value_counts().to_dict()


def check_missing(df):
    out = {c: int(df[c].isna().sum()) for c in df.columns if c != "waveform"}
    if "waveform" in df:
        out["waveform"] = sum(w is None or np.isnan(np.asarray(w, dtype=float)).any() for w in df["waveform"])
    return out


def check_amplitude(df, limit=config.AMPLITUDE_LIMIT_MV):
    if df.empty:
        return {"limit_mv": limit, "n_flagged": 0, "flagged_records": {}}
    mask = (df["wf_max"] > limit) | (df["wf_min"] < -limit)
    return {"limit_mv": limit, "n_flagged": int(mask.sum()), "flagged_records": _by_record(df, mask)}


def check_rr(df, max_ms=3000.0, fs=config.FS):
    # RR > 3 s is mostly annotation gaps over unreadable signal, but real pauses exist too -> warning only
    if df.empty:
        return {"max_ms": max_ms, "n_flagged": 0, "flagged_records": {}}
    limit = max_ms * fs / 1000
    mask = (df["rr_prev"] > limit) | (df["rr_next"] > limit)
    return {
        "max_ms": max_ms,
        "n_flagged": int(mask.sum()),
        "max_rr_ms": float(max(df["rr_prev"].max(), df["rr_next"].max()) * 1000 / fs),
        "flagged_records": _by_record(df, mask),
    }


def check_flat(df, tol=1e-6):
    if df.empty:
        return {"n_flat": 0}
    mask = df["wf_std"] < tol
    return {"n_flat": int(mask.sum()), "flat_records": _by_record(df, mask)}


def check_waveform_length(df, expected=config.WINDOW_SIZE):
    return {"expected": expected, "n_wrong_length": sum(len(w) != expected for w in df["waveform"])}


def check_duplicates(df):
    if df.empty:
        return {"duplicate_keys": 0, "duplicate_waveforms": 0}
    wf_hash = df["waveform"].map(lambda w: hash(np.asarray(w, dtype=np.float32).tobytes()))
    return {
        "duplicate_keys": int(df.duplicated(subset=["record_id", "beat_idx"]).sum()),
        "duplicate_waveforms": int(wf_hash.duplicated().sum()),
    }


def check_labels(df):
    return {"n_invalid_labels": int((~df["label"].isin(range(config.NUM_CLASSES))).sum())}


def run_checks(df):
    res = {
        "n_beats": len(df),
        "n_records": df["record_id"].nunique(),
        "empty": df.empty,
        "missing": check_missing(df) if not df.empty else {},
        "amplitude": check_amplitude(df),
        "rr_outliers": check_rr(df),
        "flat": check_flat(df),
        "waveform_length": check_waveform_length(df),
        "duplicates": check_duplicates(df),
        "labels": check_labels(df),
    }
    # amplitude, RR and duplicate-waveform flags are warnings; these are hard failures
    res["passed"] = bool(
        not df.empty
        and res["flat"]["n_flat"] == 0
        and res["waveform_length"]["n_wrong_length"] == 0
        and res["duplicates"]["duplicate_keys"] == 0
        and res["labels"]["n_invalid_labels"] == 0
        and sum(res["missing"].get(c, 0) for c in ["record_id", "label", "waveform"]) == 0
    )
    return res


QUERIES = {
    "class_distribution": """
        SELECT split, aami, COUNT(*) AS n,
               ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY split), 2) AS pct
        FROM beats GROUP BY split, aami ORDER BY split, aami""",
    "record_aggregates": """
        SELECT split, record_id, COUNT(*) AS beats,
               ROUND(AVG(local_hr), 1) AS mean_hr_bpm,
               ROUND(AVG(rr_prev) / 360.0 * 1000, 1) AS mean_rr_ms,
               ROUND(AVG(qrs_proxy), 3) AS mean_qrs_mv,
               ROUND(100.0 * AVG(CASE WHEN label <> 0 THEN 1 ELSE 0 END), 2) AS pct_abnormal
        FROM beats GROUP BY split, record_id ORDER BY split, record_id""",
    "class_by_record": """
        PIVOT (SELECT split, record_id, aami FROM beats)
        ON aami IN ('N', 'S', 'V', 'F') USING COUNT(*)
        GROUP BY split, record_id ORDER BY split, record_id""",
    "hr_by_class": """
        SELECT aami, ROUND(AVG(local_hr), 1) AS mean_hr, ROUND(MEDIAN(local_hr), 1) AS median_hr,
               ROUND(STDDEV(local_hr), 1) AS sd_hr
        FROM beats GROUP BY aami ORDER BY aami""",
}


def duckdb_analytics(ds1=config.DS1_PARQUET, ds2=config.DS2_PARQUET):
    cols = "record_id, label, rr_prev, local_hr, qrs_proxy, ['N','S','V','F'][label + 1] AS aami"
    with duckdb.connect() as con:
        con.execute(f"""
            CREATE VIEW beats AS
            SELECT 'DS1' AS split, {cols} FROM read_parquet('{ds1.as_posix()}')
            UNION ALL
            SELECT 'DS2' AS split, {cols} FROM read_parquet('{ds2.as_posix()}')""")
        return {name: con.execute(q).df() for name, q in QUERIES.items()}


def _png(fig, save_as=None):
    if save_as:
        fig.savefig(config.FIGURES_DIR / save_as, dpi=110, bbox_inches="tight")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def class_bar(ds1, ds2):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, df, name in zip(axes, (ds1, ds2), ("DS1 (train)", "DS2 (test)")):
        counts = df["label"].value_counts().reindex(range(config.NUM_CLASSES), fill_value=0)
        ax.bar_label(ax.bar(config.CLASS_NAMES, counts.values, color="#1f5fa8"), fmt="%d", fontsize=8)
        ax.set(yscale="log", title=name, ylabel="beats (log scale)")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _png(fig, "class_distribution.png")


def corr_heatmap(df):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        df[config.FEATURE_COLUMNS].corr(), cmap="RdBu_r", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f",
        annot_kws={"size": 7}, ax=ax, square=True, cbar_kws={"shrink": 0.7},
    )  # fmt: skip
    ax.set_title("Feature correlation (DS1)")
    fig.tight_layout()
    return _png(fig, "feature_correlation.png")


def _table(df):
    return df.to_html(index=False, classes="tbl", border=0, float_format=lambda v: f"{v:,.3f}")


HTML_TEMPLATE = """<!doctype html><html><head><meta charset="utf-8">
<title>PulseML Data Quality Report</title>
<style>
body{{font-family:Segoe UI,Helvetica,Arial,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#1b2733;background:#fff}}
h1{{color:#1f5fa8;margin-bottom:4px}} h2{{color:#1f5fa8;border-bottom:2px solid #e3ecf7;padding-bottom:4px;margin-top:32px}}
.banner{{background:#fdecea;border:1px solid #d93025;color:#a50e0e;padding:10px 14px;border-radius:6px;font-weight:600}}
.kpis{{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0}} .kpi{{background:#f3f7fc;border-radius:8px;padding:10px 16px;min-width:150px}}
.kpi b{{display:block;font-size:22px;color:#1f5fa8}} .ok{{color:#137333}} .bad{{color:#c5221f}}
table.tbl{{border-collapse:collapse;font-size:13px;margin:8px 0}} .tbl th,.tbl td{{padding:4px 10px;border-bottom:1px solid #e6e9ee;text-align:right}}
.tbl th{{background:#f3f7fc}} img{{max-width:100%}} .scroll{{max-height:420px;overflow:auto}} code{{background:#f3f7fc;padding:1px 4px}}
</style></head><body>
<h1>PulseML: Data Quality Report</h1>
<p>Generated {created} · MIT-BIH Arrhythmia Database · AAMI EC57 classes</p>
<div class="banner">⚠️ Research demonstration only. Not for clinical use.</div>
<div class="kpis">{kpis}</div>
<h2>1. Pipeline lineage</h2>{lineage}
<h2>2. Class distribution</h2><img src="data:image/png;base64,{class_png}">{class_tbl}
<h2>3. Quality checks</h2>{checks}
<h2>4. Missing values</h2>{missing}
<h2>5. Heart rate by class (DuckDB)</h2>{hr}
<h2>6. Per-record beat counts (DuckDB pivot)</h2><div class="scroll">{per_record}</div>
<h2>7. Record-level aggregates (DuckDB)</h2><div class="scroll">{rec_agg}</div>
<h2>8. Feature correlation</h2><img src="data:image/png;base64,{corr_png}">
</body></html>"""


def checks_table(checks, leak):
    rows = []
    for split in ("DS1", "DS2"):
        c = checks[split]
        rows += [
            (split, f"Amplitude outside ±{config.AMPLITUDE_LIMIT_MV} mV", c["amplitude"]["n_flagged"], "warn"),
            (split, "Flat / all-zero beats", c["flat"]["n_flat"], "fail"),
            (split, "RR interval > 3 s (annotation gap / pause)", c["rr_outliers"]["n_flagged"], "warn"),
            (split, f"Waveform length != {config.WINDOW_SIZE}", c["waveform_length"]["n_wrong_length"], "fail"),
            (split, "Duplicate (record, beat) keys", c["duplicates"]["duplicate_keys"], "fail"),
            (split, "Duplicate waveforms", c["duplicates"]["duplicate_waveforms"], "warn"),
            (split, "Invalid labels", c["labels"]["n_invalid_labels"], "fail"),
        ]
    rows.append(("DS1∩DS2", "Records in both splits (patient leakage)", len(leak), "fail"))
    out = "<table class='tbl'><tr><th>Split</th><th>Check</th><th>Count</th><th>Status</th></tr>"
    for split, name, n, severity in rows:
        status = severity if n else "ok"
        out += (
            f"<tr><td>{split}</td><td style='text-align:left'>{html.escape(name)}</td><td>{n:,}</td>"
            f"<td class='{'ok' if status == 'ok' else 'bad'}'>{status.upper()}</td></tr>"
        )
    out += "</table>"

    def flags(key):
        merged = checks["DS1"][key]["flagged_records"] | checks["DS2"][key]["flagged_records"]
        return ", ".join(f"<code>{k}</code>: {v}" for k, v in sorted(merged.items()))

    if amp := flags("amplitude"):
        out += f"<p>Amplitude flags by record: {amp} (flagged, not removed; typically motion artefact or high-voltage QRS).</p>"
    if rr := flags("rr_outliers"):
        longest = max(checks[s]["rr_outliers"]["max_rr_ms"] for s in ("DS1", "DS2")) / 1000
        out += (
            f"<p>RR &gt; 3 s by record: {rr} (longest {longest:.1f} s). These are mostly annotation gaps over "
            "unreadable segments; the affected beats are kept but their RR features (and the patient's local mean RR) "
            "are distorted.</p>"
        )
    return out


def build_report():
    ensure_dirs()
    if not config.DS1_PARQUET.exists():
        raise FileNotFoundError("Feature store missing, run `python -m src.preprocess` first.")
    ds1, ds2 = pd.read_parquet(config.DS1_PARQUET), pd.read_parquet(config.DS2_PARQUET)
    stats = pd.read_parquet(config.STATS_PARQUET)

    checks = {"DS1": run_checks(ds1), "DS2": run_checks(ds2)}
    leak = sorted(set(ds1["record_id"]) & set(ds2["record_id"]))
    checks["patient_leakage"] = {"overlapping_records": leak, "passed": not leak}
    passed = checks["DS1"]["passed"] and checks["DS2"]["passed"] and not leak
    for split in ("DS1", "DS2"):
        c = checks[split]
        log.info(
            "%s: %d beats, %d amplitude flags, %d flat, %d dup keys, passed=%s",
            split, c["n_beats"], c["amplitude"]["n_flagged"], c["flat"]["n_flat"], c["duplicates"]["duplicate_keys"], c["passed"],
        )  # fmt: skip
    log.info("Patient leakage: %s", leak or "none")

    sql = duckdb_analytics()
    for name, df in sql.items():
        print(f"\n=== {name} ===\n{df.to_string(index=False, max_rows=60)}")

    lineage_cols = ["total_annotations", "non_beat_annotations", "beat_annotations", "beats_extracted"]
    lineage_cols += ["excluded_class", "excluded_no_rr", "excluded_boundary"]
    lineage = stats.groupby("split")[lineage_cols].sum().reset_index()
    missing = pd.DataFrame(
        {
            "column": list(checks["DS1"]["missing"]),
            "DS1_missing": list(checks["DS1"]["missing"].values()),
            "DS2_missing": [checks["DS2"]["missing"].get(k, 0) for k in checks["DS1"]["missing"]],
        }
    )
    n_records = ds1["record_id"].nunique() + ds2["record_id"].nunique()
    kpis = [
        ("Records processed", n_records),
        ("Beats extracted", f"{len(ds1) + len(ds2):,}"),
        ("DS1 / DS2 beats", f"{len(ds1):,} / {len(ds2):,}"),
        ("Patient leakage", ", ".join(leak) or "none"),
        ("Overall status", "PASS" if passed else "FAIL"),
    ]
    page = HTML_TEMPLATE.format(
        created=utc_now(),
        kpis="".join(f"<div class='kpi'>{k}<b>{v}</b></div>" for k, v in kpis),
        lineage=_table(lineage),
        class_png=class_bar(ds1, ds2),
        class_tbl=_table(sql["class_distribution"]),
        checks=checks_table(checks, leak),
        missing=_table(missing),
        hr=_table(sql["hr_by_class"]),
        per_record=_table(sql["class_by_record"]),
        rec_agg=_table(sql["record_aggregates"]),
        corr_png=corr_heatmap(ds1),
    )
    (config.REPORTS_DIR / "data_quality.html").write_text(page, encoding="utf-8")

    summary = {
        "created_at": utc_now(),
        "total_records": n_records,
        "total_beats": len(ds1) + len(ds2),
        "checks": checks,
        "class_distribution": sql["class_distribution"].to_dict(orient="records"),
        "lineage": lineage.to_dict(orient="records"),
        "hr_by_class": sql["hr_by_class"].to_dict(orient="records"),
        "passed": bool(passed),
    }
    save_json(summary, config.REPORTS_DIR / "data_quality.json")
    log.info("Wrote data_quality.html/.json (passed=%s)", passed)
    return summary


if __name__ == "__main__":
    build_report()
