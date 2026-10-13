r"""Builds docs/presentation/PulseML_Walkthrough.pdf, a slide-style guide to the whole project.

    pip install reportlab pygments
    python scripts/walkthrough_figures.py
    python scripts/build_walkthrough.py

Code snippets are read from the repo at build time, so they always match the current code.
Uses the Segoe UI and Consolas fonts from C:\Windows\Fonts.
"""

import ast
import json
import sys
from pathlib import Path

from pygments.lexers import PythonLexer
from pygments.token import Comment, Keyword, Name, Number, Operator, String, Token
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402

FIGS = ROOT / "build" / "walkthrough_figures"
OUT = ROOT / "docs" / "presentation" / "PulseML_Walkthrough.pdf"
W, H = 960, 540

FONTS = Path(r"C:\Windows\Fonts")
pdfmetrics.registerFont(TTFont("Sans", str(FONTS / "segoeui.ttf")))
pdfmetrics.registerFont(TTFont("SansB", str(FONTS / "segoeuib.ttf")))
pdfmetrics.registerFont(TTFont("SansL", str(FONTS / "segoeuisl.ttf")))
pdfmetrics.registerFont(TTFont("Mono", str(FONTS / "consola.ttf")))
pdfmetrics.registerFont(TTFont("MonoB", str(FONTS / "consolab.ttf")))
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="SansB", italic="Sans", boldItalic="SansB")

# pastel palette (validated categorical: blue, coral, lavender, sage) + tints for cards
C = {k: HexColor(v) for k, v in {
    "bg": "#FBF9F6", "slate": "#94A3B8", "ink": "#2E3440", "muted": "#6B7280", "line": "#E7E5E0", "white": "#FFFFFF",
    "blue": "#5B95E0", "coral": "#F0957A", "lav": "#A684D6", "sage": "#5DBB86", "soft": "#CBD5E1",
    "blueT": "#EAF2FC", "coralT": "#FDEEE8", "lavT": "#F2EDFA", "sageT": "#E8F5ED", "butterT": "#FBF4DF", "butter": "#E2B33F",
    "code": "#F5F6FA", "codeLn": "#B4BAC6",
}.items()}  # fmt: skip
TINT = {"blue": "blueT", "coral": "coralT", "lav": "lavT", "sage": "sageT", "butter": "butterT"}
CODE_COLORS = {
    Keyword: "#8A5CC7", Name.Builtin: "#8A5CC7", Name.Function: "#2F6FC0", Name.Class: "#2F6FC0",
    String: "#3D8A5F", Number: "#C2603E", Comment: "#9AA3B2", Operator: "#6B7280", Name.Decorator: "#C2603E",
}  # fmt: skip

c = canvas.Canvas(str(OUT), pagesize=(W, H))
c.setTitle("PulseML: end-to-end walkthrough")
c.setAuthor("PulseML")
page_no = 0


def style(size=12, color="ink", font="Sans", leading=None):
    return ParagraphStyle(
        "s", fontName=font, fontSize=size, leading=leading or size * 1.38, textColor=C[color], alignment=TA_LEFT
    )


def para(x, y_top, w, text, size=12, color="ink", font="Sans", leading=None):
    """Draw wrapped rich text with its top at y_top. Returns the height used."""
    p = Paragraph(text, style(size, color, font, leading))
    _, h = p.wrap(w, 1000)
    p.drawOn(c, x, y_top - h)
    return h


def rrect(x, y, w, h, fill, r=10, stroke=None):
    c.setFillColor(C[fill])
    if stroke:
        c.setStrokeColor(C[stroke])
        c.setLineWidth(1)
    c.roundRect(x, y, w, h, r, stroke=1 if stroke else 0, fill=1)


def new_slide(title, section=None, subtitle=None, accent="blue"):
    global page_no
    if page_no:
        c.showPage()
    page_no += 1
    c.setFillColor(C["bg"])
    c.rect(0, 0, W, H, stroke=0, fill=1)
    if section:
        c.setFont("SansB", 9)
        tw = pdfmetrics.stringWidth(section.upper(), "SansB", 9) + 18
        rrect(48, H - 52, tw, 18, TINT[accent], r=9)
        c.setFillColor(C[accent])
        c.drawString(57, H - 46, section.upper())
    c.setFillColor(C["ink"])
    c.setFont("SansB", 25)
    c.drawString(48, H - 88, title)
    if subtitle:
        para(48, H - 98, 820, subtitle, 12.5, "muted")
    c.setFont("Sans", 8.5)
    c.setFillColor(C["muted"])
    c.drawString(48, 20, "PulseML · end-to-end walkthrough · research demonstration only, not for clinical use")
    c.drawRightString(W - 48, 20, str(page_no))


def image(name, x, y_top, w=None, h=None):
    img = ImageReader(str(FIGS / f"{name}.png"))
    iw, ih = img.getSize()
    if w and not h:
        h = w * ih / iw
    elif h and not w:
        w = h * iw / ih
    c.drawImage(img, x, y_top - h, w, h, mask="auto")
    return w, h


def card(x, y_top, w, h, color, title, body, size=11):
    rrect(x, y_top - h, w, h, TINT[color], r=12)
    c.setFillColor(C[color])
    c.circle(x + 18, y_top - 22, 4, stroke=0, fill=1)
    c.setFillColor(C["ink"])
    c.setFont("SansB", 13)
    c.drawString(x + 30, y_top - 27, title)
    para(x + 16, y_top - 40, w - 32, body, size)


def bullets(x, y_top, w, items, size=12, color="blue", gap=7):
    y = y_top
    for it in items:
        c.setFillColor(C[color])
        c.circle(x + 4, y - size * 0.62, 2.6, stroke=0, fill=1)
        y -= para(x + 16, y, w - 16, it, size) + gap
    return y_top - y


def snippet(path, names=None, start=None, end=None):
    """Source text of top-level functions/classes in a file (kept in sync with the repo), or a line range."""
    src = (ROOT / path).read_text(encoding="utf-8")
    lines = src.splitlines()
    if names is None:
        return "\n".join(lines[start - 1 : end])
    parts = []
    for node in ast.parse(src).body:
        if getattr(node, "name", None) in names:
            first = node.lineno - 1 - len(getattr(node, "decorator_list", []))
            parts.append("\n".join(lines[first : node.end_lineno]))
    return "\n\n".join(parts)


def token_color(tt):
    while tt is not Token:
        if tt in CODE_COLORS:
            return CODE_COLORS[tt]
        tt = tt.parent
    return "#2E3440"


def code(x, y_top, w, src, size=8.6, title=None, max_lines=None):
    """Syntax-coloured code block with line numbers. Long lines wrap with an indent."""
    lines = src.split("\n")[:max_lines]
    char_w = pdfmetrics.stringWidth("M", "Mono", size)
    gutter = char_w * 3.2 + 10
    cols = int((w - gutter - 16) / char_w)
    rows = []  # (line number or None, [(text, color), ...])
    toks_by_line = [[] for _ in lines]
    li = 0
    for tt, val in PythonLexer(stripnl=False, ensurenl=False).get_tokens("\n".join(lines)):
        for k, piece in enumerate(val.split("\n")):
            if k:
                li += 1
            if piece and li < len(lines):
                toks_by_line[li].append((piece, token_color(tt)))
    for n, toks in enumerate(toks_by_line, 1):
        cur, used, first = [], 0, True
        for text, col in toks:
            while text:
                room = cols - used
                if len(text) <= room:
                    cur.append((text, col))
                    used += len(text)
                    text = ""
                else:
                    cur.append((text[:room], col))
                    rows.append((n if first else None, cur))
                    first, cur, used, text = False, [("      ", "#2E3440")], 6, text[room:]
        rows.append((n if first else None, cur))
    lead = size * (1.34 if size < 8 else 1.42)
    top_pad = 22 if title else 10
    h = len(rows) * lead + top_pad + 8
    rrect(x, y_top - h, w, h, "code", r=10)
    if title:
        c.setFont("SansB", 8.5)
        c.setFillColor(C["muted"])
        c.drawString(x + 12, y_top - 15, title)
    y = y_top - top_pad - size
    for n, toks in rows:
        if n is not None:
            c.setFont("Mono", size)
            c.setFillColor(C["codeLn"])
            c.drawRightString(x + gutter - 4, y, str(n))
        cx = x + gutter + 4
        for text, col in toks:
            c.setFillColor(HexColor(col))
            c.setFont("Mono", size)
            c.drawString(cx, y, text)
            cx += pdfmetrics.stringWidth(text, "Mono", size)
        y -= lead
    return h


def notes(x, y_top, w, items, size=10.6, color="blue"):
    """Numbered explanations keyed to code line numbers: [(\"3\", \"text\"), ...]."""
    y = y_top
    badge = max(max(pdfmetrics.stringWidth(r, "MonoB", 8) + 10, 20) for r, _ in items)
    for ref, text in items:
        c.setFont("MonoB", 8)
        rrect(x, y - 15, badge, 15, TINT[color], r=7)
        c.setFillColor(C[color])
        c.drawCentredString(x + badge / 2, y - 11, ref)
        y -= para(x + badge + 10, y, w - badge - 10, text, size) + 7
    return y_top - y


def shell(x, y_top, w, lines, size=9.5, title="terminal"):
    lead = size * 1.55
    h = len(lines) * lead + 34
    rrect(x, y_top - h, w, h, "ink", r=10)
    c.setFont("SansB", 8.5)
    c.setFillColor(HexColor("#9AA3B2"))
    c.drawString(x + 12, y_top - 15, title)
    y = y_top - 22 - size
    for ln in lines:
        if ln.startswith("#"):
            c.setFillColor(HexColor("#8C96A8"))
        else:
            c.setFillColor(HexColor("#E8ECF3"))
        c.setFont("Mono", size)
        c.drawString(x + 14, y, ln)
        y -= lead
    return h


def box(x, y, w, h, color, title, sub=None, title_size=11.5):
    rrect(x, y, w, h, TINT[color], r=10, stroke=None)
    c.setFillColor(C[color])
    c.rect(x, y + 8, 3.5, h - 16, stroke=0, fill=1)
    c.setFillColor(C["ink"])
    c.setFont("SansB", title_size)
    c.drawCentredString(x + w / 2, y + h / 2 + (4 if sub else -4), title)
    if sub:
        c.setFont("Sans", 8.8)
        c.setFillColor(C["muted"])
        c.drawCentredString(x + w / 2, y + h / 2 - 11, sub)


def arrow(x1, y1, x2, y2, color="muted"):
    import math

    c.setStrokeColor(C[color])
    c.setFillColor(C[color])
    c.setLineWidth(1.4)
    c.line(x1, y1, x2, y2)
    a = math.atan2(y2 - y1, x2 - x1)
    s = 6
    p = c.beginPath()
    p.moveTo(x2, y2)
    p.lineTo(x2 - s * math.cos(a - 0.4), y2 - s * math.sin(a - 0.4))
    p.lineTo(x2 - s * math.cos(a + 0.4), y2 - s * math.sin(a + 0.4))
    p.close()
    c.drawPath(p, stroke=0, fill=1)


def stat(x, y_top, w, value, label, color="blue"):
    rrect(x, y_top - 74, w, 74, TINT[color], r=12)
    c.setFillColor(C["ink"])
    c.setFont("SansB", 26)
    c.drawString(x + 16, y_top - 38, value)
    para(x + 16, y_top - 46, w - 28, label, 9.5, "muted")


def table(x, y_top, col_w, rows, size=10.5, header_color="blue", row_h=None, highlight=None):
    """rows[0] is the header. highlight = index of a row to tint."""
    y = y_top
    for i, row in enumerate(rows):
        hs = [Paragraph(str(v), style(size, "ink", "SansB" if i == 0 else "Sans")) for v in row]
        rh = max(p.wrap(w - 14, 500)[1] for p, w in zip(hs, col_w)) + 12
        rh = max(rh, row_h or 0)
        if i == 0:
            rrect(x, y - rh, sum(col_w), rh, TINT[header_color], r=6)
        elif highlight == i:
            rrect(x, y - rh, sum(col_w), rh, "blueT", r=6)
        else:
            c.setStrokeColor(C["line"])
            c.setLineWidth(0.8)
            c.line(x, y - rh, x + sum(col_w), y - rh)
        cx = x
        for p, w in zip(hs, col_w):
            p.drawOn(c, cx + 7, y - 6 - p.wrap(w - 14, 500)[1])
            cx += w
        y -= rh
    return y_top - y


ev = json.loads(config.EVAL_METRICS.read_text())
comp = {r["model"]: r for r in ev["comparison"]}
calrep = json.loads((config.REPORTS_DIR / "calibration_report.json").read_text())
tmeta = json.loads(config.TRAINING_METADATA.read_text())
hyb = comp["Hybrid"]

# ---------------------------------------------------------------- 1 title
page_no += 1
c.setFillColor(C["bg"])
c.rect(0, 0, W, H, stroke=0, fill=1)
for i, (col, r) in enumerate((("blueT", 210), ("coralT", 150), ("lavT", 110), ("sageT", 70))):
    c.setFillColor(C[col])
    c.circle(800 - i * 18, 330 + i * 10, r, stroke=0, fill=1)
rrect(548, 240, 384, 104, "white", r=12)
image("class_shapes", 560, 334, w=360)
c.setFillColor(C["blue"])
c.setFont("SansB", 11)
c.drawString(64, 400, "END-TO-END PROJECT WALKTHROUGH")
c.setFillColor(C["ink"])
c.setFont("SansB", 54)
c.drawString(60, 330, "PulseML")
para(
    64,
    312,
    470,
    "Teaching a computer to read heartbeats: data, code, models and results, explained step by step.",
    17,
    "muted",
)
bullets(64, 220, 440, [
    "Learn what the dataset is and why it is hard",
    "Read the real code, line by line",
    "Run every step yourself",
    "Understand why each model was chosen, and what the results mean",
], size=12.5)  # fmt: skip
c.setFont("Sans", 9)
c.setFillColor(C["muted"])
c.drawString(64, 40, "Research demonstration only. Not a medical device and not for clinical use.")

# ---------------------------------------------------------------- 2 agenda
new_slide("What's inside", "roadmap")
sections = [
    ("blue", "1  The problem", "What an ECG is, and the 4 kinds of heartbeat"),
    ("coral", "2  The dataset", "MIT-BIH: where the data comes from and its traps"),
    ("lav", "3  Run it yourself", "Setup and the 7 commands of the pipeline"),
    ("sage", "4  Data pipeline code", "Filtering, cutting beats, features, checks"),
    ("blue", "5  The models", "Why 4 models, and how each one works"),
    ("coral", "6  Training", "Class imbalance, augmentation, early stopping"),
    ("lav", "7  Results", "What worked, what didn't, per class and per patient"),
    ("sage", "8  Trust & serving", "Calibration, prediction sets, API, dashboard"),
]
for i, (col, t, s) in enumerate(sections):
    x = 48 + (i % 4) * 219
    y = 400 - (i // 4) * 150
    rrect(x, y - 120, 205, 120, TINT[col], r=14)
    c.setFillColor(C[col])
    c.circle(x + 22, y - 24, 6, stroke=0, fill=1)
    c.setFillColor(C["ink"])
    c.setFont("SansB", 14)
    c.drawString(x + 18, y - 56, t)
    para(x + 18, y - 66, 175, s, 10.5, "muted")

# ---------------------------------------------------------------- 3 problem
new_slide("The problem: classify every heartbeat", "1 · the problem",
          "An ECG records the heart's electricity. Each beat makes a spike (the R-peak). The gap between spikes is the RR interval.")  # fmt: skip
image("ecg_strip", 48, 420, w=560)
card(630, 420, 282, 165, "blue", "The task",
     "Given one beat (0.78 s of signal) plus its timing, say which of 4 beat types it is. "
     "A doctor reviewing a 24-hour recording sees ~100,000 beats, so automatic sorting helps them find the rare abnormal ones.")  # fmt: skip
card(630, 240, 282, 120, "coral", "Why it is hard",
     "~90% of beats are normal, abnormal beats look different in every person, and we must work on <b>people the model has never seen</b>.")  # fmt: skip
para(48, 250, 560,
     "<b>Two kinds of clue</b> tell beats apart:<br/>"
     "• <b>Shape</b> (morphology): is the spike narrow and tall, or wide and odd?<br/>"
     "• <b>Timing</b> (rhythm): did the beat arrive early compared with this person's usual rhythm?<br/>"
     "The whole project is about combining these two clues well.", 12.5)  # fmt: skip

# ---------------------------------------------------------------- 4 classes
new_slide("The 4 beat classes (AAMI EC57 standard)", "1 · the problem",
          "Typical shape of each class (median of real beats, shaded band = middle 50%). All beats are centred on the R-peak.")  # fmt: skip
image("class_shapes", 48, 405, w=864)
rows = [
    ["Class", "Meaning", "Where it starts", "Main clue", "Share of beats"],
    ["N", "Normal (incl. bundle branch block)", "Normal pathway", "Baseline", "~89%"],
    ["S", "Supraventricular ectopic", "Upper chambers, too early", "Arrives early, normal shape", "~3%"],
    ["V", "Ventricular ectopic", "Lower chambers", "Wide, odd shape, early", "~7%"],
    ["F", "Fusion of V and normal", "Both at once", "Mix of N and V shapes", "~0.8%"],
]
table(48, 190, [70, 250, 200, 220, 124], rows, size=10.5)

# ---------------------------------------------------------------- 5 dataset facts
new_slide("The dataset: MIT-BIH Arrhythmia Database", "2 · the dataset", accent="coral",
          subtitle="The standard public benchmark for heartbeat classification, freely available from PhysioNet.")  # fmt: skip
stat(48, 420, 200, "48", "half-hour recordings from 47 people (Boston, 1975–79)", "coral")
stat(260, 420, 200, "360 Hz", "samples per second, 2 leads (we use lead MLII)", "blue")
stat(472, 420, 200, "100,630", "labelled beats we extract and use", "lav")
stat(684, 420, 228, "≥ 2", "cardiologists annotated every single beat", "sage")
para(48, 320, 420, "<b>What a record contains</b>", 13)
bullets(48, 298, 420, [
    "<b>.dat</b>: the raw voltage signal",
    "<b>.hea</b>: header (sampling rate, lead names, gain)",
    "<b>.atr</b>: the experts' annotations: the position of every R-peak and a symbol for its beat type (N, A, V, F …)",
], size=11.5)  # fmt: skip
para(500, 320, 412, "<b>What we remove, and why</b>", 13)
bullets(500, 298, 412, [
    "<b>4 paced records</b> (102, 104, 107, 217): a pacemaker shapes those beats; the AAMI standard excludes them → 44 records",
    "<b>Unclassifiable beats</b> (Q) and non-beat notes (rhythm changes, noise marks)",
    "First/last beat of each record (no neighbour, so no RR interval)",
], size=11.5, color="coral")  # fmt: skip
rrect(48, 48, 864, 50, "butterT", r=10)
para(64, 88, 840, "<b>Lineage:</b> 103,724 annotations → 100,733 beat annotations → <b>100,630 beats</b> "
     "(15 excluded as paced/unclassifiable, 88 first/last beats). Every count is logged and checked in the data-quality report.", 11)  # fmt: skip

# ---------------------------------------------------------------- 6 imbalance + split
new_slide("Trap #1: extreme class imbalance", "2 · the dataset", accent="coral",
          subtitle="Train set (DS1) and test set (DS2) beat counts. Note the log scale: F is ~100× rarer than N.")  # fmt: skip
image("class_counts", 48, 420, w=500)
card(580, 420, 332, 115, "coral", "The accuracy trap",
     "A lazy model that always answers \"N\" scores <b>89% accuracy</b> on DS2 while finding zero abnormal beats. "
     "So accuracy is never reported alone.")  # fmt: skip
card(580, 290, 332, 118, "blue", "What we do about it",
     "• Score models with <b>macro-F1</b> (every class counts equally)<br/>"
     "• Weight the loss so mistakes on rare classes cost more<br/>"
     "• Report recall and precision for each class separately")  # fmt: skip

# ---------------------------------------------------------------- 7 patient split
new_slide("Trap #2: the same person must never be in train and test", "2 · the dataset", accent="coral",
          subtitle="Beats from one person look alike. If they are shuffled into both sets, the model memorises people instead of learning beats.")  # fmt: skip
# left: random split (wrong)
rrect(48, 120, 410, 290, "coralT", r=14)
c.setFillColor(C["ink"])
c.setFont("SansB", 14)
c.drawString(68, 380, "Random beat split  (the common mistake)")
for i in range(36):
    col = ["blue", "coral", "lav", "sage"][i % 4]
    c.setFillColor(C[col])
    x, y = 78 + (i % 12) * 30, 330 - (i // 12) * 30
    c.circle(x, y, 8, stroke=0, fill=1)
c.setStrokeColor(C["ink"])
c.setDash(4, 3)
c.line(78 + 8.5 * 30, 350, 78 + 8.5 * 30, 250)
c.setDash()
para(68, 230, 370, "Colours = patients. Every patient appears on both sides of the line, so the test is easy. "
     "Papers that do this report <b>&gt;99% accuracy</b> that collapses on new people.", 11)  # fmt: skip
# right: inter-patient split (ours)
rrect(502, 120, 410, 290, "sageT", r=14)
c.setFillColor(C["ink"])
c.setFont("SansB", 14)
c.drawString(522, 380, "Inter-patient split  (what we do)")
for i in range(36):
    col = ["blue", "lav", "sage", "coral", "butter", "slate"][(i % 12) // 2]
    c.setFillColor(C[col])
    x, y = 532 + (i % 12) * 30, 330 - (i // 12) * 30
    c.circle(x, y, 8, stroke=0, fill=1)
c.setStrokeColor(C["ink"])
c.setDash(4, 3)
c.line(532 + 5.5 * 30, 350, 532 + 5.5 * 30, 250)
c.setDash()
c.setFont("Sans", 9.5)
c.setFillColor(C["muted"])
c.drawString(532, 252, "DS1: 22 people (train)")
c.drawString(712, 252, "DS2: 22 other people (test)")
para(522, 226, 370, "de Chazal et al. (2004) split: <b>no person is in both sets</b>. Scores are lower but honest. "
     "The code asserts this, and the validation step checks it again.", 11)  # fmt: skip
para(
    48,
    100,
    864,
    "The same rule applies inside training: validation and cross-validation also hold out <b>whole patients</b>, never single beats.",
    11.5,
    "muted",
)

# ---------------------------------------------------------------- 8 pipeline
new_slide("The pipeline at a glance", "3 · run it yourself", accent="lav",
          subtitle="Each box is one Python module in src/ that you run with python -m src.&lt;name&gt;. Each step writes files the next step reads.")  # fmt: skip
steps = [
    ("blue", "download", "fetch 48 records"), ("coral", "preprocess", "filter, cut, features"),
    ("lav", "validate", "quality checks"), ("sage", "train", "4 models on DS1"),
    ("blue", "evaluate", "score on DS2"), ("coral", "calibrate", "honest probabilities"),
]  # fmt: skip
bw, gap = 128, 19
for i, (col, t, s) in enumerate(steps):
    x = 48 + i * (bw + gap)
    box(x, 330, bw, 62, col, t, s)
    if i:
        arrow(x - gap + 2, 361, x - 2, 361)
outs = [
    "data/raw/*.dat",
    "beats_ds1/2.parquet",
    "data_quality.html",
    "models/*.pt, *.joblib",
    "metrics, figures",
    "calibration.json",
]
c.setFont("Mono", 8.2)
c.setFillColor(C["muted"])
for i, o in enumerate(outs):
    c.drawCentredString(48 + i * (bw + gap) + bw / 2, 314, o)
# serving row
box(250, 170, 150, 62, "lav", "app/api.py", "FastAPI /predict")
box(430, 170, 150, 62, "sage", "app/dashboard.py", "Streamlit app")
box(610, 170, 150, 62, "blue", "Docker", "both, one command")
arrow(48 + 3.5 * (bw + gap), 302, 340, 236)
arrow(48 + 5.5 * (bw + gap), 302, 520, 236)
para(48, 120, 864, "Optional research steps: <b>experiments</b> (ablation with 3 seeds) and <b>experiments --cv-select</b> "
     "(5-fold patient-grouped cross-validation that picked the final model).", 11.5, "muted")  # fmt: skip

# ---------------------------------------------------------------- 9 setup + commands
new_slide("Run it yourself: setup and commands", "3 · run it yourself", accent="lav")
shell(48, 440, 420, [
    "# 1. get the code and a virtual environment",
    "git clone <repo-url> pulseml && cd pulseml",
    "python -m venv .venv",
    ".venv\\Scripts\\activate        # Windows",
    "pip install -r requirements.txt",
    "",
    "# 2. (optional) GPU build of PyTorch",
    "pip install torch --index-url \\",
    "  https://download.pytorch.org/whl/cu126",
], title="setup")  # fmt: skip
shell(492, 440, 420, [
    "python -m src.download      # ~100 MB, once",
    "python -m src.preprocess    # ~1 min",
    "python -m src.validate      # quality report",
    "python -m src.train         # ~2 min on GPU",
    "python -m src.evaluate      # DS2 scores + figures",
    "python -m src.calibrate     # ~10 min on GPU",
    "pytest tests/ -q            # 82 tests",
], title="the pipeline, in order")  # fmt: skip
shell(48, 236, 420, [
    "uvicorn app.api:app --reload    # localhost:8000/docs",
    "streamlit run app/dashboard.py  # localhost:8501",
    "",
    "docker compose up --build       # both, in containers",
], title="run the apps")  # fmt: skip
card(492, 236, 420, 118, "sage", "Good to know",
     "Trained models and results are already in <font face='Mono'>outputs/</font>, so the API and dashboard "
     "work straight after cloning; you only need the pipeline to retrain. Training uses the GPU automatically if one is available.", 11)  # fmt: skip

# ---------------------------------------------------------------- 10 repo map
new_slide("Where everything lives", "3 · run it yourself", accent="lav")
tree = [
    ("src/config.py", "every constant: paths, record lists, filter settings, hyperparameters"),
    ("src/download.py", "fetches MIT-BIH from PhysioNet (skips files you already have)"),
    ("src/preprocess.py", "filter → find beats → label → cut windows → features → Parquet"),
    ("src/validate.py", "data-quality checks, DuckDB SQL summaries, HTML report"),
    ("src/dataset.py", "turns Parquet rows into PyTorch tensors; patient templates"),
    ("src/models.py", "LogReg, XGBoost, ECGCNN, HybridCNN, SeedEnsemble, calibration maps"),
    ("src/train.py", "patient-level splits, grid search, the training loop"),
    ("src/experiments.py", "ablation study and cross-validated model selection"),
    ("src/evaluate.py", "DS2 metrics, figures, model card"),
    ("src/calibrate.py", "calibrated probabilities + conformal prediction sets"),
    ("app/", "the web API (api.py) and the Streamlit dashboard (dashboard.py)"),
    ("scripts/", "builds this walkthrough PDF"),
    ("tests/", "82 automatic checks (run with pytest)"),
]
y = 440
for i, (f, d) in enumerate(tree):
    col = ["blue", "coral", "lav", "sage"][i % 4]
    c.setFillColor(C[col])
    c.circle(58, y - 6, 3.5, stroke=0, fill=1)
    c.setFont("MonoB", 11)
    c.setFillColor(C["ink"])
    c.drawString(72, y - 10, f)
    c.setFont("Sans", 11.5)
    c.setFillColor(C["muted"])
    c.drawString(300, y - 10, d)
    y -= 30

# ---------------------------------------------------------------- 11 filtering code
new_slide("Step 1 · Clean the signal (band-pass filter)", "4 · data pipeline", accent="sage",
          subtitle="Breathing makes the line drift slowly; muscles and mains power add fast fuzz. We keep only 0.5–40 Hz.")  # fmt: skip
h = code(48, 405, 470, snippet("src/preprocess.py", ["bandpass_filter"]), size=8.8, title="src/preprocess.py")
notes(48, 405 - h - 14, 470, [
    ("1", "<b>x</b> is the raw voltage. The defaults come from config: 360 Hz sampling, keep 0.5–40 Hz, filter order 4."),
    ("2", "Design a <b>Butterworth</b> band-pass filter (smooth, no ripples). <b>sos</b> = second-order sections, the numerically stable form."),
    ("3", "<b>sosfiltfilt</b> runs the filter forward and then backward, which cancels the time delay (zero phase). The R-peak stays on exactly the same sample."),
], color="sage")  # fmt: skip
image("filtering", 540, 410, w=372)

# ---------------------------------------------------------------- 12 windows + zscore
new_slide("Step 2 · Cut one window per beat and normalise it", "4 · data pipeline", accent="sage",
          subtitle="Each beat becomes 280 numbers: 140 samples before the R-peak and 140 after (0.78 s), rescaled to mean 0 and spread 1.")  # fmt: skip
code(48, 405, 470, snippet("src/preprocess.py", ["extract_window", "zscore"]), size=9.2, title="src/preprocess.py")
notes(540, 405, 372, [
    ("1–2", "Window = from 140 samples before the R-peak to 140 after it."),
    ("3", "If the window would run past the start or end of the recording, return <b>None</b> and skip the beat. We never pad with made-up signal."),
    ("5–6", "<b>zscore</b>: convert to floating point."),
    ("7", "Subtract the mean and divide by the standard deviation. Every beat then has mean 0 and spread 1, which removes gain differences between people and devices. The tiny <b>eps</b> stops a flat line from dividing by zero."),
], color="sage")  # fmt: skip
rrect(48, 118, 470, 82, "sageT", r=12)
para(64, 188, 440, "<b>Why normalise?</b> One patient's electrodes may give 2 mV spikes, another's 0.8 mV. "
     "Without z-scoring the model would learn \"big = person A\" instead of learning beat shapes.", 11.5)  # fmt: skip

# ---------------------------------------------------------------- 13 process_signal
new_slide("Step 3 · Turn one recording into labelled beats", "4 · data pipeline", accent="sage")
src = snippet("src/preprocess.py", ["process_signal"]).split("\n")
cut = next(i for i, ln in enumerate(src) if ln.strip().startswith("excluded ="))
code(48, 430, 520, "\n".join(src[:cut]).rstrip(), size=7.2, title="src/preprocess.py · process_signal (first part)")
notes(590, 430, 322, [
    ("4", "Mark which annotations are beats (others are notes such as rhythm changes)."),
    ("5–7", "Keep the R-peak positions and symbols of the beats."),
    ("8", "RR intervals = gaps between neighbouring R-peaks."),
    ("12–15", "Map the symbol to N/S/V/F. Paced or unknown beats are dropped, but still count for timing."),
    ("16–18", "First and last beat have no neighbour, so no RR: drop."),
    ("19–22", "Cut the 280-sample window; skip it if it runs off the edge."),
    ("23", "This patient's usual RR = mean of the <b>previous</b> 32 intervals only (no peeking ahead)."),
    ("24–34", "One row per beat: ids, label, z-scored waveform and 16 features."),
    ("35", "Count beats per class for the lineage report."),
], size=9.8, color="sage")  # fmt: skip

# ---------------------------------------------------------------- 14 rr features
new_slide("Step 4 · Timing features, relative to each patient", "4 · data pipeline", accent="sage",
          subtitle="A heart rate of 90 bpm is normal for one person and fast for another, so timing is compared with the patient's own rhythm.")  # fmt: skip
h = code(48, 405, 440, snippet("src/preprocess.py", ["rr_features"]), size=8.8, title="src/preprocess.py")
notes(48, 405 - h - 14, 440, [
    ("3", "No history yet? Use rr_prev as the patient's usual RR."),
    ("7–8", "Ratio of the two gaps, and heart rate in beats per minute (360 × 60 ÷ RR)."),
    ("10", "<b>rr_prev_norm</b> &lt; 1 means the beat came early <i>for this person</i>: the key clue for S beats."),
    ("11", "<b>rr_next_norm</b> &gt; 1 means a pause after the beat (typical after V)."),
], size=10.2, color="sage")  # fmt: skip
image("rr_norm", 510, 410, w=402)
para(510, 205, 402, "Normal beats arrive on time (1.00). S and V beats arrive ~25% early. "
     "Adding these 3 patient-relative features raised logistic regression from 0.51 to <b>0.59</b> macro-F1 on unseen patients.", 11)  # fmt: skip

# ---------------------------------------------------------------- 15 other features + template
new_slide("Step 5 · Shape features and the patient template", "4 · data pipeline", accent="sage",
          subtitle="Besides timing, 9 numbers describe the shape. The deep model also gets the patient's own 'usual beat' to compare against.")  # fmt: skip
rows = [
    ["Feature", "What it measures"],
    ["wf_mean, wf_std, wf_min, wf_max", "Level, spread and extremes of the (un-normalised) beat in mV"],
    ["wf_skew, wf_kurtosis", "Lopsidedness and peakiness of the shape"],
    ["qrs_proxy", "Tallest point in the middle 40 samples (the QRS spike)"],
    ["pre_energy, post_energy", "Signal energy before the spike (P-wave) and after it (T-wave)"],
]
table(48, 420, [190, 250], rows, size=9.6, header_color="sage")
code(48, 250, 440, snippet("src/dataset.py", ["template_matrix"]), size=7.8, title="src/dataset.py")
image("template", 510, 420, w=402)
notes(510, 285, 402, [
    ("7–9", "For each patient, take their beats in time order."),
    ("11", "The first beat has no history, so it is its own template."),
    ("12–13", "Template = sample-by-sample <b>median of the previous 16 beats</b>. Only the past is used, so this works live."),
], size=10, color="sage")  # fmt: skip
para(510, 150, 402, "The median ignores the odd abnormal beat, so the template is the person's typical normal beat. "
     "Beat − template highlights exactly what is unusual.", 10.5, "muted")  # fmt: skip

# ---------------------------------------------------------------- 16 validate
new_slide("Step 6 · Check the data before training", "4 · data pipeline", accent="sage",
          subtitle="python -m src.validate runs automatic checks and writes outputs/reports/data_quality.html. All checks pass.")  # fmt: skip
checks = [
    ("Missing values", "no NaN in any column or waveform", "0"),
    ("Amplitude", "beats outside ±5 mV (warning only)", "0"),
    ("Flat signal", "lead-off / all-zero beats", "0"),
    ("Window length", "every waveform exactly 280 samples", "0"),
    ("Duplicates", "same (record, beat) key twice", "0"),
    ("Label validity", "labels outside 0–3", "0"),
    ("RR > 3 s", "long gaps: real pauses (record 232) or unreadable stretches", "80 flagged"),
    ("Patient leakage", "a record in both DS1 and DS2", "none"),
]
y = 405
for i, (n, d, v) in enumerate(checks):
    x = 48 + (i % 2) * 438
    yy = y - (i // 2) * 74
    rrect(x, yy - 62, 424, 62, "sageT" if v in ("0", "none") else "butterT", r=10)
    c.setFillColor(C["ink"])
    c.setFont("SansB", 12.5)
    c.drawString(x + 16, yy - 25, n)
    c.setFont("Sans", 10)
    c.setFillColor(C["muted"])
    c.drawString(x + 16, yy - 44, d)
    c.setFont("SansB", 14)
    c.setFillColor(C["sage"] if v in ("0", "none") else C["coral"])
    c.drawRightString(x + 408, yy - 34, v)
para(48, 90, 864, "The checks are plain functions on a DataFrame (check_missing, check_flat, …), so the tests feed them "
     "deliberately broken data (all-zero beats, 12 mV spikes, duplicates) and confirm each one is caught. DuckDB SQL "
     "produces the per-record summary tables in the report.", 11, "muted")  # fmt: skip

# ---------------------------------------------------------------- 17 why these models
new_slide("Why these 4 models?", "5 · the models",
          subtitle="Each model answers a question. Together they show which ingredients actually matter on new patients.")  # fmt: skip
rows = [
    ["Model", "What it looks at", "Why it is here", "Macro-F1 (DS2)"],
    [
        "Logistic regression",
        "16 hand-made features",
        "Simplest sensible baseline: can good features alone do the job? Fast and easy to explain.",
        f"{comp['LogReg']['macro_f1']:.3f}",
    ],
    [
        "XGBoost",
        "Same 16 features",
        'Decision trees catch non-linear combinations ("early AND wide"). Standard strong choice for tables.',
        f"{comp['XGBoost']['macro_f1']:.3f}",
    ],
    [
        "1D CNN (specified)",
        "Raw beat shape only",
        "Learns shape patterns by itself, with no hand-made features. But it cannot see timing.",
        f"{comp['CNN']['macro_f1']:.3f}",
    ],
    [
        "<b>Hybrid-rel ensemble</b>",
        "Shape + difference from the patient's usual beat + patient-relative timing",
        "Combines both clues, each compared with the patient's own baseline. 3 copies vote.",
        f"<b>{hyb['macro_f1']:.3f}</b>",
    ],
]
table(48, 420, [160, 220, 384, 100], rows, size=10.5, highlight=4)
image("macro_f1", 48, 200, h=150)
para(400, 190, 512, "<b>Lesson:</b> the gain came from giving the model the <i>right context</i> (timing and the patient's own normal beat), "
     "not from a bigger model. XGBoost even kept choosing its smallest trees: with only 22 training people, simpler generalises better.", 11.5)  # fmt: skip

# ---------------------------------------------------------------- 18 LogReg & XGBoost
new_slide("Baselines: logistic regression and XGBoost", "5 · the models")
h = code(48, 440, 430, snippet("src/models.py", ["make_logreg", "make_xgb"]), size=8.8, title="src/models.py")
notes(48, 440 - h - 14, 430, [
    ("2", "<b>LogisticRegression</b> learns one weight per feature per class. <b>class_weight=\"balanced\"</b> makes rare classes count more. <b>C</b> controls how simple the model stays."),
    ("3", "<b>StandardScaler</b> first puts every feature on the same scale."),
    ("6–10", "<b>XGBoost</b>: hundreds of small decision trees, each fixing the previous ones' mistakes. Rows and columns are subsampled (0.8) to reduce overfitting."),
], size=10.2)  # fmt: skip
card(500, 440, 412, 150, "blue", "How the settings were chosen",
     "<b>GridSearchCV</b> tries every combination (LogReg C in {0.01, 0.1, 1, 10}; XGBoost depth, trees, learning rate) "
     "with 3-fold cross-validation <b>grouped by patient</b>, scored by macro-F1, on DS1 only.<br/><br/>"
     "Winners: LogReg C = 0.01 (strongest simplicity), XGBoost depth 2, 100 trees, rate 0.05.", 10.8)  # fmt: skip
card(500, 270, 412, 120, "coral", "Strengths and weaknesses",
     f"LogReg: best N/V balance among baselines (macro-F1 {comp['LogReg']['macro_f1']:.3f}).<br/>"
     f"XGBoost: best macro-AUC of the baselines ({comp['XGBoost']['macro_auc']:.3f}) and catches 93% of S beats, "
     "but raises many false alarms on normal beats (N recall 0.76).", 10.8)  # fmt: skip

# ---------------------------------------------------------------- 19 CNN
new_slide("The 1D CNN: learning shapes directly", "5 · the models",
          subtitle="A convolution slides small learned pattern-detectors along the beat; deeper layers combine them into bigger patterns.")  # fmt: skip
h = code(48, 405, 520, snippet("src/models.py", ["conv_block", "conv_trunk"]), size=8.4, title="src/models.py")
notes(48, 405 - h - 14, 520, [
    ("2", "One block = <b>Conv1d</b> (k-sample pattern detectors) → <b>BatchNorm</b> (keeps numbers in a stable range) → <b>ReLU</b> (keeps only positive evidence). padding = k // 2 keeps the length."),
    ("3", "<b>MaxPool</b> halves the length (keeps the strongest response). The last block instead averages over the whole beat."),
    ("5–7", "Three blocks: 32 detectors of width 7 → 64 of width 5 → 128 of width 3. Output: 128 numbers describing the beat."),
], size=10.2)  # fmt: skip
# architecture strip
y0 = 150
blocks = [("beat", "1 × 280", "soft"), ("conv 7", "32 × 140", "blue"), ("conv 5", "64 × 70", "blue"),
          ("conv 3", "128 × 1", "blue"), ("dense", "64", "lav"), ("output", "N S V F", "coral")]  # fmt: skip
x = 600
for i, (t, s, col) in enumerate(blocks):
    yy = 400 - i * 52
    box(x, yy - 40, 150, 40, col if col != "soft" else "butter", t, s, title_size=10.5)
    if i:
        arrow(x + 75, yy + 10, x + 75, yy + 1)
para(770, 400, 142, f"{tmeta['cnn']['n_parameters']:,} learned numbers (parameters).<br/><br/>Dropout (50% then 30%) "
     "randomly switches off parts while training so the network can't memorise.", 10, "muted")  # fmt: skip
para(48, 100, 520, f"<b>Result:</b> macro-F1 {comp['CNN']['macro_f1']:.3f}. Shape alone is not enough: the CNN cannot tell an "
     "early-but-normal-looking S beat from a normal one, and it found almost no F beats.", 11)  # fmt: skip

# ---------------------------------------------------------------- 20 Hybrid
new_slide("The recommended model: HybridCNN-rel", "5 · the models",
          subtitle="Two inputs, two branches, one decision. The beat goes in as 2 channels: the beat itself, and beat − patient template.")  # fmt: skip
src = snippet("src/models.py", ["HybridCNN"]).split("\n")
code(48, 405, 500, "\n".join(src), size=7.4, title="src/models.py · HybridCNN")
notes(570, 405, 342, [
    ("11", "The same conv trunk as the CNN; with in_channels = 2 it reads beat and beat − template."),
    ("12–18", "A small network for the 7 timing features → 32 numbers."),
    ("19–21", "Join 128 shape + 32 timing numbers → 64 → 4 class scores."),
    ("22–23", "The timing features' mean and spread (from training data) are stored <b>inside</b> the model file, so the API can't scale them differently."),
    ("29–33", "Forward pass: standardise timing, run both branches, concatenate, classify."),
], size=10)  # fmt: skip
rrect(570, 60, 342, 110, "blueT", r=12)
para(586, 160, 312, f"<b>{tmeta['hybrid']['n_parameters']:,}</b> parameters in total: 3 copies (seeds 42, 7, 2024) "
     "averaged by <b>SeedEnsemble</b>, which reduces luck from random initialisation.", 10.8)  # fmt: skip

# ---------------------------------------------------------------- 21 imbalance in training
new_slide("Training (1/3): making rare classes count", "6 · training", accent="coral")
h = code(48, 440, 430, snippet("src/train.py", ["class_weights"]), size=9.2, title="src/train.py")
notes(48, 440 - h - 14, 430, [
    ("3", "Avoid dividing by zero if a class is missing."),
    ("4", "weight = total ÷ (4 × count of that class). Rare class → big weight. A perfectly balanced dataset would give weight 1 for every class."),
], color="coral")  # fmt: skip
cw = tmeta["cnn"]["class_weights"]
for i, k in enumerate(config.CLASS_NAMES):
    x = 48 + i * 108
    rrect(x, 150, 98, 78, ["blueT", "coralT", "lavT", "sageT"][i], r=10)
    c.setFillColor(C["ink"])
    c.setFont("SansB", 20)
    c.drawString(x + 12, 195, f"{cw[k]:.2f}")
    c.setFont("Sans", 10)
    c.setFillColor(C["muted"])
    c.drawString(x + 12, 172, f"weight of {k}")
para(
    48,
    135,
    430,
    'A missed F beat costs ~110× more than a missed N beat. Without this, the network learns to say "N" all the time.',
    11,
    "muted",
)
card(500, 440, 412, 150, "coral", "Data augmentation (training only)",
     "Each batch gets small random changes that keep the label true:<br/>"
     "• shift the beat by up to ±5 samples (R-peak not perfectly centred)<br/>"
     "• gain ×0.85–1.15 (stronger/weaker electrodes)<br/>"
     "• a little noise and slow baseline drift<br/>"
     "This imitates differences between people and devices.", 10.8)  # fmt: skip
card(500, 270, 412, 95, "blue", "Optimiser",
     "<b>Adam</b>, learning rate 0.001, weight decay 0.0001 (gently pulls weights toward zero), batches of 256 beats.", 10.8)  # fmt: skip

# ---------------------------------------------------------------- 22 training loop
new_slide("Training (2/3): the training loop", "6 · training", accent="coral")
code(
    48,
    440,
    500,
    snippet("src/train.py", ["run_epoch"]),
    size=8.6,
    title="src/train.py · one pass over the data (an epoch)",
)
notes(572, 440, 340, [
    ("2–3", "Training mode if an optimiser is given (dropout on), otherwise evaluation mode."),
    ("5", "Only track gradients while training (saves memory when evaluating)."),
    ("6", "Loop over batches of 256 beats: waveform, timing features, label."),
    ("7–8", "Augment the batch (training only)."),
    ("10–11", "Predict class scores (logits) and measure how wrong they are (weighted cross-entropy)."),
    ("13–15", "Back-propagation: work out how each weight should change, then nudge it."),
    ("19–21", "Summarise the epoch: loss, accuracy and macro-F1."),
], size=10.2, color="coral")  # fmt: skip

new_slide("Training (3/3): when to stop", "6 · training", accent="coral",
          subtitle="Train too little and the model is vague; too long and it memorises the training patients. Held-out patients decide.")  # fmt: skip
image("training_curve", 48, 410, w=470)
card(540, 410, 372, 100, "coral", "Early stopping (CNN and ablations)",
     "After every epoch, measure macro-F1 on held-out patients. Keep the best weights; stop after 7 epochs with no improvement.", 11)  # fmt: skip
card(540, 290, 372, 118, "blue", "Final hybrid model",
     "Cross-validation showed ~7 epochs is best, so it is retrained on <b>all 22</b> training patients for exactly 7 epochs, "
     "3 times with different seeds, and the 3 copies are averaged.", 11)  # fmt: skip
para(48, 170, 470, "Notice the gap: training F1 keeps rising while validation F1 stalls. That gap is the model starting to memorise "
     "the 18 training patients instead of learning beats that generalise.", 11, "muted")  # fmt: skip

# ---------------------------------------------------------------- 23 model selection
new_slide("Choosing the final model fairly", "6 · training", accent="coral",
          subtitle="Validation on only 4 patients was too noisy to separate the best variants, so the final choice used 5-fold cross-validation.")  # fmt: skip
image("cv_select", 48, 405, w=470)
bullets(540, 400, 372, [
    "Split the 22 training patients into 5 groups. Train on 4 groups, predict the 5th, rotate.",
    "Pool all out-of-fold predictions and compute one macro-F1 per variant.",
    "<b>Winner: Hybrid-rel + augmentation (0.583)</b>, the variant now in the API.",
    "The test patients (DS2) were never used to choose anything.",
], size=11.5, color="coral")  # fmt: skip
rrect(48, 100, 864, 80, "butterT", r=12)
para(66, 168, 830, "<b>Honesty note:</b> the switch from the 4-patient validation split to cross-validation was made after seeing that "
     "validation and test disagreed. It is documented in docs/EXPERIMENTS.md. Seed-to-seed differences (±0.02–0.04 macro-F1) are "
     "as big as the gaps between top variants, which is why every comparison uses 3 seeds or 5 folds.", 11)  # fmt: skip

# ---------------------------------------------------------------- 24 metrics explained
new_slide("How to read the scores", "7 · results", accent="lav")
defs = [
    (
        "blue",
        "Recall (sensitivity)",
        "Of all real S beats, what share did we find? Missing abnormal beats is the dangerous error.",
    ),
    ("coral", "Precision", "Of all beats we called S, what share really were S? Low precision = many false alarms."),
    ("lav", "F1", "One number balancing precision and recall (their harmonic mean). 1 is perfect."),
    (
        "sage",
        "Macro-F1",
        "Average F1 over N, S, V and F with equal weight. Our main score, because it can't be gamed by predicting N.",
    ),
    (
        "blue",
        "ROC-AUC",
        "Chance that a random abnormal beat gets a higher score than a random normal one. 0.5 = guessing, 1 = perfect ranking.",
    ),
    ("coral", "Specificity", "Of all beats that are NOT S, what share did we correctly leave alone?"),
]
for i, (col, t, d) in enumerate(defs):
    x = 48 + (i % 3) * 292
    y = 430 - (i // 3) * 140
    card(x, y, 278, 122, col, t, d, size=11)

# ---------------------------------------------------------------- 25 results headline
new_slide("Results on 22 people the model never saw", "7 · results", accent="lav",
          subtitle=f"DS2: {ev['n_test_beats']:,} beats. Hybrid-rel ensemble compared with the three other models.")  # fmt: skip
stat(48, 410, 200, f"{hyb['macro_f1']:.3f}", "macro-F1 (best of all models)", "blue")
stat(260, 410, 200, f"{hyb['macro_auc']:.3f}", "macro ROC-AUC (ranking quality)", "lav")
stat(472, 410, 200, f"{hyb['V_recall']:.0%}", "of V beats found", "sage")
stat(684, 410, 228, f"{hyb['S_recall']:.0%}", "of S beats found", "coral")
image("recall", 48, 320, w=560)
para(630, 315, 282, "<b>Read the chart:</b> each group is a beat class, each bar a model. The blue Hybrid is the most balanced: "
     "best on V, near the top on S and F, without XGBoost's many false alarms on N.<br/><br/>"
     "The CNN finds no F beats at all (bar at zero).", 11)  # fmt: skip

# ---------------------------------------------------------------- 26 confusion
new_slide("Where the Hybrid model goes wrong", "7 · results", accent="lav",
          subtitle="Confusion matrix: rows = expert label, columns = model's answer. Each row adds up to 1. Diagonal = correct.")  # fmt: skip
image("confusion", 48, 410, h=330)
bullets(450, 400, 462, [
    "<b>V is excellent</b>: 96% found, and few other beats are mistaken for V.",
    "<b>S is found (84%) but over-called</b>: 5% of normal beats (2,173) are flagged S. Because normal beats are so common, more S alarms are false than true (precision 0.41).",
    "<b>F is the weakest class</b>: 81% of F beats are found, but 14% of normal beats (6,397) are also called F, so almost every F alarm is false (precision 0.05). Nearly all training F beats come from one patient (record 208).",
    "In a real workflow these false alarms are reviewed by a human; missing a dangerous beat is worse than a false alarm.",
], size=11.3, color="lav")  # fmt: skip

# ---------------------------------------------------------------- 27 per patient + ablation
new_slide("Results differ from person to person", "7 · results", accent="lav",
          subtitle="Per-patient accuracy of the Hybrid model. Averages hide this, so every evaluation also reports each record.")  # fmt: skip
image("per_patient", 48, 410, w=560)
para(630, 405, 282, "5 of 22 patients are more than 10 points below the overall accuracy (coral). "
     "They have unusual baselines or many borderline beats. This is the main remaining weakness, and the motivation "
     "for patient-specific adaptation in future work.", 11)  # fmt: skip
image("feature_ablation", 48, 225, h=170)
para(420, 210, 492, "<b>Ablation</b> = remove one ingredient and measure the drop. Adding the 3 patient-normalised timing features was the single biggest win "
     "for logistic regression (0.506 → 0.586).", 11)  # fmt: skip

# ---------------------------------------------------------------- 28 calibration
new_slide("Can we trust the percentages? Calibration", "8 · trust & serving",
          subtitle="Rare-class weighting inflates the raw S and F percentages. Calibration rescales them so that a stated 80% is right about 80% of the time.", accent="sage")  # fmt: skip
h = code(48, 405, 430, snippet("src/models.py", ["apply_calibration"]), size=9, title="src/models.py")
notes(48, 405 - h - 14, 430, [
    ("3", "Divide the log-probabilities by a <b>temperature</b> T (T &gt; 1 softens over-confident answers)."),
    ("4–5", "Add a small <b>bias per class</b> (BCTS): undoes the push toward rare classes from class weighting."),
    ("6–7", "Softmax turns the scores back into probabilities that add up to 1."),
], size=10.2, color="sage")  # fmt: skip
image("calibration", 500, 410, w=412)
para(48, 150, 430, f"T and the biases are fitted on <b>training-set out-of-fold predictions only</b>. On unseen patients the "
     f"log-loss drops from {calrep['ds2']['metrics']['raw']['nll']:.3f} to {calrep['ds2']['metrics']['bcts']['nll']:.3f}. "
     "Labels still come from the raw model, because the calibrated argmax would almost never say S.", 10.8)  # fmt: skip

# ---------------------------------------------------------------- 29 conformal
new_slide("Prediction sets: \"it's N or S\"", "8 · trust & serving", accent="sage",
          subtitle="Instead of one answer, return every class that is still plausible, with a target of containing the truth 90% of the time.")  # fmt: skip
h = code(48, 405, 450, snippet("src/calibrate.py", ["conformal_qhat"]), size=9, title="src/calibrate.py")
notes(48, 405 - h - 14, 450, [
    ("4", "One threshold per class (\"Mondrian\" conformal), so rare classes get their own guarantee."),
    ("5", "Score = 1 − probability given to the true class: how surprised the model was."),
    ("7", "Take the score below which 90% of that class's examples fall."),
], size=10.4, color="sage")  # fmt: skip
card(520, 405, 392, 105, "sage", "Rule at prediction time",
     "Class k goes in the set if 1 − p<sub>k</sub> ≤ threshold<sub>k</sub>.<br/>"
     "One class in the set = confident. Two or three = uncertain, so flag the beat for a human.", 11)  # fmt: skip
s = calrep["ds2"]["sets"]
card(520, 285, 392, 125, "coral", "How well it worked",
     f"On training patients (out-of-fold): exactly 90% for every class.<br/>"
     f"On new patients: {s['coverage']:.0%} overall (S/V/F ≈ 98%, N {s['N_coverage']:.0%}), average set size {s['mean_set_size']:.2f}. "
     "The guarantee weakens because new people are not exactly like the training people. This is reported, not hidden.", 10.8)  # fmt: skip

# ---------------------------------------------------------------- 30 API
new_slide("Serving: the prediction API", "8 · trust & serving", accent="sage")
src = snippet("app/api.py", ["predict"]).split("\n")
stop = next(i for i, ln in enumerate(src) if ln.strip().startswith("cal = calibration"))
code(
    48,
    440,
    500,
    "\n".join(src[:stop]).rstrip(),
    size=7.5,
    title="app/api.py · predict (first half; then calibration, logging, response)",
)
notes(570, 440, 342, [
    ("3", "Return 503 if no model is loaded."),
    ("4–5", "Z-score the beat exactly as in training."),
    ("8–15", "If timing is given, use the Hybrid. Convert ms to samples and reuse the <b>same rr_features</b> function as training."),
    ("16–20", "Second channel = beat − patient template (or zero difference if none is sent)."),
    ("22–24", "Run the model without gradients and turn scores into probabilities."),
], size=9.8, color="sage")  # fmt: skip
shell(570, 205, 342, [
    'POST /predict  {"waveform": [280 numbers],',
    '  "rr_context": {"rr_prev_ms": 620,',
    '     "rr_next_ms": 1050, "rr_local_avg_ms": 830}}',
    "",
    '-> "predicted_class": "S",',
    '   "prediction_set": ["N", "S"], ...',
], size=8.2, title="example")  # fmt: skip
c.setFont("Sans", 9.5)
c.setFillColor(C["muted"])
c.drawString(
    48,
    50,
    "Every response carries the header X-Disclaimer: Research demonstration only. Not for clinical use. Logs keep only a hash of the waveform.",
)

# ---------------------------------------------------------------- 31 dashboard + docker + tests
new_slide("Dashboard, Docker and tests", "8 · trust & serving", accent="sage")
card(48, 430, 278, 225, "blue", "Streamlit dashboard",
     "<b>Data Explorer</b>: class counts, typical shapes, feature distributions, quality report.<br/><br/>"
     "<b>Model Performance</b>: all metrics, curves, ablation and per-patient results.<br/><br/>"
     "<b>Live Prediction</b>: draw a test beat or paste your own; see each model's answer, the prediction set, and which samples drove the decision (saliency).", 10.5)  # fmt: skip
card(341, 430, 278, 225, "lav", "Docker",
     "One command starts the API and the dashboard in containers:<br/><font face='Mono'>docker compose up --build</font><br/><br/>"
     "CPU-only PyTorch, runs as a non-root user, health checks included. The container reproduces the GPU-trained results exactly on all 400 demo beats.", 10.5)  # fmt: skip
card(634, 430, 278, 225, "sage", "82 automatic tests",
     "<font face='Mono'>pytest tests/ -q</font><br/><br/>"
     "• the filter doesn't move R-peaks<br/>• windows are 280 samples<br/>• labels map correctly<br/>• broken data is caught<br/>"
     "• models load and predict<br/>• the API rejects bad input<br/>• logs never contain the waveform", 10.5)  # fmt: skip
para(48, 150, 864, "Reproducibility: one seed (42) for Python, NumPy and PyTorch; data and model files are fingerprinted with SHA-256 in "
     "training_metadata.json and evaluation_metrics.json, so you can prove which data produced which model.", 11, "muted")  # fmt: skip

# ---------------------------------------------------------------- 32 journey
new_slide("How the project improved, iteration by iteration", "wrap-up", accent="lav")
rows = [
    ["#", "Change", "Effect"],
    [
        "0",
        "Pipeline as specified: LogReg/XGBoost on 13 features, CNN on the waveform",
        "CNN 0.41 macro-F1 on unseen patients",
    ],
    ["1", "Bug fix: the validation split had taken 90% of F beats out of training", "F beats in training: 37 → 380"],
    ["2", "Patient-normalised RR features, Hybrid CNN, augmentation", "LogReg 0.51 → 0.59; Hybrid ~0.62"],
    ["3", "Patient template channel (beat − usual beat)", "Hybrid-rel 0.64–0.66 on DS2 (3 seeds)"],
    [
        "4",
        "5-fold patient-grouped CV for the final choice; 3-seed ensemble refit on all DS1",
        "0.603 macro-F1, AUC 0.963 (selected without DS2)",
    ],
    ["5", "Calibration + conformal prediction sets", "Log-loss 0.477 → 0.158; 86% set coverage"],
]
table(48, 440, [40, 520, 304], rows, size=10.8, header_color="lav")
para(48, 130, 864, "Each step is recorded with numbers in docs/EXPERIMENTS.md, including the steps that did not help, "
     "so the final model's choice can be checked by anyone.", 11.5, "muted")  # fmt: skip

# ---------------------------------------------------------------- 33 limits / future
new_slide("Limitations and what to try next", "wrap-up", accent="coral")
card(48, 430, 420, 330, "coral", "Limitations (be honest about them)",
     "• Small, old dataset: 47 people recorded in 1975–79<br/>"
     "• One lead (MLII) instead of the 12 a cardiologist uses<br/>"
     "• Uses expert-marked R-peak positions; a real system needs its own beat detector<br/>"
     "• F beats come almost entirely from one patient<br/>"
     "• S precision is low (many false alarms)<br/>"
     "• Not tested on any other hospital's data<br/>"
     "• Not a medical device: research and learning only", 11.2)  # fmt: skip
card(492, 430, 420, 330, "sage", "Next steps (from docs/FUTURE_WORK.md)",
     "1. Test on a second database (INCART) to see if results transfer<br/>"
     "2. Learn the patient comparison (Siamese / prototype networks)<br/>"
     "3. Sequence model over many beats (rhythm end to end)<br/>"
     "4. Focal loss and patient-balanced sampling for fewer false alarms<br/>"
     "5. MiniRocket: a fast, strong time-series baseline<br/>"
     "6. Plug in a QRS detector for true end-to-end use<br/>"
     "7. Self-supervised pre-training on large unlabelled ECG sets", 11.2)  # fmt: skip

# ---------------------------------------------------------------- 34 glossary
new_slide("Cheat sheet", "wrap-up")
terms = [
    ("ECG", "recording of the heart's electrical activity"),
    ("R-peak", "the tall spike of each beat; our anchor point"),
    ("RR interval", "time between two R-peaks"),
    ("AAMI EC57", "standard that defines the N/S/V/F classes"),
    ("DS1 / DS2", "train / test patient groups (de Chazal split)"),
    ("Band-pass filter", "keeps only frequencies in a range (0.5–40 Hz)"),
    ("Z-score", "rescale to mean 0, spread 1"),
    ("Feature", "a number describing a beat (e.g. rr_ratio)"),
    ("Epoch", "one pass over all training data"),
    ("Class weights", "make mistakes on rare classes cost more"),
    ("Augmentation", "random realistic changes to training beats"),
    ("Early stopping", "keep the epoch that did best on held-out patients"),
    ("Cross-validation", "rotate which patients are held out, average"),
    ("Ensemble", "average several models' answers"),
    ("Calibration", "make 80% confidence mean right 80% of the time"),
    ("Prediction set", "all classes still plausible at 90% coverage"),
]
for i, (t, d) in enumerate(terms):
    x = 48 + (i % 2) * 438
    y = 440 - (i // 2) * 46
    rrect(x, y - 38, 424, 38, ["blueT", "coralT", "lavT", "sageT"][(i // 2) % 4], r=9)
    c.setFillColor(C["ink"])
    c.setFont("SansB", 11.5)
    c.drawString(x + 14, y - 24, t)
    c.setFont("Sans", 10.5)
    c.setFillColor(C["muted"])
    c.drawString(x + 150, y - 24, d)

# ---------------------------------------------------------------- 35 recap
new_slide("Recap", "wrap-up")
recap = [
    ("blue", "Data", "MIT-BIH, 44 records, 100,630 beats, split by patient so tests are honest."),
    ("coral", "Pipeline", "Filter → cut 280-sample windows → z-score → 16 features + patient template → checks."),
    (
        "lav",
        "Models",
        "LogReg, XGBoost, CNN, and the winning Hybrid-rel ensemble (shape + timing, both patient-relative).",
    ),
    (
        "sage",
        "Results",
        f"Macro-F1 {hyb['macro_f1']:.3f}, AUC {hyb['macro_auc']:.3f}, 96% of V and 84% of S beats found on unseen people.",
    ),
    ("blue", "Trust", "Calibrated percentages, 90% prediction sets, every limitation documented."),
    ("coral", "Use it", "API + dashboard + Docker + 82 tests; every command is on page 9."),
]
y = 430
for col, t, d in recap:
    c.setFillColor(C[col])
    c.circle(62, y - 10, 7, stroke=0, fill=1)
    c.setFillColor(C["ink"])
    c.setFont("SansB", 15)
    c.drawString(82, y - 15, t)
    para(200, y - 2, 700, d, 13)
    y -= 58
c.setFont("Sans", 10)
c.setFillColor(C["muted"])
c.drawString(
    48,
    50,
    "More detail: README.md · docs/METHODOLOGY.md · docs/EXPERIMENTS.md · docs/API.md · docs/FUTURE_WORK.md · docs/model_card.md",
)

c.save()
print(f"wrote {OUT} ({page_no} pages)")
