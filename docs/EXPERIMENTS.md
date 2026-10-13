# Experiment Log

> ⚠️ Research demonstration only. Not for clinical use.

Chronological record of every modelling iteration, including the ones that didn't help, so results can be audited.

**Rules applied throughout**
1. Train on DS1 only. DS2 (22 unseen patients) is never used to choose models, features, epochs or thresholds.
2. Deep variants are compared on mean **validation** macro-F1 over 3 seeds (42, 7, 2024). Classical feature sets are compared on record-grouped 3-fold CV on DS1.
3. DS2 numbers are reported for *every* variant so you can see when validation and test disagree.

Reproduce everything: `python -m src.experiments` (≈15 min on an RTX 3060) and `python -m src.experiments --cv-select` (≈10 min). Raw results are in
`outputs/reports/experiments.{csv,json}`, `experiments_cv.{csv,json}` and the `*_log*.txt` files.

---

## Iteration 0: specified pipeline

The model families and hyperparameters specified up front: LogReg and XGBoost on 13 engineered features; a 1D CNN on the raw beat.

| Model | CV / val macro-F1 | DS2 macro-F1 | DS2 S recall | DS2 V recall | DS2 F recall |
|---|---|---|---|---|---|
| LogReg (13 feat.) | 0.458 (CV) | 0.506 | 0.33 | 0.84 | 0.40 |
| XGBoost (13 feat.) | 0.500 (CV) | 0.470 | 0.35 | 0.89 | 0.56 |
| CNN (waveform) | 0.464 (val) | 0.408 | 0.71 | 0.80 | 0.00 |

**Findings.** V beats generalise well (AUC ≈ 0.98). S and F do not. The CNN has no rhythm information and
"buys" S recall by labelling 34% of normal beats as S. XGBoost's grid search picked its smallest
configuration, a sign that less capacity generalises better across only 22 patients.

## Iteration 1: bug fix: validation split stripped the F class from training

**Problem found:** the first CNN run used `StratifiedGroupKFold`. Its validation fold contained record 208,
which holds **372 of DS1's 414 F beats**, so the CNN trained on only 37 F beats.

**Fix:** `patient_level_val_split` now searches 5,000 seeded record subsets for the one whose per-class
validation share is closest to 20%, while every class keeps ≥50% of its beats in training. Validation is
now records 114, 124, 205, 223 (34 F beats in validation, 380 in training). There is a regression test with a
synthetic "one patient holds the minority class" scenario.

**Also:** the environment had CPU-only PyTorch. Installing the CUDA 12.6 wheel cut CNN training from
~6 s to ~1.5 s per epoch on the RTX 3060, which made the multi-seed ablations below practical.

## Iteration 2: patient-normalised RR features + hybrid CNN + augmentation

**Hypothesis.** S beats are premature *relative to the patient's own rhythm*. Raw RR intervals mix
that signal with between-patient heart-rate differences (a 600 ms RR is premature at 60 bpm and normal at 100 bpm).

**Changes**
* New causal features: `rr_local_avg` (mean of the last ≤32 RR intervals), `rr_prev_norm`, `rr_next_norm`. They use unlabelled timing only, so they can be computed at inference time.
* `HybridCNN`: the specified conv trunk plus an MLP over 7 RR features.
* Optional waveform augmentation: shift ±5 samples, gain ±15%, noise, baseline drift.

### Feature ablation (classical models; fixed hyperparameters)

| Model · features | DS1 grouped-CV macro-F1 | DS2 macro-F1 | DS2 S recall | DS2 V recall | DS2 F recall |
|---|---|---|---|---|---|
| LogReg · 13 features | 0.458 ± 0.057 | 0.506 | 0.33 | 0.84 | 0.41 |
| LogReg · **16 features (+ normalised RR)** | **0.487 ± 0.050** | **0.586** | **0.84** | 0.85 | 0.42 |
| XGBoost · 13 features | 0.477 ± 0.009 | 0.470 | 0.35 | 0.89 | 0.56 |
| XGBoost · **16 features (+ normalised RR)** | **0.510 ± 0.034** | **0.512** | 0.57 | 0.92 | 0.79 |

Normalised RR improves CV for both models (the selection criterion), and DS2 agrees. LogReg's S recall
goes from 0.33 to 0.84. **Adopted.**

### Deep-model ablation (3 seeds each)

| Variant | Val macro-F1 (mean ± sd) | DS2 macro-F1 (mean ± sd) | DS2 S recall | DS2 V recall | DS2 F recall |
|---|---|---|---|---|---|
| CNN (waveform) | 0.488 ± 0.021 | 0.416 ± 0.015 | 0.34 | 0.82 | 0.00 |
| CNN + augmentation | 0.485 ± 0.023 | 0.412 ± 0.016 | 0.18 | 0.86 | 0.01 |
| Hybrid (waveform + RR) | 0.611 ± 0.022 | **0.620 ± 0.035** | 0.67 | 0.93 | 0.45 |
| Hybrid + augmentation | **0.615 ± 0.018** | 0.594 ± 0.007 | 0.74 | 0.95 | 0.12 |

Per-seed runs:

| Variant | Seed | Val F1 | DS2 F1 | Best epoch |
|---|---|---|---|---|
| CNN | 42 / 7 / 2024 | 0.464 / 0.504 / 0.497 | 0.408 / 0.433 / 0.407 | 13 / 14 / 7 |
| CNN + aug | 42 / 7 / 2024 | 0.459 / 0.491 / 0.505 | 0.401 / 0.431 / 0.405 | 7 / 8 / 10 |
| Hybrid | 42 / 7 / 2024 | 0.633 / 0.613 / 0.589 | 0.580 / 0.641 / 0.641 | 15 / 6 / 1 |
| Hybrid + aug | 42 / 7 / 2024 | 0.625 / 0.626 / 0.595 | 0.587 / 0.601 / 0.595 | 12 / 23 / 8 |

**Findings**
* **Adding rhythm is the single biggest gain:** +0.12 validation and +0.20 DS2 macro-F1 over the waveform-only CNN, with consistent gains on S, V and F.
* **Augmentation does not help the CNN** and changes the hybrid only marginally: +0.004 on validation, which is well inside seed noise, and −0.026 on DS2. Augmentation trades F recall for S recall.
* **Validation noise is the limiting factor.** With 4 validation patients, the seed-to-seed sd (≈0.02) is larger than the gap between the two hybrid variants. One hybrid run peaked at epoch 1. The documented protocol (choose on validation) was followed anyway; the disagreement is reported rather than overridden with test data.

## Iteration 3: patient-relative morphology (patient template)

**Hypothesis.** Absolute beat shape differs a lot between patients and leads, while *change from this patient's
usual beat* transfers better. That is how a cardiologist reads a strip: "this beat looks different from the others".

**Change.** `HybridCNN-rel`: a second input channel = beat − (sample-wise median of the patient's previous 16
beats). It is causal and label-free, so it's available at inference time (the API takes it as an optional
`patient_template`). A regression test checks that templates never use future beats and never mix records.

| Variant | Val macro-F1 (3 seeds) | DS2 macro-F1 (3 seeds) | Per-seed val | Per-seed DS2 |
|---|---|---|---|---|
| Hybrid-rel | 0.598 ± 0.027 | **0.657 ± 0.041** | 0.623 / 0.603 / 0.569 | 0.682 / 0.610 / 0.680 |
| Hybrid-rel + augmentation | 0.597 ± 0.017 | 0.640 ± 0.028 | 0.579 / 0.601 / 0.613 | 0.616 / 0.633 / 0.670 |

**Finding.** On the 4-patient validation split, the patient-template variants look *slightly worse* than
the plain hybrid (0.598 vs 0.611–0.615). On DS2 they look *better* (0.64–0.66 vs 0.59–0.62). Several runs
peaked at epoch 1. Under the round-2 rule (validation only), the selection is still **Hybrid + augmentation**.

## Iteration 4: sturdier model selection: 5-fold record-grouped CV

**Why.** The top four variants differ by less than one seed-sd on validation, and validation disagrees with
test. A 4-patient validation set is too small to choose between them. Choosing on DS2 would be
test-set selection and invalidate the reported numbers, so the selection criterion itself was upgraded.

> **Disclosure.** This change was made *after* seeing the round-2 validation/DS2 disagreement. CV uses only
> DS1, but the decision to run it is a researcher degree of freedom, so the DS2 result of the final model
> should be read as slightly optimistic.

**Protocol.** `python -m src.experiments --cv-select`: 5-fold `StratifiedGroupKFold` over all 22 DS1 patients
(4–5 held-out patients per fold), early stopping on the held-out fold (equally optimistic for every
variant), and macro-F1 computed on the **pooled out-of-fold predictions** of all 50,969 DS1 beats.

| Variant | Pooled OOF macro-F1 | Per-fold mean ± sd | OOF S recall | OOF V recall | OOF F recall |
|---|---|---|---|---|---|
| Hybrid | 0.530 | 0.561 ± 0.125 | 0.66 | 0.89 | 0.21 |
| Hybrid + augmentation | 0.557 | 0.570 ± 0.122 | 0.70 | 0.89 | 0.23 |
| Hybrid-rel | 0.578 | 0.600 ± 0.089 | 0.54 | 0.90 | 0.29 |
| **Hybrid-rel + augmentation** | **0.583** | 0.582 ± 0.082 | 0.51 | 0.90 | 0.32 |

CV favours the patient-template variants (+0.02–0.05), with noticeably lower variance across folds. That
agrees with DS2. **Selected: Hybrid-rel + augmentation.**

**Final recipe.** The best epochs across folds were 14, 10, 2, 2 and 7, so early stopping on a small validation set is
unreliable. The final model is therefore **refit on all 22 DS1 patients for 7 epochs** (the median) with no held-out
data, as a **3-seed ensemble** (42, 7, 2024; softmax-averaged) to cancel seed variance.

## Iteration 5: calibrated probabilities and conformal prediction sets

**Goal:** make the hybrid's probabilities trustworthy and attach a coverage guarantee, without touching DS2.
`python -m src.calibrate` fits everything on **DS1 out-of-fold predictions** (the same 5-fold record-grouped CV,
final recipe) and only then reports on DS2.

| Method (OOF, DS1) | NLL | Brier | ECE | Macro-F1 (argmax) |
|---|---|---|---|---|
| Raw ensemble | 0.393 | 0.181 | 0.031 | 0.532 |
| Temperature scaling (T) | 0.387 | 0.182 | 0.041 | 0.532 |
| **BCTS** (T + per-class bias) | **0.175** | **0.085** | **0.012** | 0.451 |

BCTS (T = 1.66) mostly undoes the inverse-frequency class weighting, so its probabilities are much better but its
argmax almost never predicts S or F (OOF S recall 0.003). **Decision:** serve BCTS probabilities, keep the
**raw argmax** for the label (`use_calibrated_argmax: false` in `outputs/models/calibration.json`).

**Conformal sets:** Mondrian (class-conditional) split conformal, score 1 − p_y, α = 0.1, giving
qhat = [0.079, 0.991, 0.744, 0.990]. OOF coverage is 0.900 for every class; mean set size 1.38.

**DS2 (unseen patients):** NLL 0.477 → 0.158, Brier 0.273 → 0.078, ECE 0.021 → 0.021. Set coverage 0.860
(N 0.845, S 0.979, V 0.982, F 0.985), mean set size 1.61, 47% singletons. **Caveat:** N falls short of 90% because
split conformal assumes exchangeability, and new patients are not exchangeable with DS1 patients; the guarantee
holds within the DS1 patient population only. Figure: `outputs/figures/hybrid_calibration.png`.

## Final results on DS2 (`python -m src.evaluate`)

| Model | Accuracy | Macro-F1 | Weighted-F1 | Macro-AUC | N rec. | S rec. | V rec. | F rec. |
|---|---|---|---|---|---|---|---|---|
| LogReg (16 features) | 0.850 | 0.586 | 0.890 | 0.910 | 0.854 | 0.842 | 0.854 | 0.420 |
| XGBoost (16 features; depth 2, 100 trees) | 0.778 | 0.543 | 0.836 | 0.941 | 0.764 | 0.935 | 0.866 | 0.832 |
| CNN (specified, waveform only) | 0.628 | 0.408 | 0.723 | 0.712 | 0.617 | 0.712 | 0.804 | 0.003 |
| **Hybrid-rel ensemble (recommended)** | 0.807 | **0.603** | 0.865 | **0.964** | 0.794 | 0.845 | **0.964** | 0.809 |

Recommended model, per class on DS2: N F1 0.883 (precision 0.995), **S F1 0.556** (precision 0.414, recall 0.845),
**V F1 0.884** (precision 0.817, recall 0.964), F F1 0.087 (precision 0.046, recall 0.809).

**Reading these numbers**
* Against the specified CNN: macro-F1 +0.19, macro-AUC 0.71 → 0.96, V recall 0.80 → 0.96, and F recall 0.00 → 0.81.
* The final ensemble (0.603) sits below the mean of the individual ablation runs (0.640). Those runs used early
  stopping on validation; the refit uses fixed epochs. This is the protocol's honest output and
  wasn't tuned further against DS2.
* LogReg on the 16 features is surprisingly close (0.586): rhythm features, not network depth, carry most of the gain.
* **F remains unsolved.** Recall is high, but precision is about 5% because many N beats in a few DS2 patients look like
  fusion. There aren't enough distinct F patients to learn the class.
* **S precision (0.41)** is the main remaining failure mode: about 2,200 of 44,218 N beats are called S. Calibration and
  per-patient adaptation in [FUTURE_WORK.md](FUTURE_WORK.md) target this directly.

---

## Summary of what was learned

1. **Splitting by patient changes the problem.** Every honest number here is far below the >99%
   beat-level accuracy often quoted for MIT-BIH, and that gap is the point.
2. **Context relative to the patient is what generalises:** normalised RR (rhythm) gave the largest gain;
   deviation from the patient's own template beat (morphology) gave the next.
3. **Capacity and augmentation are second-order.** XGBoost repeatedly chose its smallest configuration, and
   augmentation changed results by less than seed noise. The bottleneck is patient diversity (22 training patients),
   not optimisation.
4. **Model-selection variance is about as large as the effects being measured.** A 4-patient validation set
   disagreed with both CV and test. Use patient-grouped CV for decisions, report several seeds, treat sub-0.03 differences
   as ties, and disclose any change of protocol.

Next steps are prioritised in [FUTURE_WORK.md](FUTURE_WORK.md).
