# Model Card: PulseML ECG Beat Classifier

> ⚠️ **RESEARCH DEMONSTRATION ONLY. NOT FOR CLINICAL DIAGNOSIS.** This is not a medical device and has
> not been cleared or approved by the FDA or any regulator. Do not use it for patient care.

*Generated automatically by `python -m src.evaluate` on 2026-10-13T03:42:42+00:00.*
Model version `1.0.0` · CNN checkpoint `deb5a42fa9ca` · Hybrid checkpoint `c409c9b505b1`.

## Model details

| | |
|---|---|
| Task | Single-beat arrhythmia classification into AAMI EC57 classes N, S, V, F |
| Input | 280-sample (0.78 s @ 360 Hz) MLII window centred on the annotated R-peak, 0.5–40 Hz Butterworth band-passed, per-beat z-scored |
| Recommended model | **HybridCNN-rel (2-channel trunk: beat + deviation from patient template; RR MLP), 3-seed ensemble**: the CNN trunk below plus an MLP over 7 RR-interval features (raw and patient-normalised), 143,628 parameters in total, augmentation=True, refit on all 22 DS1 patients for 7 epochs per member (seeds [42, 7, 2024]). Served by the API when `rr_context` is supplied |
| Specified baseline | 1D CNN: Conv1d(1→32,k7)→BN→ReLU→MaxPool → Conv1d(32→64,k5)→BN→ReLU→MaxPool → Conv1d(64→128,k3)→BN→ReLU→GAP → Dropout(0.5)→Linear(128→64)→ReLU→Dropout(0.3)→Linear(64→4) |
| Baseline CNN parameters | 44,228 |
| Training (both) | Adam (lr 1e-3, wd 1e-4), class-weighted cross-entropy, batch 256, baseline CNN early-stopped on validation macro-F1 (patience 7, best epoch 13 of 20); hybrid with fixed epochs from CV and waveform augmentation |
| Classical baselines | Logistic regression and XGBoost on 16 engineered features (RR intervals, RR ratio, local HR, patient-normalised RR, waveform statistics, QRS proxy, pre/post-beat energy). Best params: LogReg {'clf__C': 0.01}, XGBoost {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 100} |
| Seed | 42 (python, numpy, torch) |

## Training data and split methodology

* **Source:** MIT-BIH Arrhythmia Database (PhysioNet): 48 half-hour two-channel ambulatory ECG recordings from 47 subjects, 360 Hz, cardiologist beat annotations.
* **Exclusions:** paced records 102, 104, 107, 217 (AAMI EC57); beats labelled `/`, `f`, `Q` and other unclassifiable beats; beats whose window crosses the recording boundary or that lack a neighbouring beat for RR computation.
* **Inter-patient split (de Chazal et al., 2004):** DS1 (22 records) for training, DS2 (22 different records) held out for testing. **No record (and therefore no patient) appears in both.** Beat-level random splits leak patient-specific morphology and inflate metrics substantially.
* **Validation:** ~20% of DS1 *records* (114, 124, 205, 223), chosen so every class keeps ≥50% of its beats in training, held out for early stopping and variant selection. Baseline hyperparameters tuned with 3-fold `StratifiedGroupKFold` grouped by record.
* **Class imbalance:** DS1 is ~90% N. Handled with inverse-frequency class weights (CNN, XGBoost sample weights) and `class_weight='balanced'` (LogReg). Macro-F1 is the primary metric.

## Intended use

* Research, education and portfolio demonstration of a reproducible biomedical ML pipeline.
* **Out of scope:** any clinical, diagnostic, triage, screening or monitoring use; use on patients; use on signals from devices, leads or sampling rates other than those described above.

## Metrics on DS2 (held-out patients, 49,661 beats)

| model | accuracy | macro_f1 | weighted_f1 | macro_auc | N_recall | S_recall | V_recall | F_recall | N_f1 | S_f1 | V_f1 | F_f1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LogReg | 0.850 | 0.586 | 0.890 | 0.910 | 0.854 | 0.842 | 0.854 | 0.420 | 0.917 | 0.537 | 0.819 | 0.072 |
| XGBoost | 0.778 | 0.543 | 0.836 | 0.941 | 0.764 | 0.935 | 0.866 | 0.832 | 0.865 | 0.486 | 0.722 | 0.101 |
| CNN | 0.628 | 0.408 | 0.723 | 0.712 | 0.617 | 0.712 | 0.804 | 0.003 | 0.753 | 0.140 | 0.737 | 0.002 |
| Hybrid | 0.807 | 0.603 | 0.865 | 0.963 | 0.794 | 0.845 | 0.964 | 0.809 | 0.883 | 0.556 | 0.884 | 0.087 |

### Per-class metrics

**LogReg**

| class | precision | recall | f1 | specificity | support |
|---|---|---|---|---|---|
| N | 0.991 | 0.854 | 0.917 | 0.935 | 44218 |
| S | 0.395 | 0.842 | 0.537 | 0.950 | 1836 |
| V | 0.787 | 0.854 | 0.819 | 0.984 | 3219 |
| F | 0.039 | 0.420 | 0.072 | 0.919 | 388 |

**XGBoost**

| class | precision | recall | f1 | specificity | support |
|---|---|---|---|---|---|
| N | 0.997 | 0.764 | 0.865 | 0.982 | 44218 |
| S | 0.328 | 0.935 | 0.486 | 0.926 | 1836 |
| V | 0.618 | 0.866 | 0.722 | 0.963 | 3219 |
| F | 0.054 | 0.832 | 0.101 | 0.884 | 388 |

**CNN**

| class | precision | recall | f1 | specificity | support |
|---|---|---|---|---|---|
| N | 0.965 | 0.617 | 0.753 | 0.820 | 44218 |
| S | 0.077 | 0.712 | 0.140 | 0.674 | 1836 |
| V | 0.681 | 0.804 | 0.737 | 0.974 | 3219 |
| F | 0.001 | 0.003 | 0.002 | 0.986 | 388 |

**Hybrid**

| class | precision | recall | f1 | specificity | support |
|---|---|---|---|---|---|
| N | 0.995 | 0.794 | 0.883 | 0.970 | 44218 |
| S | 0.414 | 0.845 | 0.556 | 0.954 | 1836 |
| V | 0.817 | 0.964 | 0.884 | 0.985 | 3219 |
| F | 0.046 | 0.809 | 0.087 | 0.868 | 388 |

### Ablation (how the recommended model was chosen)

Deep-model variants, mean ± sd over seeds [42, 7, 2024]. Selection used **validation macro-F1 only**; selected: **Hybrid + augmentation**. DS2 is shown for transparency.

| variant | val_f1_mean | val_f1_std | ds2_f1_mean | ds2_f1_std |
|---|---|---|---|---|
| CNN (waveform) | 0.488 | 0.021 | 0.416 | 0.015 |
| CNN + augmentation | 0.485 | 0.023 | 0.412 | 0.016 |
| Hybrid (waveform + RR) | 0.611 | 0.022 | 0.620 | 0.035 |
| Hybrid + augmentation | 0.615 | 0.018 | 0.594 | 0.007 |
| Hybrid-rel (+ patient template) | 0.598 | 0.027 | 0.657 | 0.041 |
| Hybrid-rel + augmentation | 0.597 | 0.017 | 0.640 | 0.028 |

The top variants were within one seed-sd on the 4-patient validation split, so the final choice used **5-fold record-grouped CV over all 22 DS1 patients** (pooled out-of-fold macro-F1). Selected: **Hybrid-rel + augmentation**, refit on all of DS1 for 7 epochs (median CV best epoch) as a 3-seed ensemble.

| variant | cv_pooled_macro_f1 | cv_fold_macro_f1_mean | cv_fold_macro_f1_std |
|---|---|---|---|
| Hybrid (waveform + RR) | 0.530 | 0.561 | 0.125 |
| Hybrid + augmentation | 0.557 | 0.570 | 0.122 |
| Hybrid-rel (+ patient template) | 0.578 | 0.600 | 0.089 |
| Hybrid-rel + augmentation | 0.583 | 0.582 | 0.082 |

Full iteration log: `docs/EXPERIMENTS.md`.

### Patient-level analysis

Records whose accuracy falls more than 10 percentage points below the model's overall DS2 accuracy:
* **LogReg:** 202, 213, 214, 219, 233
* **XGBoost:** 117, 213, 219, 233
* **CNN:** 111, 113, 117, 212, 213, 222, 228
* **Hybrid:** 202, 210, 214, 222, 233

Performance varies substantially between patients; aggregate metrics hide this. See `outputs/figures/per_record_accuracy.png`.

## Limitations

* **Small dataset:** 44 usable recordings from 47 subjects; minority classes (especially F and S) have few examples and are concentrated in a handful of patients, so per-class estimates have wide uncertainty.
* **Single lead only:** uses MLII; clinicians use 12-lead context. Record 114's MLII is on channel 1 and is handled explicitly.
* **Requires known R-peak locations:** the pipeline uses expert annotated R-peaks; a deployed system would need a QRS detector, whose errors would propagate.
* **Curated lab annotations vs. real-world noise:** MIT-BIH annotations were made by cardiologists on selected recordings; real ambulatory/wearable data has more artefact, different hardware and different filtering.
* **Not validated on independent populations** (e.g. INCART, European ST-T, wearable data); generalisation is unknown.
* **Calibration:** class-weighted training shifts raw probabilities toward minority classes. The API corrects this for the hybrid with BCTS fitted on DS1 out-of-fold predictions (`python -m src.calibrate`; DS2 NLL 0.477 → 0.158) and returns Mondrian conformal prediction sets targeting 90% coverage; on DS2 they cover 86% (N 84.5%) because new patients are not exchangeable with training patients. Even calibrated scores are not probabilities of disease.
* **Morphology-only CNN:** the specified CNN sees a single beat and no RR context, which is the main discriminator for S beats; it is kept as a baseline. The hybrid model fixes most of this but needs RR context (and, for patient-normalised features, ~30 s of rhythm history) at inference.
* **F class:** almost all training F beats come from one patient (record 208); F performance is unreliable for every model.
* **Model selection noise:** only 4 validation patients; validation macro-F1 varies by ±0.02–0.04 across seeds, which is about the same size as the differences between the top variants.

## Ethical considerations

* **NOT for clinical diagnosis. No FDA clearance (or CE marking).** Every API response and dashboard view repeats this disclaimer.
* Misclassification of V or S beats could lead to false reassurance or unnecessary alarm if misused.
* The API logs only a hash of each input waveform, never the raw signal.

## Bias and representativeness

* MIT-BIH was recorded at Boston's Beth Israel Hospital between **1975 and 1979**: 25 men aged 32–89 and 22 women aged 23–89. Records 201 and 202 come from the same subject.
* The cohort is small, from a single centre and one era of recording hardware, and is not documented by race or ethnicity; performance on other demographics, ages (e.g. paediatrics), or comorbidity profiles is unknown.
* 23 of the records were deliberately selected to include clinically significant but less common arrhythmias, so class priors differ from any real screening population.

## Handling PHI in a real deployment

MIT-BIH is public and de-identified, so no PHI is processed here. In a real clinical environment this pipeline would need:

* HIPAA-compliant infrastructure (BAA-covered cloud, encryption at rest and in transit, VPC isolation).
* De-identification per HIPAA Safe Harbor/Expert Determination before data reaches research environments; date shifting and removal of device serials and timestamps embedded in waveform headers.
* Role-based access control, audit logging of every data access, and minimum-necessary data use.
* IRB approval and data-use agreements; no raw waveforms or identifiers in application logs (the API already logs only hashes).
* Model monitoring for data drift, and a documented human-in-the-loop review process before any clinical use.

## Citation

* Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. *IEEE Eng in Med and Biol* 20(3):45-50 (2001).
* Goldberger A, et al. PhysioBank, PhysioToolkit, and PhysioNet. *Circulation* 101(23):e215–e220 (2000).
* de Chazal P, O'Dwyer M, Reilly RB. Automatic classification of heartbeats using ECG morphology and heartbeat interval features. *IEEE Trans Biomed Eng* 51(7):1196-1206 (2004).
