# Future Work, Alternatives and Roadmap

> ⚠️ Research demonstration only. Nothing here describes a validated clinical system.

This document records what PulseML does **not** do yet, the alternatives that were considered, and a
prioritised roadmap. Items are tagged by expected impact on the core problem: generalising to
**unseen patients** (the de Chazal inter-patient setting). The current ablation results are in [EXPERIMENTS.md](EXPERIMENTS.md).

---

## 1. Where the current system is weakest

| Weakness | Evidence | Root cause |
|---|---|---|
| **F (fusion) beats** | F recall ≈ 0 for the specified CNN; ≈0.81 recall but ~5% precision for the recommended model | ~90% of DS1's F beats come from one patient (record 208), so there are too few distinct examples to learn F |
| **S precision** | Recommended model: S recall 0.85 but precision 0.41 (~2,200 N beats called S) | S beats differ from N mainly by timing; some patients have irregular N rhythms (e.g. sinus arrhythmia) that look premature |
| **Patient-level variance** | 5–7 DS2 records per model sit >10 pp below the model's overall accuracy | Morphology differs a lot between patients, and 22 training patients cover too little of that variation |
| **Model selection noise** | Validation macro-F1 varies ±0.02–0.04 across seeds; CV fold sd ≈ 0.08–0.12 | Only 22 training patients in total; addressed partly by grouped CV + seed ensembling |
| **Calibration** | Reliability curves sit far from the diagonal for S/F | Class-weighted training inflates minority-class probabilities |

---

## 2. Alternative models (and why they were or weren't chosen)

### 2.1 Classical / feature-based
| Alternative | Notes | Status |
|---|---|---|
| **LDA on morphology + RR features** (de Chazal 2004) | The original benchmark for this split; strong, simple, well calibrated | Worth adding as a literature baseline |
| **SVM (RBF) with wavelet features** | Popular in 2010s papers; DWT coefficients of the beat capture QRS/T shape | Candidate; needs careful grouped CV on `C`/`gamma` |
| **Random forest / LightGBM / CatBoost** | Similar to XGBoost; CatBoost handles class imbalance well | Low expected gain over the tuned XGBoost |
| **MiniRocket / ROCKET** (random convolutional kernels + ridge) | Very fast, state of the art on many UCR time-series benchmarks, few hyperparameters | **High-value next experiment**: cheap, and it reduces overfitting to 22 patients |

### 2.2 Deep learning
| Alternative | Notes | Status |
|---|---|---|
| **Hybrid CNN** (waveform + RR features) | Implemented: the best deep model in the ablation | ✅ shipped |
| **ResNet-1D / InceptionTime** | Deeper residual or multi-scale kernels; standard for PTB-XL | Likely to overfit 22 patients without far more data or pretraining |
| **Multi-beat context models** (e.g. 3-beat windows, BiLSTM/GRU or a Transformer over a sequence of beat embeddings) | Learns rhythm directly instead of through hand-made RR features | **High value**: closer to how cardiologists read rhythm strips |
| **Temporal Convolutional Network over the continuous signal** | Sequence labelling with no fixed beat windows; can also detect QRS | Medium value; needs a different label format |
| **Patient-template input** (beat − median of the patient's previous 16 beats) | A simple form of "compare with this patient's normal beat" | ✅ shipped (HybridCNN-rel; best in 5-fold CV) |
| **Siamese / prototype networks** | Learn the comparison against the patient's dominant beat instead of hand-coding the subtraction | **High value**: natural extension of the shipped template channel |
| **Foundation ECG models** (self-supervised pretraining on large unlabelled ECG corpora, then fine-tuning) | Contrastive (SimCLR/CLOCS), masked-signal or HuBERT-style pretraining; published open ECG foundation models exist | Best long-term option; needs large external datasets (PTB-XL, CODE, Icentia11k) and GPU time |

### 2.3 Training techniques
| Technique | Expected effect |
|---|---|
| **Focal loss / LDAM / logit adjustment** instead of inverse-frequency weights | Better precision–recall balance for S/F; less over-prediction of minority classes |
| **Class-balanced or patient-balanced sampling** | Stops record 208 dominating F, and stops high-beat-count patients dominating N |
| **Mixup / CutMix on beats** | Smoother decision boundaries; mixed evidence so far in ECG work |
| **Grouped k-fold model selection** | ✅ shipped (5-fold, pooled OOF macro-F1). Next: nested CV for an unbiased estimate of the selection itself, and leave-one-patient-out |
| **Seed ensembles / snapshot ensembles** | ✅ shipped: the recommended model is a 3-seed ensemble. Snapshot or cross-fold ensembles (the 5 CV-fold models) are a cheap next step |
| **Patient-specific adaptation** (fine-tune on the first ~5 minutes of each new patient's labelled beats, as in AAMI's "local" protocol) | Large published gains on S/V; needs an annotation workflow, a realistic setup for Holter review |
| **Semi-supervised / test-time adaptation** (pseudo-labels or entropy minimisation on an unlabelled new patient) | No labels needed from the new patient; has to be guarded against confirmation bias |

### 2.4 Calibration and uncertainty
* ✅ **Shipped (iteration 5): temperature scaling and BCTS** fitted on DS1 out-of-fold predictions; BCTS cuts DS2 NLL 0.477 → 0.158. Next: Dirichlet / matrix scaling, and calibration that keeps the S/F decision threshold.
* ✅ **Shipped: Mondrian split-conformal prediction sets** (α = 0.1). DS2 coverage 0.86 (N 0.845) because unseen patients are not exchangeable with DS1. Next: patient-weighted or adaptive conformal (e.g. online recalibration per patient) to restore the guarantee under shift.
* **Monte Carlo dropout / deep ensembles**, used to flag low-confidence beats for human review.

---

## 3. Data roadmap
| Dataset | Use |
|---|---|
| **INCART 12-lead Arrhythmia DB** (PhysioNet, 75 records, 257 Hz) | **External validation**: the single most important missing evaluation. Needs resampling to 360 Hz and lead matching |
| **MIT-BIH Supraventricular Arrhythmia DB** | More S beats, directly attacks the weakest class |
| **European ST-T, AHA DB** | Additional V/F examples, different hardware |
| **PTB-XL, Chapman-Shaoxing, CODE-15%** | Large 12-lead datasets for pretraining (record-level labels) |
| **Icentia11k** | ~11,000 patients of single-lead wearable data with beat labels; ideal for pretraining a beat model |

Every new dataset needs the same treatment as MIT-BIH: patient-level splits, a data-quality report and lineage counts.

---

## 4. Pipeline and engineering
| Item | Detail |
|---|---|
| **QRS detection** | The pipeline currently uses expert-annotated R-peaks. Add `wfdb.processing.XQRS` or Pan-Tompkins and report end-to-end metrics **including detection errors** (AAMI requires Se/+P for QRS detection) |
| **Multi-lead input** | Use the second channel (V1/V2/V4/V5) as a second CNN input channel; lead availability differs by record |
| **Streaming inference** | A `/predict-stream` endpoint that accepts a continuous signal, runs QRS detection, keeps the per-patient running mean RR server-side, and emits beat labels |
| **Experiment tracking** | MLflow or Weights & Biases for runs, params and artefacts; DVC for data versioning |
| **Model registry and promotion** | Promote a model only if the validation gate passes and DS2 does not regress; store model card, hash and metrics together |
| **Drift monitoring** | Log feature distributions (RR, amplitude) per request batch (aggregates only, never raw signals) and alert on PSI/KS drift |
| **Model export** | TorchScript/ONNX for the API, with INT8 quantisation for edge/wearable deployment (the CNN has ~44k parameters, which suits microcontrollers) |
| **Load testing and SLOs** | Locust/k6 against `/predict`; p95 latency targets; horizontal scaling behind a gateway |
| **Security** | Auth on the API (OAuth2/JWT), rate limiting, CORS restricted to known origins, signed model artefacts |

---

## 5. Path toward real clinical use (out of scope, documented for completeness)

1. **Interoperability:** ingest via HL7 FHIR (`Observation` with `SampledData`) or DICOM-ECG; return results as FHIR `Observation`s with provenance.
2. **Privacy:** HIPAA-compliant hosting with a BAA, encryption, audit logs, de-identification before research use, IRB approval. See the model card.
3. **Quality and regulatory:** design controls under ISO 13485, software life cycle per IEC 62304, risk management per ISO 14971, and AAMI EC57 / IEC 60601-2-47 performance testing. Arrhythmia-detection software is a regulated medical device (Software as a Medical Device); in the US that usually means a 510(k) with a predicate device, plus a Predetermined Change Control Plan if models will be retrained.
4. **Clinical validation:** a prospective, multi-site study on the intended population and device, with subgroup analysis (age, sex, ethnicity, comorbidities, device type).
5. **Human factors:** clinician-in-the-loop review UI, alarm-fatigue analysis, and clear communication of uncertainty.

---

## 6. Prioritised next steps

| # | Item | Effort | Expected impact |
|---|---|---|---|
| 1 | External validation on INCART | M | Shows whether any of this generalises; essential for credibility |
| 2 | Learned patient-relative comparison (Siamese/prototype), extending the shipped template channel | M | Further gains on S/V/F for unseen patients |
| 3 | Multi-beat sequence model (beat embeddings + BiGRU) | M | Learns rhythm end to end |
| 4 | ~~Temperature scaling + split-conformal prediction sets~~ ✅ shipped (iteration 5) | S | Calibrated, safer outputs in the API |
| 5 | Focal loss + patient-balanced sampling | S | Fewer N→S false positives |
| 6 | MiniRocket baseline | S | Strong, cheap comparison point |
| 7 | QRS detector integration + end-to-end metrics | M | Removes reliance on expert R-peaks |
| 8 | Self-supervised pretraining on Icentia11k | L | Biggest long-term gain |
