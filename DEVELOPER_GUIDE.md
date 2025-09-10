# Developer Guide — PredictRisk (Educational)

This guide documents data, training, artifacts, triage logic, runtime risk engine, deployment, and testing.

## Scope & Disclaimer
- Purpose: **awareness & early care-seeking** using **non-lab** inputs.
- Not a medical device. Not diagnostic. Models trained on **synthetic** data.

## Deployable layout (keep repo small)

```
app.py                        # Streamlit app (or app_lightweight.py)
artifacts/                    # tiny model artifacts used at runtime
  afib_beta_summary.npz
  heart_disease_beta_summary.npz
  heart_failure_beta_summary.npz
  hypertension_beta_summary.npz
  pad_beta_summary.npz
  stroke_beta_summary.npz
logo.png
requirements.txt
runtime.txt                   # 3.12
.streamlit/
  config.toml                 # theme
README.md
DEVELOPER_GUIDE.md
```

> Exclude datasets / notebooks / heavy training code from deploy branches to stay within GitHub/Streamlit limits.

## Features / Encoding
- Numeric: `age, sex(0/1), bmi, systolic_bp, diastolic_bp, heart_rate, sleep_hours, stress_score, physical_activity(0/1), family_history_heart_disease(0/1), diabetes_history(0/1), kidney_disease(0/1), substance_abuse(0/1)`
- Categorical (baseline first):
  - `smoking_status`: baseline **Never**; dummies **Former**, **Current**
  - `alcohol_use`: baseline **None**; dummies **Moderate**, **Excessive**

## Targets (one-vs-rest)
`risk_stroke`, `risk_heart_disease`, `risk_hypertension`, `risk_heart_failure`, `risk_afib`, `risk_pad`.

## Training (Laplace logistic)
- Fit logistic regression per condition on the features above.
- Compute MAP and Hessian → **Laplace approximation**: `β ~ N(μ, Σ)`.
- Save `artifacts/<cond>_beta_summary.npz` with keys: `mean`, `cov`, `feature_order`.
- Artifacts are **KB-sized**, ideal for Streamlit.

## Runtime risk engine
Given user vector `x`:
1. Sample `β^(s) ~ N(μ, Σ)` for `s = 1…S` (S≈3000–5000).
2. Compute `p^(s) = σ(β^(s)·x)` with `σ(z)=1/(1+e^{-z})`.
3. Report `p̄ = mean(p^(s))` as **risk %** and the 2.5–97.5th percentiles as the **95% credible interval**.
4. Categorize:
   - **Low:** `p̄ < 0.15`
   - **Moderate:** `0.15 ≤ p̄ < 0.40`
   - **High:** `p̄ ≥ 0.40`
5. Stability: symmetrize Σ (`0.5*(Σ+Σᵀ)`), add εI if needed.

## Safety Check (triage) — independent of risk
- **Emergency:** SBP ≥180 or DBP ≥110; severe chest pain + difficulty breathing; stroke-like signs.
- **Urgent:** SBP ≥140 or DBP ≥90; HR <50 or >120; concerning symptom clusters.
- **Routine:** none of the above.  
Overall Recommendation = conservative combination (**triage first**, then risk).

## UI
- Inputs start blank with placeholders; validation prevents empty assessment.
- Red **Assess Risk** primary button.
- Always show **Risk %**, **category**, **95% CI**, and a progress bar.
- Mismatch explainer if triage is Urgent/Emergency but risk is Low.
- Guidance composed from triage/risk + factors (BP/HR, BMI, smoking, diabetes/CKD).

## PDF
- Header: logo + title + timestamp
- Safety Check (urgency + reasons + BP/HR category)
- Condition Risk (% + category + CI)
- Overall Recommendation
- Input Summary
- Clinical Guidance
- **Footer-only disclaimer** + © Taiwo Michael Ayeni

## Performance
- `draws=3000` is usually enough; 5000 for extra smoothness.
- For further speed, cache a Cholesky factor of Σ across conditions.

## Testing profiles (smoke tests)
- **A (Low, Routine):** Age 35, SBP/DBP 118/76, HR 72, BMI 23, non-smoker, no symptoms → Low; Routine.
- **B (Moderate, Urgent):** Age 58, SBP/DBP 150/92, HR 96, BMI 29, Former smoker, chest pain + cold sweat → Moderate; Urgent.
- **C (High, Emergency):** Age 70, SBP/DBP 182/112, HR 126, BMI 31, Current smoker, severe chest pain + difficulty breathing → High; Emergency.

## Versioning & releases
- Semantic Versioning (MAJOR.MINOR.PATCH). Current version: see `VERSION`.
- Tag releases: `git tag vX.Y.Z && git push --tags`.

## Ethics
- Synthetic data; educational use only.
- Keep disclaimers prominent; don’t present as diagnostic.
