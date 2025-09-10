# PredictRisk — Cardiovascular Diagnostic Tool (Educational)

**PredictRisk** helps people understand cardiovascular risk and **when to seek care** using **non-lab** inputs (vitals, history, symptoms).  
It is **not** a diagnosis and does **not** replace clinical care.

> © Taiwo Michael Ayeni — Educational and awareness tool only.

---

## What the app shows

- **Safety Check (triage):** Uses **blood pressure, heart rate, and red-flag symptoms** to recommend how quickly to seek care.
  - **Emergency:** BP ≥ **180/110 mmHg** or stroke-like signs / severe chest pain + difficulty breathing  
  - **Urgent:** BP ≥ **140/90**, HR <50 or >120, or concerning symptom clusters  
  - **Routine:** none of the above
- **Condition Risk (you select one):** Bayesian logistic (Laplace) estimate shown as **risk %**, **95% credible interval**, and **Low / Moderate / High**.
  - Bands: **Low <15%**, **Moderate 15–39%**, **High ≥40%**
- **Clinical Guidance (educational):** concise, condition-aware suggestions (BP rechecks, rhythm review, cessation support, activity, weight, etc.).
- **PDF Report:** header with logo, Safety Check, condition risk, input summary, guidance, **footer-only disclaimer** + ©.

---

## Inputs (no lab tests)

Age, Sex, Height/Weight (BMI), Systolic/Diastolic BP, Heart Rate, Smoking (Never/Former/Current),
Alcohol (None/Moderate/Excessive), Physical activity, Sleep, Stress;
History (Family heart disease, Diabetes, CKD, Substance use);
Symptoms (chest pain, breathlessness, palpitations, dizziness/fainting, stroke signs, leg swelling, claudication).

---

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py     # or app_lightweight.py if that’s your filename
```

Python **3.12** (pinned in `runtime.txt`). No secrets required.

---

## Deploy on Streamlit Cloud

- **Repository:** `ayeni-T/PredictRisk`
- **Branch:** `revamp` (staging) or `main` (production)
- **Main file:** `app.py` (or `app_lightweight.py`)
- **Python:** 3.12 (via `runtime.txt`)

---

## How it works (short)

- One logistic model per condition; trained on **synthetic** non-lab data.
- **Laplace** posterior → tiny artifacts (μ, Σ). At runtime we sample coefficients to get risk% + 95% CI.
- **Safety Check is independent** and takes priority for the **Overall Recommendation**.

---

## Disclaimer

PredictRisk provides educational estimates and triage guidance only.  
It is **not** a diagnosis and does **not** replace clinical evaluation.  
If symptoms are severe or worsening, **seek immediate medical care**.
