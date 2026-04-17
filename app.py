# app.py
# PredictRisk: Cardiovascular Diagnostic Tool (educational)
# Revamp v3 — Prof. Yeh's suggestions implemented:
#   1. N/A / Unsure option for each covariate (excluded from prediction)
#   2. Credible interval plots — posterior predictive density + coefficient forest plot

import streamlit as st
import numpy as np
from pathlib import Path
from datetime import datetime
from textwrap import wrap
import io

APP_DIR = Path(__file__).parent

# --------------------------- Logo detection ---------------------------
LOGO_CANDIDATES = [
    APP_DIR / "logo.png",
    APP_DIR / "assets" / "logo.png",
    APP_DIR / "static" / "logo.png",
]

def get_logo_path_str():
    for p in LOGO_CANDIDATES:
        if p.exists():
            return str(p)
    return None

logo_path_str = get_logo_path_str()

# --------------------------- Page setup ---------------------------
st.set_page_config(
    page_title="PredictRisk: Cardiovascular Diagnostic Tool",
    page_icon=logo_path_str if logo_path_str else "🧠",
    layout="centered",
)

st.markdown("""
<style>
div.stButton > button[kind="primary"] { background-color:#d32f2f; color:white; border:0; border-radius:14px; padding:0.9rem 1.25rem; font-size:1.05rem; font-weight:600; width:100%; }
div.stButton > button[kind="primary"]:hover { background-color:#b71c1c; color:white; }
div.stButton > button:first-child { background-color:#d32f2f; color:white; border:0; border-radius:14px; padding:0.9rem 1.25rem; font-size:1.05rem; font-weight:600; width:100%; }
div.stButton > button:first-child:hover { background-color:#b71c1c; color:white; }
</style>
""", unsafe_allow_html=True)

# Header
if logo_path_str:
    col_logo, col_title = st.columns([1, 6])
    with col_logo:
        st.image(logo_path_str, use_container_width=True)
    with col_title:
        st.title("PredictRisk: Cardiovascular Diagnostic Tool")
else:
    st.title("🧠 PredictRisk: Cardiovascular Diagnostic Tool")

ARTIFACT_DIR = APP_DIR / "artifacts"

CONDITIONS = {
    "Stroke": "stroke",
    "Hypertension": "hypertension",
    "Heart Failure": "heart_failure",
    "Atrial Fibrillation (AFib)": "afib",
    "Peripheral Artery Disease (PAD)": "pad",
    "Angina": "angina",
    "Myocardial Infarction (MI)": "myocardial_infarction",
    "Arrhythmia": "arrhythmia",
    "Cardiomyopathy": "cardiomyopathy",
}

# --------------------------- Artifact loader ---------------------------
@st.cache_resource
def load_artifact(cond_key: str):
    npz_path = ARTIFACT_DIR / f"{cond_key}_beta_summary.npz"
    if not npz_path.exists():
        st.error(f"Artifact not found: {npz_path}. Train models first or place NPZs in ./artifacts")
        st.stop()
    pkg = np.load(npz_path, allow_pickle=True)
    mu = pkg["mean"].astype("float32")
    cov = pkg["cov"].astype("float32")
    order_arr = pkg["feature_order"]
    try:
        order = [str(x) for x in order_arr.tolist()]
    except Exception:
        order = [str(x) for x in order_arr]
    return mu, cov, order

# --------------------------- Helpers ---------------------------
def categorize_bp(sbp: int, dbp: int):
    if sbp >= 180 or dbp >= 110:
        return "Hypertensive Crisis", "red", "≥180/110"
    if sbp >= 140 or dbp >= 90:
        return "Stage 2 Hypertension", "red", "≥140 or ≥90"
    if (130 <= sbp <= 139) or (80 <= dbp <= 89):
        return "Stage 1 Hypertension", "orange", "130–139 or 80–89"
    if (120 <= sbp <= 129) and dbp < 80:
        return "Elevated", "gold", "SBP 120–129 & DBP <80"
    if sbp < 120 and dbp < 80:
        return "Normal", "green", "<120/<80"
    return "Unclassified", "gray", "Check values"

def categorize_hr(hr: int):
    if hr < 50:
        return "Bradycardia (Marked)", "orange", "<50"
    if 50 <= hr < 60:
        return "Bradycardia", "gold", "50–59"
    if 60 <= hr <= 100:
        return "Normal", "green", "60–100"
    if 100 < hr <= 120:
        return "Tachycardia", "gold", "101–120"
    if hr > 120:
        return "Tachycardia (Marked)", "red", ">120"
    return "Unclassified", "gray", "Check values"

def assess_clinical_urgency_enhanced(sbp: int, dbp: int, hr: int, symptoms: list):
    urgency = "routine"
    reasons, tags = [], []
    S = set(symptoms)

    if sbp >= 180 or dbp >= 110:
        urgency = "emergency"; reasons.append("Hypertensive crisis (≥180/110)"); tags.append("Hypertension")
    elif sbp >= 140 or dbp >= 90:
        if urgency == "routine": urgency = "urgent"
        reasons.append("Elevated BP (≥140/90)"); tags.append("Hypertension")

    if hr > 120:
        if urgency == "routine": urgency = "urgent"
        reasons.append("Tachycardia (>120 bpm)"); tags.append("Arrhythmia")
    elif hr < 50:
        if urgency == "routine": urgency = "urgent"
        reasons.append("Bradycardia (<50 bpm)"); tags.append("Arrhythmia")

    if "severe_chest_pain" in S or "difficulty_breathing" in S:
        urgency = "emergency"; reasons.append("Severe chest pain or difficulty breathing"); tags.append("Possible MI/HF")
    if {"weak_limb","slurred_speech","face_droop"} & S:
        urgency = "emergency"; reasons.append("Possible stroke symptoms"); tags.append("Stroke")
    if {"chest_pain","sudden_sweating","cold_sweat","lightheadedness","dizziness","nausea","vomiting"} & S:
        if urgency == "routine": urgency = "urgent"
        reasons.append("Ischemic symptoms"); tags.append("Possible Angina/MI")
    if {"shortness_breath","orthopnea","persistent_cough","leg_swelling","facial_swelling"} & S:
        if urgency == "routine": urgency = "urgent"
        reasons.append("Heart failure symptoms"); tags.append("Heart Failure")
    if {"palpitations","fainting","dizziness"} & S:
        if urgency == "routine": urgency = "urgent"
        reasons.append("Arrhythmia symptoms"); tags.append("Arrhythmia")
    if "claudication" in S:
        reasons.append("Claudication (PAD screening)"); tags.append("PAD")

    return urgency, sorted(set(reasons)), sorted(set(tags))

def predict_prob(mu, cov, x_vec, draws=5000, seed=0):
    rng = np.random.default_rng(seed)
    cov = np.asarray(cov, dtype="float64")
    cov = 0.5 * (cov + cov.T)
    eps = 1e-8
    for _ in range(3):
        try:
            draws_mat = rng.multivariate_normal(mu, cov, size=draws)
            break
        except np.linalg.LinAlgError:
            cov = cov + np.eye(len(mu)) * eps
            eps *= 10
    lin = (draws_mat @ x_vec).astype("float32")
    p = 1.0 / (1.0 + np.exp(-lin))
    return float(p.mean()), float(np.percentile(p, 2.5)), float(np.percentile(p, 97.5)), p  # <-- also return samples

PRETTY_FEATURE = {
    "age": "Age",
    "sex": "Male sex",
    "bmi": "BMI",
    "physical_activity": "Physically active",
    "systolic_bp": "Systolic BP",
    "diastolic_bp": "Diastolic BP",
    "heart_rate": "Heart rate",
    "sleep_hours": "Sleep (hours/night)",
    "stress_score": "Stress (1–10)",
    "family_history_heart_disease": "Family history of heart disease",
    "diabetes_history": "Diabetes (diagnosed)",
    "kidney_disease": "Chronic kidney disease",
    "substance_abuse": "Substance use",
    "smoking_status_former": "Smoking: Former",
    "smoking_status_current": "Smoking: Current",
    "alcohol_use_moderate": "Alcohol: Moderate",
    "alcohol_use_excessive": "Alcohol: Excessive",
}
def pretty_feature(n: str) -> str:
    n = n.replace("C(", "").replace(")", "").replace("[T.1]", "").replace("[T.2]", "")
    return PRETTY_FEATURE.get(n, n.replace("_", " ").title())

def value_for_feature(fname: str, base: dict, smoke_cat: int, alcohol_cat: int):
    if fname == "Intercept":
        return 1.0
    if fname in base:
        return float(base[fname])
    if fname == "smoking_status_former":   return 1.0 if smoke_cat == 1 else 0.0
    if fname == "smoking_status_current":  return 1.0 if smoke_cat == 2 else 0.0
    if fname == "alcohol_use_moderate":    return 1.0 if alcohol_cat == 1 else 0.0
    if fname == "alcohol_use_excessive":   return 1.0 if alcohol_cat == 2 else 0.0
    f = fname.lower()
    if "smok" in f:
        if ("t.1" in f) or ("former" in f):  return 1.0 if smoke_cat == 1 else 0.0
        if ("t.2" in f) or ("current" in f): return 1.0 if smoke_cat == 2 else 0.0
    if "alcohol" in f:
        if ("t.1" in f) or ("moderate" in f):  return 1.0 if alcohol_cat == 1 else 0.0
        if ("t.2" in f) or ("excessive" in f): return 1.0 if alcohol_cat == 2 else 0.0
    return 0.0

def risk_category(p: float):
    if p >= 0.40:  return "High", "red"
    if p >= 0.15:  return "Moderate", "orange"
    return "Low", "green"

def overall_recommendation(urgency: str, risk_cat: str):
    if urgency == "emergency":
        return ("Overall Recommendation: Seek emergency care now.", "red")
    if urgency == "urgent":
        return ("Overall Recommendation: Get prompt medical review (today–48h).", "orange")
    if risk_cat == "High":
        return ("Overall Recommendation: Book a clinician review soon and discuss risk reduction.", "orange")
    if risk_cat == "Moderate":
        return ("Overall Recommendation: Plan a routine check-in and address modifiable risks.", "gold")
    return ("Overall Recommendation: Maintain healthy habits and recheck periodically.", "green")

COND_GUIDE = {
    "stroke": {
        "High": [
            "Know FAST signs (Face droop, Arm weakness, Speech trouble); call for emergency care if symptoms occur.",
            "Discuss blood pressure control and whether antiplatelet therapy is appropriate."
        ],
        "Moderate": [
            "Review blood pressure goals, smoking cessation, and diabetes control with a clinician.",
            "Learn FAST signs and when to seek urgent care."
        ],
        "Low": ["Maintain BP <120/80, stay active, and know FAST signs."],
    },
    "hypertension": {
        "High": [
            "Record home BP (morning and evening for 1 week) and review targets with a clinician.",
            "Limit salt, maintain healthy weight, and follow a DASH-style diet."
        ],
        "Moderate": ["Re-check BP after 5 minutes rest; keep a log and discuss lifestyle changes."],
        "Low": ["Continue regular checks and healthy habits."],
    },
    "heart_failure": {
        "High": [
            "Discuss breathlessness, swelling, and daily weight tracking with a clinician.",
            "Review salt and fluid guidance; know when to seek urgent care."
        ],
        "Moderate": ["Check for swelling or night cough; review blood pressure control and activity plan."],
        "Low": ["Be aware of symptoms (leg swelling, breathlessness) and seek review if they develop."],
    },
    "afib": {
        "High": [
            "Discuss rhythm monitoring (ECG/ambulatory) and stroke prevention as advised by your clinician.",
            "Limit alcohol and stimulants; report palpitations, dizziness, or fainting."
        ],
        "Moderate": ["Learn pulse checks; review triggers (caffeine, alcohol) and thyroid evaluation if advised."],
        "Low": ["Know how to check pulse; seek review if it becomes irregular or fast."],
    },
    "pad": {
        "High": [
            "A supervised walking program and foot care are important; review smoking cessation if relevant.",
            "Discuss blood pressure, glucose control, and statin/antiplatelet use with a clinician."
        ],
        "Moderate": ["Start a gradual walking plan; maintain foot care; discuss preventive therapy if advised."],
        "Low": ["Stay active; note any calf pain that appears with walking and improves with rest."],
    },
    "angina": {
        "High": [
            "Discuss a chest pain action plan and review blood pressure and cholesterol management.",
            "Urgent assessment if chest pain is new, worsening, or occurs at rest."
        ],
        "Moderate": ["Assess exercise triggers and plan graded activity; review preventive medications if advised."],
        "Low": ["Maintain heart-healthy habits; know warning signs requiring urgent care."],
    },
    "myocardial_infarction": {
        "High": ["New/worsening chest pain with sweating or nausea requires emergency evaluation."],
        "Moderate": ["Discuss risk reduction (BP, lipids, smoking); know urgent symptoms."],
        "Low": ["Continue preventive habits and routine check-ins."],
    },
    "arrhythmia": {
        "High": ["Seek urgent review for palpitations with dizziness or fainting; ECG monitoring may be needed."],
        "Moderate": ["Limit alcohol/stimulants; learn pulse checks."],
        "Low": ["Know symptoms that warrant review (sustained rapid or irregular pulse)."],
    },
    "cardiomyopathy": {
        "High": ["Discuss breathlessness, swelling, and daily weights; urgent care if symptoms escalate."],
        "Moderate": ["Review BP control, activity plan, and medication adherence with a clinician."],
        "Low": ["Maintain heart-healthy lifestyle; seek review if new symptoms develop."],
    },
}

def compose_guidance(cond_key, urgency, risk_cat, bp_cat, hr_cat, bmi,
                     smoke_cat, alcohol_cat, physical_active_flag, selected_symptoms, histories):
    G = []
    if urgency == "emergency":
        G.append("Emergency symptoms detected — seek immediate medical care. Do not drive yourself.")
    elif urgency == "urgent":
        G.append("Seek prompt medical review (today–48 hours), especially if symptoms are new or worsening.")
    else:
        G.append("Arrange routine review with a clinician to discuss your cardiovascular risk profile.")

    for line in COND_GUIDE.get(cond_key, {}).get(risk_cat, []):
        G.append(line)

    if bp_cat != "Normal":
        G.append("Re-check blood pressure after 5 minutes of rest; keep a log and discuss with a clinician.")
    if hr_cat.startswith("Tachy") or hr_cat.startswith("Brady"):
        G.append("An ECG may help evaluate heart rhythm; discuss if palpitations, dizziness, or fainting occur.")
    if smoke_cat == 2:
        G.append("Support for smoking cessation can meaningfully reduce cardiovascular risk.")
    if alcohol_cat == 2:
        G.append("Reducing alcohol intake can help blood pressure and heart rhythm.")
    if physical_active_flag == 0.0:
        G.append("Aim for regular moderate activity if cleared (e.g., brisk walking).")
    if bmi >= 25:
        G.append("Heart-healthy nutrition and weight management can lower risk.")
    if histories.get("diabetes"):
        G.append("Keep diabetes well-controlled; review targets and medications with your clinician.")
    if histories.get("kidney"):
        G.append("Chronic kidney disease increases risk — ensure regular follow-up.")
    if histories.get("family_history"):
        G.append("With a family history of heart disease, discuss earlier or more frequent screening.")

    S = set(selected_symptoms)
    if {"chest_pain", "severe_chest_pain"} & S:
        G.append("Chest pain with sweating, nausea, or shortness of breath warrants urgent assessment.")
    if {"weak_limb", "slurred_speech", "face_droop"} & S:
        G.append("Stroke-like symptoms require emergency evaluation immediately.")

    seen = set(); out = []
    for s in G:
        if s not in seen:
            seen.add(s); out.append(s)
    return out

def parse_float(s):
    if s is None or s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None

def num_input_with_na(label, key, placeholder, help=None):
    """
    Number input with an N/A checkbox. Returns (value_or_None, is_na).
    """
    col_inp, col_na = st.columns([5, 1])
    with col_na:
        st.markdown("<div style='padding-top:28px'></div>", unsafe_allow_html=True)
        is_na = st.checkbox("N/A", key=f"na_{key}",
                            help="Check if this value is unavailable or unknown")
    with col_inp:
        if is_na:
            st.text_input(label, value="Not available / Unsure", disabled=True, key=f"dis_{key}")
            return None, True
        else:
            val = parse_float(st.text_input(label, key=f"val_{key}", placeholder=placeholder, help=help))
            return val, False

def select_with_placeholder_and_na(label, options, key, help=None):
    """
    Selectbox with N/A option. Returns (choice_or_None, is_na).
    """
    col_inp, col_na = st.columns([5, 1])
    with col_na:
        st.markdown("<div style='padding-top:28px'></div>", unsafe_allow_html=True)
        is_na = st.checkbox("N/A", key=f"na_{key}",
                            help="Check if this value is unavailable or unknown")
    with col_inp:
        if is_na:
            st.text_input(label, value="Not available / Unsure", disabled=True, key=f"dis_{key}")
            return None, True
        else:
            display = ["— Select —"] + options
            choice = st.selectbox(label, display, index=0, key=f"sel_{key}", help=help)
            val = None if choice == "— Select —" else choice
            return val, False

# ════════════════════════════════════════════════════════════════════
# CREDIBLE INTERVAL PLOT FUNCTION  (Prof. Zhao suggestion #2)
# ════════════════════════════════════════════════════════════════════
def render_ci_plots(prob_samples: np.ndarray, mean_p: float, lo: float, hi: float,
                    mu: np.ndarray, cov: np.ndarray, order: list, x_vec: np.ndarray,
                    missing_fields: list, condition_label: str):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from scipy.stats import gaussian_kde

    with st.expander("📊 Credible Interval Plots", expanded=True):
        st.caption(
            "These plots visualise the uncertainty in your risk estimate. "
            "The wider the distribution, the less certain the model is — "
            "this is especially relevant when some covariates are unavailable."
        )

        if missing_fields:
            st.info(
                f"⚠️ **{len(missing_fields)} covariate(s) marked N/A** ({', '.join(missing_fields)}) "
                "were excluded from the prediction. The credible interval below is therefore "
                "wider than it would be with complete data."
            )

        # ── Plot 1: Posterior predictive density ────────────────────────────
        st.markdown("#### Posterior Predictive Distribution of Risk")
        st.caption(
            "Each value on the x-axis is a plausible risk estimate given the patient profile "
            "and uncertainty in model coefficients. The red shaded band is the 95% credible interval."
        )

        fig1, ax1 = plt.subplots(figsize=(8, 3.2))
        ax1.hist(prob_samples, bins=80, density=True, color="#3498db", alpha=0.4)

        xs = np.linspace(max(0, lo - 0.05), min(1.0, hi + 0.05), 500)
        try:
            kde = gaussian_kde(prob_samples, bw_method=0.15)
            ax1.plot(xs, kde(xs), color="#2980b9", lw=2)
            ci_xs = np.linspace(lo, hi, 300)
            ax1.fill_between(ci_xs, kde(ci_xs), alpha=0.35, color="#e74c3c", label="95% Credible Interval")
        except Exception:
            pass

        ax1.axvline(mean_p, color="#e74c3c", lw=2, linestyle="--", label=f"Mean = {mean_p:.1%}")
        ax1.set_xlabel(f"Predicted Risk Probability — {condition_label}", fontsize=10)
        ax1.set_ylabel("Density", fontsize=10)
        ax1.set_xlim(0, 1)
        ax1.legend(fontsize=9)
        ax1.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)

        # ── Plot 2: Coefficient forest plot ─────────────────────────────────
        st.markdown("#### Coefficient Credible Intervals (log-odds scale)")
        st.caption(
            "Each row shows the posterior mean ± 95% credible interval for a model coefficient. "
            "🔴 Red = risk-increasing (CI entirely above 0). 🔵 Blue = protective (CI entirely below 0). "
            "Grey = direction uncertain (CI crosses 0). Only covariates present in the artifact are shown."
        )

        # Extract per-feature posterior marginals from the covariance diagonal
        cov_arr = np.asarray(cov, dtype="float64")
        sds = np.sqrt(np.diag(cov_arr))
        mu_arr = np.asarray(mu, dtype="float64")

        # Skip Intercept for the plot; show only non-intercept features
        plot_indices = [i for i, n in enumerate(order) if n.lower() != "intercept"]
        if not plot_indices:
            st.info("No non-intercept features to display.")
            return

        feat_labels = [pretty_feature(order[i]) for i in plot_indices]
        means_f = [mu_arr[i] for i in plot_indices]
        sds_f   = [sds[i]    for i in plot_indices]
        ci_lo_f = [m - 1.96 * s for m, s in zip(means_f, sds_f)]
        ci_hi_f = [m + 1.96 * s for m, s in zip(means_f, sds_f)]

        colors = [
            "#e74c3c" if lo_v > 0 else "#3498db" if hi_v < 0 else "#95a5a6"
            for lo_v, hi_v in zip(ci_lo_f, ci_hi_f)
        ]

        n_feat = len(plot_indices)
        fig2, ax2 = plt.subplots(figsize=(7, max(3.5, 0.5 * n_feat)))

        for i, (lo_v, hi_v, m_v, c) in enumerate(zip(ci_lo_f, ci_hi_f, means_f, colors)):
            ax2.plot([lo_v, hi_v], [i, i], color=c, lw=2.5, solid_capstyle="round")
            ax2.scatter(m_v, i, color=c, zorder=5, s=55)

        ax2.axvline(0, color="black", lw=1, linestyle="--", alpha=0.5)
        ax2.set_yticks(range(n_feat))
        ax2.set_yticklabels(feat_labels, fontsize=8)
        ax2.set_xlabel("Posterior Coefficient (log-odds)", fontsize=10)
        ax2.set_title(f"Model Coefficient Credible Intervals — {condition_label}", fontsize=10, fontweight="bold")

        red_p  = mpatches.Patch(color="#e74c3c", label="Risk-increasing (CI > 0)")
        blue_p = mpatches.Patch(color="#3498db", label="Protective (CI < 0)")
        grey_p = mpatches.Patch(color="#95a5a6", label="Uncertain direction")
        ax2.legend(handles=[red_p, blue_p, grey_p], fontsize=8, loc="lower right")
        ax2.spines[["top", "right"]].set_visible(False)
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)


# ════════════════════════════════════════════════════════════════════════════
# MAIN UI
# ════════════════════════════════════════════════════════════════════════════

st.markdown("### Select a cardiovascular condition to assess:")
condition_label = st.selectbox("", list(CONDITIONS.keys()))
cond_key = CONDITIONS[condition_label]

st.header("Enter your health details")
st.caption(
    "💡 **N/A** checkboxes: If a value is unavailable or unknown, tick N/A. "
    "That covariate will be excluded from the prediction, and the result will indicate "
    "how many variables were used. The credible interval will widen accordingly."
)

missing_fields = []   # tracks human-readable names of N/A fields

col1, col2 = st.columns(2)
with col1:
    age, age_na       = num_input_with_na("Age (years)", "age", "e.g., 50", help="Your age in completed years.")
    sex_val, sex_na   = select_with_placeholder_and_na("Sex", ["Female", "Male"], "sex",
                                                        help="Biological sex assigned at birth.")
    height_m, h_na    = num_input_with_na("Height (m)", "height_m", "e.g., 1.70", help="Measured height in metres.")
    weight_kg, w_na   = num_input_with_na("Weight (kg)", "weight_kg", "e.g., 75", help="Body weight in kilograms.")

with col2:
    systolic_bp, sbp_na  = num_input_with_na("Systolic BP (mmHg)", "sbp", "e.g., 120",
                                              help="Top number; seated, arm supported.")
    diastolic_bp, dbp_na = num_input_with_na("Diastolic BP (mmHg)", "dbp", "e.g., 80",
                                              help="Bottom number; pressure between beats.")
    heart_rate, hr_na    = num_input_with_na("Heart Rate (bpm)", "hr", "e.g., 75",
                                              help="Resting beats per minute.")

# BMI badge
bmi = None
if not h_na and not w_na and height_m and weight_kg and height_m > 0:
    bmi = weight_kg / (height_m ** 2)
    if bmi >= 30:   st.error(f"BMI: {bmi:.1f} kg/m² — OBESE")
    elif bmi >= 25: st.warning(f"BMI: {bmi:.1f} kg/m² — OVERWEIGHT")
    else:           st.success(f"BMI: {bmi:.1f} kg/m² — Normal")

# BP badge
if not sbp_na and not dbp_na and systolic_bp is not None and diastolic_bp is not None:
    bp_cat, bp_color, bp_note = categorize_bp(int(systolic_bp), int(diastolic_bp))
    msg = f"BP: {int(systolic_bp)}/{int(diastolic_bp)} mmHg — {bp_cat} ({bp_note})"
    if bp_color == "green":          st.success(msg)
    elif bp_color in ("gold","orange"): st.warning(msg)
    elif bp_color == "red":          st.error(msg)
    else:                            st.info(msg)

# HR badge
if not hr_na and heart_rate is not None:
    hr_cat, hr_color, hr_note = categorize_hr(int(heart_rate))
    msg = f"Heart Rate: {int(heart_rate)} bpm — {hr_cat} ({hr_note})"
    if hr_color == "green":             st.success(msg)
    elif hr_color in ("gold","orange"): st.warning(msg)
    elif hr_color == "red":             st.error(msg)
    else:                               st.info(msg)

st.subheader("Lifestyle & Risk Factors")
col3, col4 = st.columns(2)
with col3:
    smoking_status_lbl, smoke_na = select_with_placeholder_and_na(
        "Smoking status", ["Never", "Former", "Current"], "smoke",
        help="Current: smoke now; Former: smoked before but not now; Never: never smoked.")
    alcohol_use_lbl, alc_na = select_with_placeholder_and_na(
        "Alcohol use", ["None", "Moderate", "Excessive"], "alcohol",
        help="None; Moderate: occasional/≤1 drink a day; Excessive: frequent/heavy.")
    physical_activity, pa_na = select_with_placeholder_and_na(
        "Physically active?", ["Yes", "No"], "active",
        help="Regular moderate activity ≈150+ minutes/week or as advised.")
with col4:
    sleep_hours, sl_na   = num_input_with_na("Sleep (hours/night)", "sleep", "e.g., 7.0",
                                              help="Average nightly sleep duration.")
    stress_score, ss_na  = num_input_with_na("Stress (1–10)", "stress", "e.g., 5",
                                              help="Your perceived stress today; 1=low, 10=high.")

st.subheader("Medical History")
col5, col6 = st.columns(2)
with col5:
    family_history_heart, fh_na = select_with_placeholder_and_na(
        "Family history of heart disease?", ["No", "Yes"], "fh",
        help="Parent, brother, or sister with heart disease.")
    diabetes_history, dm_na = select_with_placeholder_and_na(
        "Diabetes (diagnosed)?", ["No", "Yes"], "dm",
        help="Previously diagnosed by a clinician.")
with col6:
    kidney_disease, ckd_na = select_with_placeholder_and_na(
        "Chronic kidney disease?", ["No", "Yes"], "ckd",
        help="Previously diagnosed CKD.")
    substance_abuse, sub_na = select_with_placeholder_and_na(
        "Substance abuse?", ["No", "Yes"], "sub",
        help="Problematic use of drugs/substances.")

st.subheader("Current Symptoms")
colA, colB, colC = st.columns(3)
with colA:
    chest_pain         = st.checkbox("Chest pain",            help="Tightness/pressure in the chest.")
    severe_chest_pain  = st.checkbox("Severe chest pain",     help="Intense, persistent pain; not eased by rest.")
    shortness_breath   = st.checkbox("Shortness of breath",   help="Breathless at rest or minimal activity.")
    lightheadedness    = st.checkbox("Lightheadedness",       help="Feeling faint or woozy.")
with colB:
    difficulty_breathing = st.checkbox("Difficulty breathing", help="Laboured breathing; hard to speak full sentences.")
    palpitations         = st.checkbox("Heart palpitations",   help="Awareness of fast or irregular heartbeat.")
    dizziness            = st.checkbox("Dizziness",            help="Spinning sensation or imbalance.")
    fainting             = st.checkbox("Fainting episodes",    help="Blackouts or sudden loss of consciousness.")
with colC:
    leg_swelling     = st.checkbox("Leg/ankle swelling",             help="Swelling (oedema) in lower legs/ankles.")
    persistent_cough = st.checkbox("Persistent cough",               help="Cough worse at night or lying down.")
    face_droop       = st.checkbox("Face droop (one-sided)",         help="Drooping on one side of the face.")
    slurred_speech   = st.checkbox("Slurred speech",                 help="Words sound unclear or garbled.")
    weak_limb        = st.checkbox("Weakness in arm/leg",            help="Sudden weakness or numbness in a limb.")
    cold_sweat       = st.checkbox("Cold sweat",                     help="Profuse sweating not due to heat/exercise.")
    sudden_sweating  = st.checkbox("Sudden sweating",                help="Unexpected sudden onset of sweating.")
    orthopnea        = st.checkbox("Orthopnea (worse lying down)",   help="Shortness of breath when lying flat.")
    claudication     = st.checkbox("Cramping leg pain with walking", help="Leg cramps during walking that improve with rest.")

# ── Assess button ─────────────────────────────────────────────────────────
# Build base dict from non-NA fields only
def build_base_and_missing():
    base = {}
    missing = []

    if age_na:        missing.append("Age")
    else:             base["age"] = float(age) if age is not None else None

    sex_flag = None
    if sex_na:        missing.append("Sex")
    elif sex_val:     sex_flag = 1.0 if sex_val == "Male" else 0.0

    bmi_val = None
    if h_na or w_na:  missing.append("Height/Weight (BMI)")
    elif height_m and weight_kg and height_m > 0:
        bmi_val = weight_kg / height_m ** 2

    if sbp_na:        missing.append("Systolic BP")
    else:             base["systolic_bp"] = float(systolic_bp) if systolic_bp is not None else None

    if dbp_na:        missing.append("Diastolic BP")
    else:             base["diastolic_bp"] = float(diastolic_bp) if diastolic_bp is not None else None

    if hr_na:         missing.append("Heart Rate")
    else:             base["heart_rate"] = float(heart_rate) if heart_rate is not None else None

    if sl_na:         missing.append("Sleep hours")
    else:             base["sleep_hours"] = float(sleep_hours) if sleep_hours is not None else None

    if ss_na:         missing.append("Stress score")
    else:             base["stress_score"] = float(stress_score) if stress_score is not None else None

    if not sex_na and sex_flag is not None:
        base["sex"] = sex_flag
    if bmi_val is not None:
        base["bmi"] = bmi_val

    smoke_cat, alcohol_cat = 0, 0
    if smoke_na:      missing.append("Smoking status")
    elif smoking_status_lbl:
        smoke_cat = {"Never": 0, "Former": 1, "Current": 2}[smoking_status_lbl]

    if alc_na:        missing.append("Alcohol use")
    elif alcohol_use_lbl:
        alcohol_cat = {"None": 0, "Moderate": 1, "Excessive": 2}[alcohol_use_lbl]

    if pa_na:         missing.append("Physical activity")
    elif physical_activity:
        base["physical_activity"] = 1.0 if physical_activity == "Yes" else 0.0

    if fh_na:         missing.append("Family history")
    elif family_history_heart:
        base["family_history_heart_disease"] = 1.0 if family_history_heart == "Yes" else 0.0

    if dm_na:         missing.append("Diabetes history")
    elif diabetes_history:
        base["diabetes_history"] = 1.0 if diabetes_history == "Yes" else 0.0

    if ckd_na:        missing.append("Kidney disease")
    elif kidney_disease:
        base["kidney_disease"] = 1.0 if kidney_disease == "Yes" else 0.0

    if sub_na:        missing.append("Substance abuse")
    elif substance_abuse:
        base["substance_abuse"] = 1.0 if substance_abuse == "Yes" else 0.0

    return base, smoke_cat, alcohol_cat, bmi_val if bmi_val else 0.0, missing

# Require at least the core safety fields (BP and HR) unless NA'd
def can_assess(base, sbp_na, dbp_na, hr_na, sbp, dbp, hr):
    """At minimum we need either the vitals or their NA flag acknowledged."""
    vitals_ok = (sbp_na or sbp is not None) and (dbp_na or dbp is not None) and (hr_na or hr is not None)
    has_some = len(base) > 0 or sbp_na or dbp_na or hr_na
    return vitals_ok and has_some

base_check, _, _, _, _ = build_base_and_missing()
ready = can_assess(base_check, sbp_na, dbp_na, hr_na, systolic_bp, diastolic_bp, heart_rate)

if not ready:
    st.warning("Please complete or mark as N/A the blood pressure and heart rate fields to assess.")
else:
    if st.button("🔴 Assess Risk", type="primary", use_container_width=True):
        base, smoke_cat, alcohol_cat, bmi_val, missing_fields = build_base_and_missing()

        # Symptoms
        selected_symptoms = [
            name for name, flag in [
                ("chest_pain", chest_pain), ("severe_chest_pain", severe_chest_pain),
                ("shortness_breath", shortness_breath), ("difficulty_breathing", difficulty_breathing),
                ("palpitations", palpitations), ("dizziness", dizziness), ("fainting", fainting),
                ("leg_swelling", leg_swelling), ("persistent_cough", persistent_cough),
                ("face_droop", face_droop), ("slurred_speech", slurred_speech), ("weak_limb", weak_limb),
                ("cold_sweat", cold_sweat), ("sudden_sweating", sudden_sweating),
                ("lightheadedness", lightheadedness), ("orthopnea", orthopnea), ("claudication", claudication),
            ] if flag
        ]

        # Safety check uses available vitals (or 120/80/75 defaults if N/A'd)
        sbp_safe = int(systolic_bp) if not sbp_na and systolic_bp else 120
        dbp_safe = int(diastolic_bp) if not dbp_na and diastolic_bp else 80
        hr_safe  = int(heart_rate)   if not hr_na  and heart_rate  else 75
        urgency, reasons, tags = assess_clinical_urgency_enhanced(sbp_safe, dbp_safe, hr_safe, selected_symptoms)

        # Risk prediction
        mu, cov, order = load_artifact(CONDITIONS[condition_label])
        x_vec = np.array(
            [value_for_feature(name, base, smoke_cat, alcohol_cat) for name in order],
            dtype="float32"
        )
        mean_p, lo, hi, prob_samples = predict_prob(mu, cov, x_vec, draws=5000)
        cat, cat_color = risk_category(mean_p)

        # Summary banner
        rec_text, rec_color = overall_recommendation(urgency, cat)
        if rec_color == "red":    st.error(rec_text)
        elif rec_color == "orange": st.warning(rec_text)
        elif rec_color == "gold":   st.info(rec_text)
        else:                       st.success(rec_text)

        # Missing covariates notice
        n_total = len([k for k in order if k.lower() != "intercept"])
        n_used  = n_total - len(missing_fields)
        if missing_fields:
            st.warning(
                f"**Partial data:** Prediction used **{n_used}/{n_total}** covariates. "
                f"Excluded (marked N/A): {', '.join(missing_fields)}. "
                "The credible interval below reflects increased uncertainty."
            )

        # Side-by-side panels
        left, right = st.columns(2)
        with left:
            icons = {"routine": "🟢", "urgent": "🟡", "emergency": "🔴"}
            st.subheader("Safety Check (independent of risk)")
            st.markdown(f"**Clinical Urgency:** {icons.get(urgency,'⚪')} {urgency.upper()}")
            st.caption("Safety Check recommends how quickly to seek care; the risk score is for the selected condition only.")
            if reasons: st.caption("Reasons: " + "; ".join(reasons))

        with right:
            st.subheader(f"Condition Risk — {condition_label}")
            st.metric("Risk Score", f"{mean_p*100:.1f}%")
            st.markdown(f"**Probability:** {mean_p:.1%}")
            st.markdown(f"**95% Credible Interval:** [{lo:.1%}, {hi:.1%}]")
            st.progress(min(max(int(round(mean_p * 100)), 0), 100))
            if cat == "High":       st.error("Risk Category: **HIGH**")
            elif cat == "Moderate": st.warning("Risk Category: **MODERATE**")
            else:                   st.success("Risk Category: **LOW**")

        if urgency in ("urgent","emergency") and cat == "Low":
            st.info("Why 'Urgent' with a low risk score? Safety Check uses vitals and red-flag symptoms to recommend how quickly to seek care. The risk score estimates the chance of this specific condition only. They are independent checks.")

        # ── CREDIBLE INTERVAL PLOTS (Prof. Zhao suggestion #2) ───────────
        render_ci_plots(
            prob_samples=prob_samples,
            mean_p=mean_p, lo=lo, hi=hi,
            mu=mu, cov=cov, order=order, x_vec=x_vec,
            missing_fields=missing_fields,
            condition_label=condition_label,
        )

        # Clinical guidance
        st.subheader("Clinical Guidance & Next Steps")
        bp_cat_g, _, _ = categorize_bp(sbp_safe, dbp_safe)
        hr_cat_g, _, _ = categorize_hr(hr_safe)
        guidance = compose_guidance(
            cond_key=cond_key, urgency=urgency, risk_cat=cat,
            bp_cat=bp_cat_g, hr_cat=hr_cat_g, bmi=bmi_val,
            smoke_cat=smoke_cat, alcohol_cat=alcohol_cat,
            physical_active_flag=base.get("physical_activity", 0.0),
            selected_symptoms=selected_symptoms,
            histories={
                "diabetes":       bool(base.get("diabetes_history", 0)),
                "kidney":         bool(base.get("kidney_disease", 0)),
                "family_history": bool(base.get("family_history_heart_disease", 0)),
            },
        )
        for g in guidance: st.markdown(f"- {g}")
        st.caption("This tool supports awareness and early care-seeking. It does not diagnose conditions.")

        # ── PDF Report ────────────────────────────────────────────────────
        def build_pdf_bytes(rec_text_in):
            try:
                from reportlab.lib.pagesizes import letter
                from reportlab.pdfgen import canvas
                from reportlab.lib.utils import ImageReader
            except Exception as e:
                return None, f"ReportLab import failed: {e}"

            buffer = io.BytesIO()
            c = canvas.Canvas(buffer, pagesize=letter)
            width, height = letter
            margin = 60
            y = height - margin

            def draw_wrapped(txt, x, leading=12, width_chars=110):
                nonlocal y
                for ln in wrap(txt, width_chars):
                    if y < margin + 30:
                        c.showPage(); y = height - margin
                    c.drawString(x, y, ln); y -= leading

            # Header
            if logo_path_str:
                try: c.drawImage(ImageReader(logo_path_str), 40, y-20, width=40, height=40, preserveAspectRatio=True, mask='auto')
                except: pass
                c.setFont("Helvetica-Bold", 16); c.drawString(90, y, "PredictRisk: Cardiovascular Diagnostic Tool")
            else:
                c.setFont("Helvetica-Bold", 18); c.drawString(40, y, "PredictRisk Report")
            c.setFont("Helvetica-Bold", 11); c.drawString(40, y-22, f"Assessment: {condition_label}")
            c.setFont("Helvetica", 10); c.drawString(300, y-22, datetime.now().strftime("Date: %Y-%m-%d  Time: %H:%M"))
            y -= 52

            c.setFont("Helvetica-Bold", 11); c.drawString(40, y, "Safety Check"); y -= 16
            c.setFont("Helvetica", 10)
            draw_wrapped(f"Clinical Urgency: {urgency.upper()}", 40)
            if reasons: draw_wrapped("Reasons: " + "; ".join(reasons), 40)
            draw_wrapped(f"BP: {sbp_safe}/{dbp_safe} mmHg — {bp_cat_g}", 40)
            draw_wrapped(f"Heart Rate: {hr_safe} bpm — {hr_cat_g}", 40)
            y -= 8

            c.setFont("Helvetica-Bold", 11); c.drawString(40, y, f"Condition Risk — {condition_label}"); y -= 16
            c.setFont("Helvetica", 10)
            draw_wrapped(f"Risk Score: {mean_p*100:.1f}%  |  Category: {cat}", 40)
            draw_wrapped(f"95% Credible Interval: [{lo:.1%}, {hi:.1%}]", 40)
            if missing_fields:
                draw_wrapped(f"Note: {len(missing_fields)} covariate(s) excluded (N/A): {', '.join(missing_fields)}", 40)
            y -= 8

            c.setFont("Helvetica-Bold", 11); c.drawString(40, y, "Overall Recommendation"); y -= 16
            c.setFont("Helvetica", 10); draw_wrapped(rec_text_in, 40); y -= 8

            c.setFont("Helvetica-Bold", 11); c.drawString(40, y, "Clinical Guidance"); y -= 16
            c.setFont("Helvetica", 10)
            for g in guidance: draw_wrapped("• " + g, 40)

            # Footer
            disclaimer = ("PredictRisk provides educational estimates and triage guidance only. "
                          "It is not a diagnosis and does not replace clinical evaluation. "
                          "If symptoms are severe or worsening, seek immediate medical care.")
            c.setFont("Helvetica", 8)
            foot_lines = wrap(disclaimer, 110)
            y_footer = 60
            for i, line in enumerate(foot_lines):
                c.drawString(40, y_footer + (len(foot_lines)-1-i)*10, line)
            c.setFont("Helvetica-Oblique", 8)
            c.drawRightString(width-40, 40, f"© {datetime.now().year} Taiwo Michael Ayeni")

            c.showPage(); c.save()
            pdf = buffer.getvalue(); buffer.close()
            return pdf, None

        pdf_bytes, pdf_err = build_pdf_bytes(rec_text)
        if pdf_bytes is None:
            st.warning(f"PDF not generated: {pdf_err}  — install with:  pip install reportlab pillow")
        else:
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_bytes,
                file_name=f"PredictRisk_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
            )

# Disclaimer footer
st.info(
    "⚠️ **Disclaimer:** PredictRisk provides educational risk estimates and triage guidance. "
    "It is **not** a diagnosis and does not replace professional medical care. "
    "If symptoms are severe or worsening, seek immediate care."
)
st.markdown(
    f"<div style='text-align:center; color:#888; margin-top:2rem;'>© {datetime.now().year} Taiwo Michael Ayeni</div>",
    unsafe_allow_html=True,
)
