# app_lightweight.py
# PredictRisk: Cardiovascular Diagnostic Tool (educational)
# - Laplace artifacts (mean, cov, feature_order) from /artifacts
# - Safety-first triage + condition-specific risk (always visible, in %)
# - Condition-specific guidance (varies by selected condition + risk)
# - PDF: footer-only disclaimer; fixed spacing; page-break aware
# - Red "Assess Risk" button
# - Optional "Explain my score" (hidden unless expanded)

import streamlit as st
import numpy as np
from pathlib import Path
from datetime import datetime
from textwrap import wrap
import io

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

LOGO_CANDIDATES = [
    BASE_DIR / "logo.png",
    BASE_DIR / "assets" / "logo.png",
    BASE_DIR / "static" / "logo.png",
]

ARTIFACT_DIR = BASE_DIR / "artifacts"

# --------------------------- Logo detection ---------------------------
LOGO_CANDIDATES = [Path("logo.png"), Path("assets/logo.png"), Path("static/logo.png")]

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

# Make the primary button RED
st.markdown("""
<style>
/* Streamlit primary button override */
div.stButton > button[kind="primary"] { background-color:#d32f2f; color:white; border:0; }
div.stButton > button[kind="primary"]:hover { background-color:#b71c1c; color:white; }
div.stButton > button:first-child { background-color:#d32f2f; color:white; border:0; }
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


ARTIFACT_DIR = Path("artifacts")

CONDITIONS = {
    "Stroke": "stroke",
    "Heart Disease": "heart_disease",
    "Hypertension": "hypertension",
    "Heart Failure": "heart_failure",
    "Atrial Fibrillation (AFib)": "afib",
    "Peripheral Artery Disease (PAD)": "pad",
}

# --------------------------- Artifact loader ---------------------------
@st.cache_resource
def load_artifact(cond_key: str):
    npz_path = ARTIFACT_DIR / f"{cond_key}_beta_summary.npz"
    if not npz_path.exists():
        st.error(f"Artifact not found: {npz_path}. Train models first.")
        st.stop()
    pkg = np.load(npz_path, allow_pickle=True)
    mu = pkg["mean"].astype("float32")
    cov = pkg["cov"].astype("float32")
    order_arr = pkg["feature_order"]
    try:
        order = [str(x) for x in order_arr.tolist()]
    except Exception:
        order = [str(x) for x in order_arr]
    # Optional: cache Cholesky for faster sampling (invisible to users)
    # We keep cov as-is for simplicity; you can uncomment below if desired:
    # L = np.linalg.cholesky(0.5*(cov+cov.T) + np.eye(len(mu))*1e-8).astype("float32")
    # return mu, cov, order, L
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
    return float(p.mean()), float(np.percentile(p, 2.5)), float(np.percentile(p, 97.5))

def value_for_feature(fname: str, base: dict, smoke_cat: int, alcohol_cat: int):
    if fname == "Intercept":
        return 1.0
    if fname in base:
        return float(base[fname])
    f = fname.lower()
    if "smok" in f:
        if ("t.1" in f) or ("former" in f):
            return 1.0 if smoke_cat == 1 else 0.0
        if ("t.2" in f) or ("current" in f):
            return 1.0 if smoke_cat == 2 else 0.0
        return 0.0
    if "alcohol" in f:
        if ("t.1" in f) or ("moderate" in f):
            return 1.0 if alcohol_cat == 1 else 0.0
        if ("t.2" in f) or ("excessive" in f):
            return 1.0 if alcohol_cat == 2 else 0.0
        return 0.0
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

# --------- Condition-specific guidance (educational, not diagnostic) ----------
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
        "Low": [
            "Maintain BP <120/80, stay active, and know FAST signs."
        ],
    },
    "heart_disease": {
        "High": [
            "Discuss a chest pain plan, blood pressure and lipid management, and smoking cessation if applicable.",
            "A clinician may consider ECG or other testing based on history."
        ],
        "Moderate": [
            "Assess cholesterol, diet quality, and activity; discuss preventive medications if indicated."
        ],
        "Low": [
            "Maintain heart-healthy habits; know warning signs of angina."
        ],
    },
    "hypertension": {
        "High": [
            "Record home BP (morning and evening for 1 week) and review targets with a clinician.",
            "Limit salt, maintain healthy weight, and follow a DASH-style diet."
        ],
        "Moderate": [
            "Re-check BP after 5 minutes rest; keep a log and discuss lifestyle changes."
        ],
        "Low": [
            "Continue regular checks and healthy habits."
        ],
    },
    "heart_failure": {
        "High": [
            "Discuss breathlessness, swelling, and daily weight tracking with a clinician.",
            "Review salt and fluid guidance; know when to seek urgent care."
        ],
        "Moderate": [
            "Check for swelling or night cough; review blood pressure control and activity plan."
        ],
        "Low": [
            "Be aware of symptoms (leg swelling, breathlessness) and seek review if they develop."
        ],
    },
    "afib": {
        "High": [
            "Discuss rhythm monitoring (ECG/ambulatory) and stroke prevention as advised by your clinician.",
            "Limit alcohol and stimulants; report palpitations, dizziness, or fainting."
        ],
        "Moderate": [
            "Learn pulse checks; review triggers (caffeine, alcohol) and thyroid evaluation if advised."
        ],
        "Low": [
            "Know how to check pulse; seek review if it becomes irregular or fast."
        ],
    },
    "pad": {
        "High": [
            "A supervised walking program and foot care are important; review smoking cessation if relevant.",
            "Discuss blood pressure, glucose control, and statin/antiplatelet use with a clinician."
        ],
        "Moderate": [
            "Start a gradual walking plan; maintain foot care; discuss preventive therapy if advised."
        ],
        "Low": [
            "Stay active; note any calf pain that appears with walking and improves with rest."
        ],
    },
}

def compose_guidance(
    cond_key: str,
    urgency: str,
    risk_cat: str,
    bp_cat: str,
    hr_cat: str,
    bmi: float,
    smoke_cat: int,
    alcohol_cat: int,
    physical_active_flag: float,
    selected_symptoms: list,
    histories: dict,
):
    G = []
    # Safety-first line
    if urgency == "emergency":
        G.append("Emergency symptoms detected — seek immediate medical care. Do not drive yourself.")
    elif urgency == "urgent":
        G.append("Seek prompt medical review (today–48 hours), especially if symptoms are new or worsening.")
    else:
        G.append("Arrange routine review with a clinician to discuss your cardiovascular risk profile.")

    # Condition-specific block by risk
    for line in COND_GUIDE.get(cond_key, {}).get(risk_cat, []):
        G.append(line)

    # Factor-aware additions (brief, de-duplicated)
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

    # Symptom pointers
    S = set(selected_symptoms)
    if {"chest_pain", "severe_chest_pain"} & S:
        G.append("Chest pain with sweating, nausea, or shortness of breath warrants urgent assessment.")
    if {"weak_limb", "slurred_speech", "face_droop"} & S:
        G.append("Stroke-like symptoms require emergency evaluation immediately.")

    # De-duplicate & keep order
    seen = set(); out = []
    for s in G:
        if s not in seen:
            seen.add(s); out.append(s)
    return out

# ---- blank-friendly inputs ----
def parse_float(s):
    if s is None or s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None

def num_input(label, key, placeholder, help=None):
    return parse_float(st.text_input(label, key=key, placeholder=placeholder, help=help))

def select_with_placeholder(label, options, key, help=None):
    display = ["— Select —"] + options
    choice = st.selectbox(label, display, index=0, key=key, help=help)
    return None if choice == "— Select —" else choice

# --------------------------- Condition selector ---------------------------
st.markdown("### Select a cardiovascular condition to assess:")
condition_label = st.selectbox("", list(CONDITIONS.keys()))
cond_key = CONDITIONS[condition_label]

# --------------------------- Inputs (blank by default) ---------------------------
st.header("Enter your health details")

col1, col2 = st.columns(2)
with col1:
    age = num_input("Age (years)", "age", "e.g., 50", help="Your age in completed years.")
    sex = select_with_placeholder("Sex", ["Female", "Male"], "sex_sel", help="Biological sex assigned at birth.")
    height_m = num_input("Height (m)", "height_m", "e.g., 1.70", help="Measured height in metres.")
    weight_kg = num_input("Weight (kg)", "weight_kg", "e.g., 75", help="Body weight in kilograms.")
with col2:
    systolic_bp = num_input("Systolic BP (mmHg)", "sbp", "e.g., 120", help="Top number; seated, arm supported.")
    diastolic_bp = num_input("Diastolic BP (mmHg)", "dbp", "e.g., 80", help="Bottom number; pressure between beats.")
    heart_rate = num_input("Heart Rate (bpm)", "hr", "e.g., 75", help="Resting beats per minute.")

# BMI badge once both height/weight present
bmi = None
if height_m and weight_kg and height_m > 0:
    bmi = weight_kg / (height_m ** 2)
    if bmi >= 30: st.error(f"BMI: {bmi:.1f} kg/m² — OBESE")
    elif bmi >= 25: st.warning(f"BMI: {bmi:.1f} kg/m² — OVERWEIGHT")
    else: st.success(f"BMI: {bmi:.1f} kg/m² — Normal")

# BP & HR badges when present
if systolic_bp is not None and diastolic_bp is not None:
    bp_cat, bp_color, bp_note = categorize_bp(int(systolic_bp), int(diastolic_bp))
    msg = f"BP: {int(systolic_bp)}/{int(diastolic_bp)} mmHg — {bp_cat} ({bp_note})"
    if bp_color == "green": st.success(msg)
    elif bp_color in ("gold", "orange"): st.warning(msg)
    elif bp_color == "red": st.error(msg)
    else: st.info(f"BP: {int(systolic_bp)}/{int(diastolic_bp)} mmHg — {bp_cat}")

if heart_rate is not None:
    hr_cat, hr_color, hr_note = categorize_hr(int(heart_rate))
    msg = f"Heart Rate: {int(heart_rate)} bpm — {hr_cat} ({hr_note})"
    if hr_color == "green": st.success(msg)
    elif hr_color in ("gold", "orange"): st.warning(msg)
    elif hr_color == "red": st.error(msg)
    else: st.info(f"Heart Rate: {int(heart_rate)} bpm — {hr_cat}")

st.subheader("Lifestyle & Risk Factors")
col3, col4 = st.columns(2)
with col3:
    smoking_status_lbl = select_with_placeholder(
        "Smoking status", ["Never", "Former", "Current"], "smoke_sel",
        help="Current: smoke now; Former: smoked before but not now; Never: never smoked."
    )
    alcohol_use_lbl = select_with_placeholder(
        "Alcohol use", ["None", "Moderate", "Excessive"], "alcohol_sel",
        help="None; Moderate: occasional/≤1 drink a day; Excessive: frequent/heavy."
    )
    physical_activity = select_with_placeholder(
        "Physically active?", ["Yes", "No"], "active_sel",
        help="Regular moderate activity ≈150+ minutes/week or as advised."
    )
with col4:
    sleep_hours = num_input("Sleep (hours/night)", "sleep", "e.g., 7.0", help="Average nightly sleep duration.")
    stress_score = num_input("Stress (1–10)", "stress", "e.g., 5", help="Your perceived stress today; 1=low, 10=high.")

st.subheader("Medical History")
col5, col6 = st.columns(2)
with col5:
    family_history_heart = select_with_placeholder("Family history of heart disease?", ["No", "Yes"], "fh_sel",
                                                   help="Parent, brother, or sister with heart disease.")
    diabetes_history = select_with_placeholder("Diabetes (diagnosed)?", ["No", "Yes"], "dm_sel",
                                               help="Previously diagnosed by a clinician.")
with col6:
    kidney_disease = select_with_placeholder("Chronic kidney disease?", ["No", "Yes"], "ckd_sel",
                                             help="Previously diagnosed CKD.")
    substance_abuse = select_with_placeholder("Substance abuse?", ["No", "Yes"], "sub_abuse_sel",
                                              help="Problematic use of drugs/substances.")

st.subheader("Current Symptoms")
colA, colB, colC = st.columns(3)
with colA:
    chest_pain = st.checkbox("Chest pain", help="Tightness/pressure in the chest.")
    severe_chest_pain = st.checkbox("Severe chest pain", help="Intense, persistent pain; not eased by rest.")
    shortness_breath = st.checkbox("Shortness of breath", help="Breathless at rest or minimal activity.")
    lightheadedness = st.checkbox("Lightheadedness", help="Feeling faint or woozy.")
with colB:
    difficulty_breathing = st.checkbox("Difficulty breathing", help="Laboured breathing; hard to speak full sentences.")
    palpitations = st.checkbox("Heart palpitations", help="Awareness of fast or irregular heartbeat.")
    dizziness = st.checkbox("Dizziness", help="Spinning sensation or imbalance.")
    fainting = st.checkbox("Fainting episodes", help="Blackouts or sudden loss of consciousness.")
with colC:
    leg_swelling = st.checkbox("Leg/ankle swelling", help="Swelling (oedema) in lower legs/ankles.")
    persistent_cough = st.checkbox("Persistent cough", help="Cough worse at night or lying down.")
    face_droop = st.checkbox("Face droop (one-sided)", help="Drooping on one side of the face.")
    slurred_speech = st.checkbox("Slurred speech", help="Words sound unclear or garbled.")
    weak_limb = st.checkbox("Weakness in arm/leg", help="Sudden weakness or numbness in a limb.")
    cold_sweat = st.checkbox("Cold sweat", help="Profuse sweating not due to heat/exercise.")
    sudden_sweating = st.checkbox("Sudden sweating", help="Unexpected sudden onset of sweating.")
    orthopnea = st.checkbox("Orthopnea (worse lying down)", help="Shortness of breath when lying flat.")
    claudication = st.checkbox("Cramping leg pain with walking", help="Leg cramps during walking that improve with rest.")

# --------------------------- Assess ---------------------------
required_nums = [age, height_m, weight_kg, systolic_bp, diastolic_bp, heart_rate, sleep_hours, stress_score]
required_cats = [sex, smoking_status_lbl, alcohol_use_lbl, physical_activity,
                 family_history_heart, diabetes_history, kidney_disease, substance_abuse]
all_required = all(v is not None for v in required_nums + required_cats)

if not all_required:
    st.warning("Please complete all fields (numbers and selections) before assessing.")
else:
    if st.button("🔍 Assess Risk", type="primary"):
        # Collect symptoms
        selected_symptoms = []
        for name, flag in [
            ("chest_pain", chest_pain), ("severe_chest_pain", severe_chest_pain),
            ("shortness_breath", shortness_breath), ("difficulty_breathing", difficulty_breathing),
            ("palpitations", palpitations), ("dizziness", dizziness), ("fainting", fainting),
            ("leg_swelling", leg_swelling), ("persistent_cough", persistent_cough),
            ("face_droop", face_droop), ("slurred_speech", slurred_speech), ("weak_limb", weak_limb),
            ("cold_sweat", cold_sweat), ("sudden_sweating", sudden_sweating),
            ("lightheadedness", lightheadedness), ("orthopnea", orthopnea), ("claudication", claudication),
        ]:
            if flag: selected_symptoms.append(name)

        urgency, reasons, tags = assess_clinical_urgency_enhanced(
            int(systolic_bp), int(diastolic_bp), int(heart_rate), selected_symptoms
        )

        smoke_map = {"Never": 0, "Former": 1, "Current": 2}
        alcohol_map = {"None": 0, "Moderate": 1, "Excessive": 2}
        smoke_cat = smoke_map[smoking_status_lbl]
        alcohol_cat = alcohol_map[alcohol_use_lbl]

        bmi_val = (weight_kg / (height_m ** 2)) if (height_m and weight_kg) else 0.0
        base = {
            "age": float(age),
            "sex": 1.0 if sex == "Male" else 0.0,
            "bmi": float(bmi_val),
            "physical_activity": 1.0 if physical_activity == "Yes" else 0.0,
            "systolic_bp": float(systolic_bp),
            "diastolic_bp": float(diastolic_bp),
            "heart_rate": float(heart_rate),
            "sleep_hours": float(sleep_hours),
            "stress_score": float(stress_score),
            "family_history_heart_disease": 1.0 if family_history_heart == "Yes" else 0.0,
            "diabetes_history": 1.0 if diabetes_history == "Yes" else 0.0,
            "kidney_disease": 1.0 if kidney_disease == "Yes" else 0.0,
            "substance_abuse": 1.0 if substance_abuse == "Yes" else 0.0,
        }

        mu, cov, order = load_artifact(CONDITIONS[condition_label])
        x_vec = np.array([value_for_feature(name, base, smoke_cat, alcohol_cat) for name in order], dtype="float32")
        mean_p, lo, hi = predict_prob(mu, cov, x_vec, draws=5000)
        cat, cat_color = risk_category(mean_p)

        # Overall recommendation
        rec_text, rec_color = overall_recommendation(urgency, cat)
        if rec_color == "red": st.error(rec_text)
        elif rec_color == "orange": st.warning(rec_text)
        elif rec_color == "gold": st.info(rec_text)
        else: st.success(rec_text)

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
            if cat == "High": st.error("Risk Category: **HIGH**")
            elif cat == "Moderate": st.warning("Risk Category: **MODERATE**")
            else: st.success("Risk Category: **LOW**")

            # Optional lightweight explainability (hidden unless expanded)
            with st.expander("Explain my score (optional)"):
                # simple top drivers by |mu_i * x_i| (excluding Intercept)
                names = []
                vals = []
                for name, beta, x in zip(order, mu, x_vec):
                    if name.lower() == "intercept": 
                        continue
                    contrib = abs(float(beta) * float(x))
                    names.append(name); vals.append(contrib)
                if vals:
                    top_idx = np.argsort(vals)[::-1][:3]
                    st.caption("Top factors contributing to the score:")
                    for i in top_idx:
                        st.write(f"- {names[i].replace('C(','').replace(')','').replace('[T.1]','').replace('[T.2]','')}")
                else:
                    st.caption("No contributing factors to display.")

        if urgency in ("urgent", "emergency") and cat == "Low":
            st.info("Why ‘Urgent’ with a low risk score? Safety Check uses vitals and red-flag symptoms to recommend how quickly to seek care. The risk score estimates the chance of this specific condition only. They are independent checks.")

        # Guidance (condition-specific + factors)
        st.subheader("Clinical Guidance & Next Steps")
        bp_cat, _, _ = categorize_bp(int(systolic_bp), int(diastolic_bp))
        hr_cat, _, _ = categorize_hr(int(heart_rate))
        guidance = compose_guidance(
            cond_key=cond_key,
            urgency=urgency,
            risk_cat=cat,
            bp_cat=bp_cat,
            hr_cat=hr_cat,
            bmi=bmi_val,
            smoke_cat=smoke_cat,
            alcohol_cat=alcohol_cat,
            physical_active_flag=base["physical_activity"],
            selected_symptoms=selected_symptoms,
            histories={
                "diabetes": bool(base["diabetes_history"]),
                "kidney": bool(base["kidney_disease"]),
                "family_history": bool(base["family_history_heart_disease"]),
            },
        )
        for g in guidance: st.markdown(f"- {g}")
        st.caption("This tool supports awareness and early care-seeking. It does not diagnose conditions.")

        # --------------------------- PDF Report ---------------------------
        def build_pdf_bytes(rec_text_in):
            try:
                from reportlab.lib.pagesizes import letter
                from reportlab.pdfgen import canvas
                from reportlab.lib.utils import ImageReader
            except Exception as e:
                return None, f"ReportLab import failed: {e}"

            def draw_wrapped(c, text, x, y, width_chars=110, leading=12):
                """Draw wrapped text and return new y (page-break safe)."""
                lines = wrap(text, width_chars)
                for ln in lines:
                    nonlocal_y_check()
                    c.drawString(x, y, ln); y -= leading
                return y

            # page-break helper
            buffer = io.BytesIO()
            c = canvas.Canvas(buffer, pagesize=letter)
            width, height = letter
            margin = 60
            y = height - margin

            def nonlocal_y_check():
                nonlocal y
                if y < margin + 30:
                    c.showPage()
                    y = height - margin
                    # header on new page (logo + title small)
                    if logo_path_str:
                        try:
                            c.drawImage(ImageReader(logo_path_str), 40, y - 20, width=40, height=40,
                                        preserveAspectRatio=True, mask='auto')
                        except Exception:
                            pass
                        c.setFont("Helvetica-Bold", 16); c.drawString(90, y, "PredictRisk: Cardiovascular Diagnostic Tool")
                        y -= 40
                    else:
                        c.setFont("Helvetica-Bold", 18); c.drawString(40, y, "PredictRisk Report"); y -= 30

            # Header
            if logo_path_str:
                try:
                    c.drawImage(ImageReader(logo_path_str), 40, y - 20, width=40, height=40,
                                preserveAspectRatio=True, mask='auto')
                except Exception:
                    pass
                c.setFont("Helvetica-Bold", 16); c.drawString(90, y, "PredictRisk: Cardiovascular Diagnostic Tool")
            else:
                c.setFont("Helvetica-Bold", 18); c.drawString(40, y, "🧠 PredictRisk: Cardiovascular Diagnostic Tool")
            c.setFont("Helvetica-Bold", 11); c.drawString(40, y - 22, f"Assessment: {condition_label}")
            c.setFont("Helvetica", 10); c.drawString(300, y - 22, datetime.now().strftime("Date: %Y-%m-%d  Time: %H:%M"))
            y -= 52

            # Safety Check
            c.setFont("Helvetica-Bold", 11); c.drawString(40, y, "Safety Check (independent of risk)"); y -= 16
            c.setFont("Helvetica", 10); y = draw_wrapped(c, f"Clinical Urgency: {urgency.upper()}", 40, y)
            if reasons:
                y = draw_wrapped(c, "Reasons: " + "; ".join(reasons), 40, y)
            bp_cat_pdf, _, bp_note_pdf = categorize_bp(int(systolic_bp), int(diastolic_bp))
            hr_cat_pdf, _, hr_note_pdf = categorize_hr(int(heart_rate))
            y = draw_wrapped(c, f"BP: {int(systolic_bp)}/{int(diastolic_bp)} mmHg — {bp_cat_pdf} ({bp_note_pdf})", 40, y)
            y = draw_wrapped(c, f"Heart Rate: {int(heart_rate)} bpm — {hr_cat_pdf} ({hr_note_pdf})", 40, y)
            y -= 10

            # Risk
            c.setFont("Helvetica-Bold", 11); c.drawString(40, y, f"Condition Risk — {condition_label}"); y -= 16
            c.setFont("Helvetica", 10)
            y = draw_wrapped(c, f"Risk Score: {mean_p*100:.1f}%  |  Category: {cat}", 40, y)
            y = draw_wrapped(c, f"Probability: {mean_p:.1%}", 40, y)
            y = draw_wrapped(c, f"95% Credible Interval: [{lo:.1%}, {hi:.1%}]", 40, y)
            y -= 10

            # Overall Recommendation (page-break aware & spaced)
            c.setFont("Helvetica-Bold", 11); c.drawString(40, y, "Overall Recommendation"); y -= 16
            c.setFont("Helvetica", 10); y = draw_wrapped(c, rec_text_in, 40, y); y -= 8

            # Input Summary
            c.setFont("Helvetica-Bold", 11); c.drawString(40, y, "Input Summary"); y -= 16
            c.setFont("Helvetica", 10)
            for t in [
                f"Age {int(age)} • Sex {sex} • BMI {bmi_val:.1f}",
                f"SBP/DBP {int(systolic_bp)}/{int(diastolic_bp)} mmHg • HR {int(heart_rate)} bpm",
                f"Smoking: {smoking_status_lbl} • Alcohol: {alcohol_use_lbl} • Active: {physical_activity}",
                f"Sleep: {sleep_hours} h • Stress: {stress_score}/10",
                f"Family hx heart disease: {family_history_heart} • Diabetes: {diabetes_history} • CKD: {kidney_disease} • Substance: {substance_abuse}",
            ]:
                y = draw_wrapped(c, t, 40, y)
            y -= 8

            # Guidance
            c.setFont("Helvetica-Bold", 11); c.drawString(40, y, "Clinical Guidance & Next Steps"); y -= 16
            c.setFont("Helvetica", 10)
            guidance_pdf = compose_guidance(
                cond_key=cond_key, urgency=urgency, risk_cat=cat,
                bp_cat=bp_cat_pdf, hr_cat=hr_cat_pdf, bmi=bmi_val,
                smoke_cat={"Never":0,"Former":1,"Current":2}[smoking_status_lbl],
                alcohol_cat={"None":0,"Moderate":1,"Excessive":2}[alcohol_use_lbl],
                physical_active_flag=1.0 if physical_activity=="Yes" else 0.0,
                selected_symptoms=[],  # summarized via urgency/reasons
                histories={
                    "diabetes": diabetes_history=="Yes",
                    "kidney": kidney_disease=="Yes",
                    "family_history": family_history_heart=="Yes",
                },
            )
            for g in guidance_pdf:
                y = draw_wrapped(c, "• " + g, 40, y)

            # Footer: DISCLAIMER ONLY + copyright
            disclaimer = ("PredictRisk provides educational estimates and triage guidance only. "
                          "It is not a diagnosis and does not replace clinical evaluation. "
                          "If symptoms are severe or worsening, seek immediate medical care.")
            c.setFont("Helvetica", 8)
            # draw footer at fixed bottom area
            foot_lines = wrap(disclaimer, 110)
            y_footer = 60
            for i, line in enumerate(foot_lines):
                c.drawString(40, y_footer + (len(foot_lines)-1-i)*10, line)
            c.setFont("Helvetica-Oblique", 8)
            c.drawRightString(width - 40, 40, f"© {datetime.now().year} Taiwo Michael Ayeni")

            c.showPage(); c.save()
            pdf = buffer.getvalue(); buffer.close()
            return pdf, None

        pdf_bytes, pdf_err = build_pdf_bytes(rec_text)
        if pdf_bytes is None:
            st.warning(f"PDF not generated: {pdf_err}  — install with:  pip install reportlab")
        else:
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_bytes,
                file_name=f"PredictRisk_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
            )


            # Launch disclaimer (app only; PDF disclaimer is in footer)
st.info(
    "⚠️ **Disclaimer:** PredictRisk provides educational risk estimates and triage guidance. "
    "It is **not** a diagnosis and does not replace professional medical care. "
    "If symptoms are severe or worsening, seek immediate care."
)

# Footer copyright
st.markdown(
    f"<div style='text-align:center; color:#888; margin-top:2rem;'>© {datetime.now().year} Taiwo Michael Ayeni</div>",
    unsafe_allow_html=True,
)
