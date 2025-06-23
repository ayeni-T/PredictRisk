
import streamlit as st
import pandas as pd
import numpy as np
import bambi as bmb
import arviz as az
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
import datetime

os.environ["PYTENSOR_FLAGS"] = "cxx="

st.set_page_config(page_title="PredictRisk", page_icon="🧠", layout="centered")
st.title("🧠 PredictRisk: Cardiovascular Diagnostic Tool")

condition_map = {
    "Stroke": {"file": "stroke_model_fit.nc", "formula": "risk_stroke"},
    "Heart Disease": {"file": "heart_disease_model_fit.nc", "formula": "risk_heart_disease"},
    "Hypertension": {"file": "hypertension_model_fit.nc", "formula": "risk_hypertension"},
    "Heart Failure": {"file": "heart_failure_model_fit.nc", "formula": "risk_heart_failure"},
    "Atrial Fibrillation (AFib)": {"file": "afib_model_fit.nc", "formula": "risk_afib"},
    "Peripheral Artery Disease (PAD)": {"file": "pad_model_fit.nc", "formula": "risk_pad"}
}

def get_risk_advice(risk_level):
    if risk_level == "High Risk":
        return [
            "Your predicted risk is high. Please take immediate action:",
            "- Consult a licensed physician or cardiologist",
            "- Request a full cardiovascular evaluation",
            "- Manage key risk factors (smoking, inactivity, blood pressure, etc.)",
            "- Do not delay action even if you feel well"
        ]
    elif risk_level == "Moderate Risk":
        return [
            "Your risk is moderate. Consider the following steps:",
            "- Speak with your healthcare provider about prevention",
            "- Monitor your blood pressure, glucose, and stress",
            "- Adopt heart-healthy lifestyle changes",
            "- Schedule a check-up if you haven't recently"
        ]
    else:
        return [
            "Great news! Your predicted risk is low. Maintain these habits:",
            "- Stay physically active and eat healthily",
            "- Manage stress and avoid smoking/alcohol excess",
            "- Continue regular medical checkups",
            "- Educate others about cardiovascular risk awareness"
        ]

st.markdown("### Select a condition to assess:")
condition = st.selectbox("", list(condition_map.keys()))

st.header("Enter your health details:")
age = st.number_input("Age (years)", 30, 90, 50, help="Your current age in years.")
bmi = st.number_input("BMI (kg/m²)", 15.0, 45.0, 25.0, step=0.1, help="Body Mass Index — a measure of body fat.")
glucose = st.number_input("Glucose (mg/dL)", 70, 200, 100, help="Fasting blood glucose level.")
systolic_bp = st.number_input("Systolic BP (mmHg)", 90, 200, 120, help="Top number in your blood pressure reading.")
diastolic_bp = st.number_input("Diastolic BP (mmHg)", 60, 130, 80, help="Bottom number in your blood pressure reading.")
heart_rate = st.number_input("Heart Rate (bpm)", 40, 150, 75, help="Resting heart rate.")
alcohol_use = st.radio("Do you consume alcohol?", ["No", "Yes"], help="Do you currently drink alcohol?")
smoking_status = st.radio("Do you smoke?", ["No", "Yes"], help="Have you smoked recently?")
physical_activity = st.radio("Are you physically active?", ["Yes", "No"], help="Do you engage in regular physical exercise?")
sleep_hours = st.slider("Sleep Hours", 3.0, 10.0, 6.5, 0.5, help="How many hours do you sleep per day?")
stress_score = st.slider("Stress Level (1-10)", 1, 10, 5, help="Rate your daily stress level.")

input_data = {
    "age": age,
    "bmi": bmi,
    "glucose": glucose,
    "systolic_bp": systolic_bp,
    "diastolic_bp": diastolic_bp,
    "heart_rate": heart_rate,
    "alcohol_use": 1 if alcohol_use == "Yes" else 0,
    "smoking_status": 1 if smoking_status == "Yes" else 0,
    "physical_activity": 1 if physical_activity == "Yes" else 0,
    "sleep_hours": sleep_hours,
    "stress_score": stress_score
}

def predict_risk(model_file, formula, input_dict):
    df = pd.read_csv("multi_cvd_dataset.csv")
    model = bmb.Model(f"{formula} ~ age + bmi + glucose + smoking_status + physical_activity + systolic_bp + diastolic_bp + heart_rate + alcohol_use + sleep_hours + stress_score", data=df, family="bernoulli")
    idata = az.from_netcdf(model_file)
    new_data = pd.DataFrame([input_dict])
    preds = model.predict(idata=idata, data=new_data, kind="response_params", inplace=False)
    probs = preds.posterior["p"].values.flatten()
    mean_risk = np.mean(probs)
    lower, upper = np.percentile(probs, [2.5, 97.5])
    summary = az.summary(idata, kind="stats", round_to=2)
    coef_summary = summary.loc[~summary.index.str.startswith("Intercept")]
    return mean_risk, (lower, upper), coef_summary

def generate_pdf_report(condition, mean_risk, ci, risk_level, top_vars, advice_lines, interpretation_text):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica-Bold", 12)
    y = 750
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    c.drawString(50, y, "PredictRisk: Cardiovascular Risk Report")
    y -= 30
    c.drawString(50, y, f"Condition: {condition}")
    y -= 20
    c.drawString(50, y, f"Predicted Risk: {mean_risk:.2%}")
    y -= 20
    c.drawString(50, y, f"95% Credible Interval: [{ci[0]:.2%}, {ci[1]:.2%}]")
    y -= 20
    c.drawString(50, y, f"Risk Level: {risk_level}")
    y -= 30

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "What This Risk Prediction Means for You:")
    y -= 20
    c.setFont("Helvetica", 12)
    for line in interpretation_text.split("\n"):
        c.drawString(60, y, line)
        y -= 20
    y -= 10

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Recommended Actions:")
    c.setFont("Helvetica", 12)
    for line in advice_lines:
        y -= 20
        c.drawString(60, y, line)
    y -= 30

    c.setFont("Helvetica-Oblique", 10)
    c.drawString(50, y, f"Date/Time: {timestamp}")
    y -= 15
    c.drawString(50, y, "Note: This report is for educational purposes only. Consult your doctor for clinical advice.")
    c.drawRightString(540, 30, "© Taiwo Michael Ayeni")
    c.save()
    buffer.seek(0)
    return buffer

if st.button("🔍 Estimate Risk"):
    selected_model = condition_map[condition]
    mean_risk, ci, coef_summary = predict_risk(selected_model["file"], selected_model["formula"], input_data)

    st.subheader(f"🩺 Predicted Risk for {condition}: {mean_risk:.2%}")
    st.markdown(f"**95% Credible Interval:** [{ci[0]:.2%}, {ci[1]:.2%}]")

    if mean_risk >= 0.7:
        risk_level = "High Risk"
        st.error("🚨 High Risk Detected")
    elif mean_risk >= 0.4:
        risk_level = "Moderate Risk"
        st.warning("⚠️ Moderate Risk")
    else:
        risk_level = "Low Risk"
        st.success("✅ Low Risk")

    st.markdown("### 💡 Recommended Actions")
    advice_lines = get_risk_advice(risk_level)
    for line in advice_lines:
        st.markdown(f"- {line}")

    st.markdown("### 📌 What This Risk Prediction Means for You")
    interpretation_text = f"Based on your input, your predicted risk of developing {condition} is {mean_risk:.2%}.\nThis estimate reflects your current likelihood based on your health profile.\nWhile not diagnostic, it offers a data-informed view of your risk.\nFor an accurate medical interpretation, consult a certified healthcare provider."
    for line in interpretation_text.split("\n"):
        st.markdown(line)

    top_dict = {}  # For PDF structure
    pdf_buffer = generate_pdf_report(condition, mean_risk, ci, risk_level, top_dict, advice_lines, interpretation_text)
    st.download_button("📄 Download Risk Report (PDF)", data=pdf_buffer, file_name=f"{condition}_Risk_Report.pdf", mime="application/pdf")

    st.markdown("---")
    st.info("""
⚠️ **Disclaimer:**  
This tool is for educational and informational purposes only.  
It is **not a substitute for professional medical advice or diagnosis**.  
Always consult a licensed healthcare provider regarding your health.  
This model is based on simulated data and may not reflect your actual clinical risk.
""")
    st.markdown("<div style='text-align: right; font-size: 0.9em;'>© Taiwo Michael Ayeni</div>", unsafe_allow_html=True)
