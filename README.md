# 🧠 PredictRisk: Cardiovascular Diagnostic Tool

A comprehensive Bayesian machine learning application for predicting cardiovascular disease risk using personalized health data and advanced statistical modeling techniques.

## 🎯 Project Vision & Objectives

### **Personal Motivation**

This project stems from recognizing a critical gap in cardiovascular risk awareness and prevention. The reality that many individuals remain unaware of their elevated cardiovascular risk until clinical intervention becomes critical represents a significant challenge in modern healthcare. The "silent killer" nature of conditions like hypertension means that by the time symptoms appear, significant damage may have already occurred.

This recognition highlighted a fundamental healthcare challenge: **the lack of accessible, personalized risk assessment tools** that could provide early warning signs and enable preventive action. Too often, people remain unaware of their cardiovascular risk until they face a medical emergency. I realized that if we could democratize access to sophisticated risk prediction and make it as easy as checking the weather, we could save lives through early intervention and lifestyle modifications.

### **Technical Vision**

This project was developed with the goal of democratizing cardiovascular risk assessment through cutting-edge statistical methods. Healthcare decisions benefit from understanding the range of possible outcomes and the uncertainty inherent in predictions. By leveraging Bayesian statistics, PredictRisk provides comprehensive risk predictions with **credible intervals** that communicate this uncertainty effectively.

**Key Objectives:**
- **Address the critical gap in early cardiovascular risk detection** by creating accessible assessment tools
- Bridge the gap between complex statistical models and practical healthcare applications
- Demonstrate the power of Bayesian inference in medical risk prediction
- Provide transparent, interpretable risk assessments with uncertainty quantification
- Create an accessible tool for educational and research purposes in cardiovascular epidemiology
- **Empower individuals** to take proactive steps in their cardiovascular health journey

## 🔬 Statistical Methodology

### **Dataset & Data Generation**
**Important Note**: The current models are trained on **synthetically generated data** created through statistical simulation. This synthetic dataset was designed to:
- Reflect realistic distributions of cardiovascular risk factors
- Maintain appropriate correlations between variables observed in epidemiological literature
- Provide a foundation for methodology development and proof-of-concept demonstration
- Enable open-source sharing without privacy concerns

**Dataset Characteristics:**
- **Sample Size**: 500 simulated patient records
- **Variables**: 18 cardiovascular risk factors with realistic value ranges (age, BMI, lipid profiles, blood pressure, lifestyle factors, family history)
- **Outcome Variables**: 6 cardiovascular conditions with logistic regression-based risk modeling
- **Generation Method**: Statistical simulation using numpy random sampling from normal and binomial distributions with clinically plausible parameter ranges

### **Bayesian Logistic Regression Framework**
The application employs **Bayesian logistic regression models** implemented through the Bambi library (Built on PyMC). This represents a **methodologically sophisticated approach** to cardiovascular risk assessment. Each cardiovascular condition is modeled independently using:

- **Model Family**: Bernoulli likelihood with logit link function
- **Sampling Method**: No-U-Turn Sampler (NUTS) with four chains
- **Posterior Estimation**: 1,000 draws per chain after 1,000 tuning steps
- **Convergence Diagnostics**: R-hat statistics and effective sample size monitoring
- **Synthetic Data Foundation**: Statistically generated dataset using clinically plausible parameter distributions

**Key Methodological Innovation**: This Bayesian framework provides full posterior distributions for risk assessment, enabling probabilistic inference with explicit uncertainty quantification through credible intervals.

### **Predictor Variables**
The synthetic dataset contains 18 variables, but each Bayesian model uses a subset of 11 evidence-based risk factors:
- **Demographic**: Age
- **Anthropometric**: BMI  
- **Biomarkers**: Glucose levels, systolic/diastolic blood pressure, heart rate
- **Lifestyle**: Smoking status, alcohol consumption, physical activity, sleep duration
- **Psychosocial**: Stress score (1-10 scale)

*Note: Additional variables in the dataset (HDL/LDL cholesterol, triglycerides, diet score, family history) are available for future model enhancements.*

### **Uncertainty Quantification**
Our Bayesian approach provides:
- **Posterior predictive distributions** for individual risk assessment
- **95% credible intervals** representing uncertainty in predictions
- **Full posterior samples** enabling rich probabilistic inference

### **Model Validation & Diagnostics**
All models undergo rigorous validation:
- Convergence assessment via R-hat statistics (all < 1.01)
- Effective sample size evaluation (ESS > 400 for all parameters)
- Posterior predictive checks for model adequacy

**⚠️ Current Limitations:**
- Models are trained on synthetic data and require validation on real clinical datasets
- Performance metrics reflect synthetic data characteristics, not real-world clinical outcomes
- Clinical applicability pending validation studies with actual patient data

## 🔬 Methodological Approach

**PredictRisk employs advanced Bayesian statistical methods for cardiovascular risk assessment:**

- **Bayesian Statistical Foundation**: Uses sophisticated probabilistic modeling for risk prediction
- **Uncertainty Quantification**: Provides 95% credible intervals and full posterior distributions
- **Synthetic Data Methodology**: Built on statistically simulated data with transparent generation process
- **Open-Source Implementation**: Complete methodological transparency with reproducible code
- **Educational Focus**: Designed as proof-of-concept for Bayesian risk modeling techniques

This approach demonstrates the potential of Bayesian methods in healthcare applications with full uncertainty quantification.

## 🤝 Collaboration & Enhancement Opportunities

This project represents the foundation of what could become a powerful tool for cardiovascular risk research and education. **I actively welcome collaborations, suggestions, and contributions** from:

### **Researchers & Clinicians**
- **Validation studies using real clinical datasets** (highest priority need)
- Integration of additional biomarkers or genetic risk factors
- Clinical workflow integration and usability studies
- **External validation on diverse patient populations**

### **Data Scientists & Statisticians**
- Model architecture improvements and feature engineering
- Alternative Bayesian model specifications (hierarchical models, mixture models)
- Advanced uncertainty quantification techniques
- Computational optimization for larger datasets

### **Healthcare Professionals**
- Clinical interpretation guidelines and decision support integration
- User experience optimization for clinical settings
- Validation against clinical outcomes and real-world evidence
- Integration with electronic health records (EHR) systems

### **Open Source Contributors**
- Code optimization and performance improvements
- Additional visualizations and interpretability features
- API development for integration with other healthcare tools
- Documentation and educational materials enhancement

**How to Contribute:**
- 📧 Contact: ayenitaiwomichael24@gmail.com
- 🐛 Issues: Submit bug reports or feature requests via GitHub Issues
- 🔄 Pull Requests: Submit code improvements or new features
- 💬 Discussions: Join conversations about methodology and applications

**Areas for Enhancement:**
1. **External Validation**: Testing on diverse, real-world clinical datasets *(Critical Priority)*
2. **Real Data Integration**: Transitioning from synthetic to validated clinical data sources
3. **Model Expansion**: Incorporating time-to-event analysis for longitudinal risk
4. **Explainability**: Advanced SHAP or LIME integration for model interpretability
5. **Clinical Integration**: FHIR-compliant APIs for EHR integration
6. **Personalization**: Individual-level model updating with longitudinal data

## 🎯 Features

- **6 Cardiovascular Conditions**: Stroke, Heart Disease, Hypertension, Heart Failure, Atrial Fibrillation, Peripheral Artery Disease
- **Bayesian Modeling**: Uses advanced statistical modeling with uncertainty quantification
- **Risk Assessment**: Provides personalized risk scores with credible intervals
- **PDF Reports**: Generate downloadable risk assessment reports
- **User-Friendly Interface**: Clean, intuitive Streamlit interface

## 🔬 Technology Stack

- **Frontend**: Streamlit
- **Machine Learning**: Bambi (Bayesian modeling), PyMC, ArviZ
- **Data Processing**: Pandas, NumPy
- **Report Generation**: ReportLab

## 🚀 Live Demo

[Access the live application here](https://predictrisk.streamlit.app/) 

## ⚠️ Disclaimer

This tool is for **educational and informational purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a licensed healthcare provider regarding your health.

**Important Limitations:**
- **Models are trained on synthetically generated data**, not real clinical datasets
- Risk predictions have not been validated against actual clinical outcomes
- Performance characteristics may differ significantly when applied to real patient populations
- This tool should not be used for clinical decision-making without proper validation

**Current Status:** This is a proof-of-concept demonstration of Bayesian risk modeling methodology. Clinical validation with real-world data is required before any medical application.

## 📊 Model Information

The models are trained on **synthetically generated cardiovascular risk data** (500 records) created through statistical simulation with clinically plausible parameter distributions. The synthetic dataset includes the following predictors:
- **Demographics**: Age (30-80 years)
- **Anthropometrics**: BMI (normal distribution, mean=27, sd=4)
- **Biomarkers**: Glucose, HDL/LDL cholesterol, triglycerides, blood pressure, heart rate
- **Lifestyle factors**: Smoking, alcohol use, physical activity, diet score, sleep hours
- **Risk factors**: Stress levels, family history (diabetes, stroke, heart disease)

**Synthetic Data Rationale:**
- Enables open-source development without privacy concerns
- Provides consistent baseline for methodology development using realistic parameter ranges
- Allows controlled experimentation with model architectures
- Uses logistic regression relationships to generate plausible risk outcomes
- **Requires validation on real clinical data for medical applications**

## 👨‍💻 Developer

**Taiwo Michael Ayeni**

---

*Built with ❤️ using Streamlit and Bayesian Machine Learning*
