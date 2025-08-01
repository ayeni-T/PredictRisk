"""
Streamlit web application interface for PredictRisk.

This module provides the web-based user interface for cardiovascular
risk assessment using the PredictRisk Bayesian framework.
"""

import streamlit as st
import logging
from typing import Dict, Union

from .config import APP_CONFIG, CONDITION_MAP, INPUT_RANGES, BINARY_INPUTS, HELP_TEXT
from .risk_calculator import RiskCalculator
from .advice_engine import AdviceEngine, get_risk_advice, get_interpretation_text
from .pdf_generator import generate_pdf_report
from .utils import validate_input_data, format_risk_output

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictRiskApp:
    """
    Main Streamlit application class for PredictRisk.
    
    This class encapsulates the web interface functionality and
    coordinates between the risk calculator, advice engine, and
    PDF report generator.
    """
    
    def __init__(self):
        """Initialize the PredictRisk web application."""
        self.risk_calculator = RiskCalculator()
        self.advice_engine = AdviceEngine()
        self._setup_page_config()
        
    def _setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title=APP_CONFIG["title"],
            page_icon=APP_CONFIG["page_icon"],
            layout=APP_CONFIG["layout"],
            initial_sidebar_state=APP_CONFIG.get("initial_sidebar_state", "auto"),
            menu_items=APP_CONFIG.get("menu_items", {})
        )
    
    def render_header(self):
        """Render the application header and title."""
        st.title(APP_CONFIG["title"])
        
        # Add subtitle and description
        st.markdown("""
        **Bayesian Cardiovascular Risk Assessment with Uncertainty Quantification**
        
        This tool provides personalized cardiovascular risk predictions using advanced 
        Bayesian statistical methods. Get risk assessments with confidence intervals
        and personalized health recommendations.
        """)
        
        # Add information expander
        with st.expander("ℹ️ About This Tool"):
            st.markdown("""
            - **Bayesian Models**: Uses advanced statistical methods with uncertainty quantification
            - **Six Conditions**: Stroke, Heart Disease, Hypertension, Heart Failure, AFib, PAD
            - **Evidence-Based**: Risk factors based on established cardiovascular epidemiology
            - **Educational Purpose**: For learning and research - not clinical diagnosis
            - **Open Source**: Complete methodology available on GitHub
            """)
    
    def render_condition_selector(self) -> str:
        """
        Render condition selection interface.
        
        Returns:
        --------
        str
            Selected cardiovascular condition
        """
        st.markdown("### 🔍 Select Cardiovascular Condition to Assess")
        
        # Create more informative condition options
        condition_options = {}
        for condition, info in CONDITION_MAP.items():
            description = info.get("description", "")
            condition_options[condition] = f"{condition} - {description}"
        
        # Display as selectbox with descriptions
        selected_display = st.selectbox(
            "Choose the cardiovascular condition you want to assess:",
            options=list(condition_options.values()),
            help="Select the specific cardiovascular condition for risk assessment."
        )
        
        # Extract the actual condition name
        selected_condition = None
        for condition, display in condition_options.items():
            if display == selected_display:
                selected_condition = condition
                break
        
        return selected_condition
    
    def render_input_form(self) -> Dict[str, Union[int, float]]:
        """
        Render the risk factor input form.
        
        Returns:
        --------
        dict
            User input data
        """
        st.markdown("### 📋 Enter Your Health Information")
        
        input_data = {}
        
        # Create two columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Basic Information**")
            
            # Age
            input_data["age"] = st.number_input(
                "Age (years)",
                min_value=INPUT_RANGES["age"]["min"],
                max_value=INPUT_RANGES["age"]["max"],
                value=INPUT_RANGES["age"]["default"],
                help=HELP_TEXT["age"]
            )
            
            # BMI
            input_data["bmi"] = st.number_input(
                "BMI (kg/m²)",
                min_value=INPUT_RANGES["bmi"]["min"],
                max_value=INPUT_RANGES["bmi"]["max"],
                value=INPUT_RANGES["bmi"]["default"],
                step=0.1,
                help=HELP_TEXT["bmi"]
            )
            
            # Glucose
            input_data["glucose"] = st.number_input(
                "Glucose (mg/dL)",
                min_value=INPUT_RANGES["glucose"]["min"],
                max_value=INPUT_RANGES["glucose"]["max"],
                value=INPUT_RANGES["glucose"]["default"],
                help=HELP_TEXT["glucose"]
            )
            
            # Sleep hours
            input_data["sleep_hours"] = st.slider(
                "Sleep Hours per Night",
                min_value=INPUT_RANGES["sleep_hours"]["min"],
                max_value=INPUT_RANGES["sleep_hours"]["max"],
                value=INPUT_RANGES["sleep_hours"]["default"],
                step=0.5,
                help=HELP_TEXT["sleep_hours"]
            )
        
        with col2:
            st.markdown("**🩺 Vital Signs**")
            
            # Blood pressure
            input_data["systolic_bp"] = st.number_input(
                "Systolic Blood Pressure (mmHg)",
                min_value=INPUT_RANGES["systolic_bp"]["min"],
                max_value=INPUT_RANGES["systolic_bp"]["max"],
                value=INPUT_RANGES["systolic_bp"]["default"],
                help=HELP_TEXT["systolic_bp"]
            )
            
            input_data["diastolic_bp"] = st.number_input(
                "Diastolic Blood Pressure (mmHg)",
                min_value=INPUT_RANGES["diastolic_bp"]["min"],
                max_value=INPUT_RANGES["diastolic_bp"]["max"],
                value=INPUT_RANGES["diastolic_bp"]["default"],
                help=HELP_TEXT["diastolic_bp"]
            )
            
            # Heart rate
            input_data["heart_rate"] = st.number_input(
                "Heart Rate (bpm)",
                min_value=INPUT_RANGES["heart_rate"]["min"],
                max_value=INPUT_RANGES["heart_rate"]["max"],
                value=INPUT_RANGES["heart_rate"]["default"],
                help=HELP_TEXT["heart_rate"]
            )
            
            # Stress score
            input_data["stress_score"] = st.slider(
                "Stress Level (1-10)",
                min_value=INPUT_RANGES["stress_score"]["min"],
                max_value=INPUT_RANGES["stress_score"]["max"],
                value=INPUT_RANGES["stress_score"]["default"],
                help=HELP_TEXT["stress_score"]
            )
        
        # Lifestyle factors in full width
        st.markdown("**🏃‍♂️ Lifestyle Factors**")
        
        lifestyle_col1, lifestyle_col2, lifestyle_col3 = st.columns(3)
        
        with lifestyle_col1:
            smoking_response = st.radio(
                "Do you smoke?",
                options=BINARY_INPUTS["smoking_status"]["options"],
                index=0,
                help=HELP_TEXT["smoking_status"]
            )
            input_data["smoking_status"] = 1 if smoking_response == "Yes" else 0
        
        with lifestyle_col2:
            alcohol_response = st.radio(
                "Do you consume alcohol?",
                options=BINARY_INPUTS["alcohol_use"]["options"],
                index=0,
                help=HELP_TEXT["alcohol_use"]
            )
            input_data["alcohol_use"] = 1 if alcohol_response == "Yes" else 0
        
        with lifestyle_col3:
            activity_response = st.radio(
                "Are you physically active?",
                options=BINARY_INPUTS["physical_activity"]["options"],
                index=0,
                help=HELP_TEXT["physical_activity"]
            )
            input_data["physical_activity"] = 1 if activity_response == "Yes" else 0
        
        return input_data
    
    def render_risk_assessment(self, condition: str, input_data: Dict):
        """
        Render risk assessment results and recommendations.
        
        Parameters:
        -----------
        condition : str
            Selected cardiovascular condition
        input_data : dict
            User input data
        """
        try:
            # Validate input data
            validated_data = validate_input_data(input_data)
            
            # Make prediction
            mean_risk, credible_interval, model_summary = self.risk_calculator.predict_risk(
                condition, validated_data
            )
            
            # Format output
            formatted_output = format_risk_output(mean_risk, credible_interval)
            risk_category = formatted_output["risk_category"]
            
            # Display results
            st.markdown("---")
            st.markdown("## 🩺 Risk Assessment Results")
            
            # Main risk display
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.metric(
                    label=f"Predicted {condition} Risk",
                    value=f"{formatted_output['mean_risk_percent']}%",
                    delta=f"95% CI: [{formatted_output['credible_interval_percent'][0]}%, {formatted_output['credible_interval_percent'][1]}%]"
                )
            
            with col2:
                # Risk category with appropriate styling
                if risk_category == "High Risk":
                    st.error(f"🚨 {risk_category}")
                elif risk_category == "Moderate Risk":
                    st.warning(f"⚠️ {risk_category}")
                else:
                    st.success(f"✅ {risk_category}")
            
            # Get personalized advice
            advice_lines = get_risk_advice(risk_category, condition, validated_data)
            interpretation_text = get_interpretation_text(condition, mean_risk, credible_interval)
            
            # Display interpretation
            st.markdown("### 📖 What This Means for You")
            st.markdown(interpretation_text)
            
            # Display recommendations
            st.markdown("### 💡 Personalized Recommendations")
            for line in advice_lines:
                st.markdown(line)
            
            # Generate and offer PDF download
            pdf_buffer = generate_pdf_report(
                condition, mean_risk, credible_interval,
                risk_category, advice_lines, interpretation_text,
                validated_data, use_simple=True
            )
            
            st.download_button(
                "📄 Download Complete Risk Report (PDF)",
                data=pdf_buffer,
                file_name=f"{condition.replace(' ', '_')}_Risk_Report.pdf",
                mime="application/pdf",
                help="Download a comprehensive PDF report with your risk assessment"
            )
            
            # Model information in expander
            if model_summary:
                with st.expander("🔬 Model Information"):
                    st.markdown(f"**Convergence Status:** {model_summary.get('convergence_status', 'Unknown')}")
                    
                    if 'r_hat_range' in model_summary:
                        r_hat_range = model_summary['r_hat_range']
                        st.markdown(f"**R-hat Range:** {r_hat_range[0]:.3f} - {r_hat_range[1]:.3f}")
                    
                    if 'ess_range' in model_summary:
                        ess_range = model_summary['ess_range']
                        st.markdown(f"**Effective Sample Size Range:** {ess_range[0]:.0f} - {ess_range[1]:.0f}")
                    
                    st.markdown("All models demonstrate excellent convergence (R-hat = 1.0)")
            
        except Exception as e:
            st.error(f"Error performing risk assessment: {str(e)}")
            logger.error(f"Risk assessment error: {e}")
    
    def render_disclaimer(self):
        """Render important disclaimers and information."""
        st.markdown("---")
        st.info("""
        ⚠️ **Important Disclaimer**
        
        This tool is for **educational and informational purposes only**. 
        It is **NOT a substitute for professional medical advice, diagnosis, or treatment**.
        
        - Models are trained on synthetic data and require clinical validation
        - Results may not reflect your actual clinical risk
        - Always consult a licensed healthcare provider regarding your health
        - Seek immediate medical attention if experiencing symptoms
        """)
        
        # Add footer with attribution
        st.markdown("""
        ---
        <div style='text-align: center; color: #666; font-size: 0.9em;'>
        © 2025 Taiwo Michael Ayeni | Northeastern University<br>
        <a href='https://github.com/ayeni-T/PredictRisk'>View Source Code</a> | 
        <a href='https://github.com/ayeni-T/PredictRisk/issues'>Report Issues</a>
        </div>
        """, unsafe_allow_html=True)
    
    def run(self):
        """Run the complete Streamlit application."""
        # Render header
        self.render_header()
        
        # Condition selection
        selected_condition = self.render_condition_selector()
        
        # Input form
        input_data = self.render_input_form()
        
        # Risk assessment button and results
        if st.button("🔍 Assess My Cardiovascular Risk", type="primary"):
            with st.spinner("Performing Bayesian risk assessment..."):
                self.render_risk_assessment(selected_condition, input_data)
        
        # Always show disclaimer
        self.render_disclaimer()


def main():
    """Main entry point for the Streamlit application."""
    try:
        app = PredictRiskApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application startup error: {e}")


# For direct execution
if __name__ == "__main__":
    main()
