"""
Configuration settings and constants for PredictRisk application.

This module contains all configuration parameters, model mappings,
risk thresholds, and application settings used throughout the package.
"""

import os
from pathlib import Path

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

CONDITION_MAP = {
    "Stroke": {
        "file": "stroke_model_fit.nc", 
        "formula": "risk_stroke",
        "description": "Cerebrovascular accident risk assessment",
        "risk_factors": ["age", "bmi", "glucose", "smoking_status", "physical_activity", 
                        "systolic_bp", "stress_score"]
    },
    "Heart Disease": {
        "file": "heart_disease_model_fit.nc", 
        "formula": "risk_heart_disease",
        "description": "Coronary heart disease and related conditions",
        "risk_factors": ["age", "bmi", "glucose", "smoking_status", "alcohol_use", 
                        "physical_activity", "stress_score"]
    },
    "Hypertension": {
        "file": "hypertension_model_fit.nc", 
        "formula": "risk_hypertension",
        "description": "High blood pressure (>140/90 mmHg)",
        "risk_factors": ["age", "bmi", "systolic_bp", "physical_activity", "stress_score"]
    },
    "Heart Failure": {
        "file": "heart_failure_model_fit.nc", 
        "formula": "risk_heart_failure",
        "description": "Reduced cardiac pumping capacity",
        "risk_factors": ["age", "bmi", "glucose", "smoking_status", "heart_rate", 
                        "systolic_bp", "stress_score"]
    },
    "Atrial Fibrillation (AFib)": {
        "file": "afib_model_fit.nc", 
        "formula": "risk_afib",
        "description": "Irregular heart rhythm disorder",
        "risk_factors": ["age", "heart_rate", "smoking_status", "stress_score", "alcohol_use"]
    },
    "Peripheral Artery Disease (PAD)": {
        "file": "pad_model_fit.nc", 
        "formula": "risk_pad",
        "description": "Reduced blood flow to limbs",
        "risk_factors": ["age", "smoking_status", "physical_activity", "stress_score"]
    }
}

# =============================================================================
# RISK ASSESSMENT THRESHOLDS
# =============================================================================

RISK_THRESHOLDS = {
    "low": 0.0,        # 0-40%: Low risk
    "moderate": 0.4,   # 40-70%: Moderate risk  
    "high": 0.7        # 70%+: High risk
}

RISK_CATEGORIES = {
    "Low Risk": {"threshold": 0.4, "color": "green", "icon": "✅"},
    "Moderate Risk": {"threshold": 0.7, "color": "orange", "icon": "⚠️"},
    "High Risk": {"threshold": 1.0, "color": "red", "icon": "🚨"}
}

# =============================================================================
# APPLICATION CONFIGURATION
# =============================================================================

APP_CONFIG = {
    "title": "🧠 PredictRisk: Cardiovascular Diagnostic Tool",
    "page_icon": "🧠",
    "layout": "centered",
    "initial_sidebar_state": "auto",
    "menu_items": {
        'Get Help': 'https://github.com/ayeni-T/PredictRisk',
        'Report a bug': "https://github.com/ayeni-T/PredictRisk/issues",
        'About': "Bayesian Cardiovascular Risk Assessment Tool"
    }
}

# =============================================================================
# MODEL PARAMETERS
# =============================================================================

MODEL_FORMULA_TEMPLATE = (
    "{formula} ~ age + bmi + glucose + smoking_status + physical_activity + "
    "systolic_bp + diastolic_bp + heart_rate + alcohol_use + sleep_hours + stress_score"
)

BAYESIAN_CONFIG = {
    "family": "bernoulli",
    "draws": 1000,
    "tune": 1000,
    "chains": 4,
    "target_accept": 0.8,
    "max_treedepth": 10
}

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

DATA_CONFIG = {
    "synthetic_data_file": "multi_cvd_dataset.csv",
    "sample_size": 500,
    "random_seed": 42,
    "test_size": 0.2,
    "validation_size": 0.1
}

# Input validation ranges
INPUT_RANGES = {
    "age": {"min": 30, "max": 90, "default": 50, "unit": "years"},
    "bmi": {"min": 15.0, "max": 45.0, "default": 25.0, "unit": "kg/m²"},
    "glucose": {"min": 70, "max": 200, "default": 100, "unit": "mg/dL"},
    "systolic_bp": {"min": 90, "max": 200, "default": 120, "unit": "mmHg"},
    "diastolic_bp": {"min": 60, "max": 130, "default": 80, "unit": "mmHg"},
    "heart_rate": {"min": 40, "max": 150, "default": 75, "unit": "bpm"},
    "sleep_hours": {"min": 3.0, "max": 10.0, "default": 6.5, "unit": "hours"},
    "stress_score": {"min": 1, "max": 10, "default": 5, "unit": "1-10 scale"}
}

BINARY_INPUTS = {
    "smoking_status": {"options": ["No", "Yes"], "default": "No"},
    "alcohol_use": {"options": ["No", "Yes"], "default": "No"},
    "physical_activity": {"options": ["Yes", "No"], "default": "Yes"}
}

# =============================================================================
# FILE PATHS
# =============================================================================

# Get package root directory
PACKAGE_ROOT = Path(__file__).parent
PROJECT_ROOT = PACKAGE_ROOT.parent

# Data and model paths
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Ensure directories exist
for directory in [DATA_DIR, MODEL_DIR, REPORTS_DIR]:
    directory.mkdir(exist_ok=True)

# =============================================================================
# HELP TEXT AND TOOLTIPS
# =============================================================================

HELP_TEXT = {
    "age": "Your current age in years.",
    "bmi": "Body Mass Index — a measure of body fat based on height and weight.",
    "glucose": "Fasting blood glucose level in mg/dL.",
    "systolic_bp": "Top number in your blood pressure reading (pressure during heartbeat).",
    "diastolic_bp": "Bottom number in your blood pressure reading (pressure between beats).",
    "heart_rate": "Resting heart rate in beats per minute.",
    "smoking_status": "Current smoking status - have you smoked recently?",
    "alcohol_use": "Do you currently consume alcoholic beverages?",
    "physical_activity": "Do you engage in regular physical exercise?",
    "sleep_hours": "Average number of hours you sleep per night.",
    "stress_score": "Rate your daily stress level on a scale from 1 (low) to 10 (high)."
}

# =============================================================================
# ENVIRONMENT VARIABLES
# =============================================================================

# PyTensor configuration
os.environ["PYTENSOR_FLAGS"] = "cxx="

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_condition_info(condition_name):
    """Get detailed information about a specific condition."""
    return CONDITION_MAP.get(condition_name, {})

def get_risk_category(risk_score):
    """Determine risk category based on score."""
    if risk_score < RISK_THRESHOLDS["moderate"]:
        return "Low Risk"
    elif risk_score < RISK_THRESHOLDS["high"]:
        return "Moderate Risk"
    else:
        return "High Risk"

def validate_config():
    """Validate configuration settings."""
    assert len(CONDITION_MAP) == 6, "Should have 6 cardiovascular conditions"
    assert all(isinstance(v, dict) for v in CONDITION_MAP.values()), "All conditions should be dictionaries"
    assert 0 < RISK_THRESHOLDS["moderate"] < RISK_THRESHOLDS["high"] < 1, "Risk thresholds should be ordered"
    return True

# Validate configuration on import
validate_config()
