"""
PredictRisk: Bayesian Cardiovascular Risk Assessment

A comprehensive framework for cardiovascular risk prediction using 
Bayesian statistical methods with uncertainty quantification.

This package provides:
- Bayesian logistic regression models for 6 cardiovascular conditions
- Uncertainty quantification through credible intervals
- Web-based risk assessment interface
- PDF report generation
- Synthetic data generation for research and education

Author: Taiwo Michael Ayeni
Institution: Northeastern University, College of Professional Studies Analytics
Email: ayeni.t@northeastern.edu
License: MIT License
"""

__version__ = "0.1.0"
__author__ = "Taiwo Michael Ayeni"
__email__ = "ayeni.t@northeastern.edu"
__institution__ = "Northeastern University"
__license__ = "MIT"

# Import main classes and functions
from .config import CONDITION_MAP, RISK_THRESHOLDS, APP_CONFIG
from .risk_calculator import RiskCalculator
from .pdf_generator import PDFReportGenerator
from .advice_engine import get_risk_advice, get_interpretation_text
from .data_generator import generate_synthetic_data, create_risk_outcomes
from .bayesian_models import BayesianCVDModels, load_model, predict_risk
from .utils import validate_input_data, format_risk_output, calculate_credible_intervals

# Define what gets imported with "from predictrisk import *"
__all__ = [
    # Configuration
    'CONDITION_MAP',
    'RISK_THRESHOLDS', 
    'APP_CONFIG',
    
    # Core classes
    'RiskCalculator',
    'PDFReportGenerator',
    'BayesianCVDModels',
    
    # Main functions
    'get_risk_advice',
    'get_interpretation_text',
    'generate_synthetic_data',
    'create_risk_outcomes',
    'load_model',
    'predict_risk',
    
    # Utility functions
    'validate_input_data',
    'format_risk_output',
    'calculate_credible_intervals'
]

# Package metadata
__package_info__ = {
    'name': 'predictrisk',
    'version': __version__,
    'description': 'Bayesian Cardiovascular Risk Assessment with Uncertainty Quantification',
    'author': __author__,
    'author_email': __email__,
    'license': __license__,
    'url': 'https://github.com/ayeni-T/PredictRisk',
    'keywords': ['bayesian', 'cardiovascular', 'risk-assessment', 'machine-learning', 'healthcare'],
    'python_requires': '>=3.8',
}

def get_package_info():
    """Return package information dictionary."""
    return __package_info__.copy()

def print_info():
    """Print package information."""
    print(f"PredictRisk v{__version__}")
    print(f"Author: {__author__}")
    print(f"Institution: {__institution__}")
    print(f"Email: {__email__}")
    print(f"License: {__license__}")
    print("\nBayesian Cardiovascular Risk Assessment Framework")
    print("For more information: https://github.com/ayeni-T/PredictRisk")
