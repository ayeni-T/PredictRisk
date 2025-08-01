"""
Utility functions for PredictRisk package.

This module contains helper functions for data validation, formatting,
statistical calculations, and other common operations used throughout
the package.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, List, Union, Optional, Any
from .config import INPUT_RANGES, BINARY_INPUTS, RISK_THRESHOLDS

# Configure logging
logger = logging.getLogger(__name__)


def validate_input_data(input_data: Dict[str, Union[int, float, str]]) -> Dict[str, Union[int, float]]:
    """
    Validate and clean input data for risk prediction.
    
    Parameters:
    -----------
    input_data : dict
        Raw input data from user interface
        
    Returns:
    --------
    dict
        Validated and cleaned input data
        
    Raises:
    -------
    ValueError
        If input data is invalid or out of range
    """
    validated_data = {}
    errors = []
    
    # Validate numerical inputs
    for field, config in INPUT_RANGES.items():
        if field in input_data:
            value = input_data[field]
            
            # Convert to appropriate type
            try:
                if isinstance(config["default"], float):
                    value = float(value)
                else:
                    value = int(value)
            except (ValueError, TypeError):
                errors.append(f"{field}: Invalid numeric value")
                continue
            
            # Check range
            if not (config["min"] <= value <= config["max"]):
                errors.append(f"{field}: Value {value} out of range [{config['min']}, {config['max']}]")
                continue
                
            validated_data[field] = value
        else:
            # Use default value if not provided
            validated_data[field] = config["default"]
            logger.warning(f"Using default value for {field}: {config['default']}")
    
    # Validate binary inputs
    for field, config in BINARY_INPUTS.items():
        if field in input_data:
            value = input_data[field]
            
            # Convert string responses to binary
            if isinstance(value, str):
                if value.lower() in ["yes", "y", "true", "1"]:
                    validated_data[field] = 1
                elif value.lower() in ["no", "n", "false", "0"]:
                    validated_data[field] = 0
                else:
                    errors.append(f"{field}: Invalid option '{value}'. Must be one of {config['options']}")
                    continue
            elif isinstance(value, (int, float)):
                # Direct binary value
                if value in [0, 1]:
                    validated_data[field] = int(value)
                else:
                    errors.append(f"{field}: Binary value must be 0 or 1, got {value}")
                    continue
            else:
                errors.append(f"{field}: Invalid type for binary field")
                continue
        else:
            # Use default value
            default_option = config["default"]
            validated_data[field] = 1 if default_option.lower() == "yes" else 0
            logger.warning(f"Using default value for {field}: {default_option}")
    
    if errors:
        error_msg = "Input validation errors:\n" + "\n".join(errors)
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info("Input data validation successful")
    return validated_data


def calculate_credible_intervals(samples: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Calculate credible intervals from posterior samples.
    
    Parameters:
    -----------
    samples : np.ndarray
        Posterior samples
    alpha : float, default 0.05
        Significance level (0.05 for 95% CI)
        
    Returns:
    --------
    tuple
        (lower_bound, upper_bound)
    """
    if len(samples) == 0:
        logger.warning("Empty samples array provided")
        return (np.nan, np.nan)
    
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(samples, lower_percentile)
    upper_bound = np.percentile(samples, upper_percentile)
    
    return (float(lower_bound), float(upper_bound))


def format_risk_output(mean_risk: float, credible_interval: Tuple[float, float], 
                      decimals: int = 3) -> Dict[str, Any]:
    """
    Format risk prediction output for display.
    
    Parameters:
    -----------
    mean_risk : float
        Mean risk prediction
    credible_interval : tuple
        (lower_bound, upper_bound)
    decimals : int, default 3
        Number of decimal places
        
    Returns:
    --------
    dict
        Formatted risk output
    """
    # Determine risk category
    risk_category = get_risk_category(mean_risk)
    
    # Format percentages
    mean_percent = round(mean_risk * 100, decimals)
    ci_lower_percent = round(credible_interval[0] * 100, decimals)
    ci_upper_percent = round(credible_interval[1] * 100, decimals)
    
    # Calculate uncertainty width
    uncertainty_width = ci_upper_percent - ci_lower_percent
    
    return {
        "mean_risk": mean_risk,
        "mean_risk_percent": mean_percent,
        "credible_interval": credible_interval,
        "credible_interval_percent": (ci_lower_percent, ci_upper_percent),
        "risk_category": risk_category,
        "uncertainty_width": uncertainty_width,
        "formatted_output": f"{mean_percent}% [{ci_lower_percent}%-{ci_upper_percent}%]"
    }


def get_risk_category(risk_score: float) -> str:
    """
    Determine risk category based on risk score.
    
    Parameters:
    -----------
    risk_score : float
        Risk probability (0-1)
        
    Returns:
    --------
    str
        Risk category ('Low Risk', 'Moderate Risk', 'High Risk')
    """
    if risk_score < RISK_THRESHOLDS["moderate"]:
        return "Low Risk"
    elif risk_score < RISK_THRESHOLDS["high"]:
        return "Moderate Risk"
    else:
        return "High Risk"


def calculate_risk_statistics(samples: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive statistics from risk samples.
    
    Parameters:
    -----------
    samples : np.ndarray
        Posterior risk samples
        
    Returns:
    --------
    dict
        Statistical summary
    """
    if len(samples) == 0:
        return {}
    
    stats = {
        "mean": float(np.mean(samples)),
        "median": float(np.median(samples)),
        "std": float(np.std(samples)),
        "var": float(np.var(samples)),
        "min": float(np.min(samples)),
        "max": float(np.max(samples)),
        "q25": float(np.percentile(samples, 25)),
        "q75": float(np.percentile(samples, 75)),
        "iqr": float(np.percentile(samples, 75) - np.percentile(samples, 25)),
        "skewness": float(calculate_skewness(samples)),
        "kurtosis": float(calculate_kurtosis(samples))
    }
    
    # Add credible intervals
    for alpha in [0.05, 0.1, 0.2]:  # 95%, 90%, 80% CI
        ci_level = int((1 - alpha) * 100)
        lower, upper = calculate_credible_intervals(samples, alpha)
        stats[f"ci_{ci_level}_lower"] = lower
        stats[f"ci_{ci_level}_upper"] = upper
        stats[f"ci_{ci_level}_width"] = upper - lower
    
    return stats


def calculate_skewness(data: np.ndarray) -> float:
    """Calculate skewness of data."""
    if len(data) < 3:
        return np.nan
    
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    
    if std == 0:
        return 0.0
    
    n = len(data)
    skew = (n / ((n - 1) * (n - 2))) * np.sum(((data - mean) / std) ** 3)
    return skew


def calculate_kurtosis(data: np.ndarray) -> float:
    """Calculate kurtosis of data."""
    if len(data) < 4:
        return np.nan
    
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    
    if std == 0:
        return 0.0
    
    n = len(data)
    kurt = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * np.sum(((data - mean) / std) ** 4)
    kurt -= 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return kurt


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if division by zero.
    
    Parameters:
    -----------
    numerator : float
        Numerator
    denominator : float  
        Denominator
    default : float, default 0.0
        Value to return if denominator is zero
        
    Returns:
    --------
    float
        Result of division or default value
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (ZeroDivisionError, TypeError):
        return default


def create_input_summary(input_data: Dict[str, Union[int, float]]) -> str:
    """
    Create a human-readable summary of input data.
    
    Parameters:
    -----------
    input_data : dict
        Validated input data
        
    Returns:
    --------
    str
        Formatted summary string
    """
    summary_lines = []
    
    # Demographic info
    age = input_data.get("age", "Unknown")
    bmi = input_data.get("bmi", "Unknown")
    summary_lines.append(f"Patient Profile: {age} years old, BMI {bmi}")
    
    # Vital signs
    systolic = input_data.get("systolic_bp", "Unknown")
    diastolic = input_data.get("diastolic_bp", "Unknown")
    heart_rate = input_data.get("heart_rate", "Unknown")
    summary_lines.append(f"Vitals: BP {systolic}/{diastolic} mmHg, HR {heart_rate} bpm")
    
    # Biomarkers
    glucose = input_data.get("glucose", "Unknown")
    summary_lines.append(f"Glucose: {glucose} mg/dL")
    
    # Lifestyle factors
    lifestyle = []
    if input_data.get("smoking_status", 0) == 1:
        lifestyle.append("smoker")
    if input_data.get("physical_activity", 1) == 0:
        lifestyle.append("physically inactive")
    if input_data.get("alcohol_use", 0) == 1:
        lifestyle.append("alcohol use")
    
    sleep_hours = input_data.get("sleep_hours", "Unknown")
    stress_score = input_data.get("stress_score", "Unknown")
    
    lifestyle_str = ", ".join(lifestyle) if lifestyle else "no major lifestyle risk factors"
    summary_lines.append(f"Lifestyle: {lifestyle_str}")
    summary_lines.append(f"Sleep: {sleep_hours} hours, Stress level: {stress_score}/10")
    
    return "\n".join(summary_lines)


def convert_to_serializable(obj: Any) -> Any:
    """
    Convert numpy types to Python native types for JSON serialization.
    
    Parameters:
    -----------
    obj : Any
        Object to convert
        
    Returns:
    --------
    Any
        Serializable version of object
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def log_model_performance(condition: str, mean_risk: float, ci: Tuple[float, float], 
                         input_summary: str) -> None:
    """
    Log model performance and prediction details.
    
    Parameters:
    -----------
    condition : str
        Cardiovascular condition
    mean_risk : float
        Mean risk prediction
    ci : tuple
        Credible interval
    input_summary : str
        Summary of input data
    """
    risk_category = get_risk_category(mean_risk)
    
    logger.info(f"PREDICTION - {condition}")
    logger.info(f"Risk: {mean_risk:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] ({risk_category})")
    logger.info(f"Input: {input_summary}")


def check_data_quality(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Check data quality and return summary statistics.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset to check
        
    Returns:
    --------
    dict
        Data quality report
    """
    report = {
        "n_rows": len(data),
        "n_cols": len(data.columns),
        "missing_values": data.isnull().sum().to_dict(),
        "duplicate_rows": data.duplicated().sum(),
        "numeric_cols": list(data.select_dtypes(include=[np.number]).columns),
        "categorical_cols": list(data.select_dtypes(include=["object", "category"]).columns),
        "memory_usage_mb": data.memory_usage(deep=True).sum() / 1024**2
    }
    
    # Check for suspicious values
    numeric_data = data.select_dtypes(include=[np.number])
    report["outliers"] = {}
    
    for col in numeric_data.columns:
        q1 = numeric_data[col].quantile(0.25)
        q3 = numeric_data[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = ((numeric_data[col] < lower_bound) | (numeric_data[col] > upper_bound)).sum()
        report["outliers"][col] = int(outliers)
    
    return report


def generate_random_seed(base_seed: int = 42) -> int:
    """
    Generate a random seed based on current time and base seed.
    
    Parameters:
    -----------
    base_seed : int, default 42
        Base seed value
        
    Returns:
    --------
    int
        Generated random seed
    """
    import time
    return int((time.time() * 1000) % 1000000) + base_seed
