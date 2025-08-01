"""
Risk calculation and prediction module for PredictRisk.

This module handles the core Bayesian risk assessment functionality,
including model loading, prediction, and uncertainty quantification.
"""

import pandas as pd
import numpy as np
import bambi as bmb
import arviz as az
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, List
from .config import CONDITION_MAP, MODEL_FORMULA_TEMPLATE, DATA_CONFIG, BAYESIAN_CONFIG
from .utils import validate_input_data, calculate_credible_intervals

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskCalculator:
    """
    Main class for cardiovascular risk calculation using Bayesian models.
    
    This class provides methods to load trained Bayesian models, make predictions,
    and quantify uncertainty in risk assessments.
    """
    
    def __init__(self, data_file: str = None):
        """
        Initialize the RiskCalculator.
        
        Parameters:
        -----------
        data_file : str, optional
            Path to the synthetic dataset file. If None, uses default from config.
        """
        self.data_file = data_file or DATA_CONFIG["synthetic_data_file"]
        self.data = None
        self.models = {}
        self.model_summaries = {}
        self._load_data()
        
    def _load_data(self) -> None:
        """Load the synthetic cardiovascular dataset."""
        try:
            self.data = pd.read_csv(self.data_file)
            logger.info(f"Loaded dataset with {len(self.data)} records and {len(self.data.columns)} features")
        except FileNotFoundError:
            logger.error(f"Dataset file {self.data_file} not found")
            raise
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
            
    def load_model(self, condition: str) -> bmb.Model:
        """
        Load a trained Bayesian model for a specific condition.
        
        Parameters:
        -----------
        condition : str
            Name of the cardiovascular condition
            
        Returns:
        --------
        bmb.Model
            Loaded Bayesian model
        """
        if condition not in CONDITION_MAP:
            raise ValueError(f"Unknown condition: {condition}. Available: {list(CONDITION_MAP.keys())}")
            
        condition_info = CONDITION_MAP[condition]
        model_file = condition_info["file"]
        formula = condition_info["formula"]
        
        try:
            # Create the model structure
            formula_str = MODEL_FORMULA_TEMPLATE.format(formula=formula)
            model = bmb.Model(formula_str, data=self.data, family=BAYESIAN_CONFIG["family"])
            
            # Load the fitted parameters
            idata = az.from_netcdf(model_file)
            
            # Store model and inference data
            self.models[condition] = {"model": model, "idata": idata}
            
            logger.info(f"Successfully loaded model for {condition}")
            return model
            
        except FileNotFoundError:
            logger.error(f"Model file {model_file} not found")
            raise
        except Exception as e:
            logger.error(f"Error loading model for {condition}: {e}")
            raise
    
    def predict_risk(self, condition: str, input_data: Dict[str, Union[int, float]]) -> Tuple[float, Tuple[float, float], Dict]:
        """
        Predict cardiovascular risk for a given condition and input data.
        
        Parameters:
        -----------
        condition : str
            Name of the cardiovascular condition
        input_data : dict
            Dictionary containing risk factor values
            
        Returns:
        --------
        tuple
            (mean_risk, credible_interval, model_summary)
        """
        # Validate input data
        validated_data = validate_input_data(input_data)
        
        # Load model if not already loaded
        if condition not in self.models:
            self.load_model(condition)
            
        model_info = self.models[condition]
        model = model_info["model"]
        idata = model_info["idata"]
        
        try:
            # Prepare data for prediction
            new_data = pd.DataFrame([validated_data])
            
            # Make prediction
            predictions = model.predict(
                idata=idata, 
                data=new_data, 
                kind="response_params", 
                inplace=False
            )
            
            # Extract probabilities
            probs = predictions.posterior["p"].values.flatten()
            
            # Calculate statistics
            mean_risk = np.mean(probs)
            credible_interval = calculate_credible_intervals(probs)
            
            # Get model summary
            summary = self._get_model_summary(idata)
            
            logger.info(f"Prediction completed for {condition}: {mean_risk:.3f} [{credible_interval[0]:.3f}, {credible_interval[1]:.3f}]")
            
            return mean_risk, credible_interval, summary
            
        except Exception as e:
            logger.error(f"Error making prediction for {condition}: {e}")
            raise
    
    def _get_model_summary(self, idata) -> Dict:
        """
        Get summary statistics for the model.
        
        Parameters:
        -----------
        idata : arviz.InferenceData
            Inference data from the fitted model
            
        Returns:
        --------
        dict
            Model summary statistics
        """
        try:
            summary = az.summary(idata, kind="stats", round_to=3)
            coef_summary = summary.loc[~summary.index.str.startswith("Intercept")]
            
            return {
                "coefficients": coef_summary.to_dict(),
                "r_hat_range": (summary["r_hat"].min(), summary["r_hat"].max()),
                "ess_range": (summary["ess_bulk"].min(), summary["ess_bulk"].max()),
                "convergence_status": "Good" if summary["r_hat"].max() <= 1.01 else "Check convergence"
            }
        except Exception as e:
            logger.warning(f"Could not generate model summary: {e}")
            return {}
    
    def batch_predict(self, condition: str, input_data_list: List[Dict]) -> List[Tuple[float, Tuple[float, float]]]:
        """
        Make predictions for multiple input datasets.
        
        Parameters:
        -----------
        condition : str
            Name of the cardiovascular condition
        input_data_list : list
            List of input data dictionaries
            
        Returns:
        --------
        list
            List of (mean_risk, credible_interval) tuples
        """
        results = []
        
        for i, input_data in enumerate(input_data_list):
            try:
                mean_risk, ci, _ = self.predict_risk(condition, input_data)
                results.append((mean_risk, ci))
            except Exception as e:
                logger.warning(f"Failed to predict for input {i}: {e}")
                results.append((np.nan, (np.nan, np.nan)))
                
        return results
    
    def get_feature_importance(self, condition: str) -> Dict[str, float]:
        """
        Calculate feature importance based on coefficient magnitudes.
        
        Parameters:
        -----------
        condition : str
            Name of the cardiovascular condition
            
        Returns:
        --------
        dict
            Feature importance scores
        """
        if condition not in self.models:
            self.load_model(condition)
            
        idata = self.models[condition]["idata"]
        
        try:
            summary = az.summary(idata, kind="stats")
            coef_summary = summary.loc[~summary.index.str.startswith("Intercept")]
            
            # Use absolute mean values as importance scores
            importance = abs(coef_summary["mean"]).sort_values(ascending=False)
            
            return importance.to_dict()
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
            return {}
    
    def validate_model_convergence(self, condition: str) -> Dict[str, Union[bool, float, str]]:
        """
        Validate that a model has converged properly.
        
        Parameters:
        -----------
        condition : str
            Name of the cardiovascular condition
            
        Returns:
        --------
        dict
            Convergence validation results
        """
        if condition not in self.models:
            self.load_model(condition)
            
        idata = self.models[condition]["idata"]
        
        try:
            summary = az.summary(idata)
            
            # Check R-hat (should be <= 1.01)
            max_rhat = summary["r_hat"].max()
            rhat_ok = max_rhat <= 1.01
            
            # Check effective sample size (should be > 400)
            min_ess = summary["ess_bulk"].min()
            ess_ok = min_ess > 400
            
            # Check for divergences
            divergences = az.summary(idata, kind="diagnostics").get("diverging", [])
            no_divergences = len(divergences) == 0 if isinstance(divergences, list) else divergences == 0
            
            overall_convergence = rhat_ok and ess_ok and no_divergences
            
            return {
                "converged": overall_convergence,
                "max_rhat": float(max_rhat),
                "min_ess": float(min_ess),
                "rhat_ok": rhat_ok,
                "ess_ok": ess_ok,
                "no_divergences": no_divergences,
                "status": "Good convergence" if overall_convergence else "Check convergence diagnostics"
            }
            
        except Exception as e:
            logger.error(f"Error validating convergence: {e}")
            return {"converged": False, "status": f"Error: {e}"}
    
    def get_available_conditions(self) -> List[str]:
        """Get list of available cardiovascular conditions."""
        return list(CONDITION_MAP.keys())
    
    def get_condition_info(self, condition: str) -> Dict:
        """Get detailed information about a condition."""
        return CONDITION_MAP.get(condition, {})


# Convenience functions for direct use
def load_model(condition: str, data_file: str = None) -> bmb.Model:
    """
    Load a trained Bayesian model for a specific condition.
    
    Parameters:
    -----------
    condition : str
        Name of the cardiovascular condition
    data_file : str, optional
        Path to the dataset file
        
    Returns:
    --------
    bmb.Model
        Loaded Bayesian model
    """
    calculator = RiskCalculator(data_file)
    return calculator.load_model(condition)


def predict_risk(condition: str, input_data: Dict[str, Union[int, float]], 
                data_file: str = None) -> Tuple[float, Tuple[float, float], Dict]:
    """
    Predict cardiovascular risk for a given condition and input data.
    
    Parameters:
    -----------
    condition : str
        Name of the cardiovascular condition
    input_data : dict
        Dictionary containing risk factor values
    data_file : str, optional
        Path to the dataset file
        
    Returns:
    --------
    tuple
        (mean_risk, credible_interval, model_summary)
    """
    calculator = RiskCalculator(data_file)
    return calculator.predict_risk(condition, input_data)


def quick_risk_assessment(input_data: Dict[str, Union[int, float]], 
                         conditions: List[str] = None) -> Dict[str, Dict]:
    """
    Perform quick risk assessment for multiple conditions.
    
    Parameters:
    -----------
    input_data : dict
        Dictionary containing risk factor values
    conditions : list, optional
        List of conditions to assess. If None, assess all conditions.
        
    Returns:
    --------
    dict
        Risk assessment results for each condition
    """
    calculator = RiskCalculator()
    
    if conditions is None:
        conditions = calculator.get_available_conditions()
    
    results = {}
    
    for condition in conditions:
        try:
            mean_risk, ci, summary = calculator.predict_risk(condition, input_data)
            results[condition] = {
                "mean_risk": mean_risk,
                "credible_interval": ci,
                "risk_category": "Low" if mean_risk < 0.4 else "Moderate" if mean_risk < 0.7 else "High",
                "model_summary": summary
            }
        except Exception as e:
            logger.warning(f"Failed to assess {condition}: {e}")
            results[condition] = {"error": str(e)}
    
    return results
