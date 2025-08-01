"""
Synthetic cardiovascular data generation module for PredictRisk.

This module provides functions to generate realistic synthetic cardiovascular
risk datasets for research, education, and model development purposes.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

from .config import DATA_CONFIG, INPUT_RANGES, BINARY_INPUTS

# Configure logging
logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """
    Generator for synthetic cardiovascular risk datasets.
    
    This class creates realistic synthetic datasets with clinically plausible
    parameter distributions and appropriate correlations between risk factors
    and cardiovascular outcomes.
    """
    
    def __init__(self, random_seed: int = None):
        """
        Initialize the synthetic data generator.
        
        Parameters:
        -----------
        random_seed : int, optional
            Random seed for reproducibility
        """
        self.random_seed = random_seed or DATA_CONFIG["random_seed"]
        np.random.seed(self.random_seed)
        logger.info(f"Initialized SyntheticDataGenerator with seed {self.random_seed}")
    
    def generate_risk_factors(self, n_samples: int = None) -> pd.DataFrame:
        """
        Generate synthetic risk factor data.
        
        Parameters:
        -----------
        n_samples : int, optional
            Number of samples to generate
            
        Returns:
        --------
        pd.DataFrame
            Synthetic risk factor dataset
        """
        if n_samples is None:
            n_samples = DATA_CONFIG["sample_size"]
        
        logger.info(f"Generating {n_samples} synthetic risk factor records")
        
        # Set seed for reproducibility
        np.random.seed(self.random_seed)
        
        # Generate demographic variables
        age = np.random.randint(30, 80, size=n_samples)
        
        # Generate anthropometric measures
        bmi = np.random.normal(27, 4, size=n_samples)
        bmi = np.clip(bmi, 15, 45)  # Ensure realistic range
        
        # Generate biomarkers
        glucose = np.random.normal(100, 15, size=n_samples)
        glucose = np.clip(glucose, 70, 200)
        
        hdl_chol = np.random.normal(55, 10, size=n_samples)
        hdl_chol = np.clip(hdl_chol, 20, 100)
        
        ldl_chol = np.random.normal(130, 30, size=n_samples)
        ldl_chol = np.clip(ldl_chol, 50, 250)
        
        triglycerides = np.random.normal(150, 50, size=n_samples)
        triglycerides = np.clip(triglycerides, 50, 400)
        
        # Generate vital signs with realistic correlations
        # Blood pressure correlated with age and BMI
        age_factor = (age - 30) / 50  # Normalized age effect
        bmi_factor = (bmi - 25) / 10  # Normalized BMI effect
        
        systolic_bp = np.random.normal(130, 15, size=n_samples)
        systolic_bp += 10 * age_factor + 5 * bmi_factor  # Add correlations
        systolic_bp = np.clip(systolic_bp, 90, 200)
        
        diastolic_bp = np.random.normal(85, 10, size=n_samples)
        diastolic_bp += 5 * age_factor + 3 * bmi_factor
        diastolic_bp = np.clip(diastolic_bp, 60, 130)
        
        heart_rate = np.random.normal(75, 10, size=n_samples)
        heart_rate = np.clip(heart_rate, 40, 150)
        
        # Generate lifestyle factors
        alcohol_use = np.random.binomial(1, 0.3, size=n_samples)
        smoking_status = np.random.binomial(1, 0.25, size=n_samples)
        
        # Physical activity correlated with age (older = less active)
        activity_prob = 0.6 - 0.2 * age_factor
        activity_prob = np.clip(activity_prob, 0.2, 0.8)
        physical_activity = np.random.binomial(1, activity_prob, size=n_samples)
        
        diet_score = np.random.randint(1, 11, size=n_samples)
        
        # Sleep hours with realistic distribution
        sleep_hours = np.random.normal(6.5, 1.5, size=n_samples)
        sleep_hours = np.clip(sleep_hours, 3, 10)
        
        # Stress score correlated with lifestyle factors
        base_stress = np.random.randint(1, 11, size=n_samples)
        stress_modifier = smoking_status * 1 + (1 - physical_activity) * 1
        stress_score = np.clip(base_stress + stress_modifier, 1, 10)
        
        # Family history factors
        family_history_diabetes = np.random.binomial(1, 0.3, size=n_samples)
        family_history_stroke = np.random.binomial(1, 0.2, size=n_samples)
        family_history_heart_disease = np.random.binomial(1, 0.25, size=n_samples)
        
        # Create DataFrame
        data = pd.DataFrame({
            'age': age,
            'bmi': bmi,
            'glucose': glucose,
            'hdl_chol': hdl_chol,
            'ldl_chol': ldl_chol,
            'triglycerides': triglycerides,
            'systolic_bp': systolic_bp,
            'diastolic_bp': diastolic_bp,
            'heart_rate': heart_rate,
            'alcohol_use': alcohol_use,
            'smoking_status': smoking_status,
            'physical_activity': physical_activity,
            'diet_score': diet_score,
            'sleep_hours': sleep_hours,
            'stress_score': stress_score,
            'family_history_diabetes': family_history_diabetes,
            'family_history_stroke': family_history_stroke,
            'family_history_heart_disease': family_history_heart_disease
        })
        
        logger.info(f"Generated {len(data)} risk factor records with {len(data.columns)} features")
        return data
    
    def create_risk_outcomes(self, risk_factors_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate cardiovascular risk outcomes based on logistic regression relationships.
        
        Parameters:
        -----------
        risk_factors_df : pd.DataFrame
            Risk factor dataset
            
        Returns:
        --------
        pd.DataFrame
            Dataset with risk outcomes added
        """
        logger.info("Generating cardiovascular risk outcomes")
        
        df = risk_factors_df.copy()
        
        # Extract variables for readability
        age = df['age'].values
        bmi = df['bmi'].values
        glucose = df['glucose'].values
        systolic_bp = df['systolic_bp'].values
        heart_rate = df['heart_rate'].values
        smoking_status = df['smoking_status'].values
        physical_activity = df['physical_activity'].values
        alcohol_use = df['alcohol_use'].values
        stress_score = df['stress_score'].values
        ldl_chol = df['ldl_chol'].values
        triglycerides = df['triglycerides'].values
        family_history_stroke = df['family_history_stroke'].values
        family_history_heart_disease = df['family_history_heart_disease'].values
        
        # Define logistic regression relationships for each condition
        logistic_models = self._get_logistic_relationships()
        
        # Stroke risk
        logit_stroke = logistic_models['stroke'](
            age, bmi, glucose, smoking_status, family_history_stroke,
            physical_activity, systolic_bp, stress_score
        )
        
        # Heart disease risk
        logit_heart_disease = logistic_models['heart_disease'](
            age, bmi, ldl_chol, family_history_heart_disease,
            alcohol_use, smoking_status, triglycerides
        )
        
        # Heart failure risk
        logit_heart_failure = logistic_models['heart_failure'](
            age, bmi, glucose, smoking_status, family_history_heart_disease,
            stress_score, heart_rate, systolic_bp
        )
        
        # Hypertension risk
        logit_htn = logistic_models['hypertension'](
            age, bmi, systolic_bp, family_history_heart_disease,
            stress_score, physical_activity
        )
        
        # Atrial fibrillation risk
        logit_afib = logistic_models['afib'](
            age, heart_rate, smoking_status, stress_score,
            family_history_heart_disease
        )
        
        # Peripheral artery disease risk
        logit_pad = logistic_models['pad'](
            age, ldl_chol, smoking_status, stress_score,
            family_history_heart_disease, physical_activity
        )
        
        # Generate binary outcomes
        df['risk_stroke'] = np.random.binomial(1, self._sigmoid(logit_stroke))
        df['risk_heart_disease'] = np.random.binomial(1, self._sigmoid(logit_heart_disease))
        df['risk_heart_failure'] = np.random.binomial(1, self._sigmoid(logit_heart_failure))
        df['risk_hypertension'] = np.random.binomial(1, self._sigmoid(logit_htn))
        df['risk_afib'] = np.random.binomial(1, self._sigmoid(logit_afib))
        df['risk_pad'] = np.random.binomial(1, self._sigmoid(logit_pad))
        
        # Log outcome statistics
        outcome_cols = [col for col in df.columns if col.startswith('risk_')]
        for col in outcome_cols:
            prevalence = df[col].mean()
            logger.info(f"{col}: {prevalence:.3f} prevalence ({df[col].sum()} cases)")
        
        return df
    
    def _get_logistic_relationships(self) -> Dict:
        """
        Define logistic regression relationships for each cardiovascular condition.
        
        Returns:
        --------
        dict
            Dictionary of logistic relationship functions
        """
        return {
            'stroke': lambda age, bmi, glucose, smoking, fh_stroke, activity, sbp, stress: (
                -10 + 0.04*age + 0.05*bmi + 0.03*glucose + 0.7*smoking +
                0.8*fh_stroke - 0.4*activity + 0.02*sbp + 0.05*stress
            ),
            
            'heart_disease': lambda age, bmi, ldl, fh_heart, alcohol, smoking, trig: (
                -9 + 0.05*age + 0.08*bmi + 0.02*ldl + 0.9*fh_heart +
                0.6*alcohol + 0.9*smoking + 0.01*trig
            ),
            
            'heart_failure': lambda age, bmi, glucose, smoking, fh_heart, stress, hr, sbp: (
                -8 + 0.06*age + 0.1*bmi + 0.03*glucose + 0.8*smoking +
                0.7*fh_heart + 0.5*stress + 0.01*hr + 0.02*sbp
            ),
            
            'hypertension': lambda age, bmi, sbp, fh_heart, stress, activity: (
                -9 + 0.06*age + 0.1*bmi + 0.03*sbp + 0.9*fh_heart +
                0.6*stress - 0.4*activity
            ),
            
            'afib': lambda age, hr, smoking, stress, fh_heart: (
                -11 + 0.04*age + 0.03*hr + 0.7*smoking + 0.6*stress +
                0.9*fh_heart
            ),
            
            'pad': lambda age, ldl, smoking, stress, fh_heart, activity: (
                -10 + 0.05*age + 0.03*ldl + 0.9*smoking + 0.5*stress +
                0.8*fh_heart - 0.3*activity
            )
        }
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate sigmoid function with numerical stability.
        
        Parameters:
        -----------
        x : np.ndarray
            Input values
            
        Returns:
        --------
        np.ndarray
            Sigmoid transformed values
        """
        # Clip extreme values for numerical stability
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def generate_complete_dataset(self, n_samples: int = None, 
                                 output_file: str = None) -> pd.DataFrame:
        """
        Generate complete synthetic cardiovascular dataset.
        
        Parameters:
        -----------
        n_samples : int, optional
            Number of samples to generate
        output_file : str, optional
            Output CSV filename
            
        Returns:
        --------
        pd.DataFrame
            Complete synthetic dataset
        """
        # Generate risk factors
        risk_factors = self.generate_risk_factors(n_samples)
        
        # Generate outcomes
        complete_data = self.create_risk_outcomes(risk_factors)
        
        # Save to file if specified
        if output_file:
            complete_data.to_csv(output_file, index=False)
            logger.info(f"Saved synthetic dataset to {output_file}")
        
        return complete_data
    
    def validate_dataset(self, data: pd.DataFrame) -> Dict[str, Union[bool, str, Dict]]:
        """
        Validate the generated synthetic dataset.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Dataset to validate
            
        Returns:
        --------
        dict
            Validation results
        """
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {}
        }
        
        expected_columns = 18  # Risk factors
        expected_outcomes = 6   # Risk outcomes
        total_expected = expected_columns + expected_outcomes
        
        # Check basic structure
        if len(data.columns) != total_expected:
            validation_results["errors"].append(
                f"Expected {total_expected} columns, found {len(data.columns)}"
            )
            validation_results["valid"] = False
        
        # Check for missing values
        missing_values = data.isnull().sum().sum()
        if missing_values > 0:
            validation_results["warnings"].append(f"Found {missing_values} missing values")
        
        # Validate risk factor ranges
        for col, config in INPUT_RANGES.items():
            if col in data.columns:
                col_data = data[col]
                if col_data.min() < config["min"] or col_data.max() > config["max"]:
                    validation_results["warnings"].append(
                        f"{col} values outside expected range [{config['min']}, {config['max']}]"
                    )
        
        # Validate binary variables
        binary_cols = ['smoking_status', 'alcohol_use', 'physical_activity']
        outcome_cols = [col for col in data.columns if col.startswith('risk_')]
        
        for col in binary_cols + outcome_cols:
            if col in data.columns:
                unique_vals = set(data[col].unique())
                if not unique_vals.issubset({0, 1}):
                    validation_results["errors"].append(
                        f"{col} should only contain 0 and 1, found: {unique_vals}"
                    )
                    validation_results["valid"] = False
        
        # Calculate summary statistics
        validation_results["statistics"] = {
            "n_rows": len(data),
            "n_columns": len(data.columns),
            "missing_values": int(missing_values),
            "outcome_prevalences": {}
        }
        
        # Calculate outcome prevalences
        for col in outcome_cols:
            if col in data.columns:
                prevalence = data[col].mean()
                validation_results["statistics"]["outcome_prevalences"][col] = float(prevalence)
        
        if validation_results["valid"]:
            logger.info("Dataset validation passed")
        else:
            logger.warning(f"Dataset validation failed: {validation_results['errors']}")
        
        return validation_results
    
    def create_correlation_matrix(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create correlation matrix for the synthetic dataset.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Synthetic dataset
            
        Returns:
        --------
        pd.DataFrame
            Correlation matrix
        """
        # Select numerical columns
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        correlation_matrix = data[numerical_cols].corr()
        
        logger.info("Calculated correlation matrix for synthetic dataset")
        return correlation_matrix
    
    def generate_summary_statistics(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """
        Generate comprehensive summary statistics for the dataset.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Synthetic dataset
            
        Returns:
        --------
        dict
            Summary statistics by variable type
        """
        summary = {
            "continuous_variables": {},
            "binary_variables": {},
            "outcome_variables": {}
        }
        
        # Continuous variables
        continuous_vars = ['age', 'bmi', 'glucose', 'hdl_chol', 'ldl_chol', 
                          'triglycerides', 'systolic_bp', 'diastolic_bp', 
                          'heart_rate', 'sleep_hours', 'stress_score']
        
        for var in continuous_vars:
            if var in data.columns:
                col_data = data[var]
                summary["continuous_variables"][var] = {
                    "mean": float(col_data.mean()),
                    "std": float(col_data.std()),
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "median": float(col_data.median()),
                    "q25": float(col_data.quantile(0.25)),
                    "q75": float(col_data.quantile(0.75))
                }
        
        # Binary variables
        binary_vars = ['smoking_status', 'alcohol_use', 'physical_activity',
                      'family_history_diabetes', 'family_history_stroke',
                      'family_history_heart_disease']
        
        for var in binary_vars:
            if var in data.columns:
                col_data = data[var]
                summary["binary_variables"][var] = {
                    "prevalence": float(col_data.mean()),
                    "count_positive": int(col_data.sum()),
                    "count_negative": int(len(col_data) - col_data.sum())
                }
        
        # Outcome variables
        outcome_vars = [col for col in data.columns if col.startswith('risk_')]
        
        for var in outcome_vars:
            col_data = data[var]
            summary["outcome_variables"][var] = {
                "prevalence": float(col_data.mean()),
                "cases": int(col_data.sum()),
                "controls": int(len(col_data) - col_data.sum())
            }
        
        logger.info("Generated comprehensive summary statistics")
        return summary


# Convenience functions for direct use
def generate_synthetic_data(n_samples: int = 500, random_seed: int = 42, 
                           output_file: str = None) -> pd.DataFrame:
    """
    Generate synthetic cardiovascular risk dataset.
    
    Parameters:
    -----------
    n_samples : int, default 500
        Number of samples to generate
    random_seed : int, default 42
        Random seed for reproducibility
    output_file : str, optional
        Output CSV filename
        
    Returns:
    --------
    pd.DataFrame
        Complete synthetic dataset
    """
    generator = SyntheticDataGenerator(random_seed)
    return generator.generate_complete_dataset(n_samples, output_file)


def create_risk_outcomes(risk_factors_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create cardiovascular risk outcomes for a risk factor dataset.
    
    Parameters:
    -----------
    risk_factors_df : pd.DataFrame
        Risk factor dataset
        
    Returns:
    --------
    pd.DataFrame
        Dataset with outcomes added
    """
    generator = SyntheticDataGenerator()
    return generator.create_risk_outcomes(risk_factors_df)


def validate_synthetic_data(data: pd.DataFrame) -> Dict:
    """
    Validate a synthetic cardiovascular dataset.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset to validate
        
    Returns:
    --------
    dict
        Validation results
    """
    generator = SyntheticDataGenerator()
    return generator.validate_dataset(data)


def load_or_generate_data(data_file: str = "multi_cvd_dataset.csv", 
                         regenerate: bool = False) -> pd.DataFrame:
    """
    Load existing dataset or generate new one if not found.
    
    Parameters:
    -----------
    data_file : str, default "multi_cvd_dataset.csv"
        Path to dataset file
    regenerate : bool, default False
        Force regeneration even if file exists
        
    Returns:
    --------
    pd.DataFrame
        Cardiovascular risk dataset
    """
    if Path(data_file).exists() and not regenerate:
        logger.info(f"Loading existing dataset from {data_file}")
        return pd.read_csv(data_file)
    else:
        logger.info(f"Generating new synthetic dataset")
        return generate_synthetic_data(output_file=data_file)
