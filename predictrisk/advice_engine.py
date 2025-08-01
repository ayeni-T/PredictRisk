"""
Risk advice and interpretation engine for PredictRisk.

This module generates personalized health recommendations, risk interpretations,
and educational content based on risk assessment results and individual profiles.
"""

import logging
from typing import Dict, List, Tuple, Union, Optional
from .config import RISK_THRESHOLDS, HELP_TEXT, INPUT_RANGES, BINARY_INPUTS
from .utils import get_risk_category, format_risk_output

# Configure logging
logger = logging.getLogger(__name__)


class AdviceEngine:
    """
    Engine for generating personalized cardiovascular risk advice and interpretations.
    
    This class provides methods to generate tailored recommendations based on
    risk levels, individual risk profiles, and specific cardiovascular conditions.
    """
    
    def __init__(self):
        """Initialize the AdviceEngine."""
        self.advice_templates = self._load_advice_templates()
        self.interpretation_templates = self._load_interpretation_templates()
        
    def _load_advice_templates(self) -> Dict[str, Dict[str, List[str]]]:
        """Load advice templates for different risk levels and conditions."""
        return {
            "High Risk": {
                "immediate_actions": [
                    "Consult a licensed physician or cardiologist immediately",
                    "Request a comprehensive cardiovascular evaluation",
                    "Do not delay medical attention even if you feel well",
                    "Consider emergency medical attention if experiencing symptoms"
                ],
                "lifestyle_changes": [
                    "Implement immediate smoking cessation if applicable",
                    "Begin medically supervised exercise program", 
                    "Adopt heart-healthy diet (DASH or Mediterranean)",
                    "Implement stress reduction techniques (meditation, yoga)",
                    "Optimize sleep hygiene for 7-9 hours nightly",
                    "Monitor blood pressure and glucose regularly"
                ],
                "medical_management": [
                    "Discuss medication options with healthcare provider",
                    "Schedule regular follow-up appointments",
                    "Consider cardiac rehabilitation programs",
                    "Implement intensive risk factor modification"
                ]
            },
            "Moderate Risk": {
                "preventive_actions": [
                    "Schedule consultation with healthcare provider",
                    "Discuss prevention strategies with your doctor",
                    "Consider cardiovascular screening tests",
                    "Monitor key risk factors regularly"
                ],
                "lifestyle_modifications": [
                    "Increase physical activity to 150+ minutes/week",
                    "Adopt heart-healthy eating patterns",
                    "Implement effective stress management techniques",
                    "Optimize sleep duration and quality",
                    "Limit alcohol consumption if applicable",
                    "Maintain healthy body weight"
                ],
                "monitoring": [
                    "Regular blood pressure monitoring",
                    "Annual lipid panel and glucose testing",
                    "Track physical activity and fitness levels",
                    "Monitor stress levels and mental health"
                ]
            },
            "Low Risk": {
                "maintenance": [
                    "Continue current healthy lifestyle practices",
                    "Maintain regular physical activity routine",
                    "Keep up heart-healthy eating habits",
                    "Continue effective stress management"
                ],
                "prevention": [
                    "Schedule routine medical checkups",
                    "Stay informed about cardiovascular health",
                    "Maintain awareness of family health history",
                    "Educate others about cardiovascular risk prevention"
                ],
                "optimization": [
                    "Consider advanced fitness goals",
                    "Explore stress reduction techniques",
                    "Optimize nutrition for cardiovascular health",
                    "Maintain social connections and mental health"
                ]
            }
        }
    
    def _load_interpretation_templates(self) -> Dict[str, str]:
        """Load interpretation templates for different contexts."""
        return {
            "general": (
                "Based on your health profile, your predicted risk of developing {condition} "
                "is {risk_percent}%. This estimate reflects your current likelihood based on "
                "the risk factors you provided. The 95% credible interval "
                "[{ci_lower}%-{ci_upper}%] represents the uncertainty in this prediction."
            ),
            "low_risk": (
                "Your risk assessment indicates a low probability of developing {condition}. "
                "This is excellent news and suggests that your current health profile and "
                "lifestyle choices are protective against this condition."
            ),
            "moderate_risk": (
                "Your risk assessment indicates a moderate probability of developing {condition}. "
                "While this is not immediately alarming, it suggests that preventive measures "
                "and lifestyle modifications could significantly reduce your risk."
            ),
            "high_risk": (
                "Your risk assessment indicates a high probability of developing {condition}. "
                "This suggests that immediate medical consultation and aggressive risk factor "
                "modification are important for your cardiovascular health."
            ),
            "uncertainty": (
                "The credible interval width of {uncertainty_width}% indicates {uncertainty_level} "
                "uncertainty in this prediction. {uncertainty_explanation}"
            )
        }
    
    def get_risk_advice(self, risk_category: str, condition: str = None, 
                       personalized_factors: Dict = None) -> List[str]:
        """
        Generate comprehensive risk advice based on risk level and profile.
        
        Parameters:
        -----------
        risk_category : str
            Risk category ('Low Risk', 'Moderate Risk', 'High Risk')
        condition : str, optional
            Specific cardiovascular condition
        personalized_factors : dict, optional
            Individual risk factors for personalization
            
        Returns:
        --------
        list
            List of advice strings
        """
        advice_list = []
        
        if risk_category not in self.advice_templates:
            logger.warning(f"Unknown risk category: {risk_category}")
            return ["Consult with a healthcare professional for personalized advice."]
        
        templates = self.advice_templates[risk_category]
        
        # Add header
        if risk_category == "High Risk":
            advice_list.append("🚨 Your predicted risk is HIGH. Take immediate action:")
        elif risk_category == "Moderate Risk":
            advice_list.append("⚠️ Your risk is MODERATE. Consider these preventive steps:")
        else:
            advice_list.append("✅ Great news! Your predicted risk is LOW. Maintain these habits:")
        
        # Add category-specific advice
        for category, advice_items in templates.items():
            advice_list.append(f"\n{category.replace('_', ' ').title()}:")
            for item in advice_items:
                advice_list.append(f"• {item}")
        
        # Add personalized advice if factors provided
        if personalized_factors:
            personalized_advice = self._generate_personalized_advice(
                risk_category, personalized_factors, condition
            )
            if personalized_advice:
                advice_list.append("\nPersonalized Recommendations:")
                advice_list.extend([f"• {advice}" for advice in personalized_advice])
        
        return advice_list
    
    def _generate_personalized_advice(self, risk_category: str, factors: Dict, 
                                    condition: str = None) -> List[str]:
        """
        Generate personalized advice based on individual risk factors.
        
        Parameters:
        -----------
        risk_category : str
            Risk category
        factors : dict
            Individual risk factor values
        condition : str, optional
            Specific condition
            
        Returns:
        --------
        list
            Personalized advice items
        """
        personalized = []
        
        # Age-specific advice
        age = factors.get("age", 50)
        if age > 65:
            personalized.append("Given your age, prioritize regular medical monitoring")
        elif age < 40:
            personalized.append("Early prevention efforts now can significantly impact future risk")
        
        # BMI-specific advice
        bmi = factors.get("bmi", 25)
        if bmi > 30:
            personalized.append("Weight management could significantly reduce your cardiovascular risk")
        elif bmi > 25:
            personalized.append("Maintaining or achieving healthy weight would be beneficial")
        
        # Blood pressure advice
        systolic_bp = factors.get("systolic_bp", 120)
        if systolic_bp > 140:
            personalized.append("Blood pressure management is a high priority for you")
        elif systolic_bp > 130:
            personalized.append("Monitor blood pressure regularly and consider lifestyle modifications")
        
        # Lifestyle factor advice
        if factors.get("smoking_status", 0) == 1:
            personalized.append("Smoking cessation would provide the greatest risk reduction benefit")
        
        if factors.get("physical_activity", 1) == 0:
            personalized.append("Increasing physical activity could substantially lower your risk")
        
        # Sleep and stress advice
        sleep_hours = factors.get("sleep_hours", 7)
        if sleep_hours < 6:
            personalized.append("Improving sleep duration to 7-9 hours could benefit cardiovascular health")
        
        stress_score = factors.get("stress_score", 5)
        if stress_score > 7:
            personalized.append("Stress management techniques could significantly impact your risk profile")
        
        return personalized
    
    def get_interpretation_text(self, condition: str, mean_risk: float, 
                              credible_interval: Tuple[float, float],
                              risk_category: str = None) -> str:
        """
        Generate detailed interpretation text for risk assessment.
        
        Parameters:
        -----------
        condition : str
            Cardiovascular condition
        mean_risk : float
            Mean risk prediction (0-1)
        credible_interval : tuple
            (lower_bound, upper_bound) for 95% CI
        risk_category : str, optional
            Risk category (auto-determined if not provided)
            
        Returns:
        --------
        str
            Detailed interpretation text
        """
        if risk_category is None:
            risk_category = get_risk_category(mean_risk)
        
        # Format values
        risk_percent = round(mean_risk * 100, 1)
        ci_lower = round(credible_interval[0] * 100, 1)
        ci_upper = round(credible_interval[1] * 100, 1)
        uncertainty_width = round(ci_upper - ci_lower, 1)
        
        # Generate main interpretation
        interpretation_parts = []
        
        # General interpretation
        general_template = self.interpretation_templates["general"]
        general_text = general_template.format(
            condition=condition.lower(),
            risk_percent=risk_percent,
            ci_lower=ci_lower,
            ci_upper=ci_upper
        )
        interpretation_parts.append(general_text)
        
        # Risk-specific interpretation
        risk_key = risk_category.lower().replace(" ", "_")
        if risk_key in self.interpretation_templates:
            risk_specific = self.interpretation_templates[risk_key].format(
                condition=condition.lower()
            )
            interpretation_parts.append(risk_specific)
        
        # Uncertainty interpretation
        uncertainty_level, uncertainty_explanation = self._interpret_uncertainty(uncertainty_width)
        uncertainty_template = self.interpretation_templates["uncertainty"]
        uncertainty_text = uncertainty_template.format(
            uncertainty_width=uncertainty_width,
            uncertainty_level=uncertainty_level,
            uncertainty_explanation=uncertainty_explanation
        )
        interpretation_parts.append(uncertainty_text)
        
        # Add disclaimer
        disclaimer = (
            "\n\nIMPORTANT: This prediction is based on statistical modeling with synthetic data "
            "and should not be used for clinical decision-making. Always consult with a "
            "qualified healthcare provider for medical advice, diagnosis, and treatment."
        )
        interpretation_parts.append(disclaimer)
        
        return "\n\n".join(interpretation_parts)
    
    def _interpret_uncertainty(self, uncertainty_width: float) -> Tuple[str, str]:
        """
        Interpret uncertainty width and provide explanation.
        
        Parameters:
        -----------
        uncertainty_width : float
            Width of credible interval in percentage points
            
        Returns:
        --------
        tuple
            (uncertainty_level, explanation)
        """
        if uncertainty_width < 10:
            level = "low"
            explanation = (
                "This narrow range suggests the model is relatively confident in this prediction. "
                "The risk factors provided give a clear signal about your risk profile."
            )
        elif uncertainty_width < 25:
            level = "moderate"
            explanation = (
                "This moderate range reflects normal prediction uncertainty. "
                "The model accounts for natural variability in individual risk profiles."
            )
        else:
            level = "high"
            explanation = (
                "This wide range indicates higher uncertainty in the prediction. "
                "This could be due to conflicting risk factors or limited model certainty for your profile."
            )
        
        return level, explanation
    
    def get_educational_content(self, condition: str) -> Dict[str, str]:
        """
        Get educational content about a specific cardiovascular condition.
        
        Parameters:
        -----------
        condition : str
            Cardiovascular condition
            
        Returns:
        --------
        dict
            Educational content sections
        """
        education_content = {
            "Stroke": {
                "definition": "A stroke occurs when blood flow to part of the brain is blocked or reduced, preventing brain tissue from getting oxygen and nutrients.",
                "risk_factors": "Key risk factors include high blood pressure, smoking, diabetes, high cholesterol, age, and family history.",
                "prevention": "Prevention focuses on controlling blood pressure, not smoking, regular exercise, healthy diet, and managing diabetes.",
                "symptoms": "Common symptoms include sudden numbness, confusion, trouble speaking, severe headache, and difficulty walking."
            },
            "Heart Disease": {
                "definition": "Heart disease refers to several conditions affecting the heart, most commonly coronary artery disease.",
                "risk_factors": "Major risk factors include high cholesterol, high blood pressure, smoking, diabetes, obesity, and physical inactivity.",
                "prevention": "Prevention involves healthy eating, regular exercise, not smoking, limiting alcohol, and managing stress.",
                "symptoms": "Symptoms may include chest pain, shortness of breath, fatigue, and irregular heartbeat."
            },
            "Hypertension": {
                "definition": "High blood pressure (hypertension) occurs when the force of blood against artery walls is consistently too high.",
                "risk_factors": "Risk factors include age, family history, obesity, lack of physical activity, tobacco use, and high sodium intake.",
                "prevention": "Prevention involves maintaining healthy weight, regular exercise, healthy diet, limiting sodium, and managing stress.",
                "symptoms": "Often called the 'silent killer' because it usually has no symptoms until serious complications develop."
            },
            "Heart Failure": {
                "definition": "Heart failure occurs when the heart cannot pump blood effectively to meet the body's needs.",
                "risk_factors": "Risk factors include coronary artery disease, high blood pressure, diabetes, obesity, and previous heart attack.",
                "prevention": "Prevention focuses on managing underlying conditions, healthy lifestyle, and avoiding excessive alcohol and drug use.",
                "symptoms": "Symptoms include shortness of breath, fatigue, swelling in legs and ankles, and persistent cough."
            },
            "Atrial Fibrillation (AFib)": {
                "definition": "Atrial fibrillation is an irregular and often rapid heart rhythm that can increase stroke risk.",
                "risk_factors": "Risk factors include age, heart disease, high blood pressure, diabetes, sleep apnea, and excessive alcohol use.",
                "prevention": "Prevention involves managing underlying heart conditions, limiting alcohol, maintaining healthy weight, and managing stress.",
                "symptoms": "Symptoms may include heart palpitations, shortness of breath, fatigue, and chest pain."
            },
            "Peripheral Artery Disease (PAD)": {
                "definition": "PAD occurs when narrowed arteries reduce blood flow to the arms or legs.",
                "risk_factors": "Major risk factors include smoking, diabetes, high blood pressure, high cholesterol, and age over 50.",
                "prevention": "Prevention emphasizes smoking cessation, diabetes management, blood pressure control, and regular exercise.",
                "symptoms": "Common symptoms include leg pain when walking, coldness in lower leg, and slow-healing wounds on feet."
            }
        }
        
        return education_content.get(condition, {})
    
    def generate_lifestyle_recommendations(self, input_data: Dict[str, Union[int, float]], 
                                         risk_category: str) -> Dict[str, List[str]]:
        """
        Generate specific lifestyle recommendations based on individual profile.
        
        Parameters:
        -----------
        input_data : dict
            Individual risk factor values
        risk_category : str
            Risk category
            
        Returns:
        --------
        dict
            Categorized lifestyle recommendations
        """
        recommendations = {
            "diet": [],
            "exercise": [],
            "lifestyle": [],
            "monitoring": []
        }
        
        # Diet recommendations
        bmi = input_data.get("bmi", 25)
        if bmi > 25:
            recommendations["diet"].extend([
                "Focus on portion control and caloric balance",
                "Emphasize fruits, vegetables, and whole grains",
                "Limit processed foods and added sugars",
                "Consider consulting a registered dietitian"
            ])
        
        glucose = input_data.get("glucose", 100)
        if glucose > 100:
            recommendations["diet"].extend([
                "Monitor carbohydrate intake and timing",
                "Choose complex carbohydrates over simple sugars",
                "Consider Mediterranean or DASH diet patterns"
            ])
        
        # Exercise recommendations
        physical_activity = input_data.get("physical_activity", 1)
        age = input_data.get("age", 50)
        
        if physical_activity == 0:
            if age > 65:
                recommendations["exercise"].extend([
                    "Start with gentle activities like walking",
                    "Consider supervised exercise programs",
                    "Include balance and flexibility exercises",
                    "Consult physician before starting new exercise"
                ])
            else:
                recommendations["exercise"].extend([
                    "Aim for 150 minutes of moderate activity weekly",
                    "Include both aerobic and strength training",
                    "Start gradually and increase intensity over time",
                    "Consider activities you enjoy for better adherence"
                ])
        else:
            recommendations["exercise"].extend([
                "Continue current physical activity routine",
                "Consider increasing intensity or duration gradually",
                "Include variety in your exercise program"
            ])
        
        # Lifestyle recommendations
        smoking_status = input_data.get("smoking_status", 0)
        if smoking_status == 1:
            recommendations["lifestyle"].extend([
                "Smoking cessation is the single most important change you can make",
                "Consider nicotine replacement therapy or prescription medications",
                "Seek support from healthcare providers or smoking cessation programs",
                "Avoid triggers and develop coping strategies"
            ])
        
        stress_score = input_data.get("stress_score", 5)
        if stress_score > 7:
            recommendations["lifestyle"].extend([
                "Implement regular stress reduction techniques",
                "Consider mindfulness meditation or yoga",
                "Ensure adequate social support and connections",
                "Consider professional counseling if stress is overwhelming"
            ])
        
        sleep_hours = input_data.get("sleep_hours", 7)
        if sleep_hours < 6:
            recommendations["lifestyle"].extend([
                "Prioritize 7-9 hours of sleep nightly",
                "Establish consistent sleep schedule",
                "Create optimal sleep environment (dark, cool, quiet)",
                "Limit screen time before bedtime"
            ])
        
        # Monitoring recommendations
        systolic_bp = input_data.get("systolic_bp", 120)
        if systolic_bp > 130 or risk_category != "Low Risk":
            recommendations["monitoring"].extend([
                "Monitor blood pressure regularly at home",
                "Keep a log of readings to share with healthcare provider",
                "Check blood pressure at different times of day"
            ])
        
        if glucose > 100 or risk_category != "Low Risk":
            recommendations["monitoring"].extend([
                "Monitor fasting glucose levels",
                "Consider regular HbA1c testing",
                "Track how diet and exercise affect glucose levels"
            ])
        
        return recommendations
    
    def generate_risk_comparison(self, individual_risk: float, condition: str) -> str:
        """
        Generate text comparing individual risk to population averages.
        
        Parameters:
        -----------
        individual_risk : float
            Individual risk prediction (0-1)
        condition : str
            Cardiovascular condition
            
        Returns:
        --------
        str
            Risk comparison text
        """
        # Approximate population prevalence (these would ideally be age-adjusted)
        population_rates = {
            "Stroke": 0.03,  # ~3% lifetime risk
            "Heart Disease": 0.06,  # ~6% prevalence
            "Hypertension": 0.45,  # ~45% adult prevalence
            "Heart Failure": 0.02,  # ~2% prevalence
            "Atrial Fibrillation (AFib)": 0.04,  # ~4% prevalence
            "Peripheral Artery Disease (PAD)": 0.03  # ~3% prevalence
        }
        
        population_risk = population_rates.get(condition, 0.05)
        individual_percent = round(individual_risk * 100, 1)
        population_percent = round(population_risk * 100, 1)
        
        if individual_risk > population_risk * 1.5:
            comparison = f"significantly higher than the general population average of {population_percent}%"
        elif individual_risk > population_risk * 1.2:
            comparison = f"higher than the general population average of {population_percent}%"
        elif individual_risk < population_risk * 0.8:
            comparison = f"lower than the general population average of {population_percent}%"
        else:
            comparison = f"similar to the general population average of {population_percent}%"
        
        return (
            f"Your predicted {condition.lower()} risk of {individual_percent}% is {comparison}. "
            "However, remember that individual risk depends on your specific profile and may "
            "vary significantly from population averages."
        )


# Convenience functions for direct use
def get_risk_advice(risk_category: str, condition: str = None, 
                   personalized_factors: Dict = None) -> List[str]:
    """
    Get risk advice for a given risk category.
    
    Parameters:
    -----------
    risk_category : str
        Risk category ('Low Risk', 'Moderate Risk', 'High Risk')
    condition : str, optional
        Specific cardiovascular condition
    personalized_factors : dict, optional
        Individual risk factors for personalization
        
    Returns:
    --------
    list
        List of advice strings
    """
    engine = AdviceEngine()
    return engine.get_risk_advice(risk_category, condition, personalized_factors)


def get_interpretation_text(condition: str, mean_risk: float, 
                          credible_interval: Tuple[float, float]) -> str:
    """
    Get interpretation text for risk assessment.
    
    Parameters:
    -----------
    condition : str
        Cardiovascular condition
    mean_risk : float
        Mean risk prediction (0-1)
    credible_interval : tuple
        (lower_bound, upper_bound) for 95% CI
        
    Returns:
    --------
    str
        Interpretation text
    """
    engine = AdviceEngine()
    return engine.get_interpretation_text(condition, mean_risk, credible_interval)


def get_educational_content(condition: str) -> Dict[str, str]:
    """
    Get educational content about a cardiovascular condition.
    
    Parameters:
    -----------
    condition : str
        Cardiovascular condition
        
    Returns:
    --------
    dict
        Educational content sections
    """
    engine = AdviceEngine()
    return engine.get_educational_content(condition)
