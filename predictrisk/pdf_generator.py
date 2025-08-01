"""
PDF report generation module for PredictRisk.

This module handles the creation of comprehensive PDF reports containing
risk assessments, interpretations, recommendations, and educational content.
"""

import logging
from io import BytesIO
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors

from .utils import format_risk_output, get_risk_category

# Configure logging
logger = logging.getLogger(__name__)


class PDFReportGenerator:
    """
    Professional PDF report generator for cardiovascular risk assessments.
    
    This class creates comprehensive, well-formatted PDF reports containing
    risk predictions, uncertainty quantification, personalized advice,
    and educational content.
    """
    
    def __init__(self):
        """Initialize the PDF report generator."""
        pass
    
    def generate_simple_report(self, condition: str, mean_risk: float,
                             credible_interval: Tuple[float, float],
                             risk_category: str, advice_lines: List[str],
                             interpretation_text: str, input_data: Dict = None) -> BytesIO:
        """
        Generate a simple PDF report using basic canvas drawing.
        
        Parameters:
        -----------
        condition : str
            Cardiovascular condition
        mean_risk : float
            Mean risk prediction
        credible_interval : tuple
            95% credible interval
        risk_category : str
            Risk category
        advice_lines : list
            Advice recommendations
        interpretation_text : str
            Risk interpretation
        input_data : dict, optional
            Input risk factors
            
        Returns:
        --------
        BytesIO
            PDF buffer
        """
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        
        # Title
        c.setFont("Helvetica-Bold", 16)
        title_text = "PredictRisk: Cardiovascular Risk Assessment Report"
        title_width = c.stringWidth(title_text, "Helvetica-Bold", 16)
        c.drawString((width - title_width) / 2, height-50, title_text)
        
        # Metadata
        y_position = height - 100
        timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        
        c.setFont("Helvetica", 12)
        c.drawString(50, y_position, f"Report Generated: {timestamp}")
        y_position -= 20
        c.drawString(50, y_position, f"Condition Assessed: {condition}")
        y_position -= 40
        
        # Risk assessment
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_position, "Risk Assessment Results")
        y_position -= 30
        
        # Risk level with color coding
        risk_colors = {
            "High Risk": colors.red,
            "Moderate Risk": colors.orange,
            "Low Risk": colors.green
        }
        
        c.setFillColor(risk_colors.get(risk_category, colors.black))
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y_position, f"Risk Level: {risk_category}")
        y_position -= 25
        
        c.setFillColor(colors.black)
        c.setFont("Helvetica", 12)
        c.drawString(50, y_position, f"Predicted Risk: {mean_risk:.2%}")
        y_position -= 20
        c.drawString(50, y_position, f"95% Credible Interval: [{credible_interval[0]:.2%}, {credible_interval[1]:.2%}]")
        y_position -= 40
        
        # Interpretation
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y_position, "What This Means for You")
        y_position -= 25
        
        # Split interpretation into lines that fit the page
        c.setFont("Helvetica", 10)
        interpretation_lines = self._wrap_text(interpretation_text, width - 100)
        
        for line in interpretation_lines[:8]:  # Limit lines to fit page
            c.drawString(60, y_position, line)
            y_position -= 15
            if y_position < 200:  # Start new page if needed
                c.showPage()
                y_position = height - 50
        
        y_position -= 20
        
        # Recommendations
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y_position, "Personalized Recommendations")
        y_position -= 25
        
        c.setFont("Helvetica", 10)
        for line in advice_lines[:10]:  # Limit to fit page
            if y_position < 100:  # Start new page if needed
                c.showPage()
                y_position = height - 50
                
            wrapped_lines = self._wrap_text(line, width - 100)
            for wrapped_line in wrapped_lines:
                c.drawString(60, y_position, wrapped_line)
                y_position -= 15
        
        # Input summary if provided
        if input_data:
            if y_position < 200:
                c.showPage()
                y_position = height - 50
            
            y_position -= 30
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y_position, "Your Health Profile")
            y_position -= 25
            
            c.setFont("Helvetica", 10)
            
            # Basic info
            age = input_data.get("age", "N/A")
            bmi = input_data.get("bmi", "N/A")
            c.drawString(60, y_position, f"Age: {age} years, BMI: {bmi}")
            y_position -= 15
            
            # Vital signs
            systolic = input_data.get("systolic_bp", "N/A")
            diastolic = input_data.get("diastolic_bp", "N/A")
            heart_rate = input_data.get("heart_rate", "N/A")
            c.drawString(60, y_position, f"Blood Pressure: {systolic}/{diastolic} mmHg, Heart Rate: {heart_rate} bpm")
            y_position -= 15
            
            # Lifestyle factors
            smoking = "Yes" if input_data.get("smoking_status", 0) == 1 else "No"
            alcohol = "Yes" if input_data.get("alcohol_use", 0) == 1 else "No"
            activity = "Yes" if input_data.get("physical_activity", 1) == 1 else "No"
            c.drawString(60, y_position, f"Smoking: {smoking}, Alcohol: {alcohol}, Active: {activity}")
            y_position -= 30
        
        # Disclaimer
        if y_position < 150:
            c.showPage()
            y_position = height - 50
        
        c.setFont("Helvetica-Bold", 10)
        c.drawString(50, y_position, "IMPORTANT DISCLAIMER")
        y_position -= 20
        
        disclaimer_text = (
            "This report is for educational purposes only and is not a substitute for "
            "professional medical advice. Always consult your healthcare provider."
        )
        
        c.setFont("Helvetica", 9)
        disclaimer_lines = self._wrap_text(disclaimer_text, width - 100)
        for line in disclaimer_lines:
            c.drawString(50, y_position, line)
            y_position -= 12
        
        # Footer
        c.setFont("Helvetica", 8)
        c.drawString(width-200, 30, "© 2025 Taiwo Michael Ayeni - PredictRisk")
        
        c.save()
        buffer.seek(0)
        logger.info(f"Successfully generated PDF report for {condition}")
        return buffer
    
    def _wrap_text(self, text: str, max_width: float, font_size: int = 10) -> List[str]:
        """
        Wrap text to fit within specified width.
        
        Parameters:
        -----------
        text : str
            Text to wrap
        max_width : float
            Maximum width in points
        font_size : int
            Font size
            
        Returns:
        --------
        list
            List of wrapped lines
        """
        words = text.split()
        lines = []
        current_line = []
        
        # Approximate character width (rough estimate)
        char_width = font_size * 0.6
        max_chars = int(max_width / char_width)
        
        for word in words:
            # Check if adding this word would exceed line length
            test_line = ' '.join(current_line + [word])
            if len(test_line) <= max_chars:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines


# Convenience functions for direct use
def generate_pdf_report(condition: str, mean_risk: float, 
                       credible_interval: Tuple[float, float],
                       risk_category: str, advice_lines: List[str],
                       interpretation_text: str, input_data: Dict = None,
                       use_simple: bool = True) -> BytesIO:
    """
    Generate a PDF report for cardiovascular risk assessment.
    
    Parameters:
    -----------
    condition : str
        Cardiovascular condition
    mean_risk : float
        Mean risk prediction
    credible_interval : tuple
        95% credible interval  
    risk_category : str
        Risk category
    advice_lines : list
        Advice recommendations
    interpretation_text : str
        Risk interpretation
    input_data : dict, optional
        Input risk factors
    use_simple : bool, default True
        Whether to use simple canvas-based generation
        
    Returns:
    --------
    BytesIO
        PDF buffer
    """
    generator = PDFReportGenerator()
    return generator.generate_simple_report(
        condition, mean_risk, credible_interval, 
        risk_category, advice_lines, interpretation_text, input_data
    )
