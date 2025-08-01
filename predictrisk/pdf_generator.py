"""
PDF report generation module for PredictRisk.

This module handles the creation of comprehensive PDF reports containing
risk assessments, interpretations, recommendations, and educational content.
"""

import logging
from io import BytesIO
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional
from reportlab.lib.pagesizes import letter, A4
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import black, blue, red, green, orange
from reportlab.lib import colors

from .config import APP_CONFIG, RISK_THRESHOLDS
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
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
    def _setup_custom_styles(self):
        """Setup custom paragraph styles for the report."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=20,
            textColor=colors.darkblue,
            alignment=1  # Center alignment
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceBefore=15,
            spaceAfter=10,
            textColor=colors.darkblue
        ))
        
        # Risk level styles
        self.styles.add(ParagraphStyle(
            name='HighRisk',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=colors.red,
            spaceBefore=10,
            spaceAfter=10
        ))
        
        self.styles.add(ParagraphStyle(
            name='ModerateRisk', 
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=colors.orange,
            spaceBefore=10,
            spaceAfter=10
        ))
        
        self.styles.add(ParagraphStyle(
            name='LowRisk',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=colors.green,
            spaceBefore=10,
            spaceAfter=10
        ))
        
        # Disclaimer style
        self.styles.add(ParagraphStyle(
            name='Disclaimer',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.grey,
            spaceBefore=20,
            spaceAfter=10,
            leftIndent=20,
            rightIndent=20
        ))
    
    def generate_comprehensive_report(self, condition: str, mean_risk: float,
                                    credible_interval: Tuple[float, float],
                                    risk_category: str, advice_lines: List[str],
                                    interpretation_text: str, input_data: Dict,
                                    model_summary: Dict = None,
                                    educational_content: Dict = None) -> BytesIO:
        """
        Generate a comprehensive PDF report.
        
        Parameters:
        -----------
        condition : str
            Cardiovascular condition assessed
        mean_risk : float
            Mean risk prediction
        credible_interval : tuple
            95% credible interval
        risk_category : str
            Risk category
        advice_lines : list
            Personalized advice
        interpretation_text : str
            Risk interpretation
        input_data : dict
            Input risk factors
        model_summary : dict, optional
            Model performance summary
        educational_content : dict, optional
            Educational information
            
        Returns:
        --------
        BytesIO
            PDF report buffer
        """
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter,
                              rightMargin=72, leftMargin=72,
                              topMargin=72, bottomMargin=18)
        
        # Build report content
        story = []
        
        # Title and header
        story.append(Paragraph("PredictRisk: Cardiovascular Risk Assessment Report", 
                              self.styles['ReportTitle']))
        story.append(Spacer(1, 20))
        
        # Report metadata
        timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        story.append(Paragraph(f"<b>Report Generated:</b> {timestamp}", self.styles['Normal']))
        story.append(Paragraph(f"<b>Condition Assessed:</b> {condition}", self.styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Executive summary
        story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        
        # Risk assessment summary
        formatted_output = format_risk_output(mean_risk, credible_interval)
        risk_text = (
            f"Your predicted risk of developing <b>{condition}</b> is "
            f"<b>{formatted_output['mean_risk_percent']}%</b> "
            f"(95% Credible Interval: {formatted_output['credible_interval_percent'][0]}% - "
            f"{formatted_output['credible_interval_percent'][1]}%)."
        )
        
        # Use appropriate style based on risk level
        risk_style_map = {
            "High Risk": "HighRisk",
            "Moderate Risk": "ModerateRisk", 
            "Low Risk": "LowRisk"
        }
        risk_style = self.styles[risk_style_map.get(risk_category, 'Normal')]
        
        story.append(Paragraph(risk_text, risk_style))
        story.append(Paragraph(f"<b>Risk Classification:</b> {risk_category}", risk_style))
        story.append(Spacer(1, 15))
        
        # Risk interpretation
        story.append(Paragraph("Risk Interpretation", self.styles['SectionHeader']))
        
        # Split interpretation text into paragraphs
        interpretation_paragraphs = interpretation_text.split('\n\n')
        for para in interpretation_paragraphs:
            if para.strip():
                story.append(Paragraph(para.strip(), self.styles['Normal']))
                story.append(Spacer(1, 10))
        
        # Input summary table
        story.append(Paragraph("Your Health Profile", self.styles['SectionHeader']))
        input_table = self._create_input_summary_table(input_data)
        story.append(input_table)
        story.append(Spacer(1, 15))
        
        # Personalized recommendations
        story.append(Paragraph("Personalized Recommendations", self.styles['SectionHeader']))
        
        for line in advice_lines:
            if line.strip():
                if line.startswith('🚨') or line.startswith('⚠️') or line.startswith('✅'):
                    story.append(Paragraph(f"<b>{line}</b>", self.styles['Normal']))
                elif line.startswith('•') or line.startswith('-'):
                    story.append(Paragraph(line, self.styles['Normal']))
                else:
                    story.append(Paragraph(line, self.styles['Normal']))
                story.append(Spacer(1, 5))
        
        story.append(Spacer(1, 15))
        
        # Educational content if provided
        if educational_content:
            story.append(Paragraph(f"About {condition}", self.styles['SectionHeader']))
            
            for section, content in educational_content.items():
                if content:
                    story.append(Paragraph(f"<b>{section.title()}:</b> {content}", 
                                         self.styles['Normal']))
                    story.append(Spacer(1, 8))
        
        # Model information if provided
        if model_summary:
            story.append(Paragraph("Model Information", self.styles['SectionHeader']))
            
            convergence_status = model_summary.get('convergence_status', 'Unknown')
            story.append(Paragraph(f"<b>Model Convergence:</b> {convergence_status}", 
                                 self.styles['Normal']))
            
            if 'r_hat_range' in model_summary:
                r_hat_range = model_summary['r_hat_range']
                story.append(Paragraph(f"<b>R-hat Range:</b> {r_hat_range[0]:.3f} - {r_hat_range[1]:.3f}", 
                                     self.styles['Normal']))
            
            story.append(Spacer(1, 10))
        
        # Disclaimer
        disclaimer_text = (
            "<b>IMPORTANT DISCLAIMER:</b><br/><br/>"
            "This report is for educational and informational purposes only. "
            "It is <b>NOT</b> a substitute for professional medical advice, diagnosis, or treatment. "
            "The predictions are based on statistical models trained on synthetic data and "
            "may not accurately reflect your actual clinical risk.<br/><br/>"
            "Always consult with a licensed healthcare provider regarding your health. "
            "If you are experiencing symptoms or have concerns about your cardiovascular health, "
            "seek immediate medical attention.<br/><br/>"
            "This tool should be used as a starting point for discussions with healthcare "
            "professionals, not as a diagnostic instrument."
        )
        
        story.append(Spacer(1, 20))
        story.append(Paragraph(disclaimer_text, self.styles['Disclaimer']))
        
        # Footer
        story.append(Spacer(1, 20))
        story.append(Paragraph("© 2025 Taiwo Michael Ayeni - PredictRisk", 
                              self.styles['Normal']))
        story.append(Paragraph("Northeastern University, College of Professional Studies Analytics", 
                              self.styles['Normal']))
        
        # Build PDF
        try:
            doc.build(story)
            buffer.seek(0)
            logger.info(f"Successfully generated PDF report for {condition}")
            return buffer
        except Exception as e:
            logger.error(f"Error generating PDF report: {e}")
            raise
    
    def _create_input_summary_table(self, input_data: Dict) -> Table:
        """
        Create a formatted table of input data.
        
        Parameters:
        -----------
        input_data : dict
            Input risk factors
            
        Returns:
        --------
        Table
            Formatted table for PDF
        """
        # Prepare table data
        table_data = [['Risk Factor', 'Your Value', 'Unit/Category']]
        
        # Continuous variables
        continuous_vars = {
            'age': 'years',
            'bmi': 'kg/m²',
            'glucose': 'mg/dL',
            'systolic_bp': 'mmHg',
            'diastolic_bp': 'mmHg', 
            'heart_rate': 'bpm',
            'sleep_hours': 'hours',
            'stress_score': '1-10 scale'
        }
        
        for var, unit in continuous_vars.items():
            if var in input_data:
                value = input_data[var]
                if isinstance(value, float):
                    formatted_value = f"{value:.1f}"
                else:
                    formatted_value = str(value)
                table_data.append([var.replace('_', ' ').title(), formatted_value, unit])
        
        # Binary variables
        binary_vars = {
            'smoking_status': 'Smoking Status',
            'alcohol_use': 'Alcohol Use',
            'physical_activity': 'Physical Activity'
        }
        
        for var, label in binary_vars.items():
            if var in input_data:
                value = input_data[var]
                if var == 'physical_activity':
                    # Physical activity is reversed (1=Yes, 0=No)
                    formatted_value = "Yes" if value == 1 else "No"
                else:
                    # Other binary vars (1=Yes, 0=No)
                    formatted_value = "Yes" if value == 1 else "No"
                table_data.append([label, formatted_value, "Yes/No"])
        
        # Create table
        table = Table(table_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
        
        # Apply table style
        table.setStyle(TableStyle([
            # Header row
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            
            # Data rows
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
            
            # Borders
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        return table
    
    def generate_simple_report(self, condition: str, mean_risk: float,
                             credible_interval: Tuple[float, float],
                             risk_category: str, advice_lines: List[str],
                             interpretation_text: str) -> BytesIO:
        """
        Generate a simple PDF report using basic canvas drawing.
        
        This method provides a fallback for environments where ReportLab
        platypus components might not work properly.
        
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
        c.drawCentredText(width/2, height-50, "PredictRisk: Cardiovascular Risk Assessment Report")
        
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
        
        # Disclaimer
        if y_position < 150:
            c.showPage()
            y_position = height - 50
        
        y_position -= 30
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
        c.drawRightString(width-50, 30, "© 2025 Taiwo Michael Ayeni - PredictRisk")
        
        c.save()
        buffer.seek(0)
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
    
    def create_risk_summary_chart(self, mean_risk: float, credible_interval: Tuple[float, float]) -> str:
        """
        Create a text-based visualization of risk assessment.
        
        Parameters:
        -----------
        mean_risk : float
            Mean risk prediction
        credible_interval : tuple
            Credible interval bounds
            
        Returns:
        --------
        str
            Text-based chart representation
        """
        # Create a simple ASCII bar chart
        chart_width = 50
        
        # Convert to percentages
        mean_percent = mean_risk * 100
        ci_lower_percent = credible_interval[0] * 100
        ci_upper_percent = credible_interval[1] * 100
        
        # Create scale (0-100%)
        scale_marks = [0, 25, 50, 75, 100]
        scale_line = ""
        
        for i in range(chart_width + 1):
            position = (i / chart_width) * 100
            if any(abs(position - mark) < 2 for mark in scale_marks):
                scale_line += "|"
            else:
                scale_line += "-"
        
        # Create risk bar
        risk_bar = ""
        for i in range(chart_width + 1):
            position = (i / chart_width) * 100
            
            if ci_lower_percent <= position <= ci_upper_percent:
                if abs(position - mean_percent) < 1:
                    risk_bar += "●"  # Mean point
                else:
                    risk_bar += "█"  # Credible interval
            else:
                risk_bar += " "
        
        # Add labels
        chart = f"""
Risk Visualization:
0%{' '*42}100%
{scale_line}
{risk_bar}

Mean Risk: {mean_percent:.1f}%
95% CI: [{ci_lower_percent:.1f}%, {ci_upper_percent:.1f}%]
        """
        
        return chart.strip()


# Convenience functions for direct use
def generate_pdf_report(condition: str, mean_risk: float, 
                       credible_interval: Tuple[float, float],
                       risk_category: str, advice_lines: List[str],
                       interpretation_text: str, input_data: Dict = None,
                       use_simple: bool = False) -> BytesIO:
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
    use_simple : bool, default False
        Whether to use simple canvas-based generation
        
    Returns:
    --------
    BytesIO
        PDF buffer
    """
    generator = PDFReportGenerator()
    
    if use_simple or input_data is None:
        return generator.generate_simple_report(
            condition, mean_risk, credible_interval, 
            risk_category, advice_lines, interpretation_text
        )
    else:
        return generator.generate_comprehensive_report(
            condition, mean_risk, credible_interval,
            risk_category, advice_lines, interpretation_text,
            input_data
        )


def create_batch_report(assessments: List[Dict], output_filename: str = None) -> BytesIO:
    """
    Create a batch report for multiple risk assessments.
    
    Parameters:
    -----------
    assessments : list
        List of assessment dictionaries
    output_filename : str, optional
        Output filename
        
    Returns:
    --------
    BytesIO
        PDF buffer with batch report
    """
    generator = PDFReportGenerator()
    buffer = BytesIO()
    
    # Implementation would go here for batch processing
    # This is a placeholder for future enhancement
    
    logger.info("Batch report generation not yet implemented")
    return buffer
