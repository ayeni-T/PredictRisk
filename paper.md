---
title: 'PredictRisk: An Open-Source Bayesian Web Application for Cardiovascular Risk Assessment'
tags:
  - Python
  - Bayesian statistics
  - cardiovascular risk
  - machine learning
  - healthcare
  - uncertainty quantification
  - streamlit
  - web application
authors:
  - name: Taiwo Michael Ayeni
    orcid: 0000-0002-6823-1417
    affiliation: 1
affiliations:
 - name: Northeastern University, College of Professional Studies Analytics, Toronto, Canada
   index: 1
date: 25 June 2025
bibliography: paper.bib
---

# Summary

PredictRisk is a comprehensive open-source web application that provides Bayesian cardiovascular risk assessment with explicit uncertainty quantification. The software implements six independent Bayesian logistic regression models to predict major cardiovascular conditions (stroke, heart disease, hypertension, heart failure, atrial fibrillation, and peripheral artery disease) using established risk factors. Built with Streamlit for the web interface and Bambi/PyMC [@capretto2022bambi; @salvatier2016probabilistic] for Bayesian modeling, PredictRisk offers real-time probabilistic risk predictions with 95% credible intervals, personalized health recommendations, and downloadable PDF reports. The application addresses the critical need for accessible, uncertainty-aware cardiovascular risk tools that can democratize sophisticated statistical modeling for healthcare applications.

# Statement of need

Cardiovascular diseases remain the leading cause of global mortality, accounting for approximately 17.9 million deaths annually [@roth2020global]. Yet many existing risk assessment tools provide only point estimates without communicating prediction uncertainty, creating challenges for both clinical decision-making and patient understanding. Current tools like the Framingham Risk Score [@dagostino2008general] and ASCVD Risk Calculator [@goff20142013], while clinically validated, are not readily accessible outside clinical settings and do not provide uncertainty quantification.

PredictRisk fills this gap by providing an open-source, web-based platform that combines sophisticated Bayesian statistical methods [@gelman2013bayesian] with user-friendly interfaces. The software serves multiple user communities: (1) researchers studying cardiovascular risk modeling methodologies, (2) healthcare professionals seeking accessible risk assessment tools with uncertainty quantification, (3) data scientists learning Bayesian approaches to medical prediction, and (4) individuals interested in understanding their cardiovascular risk profile. By making advanced Bayesian modeling accessible through a simple web interface, PredictRisk democratizes access to uncertainty-aware risk assessment.

# Software description

PredictRisk is implemented as a modular Python application providing cardiovascular risk assessment for six conditions using 11 evidence-based risk factors including age, BMI, glucose levels, blood pressure, heart rate, lifestyle factors (smoking, alcohol use, physical activity, sleep hours), and stress levels. Each condition is modeled using independent Bayesian logistic regression implemented through the Bambi library [@capretto2022bambi], which provides a high-level interface to PyMC [@salvatier2016probabilistic].

## Bayesian Modeling Pipeline

The Bayesian models employ No-U-Turn Sampler (NUTS) [@hoffman2014nuts] with four parallel chains, 1,000 draws per chain after 1,000 tuning steps. All models demonstrate exceptional convergence properties (R-hat = 1.0) with effective sample sizes exceeding 2,570 for all parameters. The software provides full posterior distributions enabling rich probabilistic inference with 95% credible intervals for uncertainty quantification. Model analysis and diagnostics are performed using ArviZ [@kumar2019arviz] for comprehensive posterior analysis.

## Web Application Interface

The user interface is built with Streamlit, providing an intuitive, responsive design that guides users through risk factor input with helpful tooltips and validation. The interface features real-time risk assessment, uncertainty visualization, automatic risk categorization into Low Risk (<40%), Moderate Risk (40-70%), and High Risk (≥70%) categories, and personalized health recommendations based on individual risk profiles.

## Technical Implementation

The software leverages a robust technology stack with Python as the core language, Bambi/PyMC [@capretto2022bambi; @salvatier2016probabilistic] for Bayesian modeling, ArviZ [@kumar2019arviz] for posterior analysis, Streamlit for web interface, and ReportLab for PDF report generation. The application is deployed at https://predictrisk.streamlit.app/ providing immediate accessibility without installation requirements.

# Quality assurance

The software implements comprehensive quality assurance measures including rigorous model convergence diagnostics using R-hat statistics and effective sample size evaluation, input validation with bounds checking, and complete seed-based reproducibility for all synthetic data generation and model training. The modular design ensures clean separation of concerns with comprehensive documentation and robust error handling throughout the application.

# Community guidelines

PredictRisk embraces open science principles and welcomes community contributions. The project is hosted on GitHub (https://github.com/ayeni-T/PredictRisk) under the MIT License, ensuring broad accessibility and reuse. Contributors can engage through GitHub Issues for bug reports and feature requests, pull requests for code improvements, and direct collaboration opportunities for clinical validation studies using real datasets.

The project roadmap includes clinical validation through healthcare collaborations, model expansion to additional cardiovascular conditions, advanced interpretability features, and clinical integration through FHIR-compliant APIs for electronic health record systems.

# Acknowledgments

The author thanks the open-source community, particularly the PyMC, Bambi, ArviZ, and Streamlit development teams. Special recognition goes to Northeastern University for providing the academic environment that supported this research.

# References
