"""
Setup script for PredictRisk package.

This script enables pip installation and distribution of the PredictRisk
Bayesian cardiovascular risk assessment framework.
"""

from setuptools import setup, find_packages
import os
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
try:
    with open(this_directory / 'README.md', encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "Bayesian Cardiovascular Risk Assessment with Uncertainty Quantification"

# Read requirements
try:
    with open(this_directory / 'requirements.txt', encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
except FileNotFoundError:
    # Fallback requirements if file not found
    requirements = [
        "streamlit>=1.28.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "bambi>=0.13.0",
        "arviz>=0.12.0",
        "reportlab>=3.6.0",
        "scipy>=1.7.0"
    ]

# Package metadata
setup(
    name="predictrisk",
    version="0.1.0",
    author="Taiwo Michael Ayeni",
    author_email="ayeni.t@northeastern.edu",
    description="Bayesian Cardiovascular Risk Assessment with Uncertainty Quantification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ayeni-T/PredictRisk",
    project_urls={
        "Bug Reports": "https://github.com/ayeni-T/PredictRisk/issues",
        "Source": "https://github.com/ayeni-T/PredictRisk",
        "Documentation": "https://github.com/ayeni-T/PredictRisk/blob/main/README.md",
        "Live Demo": "https://predictrisk.streamlit.app/"
    },
    packages=find_packages(exclude=["tests*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Natural Language :: English"
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "flake8>=5.0",
            "mypy>=1.0",
            "jupyter>=1.0",
            "notebook>=6.4"
        ],
        "docs": [
            "sphinx>=5.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.18"
        ],
        "test": [
            "pytest>=7.0",
            "pytest-mock>=3.0",
            "coverage>=6.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "predictrisk-app=predictrisk.web_app:main",
            "predictrisk-generate-data=predictrisk.data_generator:generate_synthetic_data",
        ],
    },
    include_package_data=True,
    package_data={
        "predictrisk": [
            "*.nc",   # Model files
            "*.csv",  # Data files
        ],
    },
    data_files=[
        (".", ["*.nc", "*.csv"]),  # Include model and data files in root
    ],
    keywords=[
        "bayesian", "cardiovascular", "risk-assessment", "machine-learning",
        "healthcare", "uncertainty-quantification", "medical-informatics",
        "streamlit", "pymc", "bambi", "statistics", "epidemiology"
    ],
    zip_safe=False,
    license="MIT",
    platforms=["any"],
    maintainer="Taiwo Michael Ayeni",
    maintainer_email="ayeni.t@northeastern.edu",
)
