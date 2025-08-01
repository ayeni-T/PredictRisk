"""
Setup script for PredictRisk package.

This script enables pip installation and distribution of the PredictRisk
Bayesian cardiovascular risk assessment framework.
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

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
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
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
        ]
    },
    entry_points={
        "console_scripts": [
            "predictrisk=predictrisk.web_app:main",
            "predictrisk-generate-data=predictrisk.data_generator:main",
        ],
    },
    include_package_data=True,
    package_data={
        "predictrisk": [
            "*.nc",  # Model files
            "*.csv", # Data files
        ],
    },
    keywords=[
        "bayesian", "cardiovascular", "risk-assessment", "machine-learning",
        "healthcare", "uncertainty-quantification", "medical-informatics",
        "streamlit", "pymc", "bambi"
    ],
    zip_safe=False,
)
