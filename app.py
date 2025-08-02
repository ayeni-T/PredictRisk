"""
Simple launcher for PredictRisk using the modular package structure.

This launches the enhanced Bayesian cardiovascular risk assessment tool
with professional package architecture and improved features.
"""

try:
    from predictrisk.web_app import main
    
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    # Fallback to basic error message if package not available
    import streamlit as st
    
    st.error("Package import error - please check installation")
    st.error(f"Error details: {str(e)}")
    st.info("If you're seeing this, the package may need to be properly installed.")
