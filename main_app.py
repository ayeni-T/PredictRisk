"""
Package-based launcher for PredictRisk using modular structure.

This demonstrates the professional package structure for JOSS submission
while keeping the original app.py working for the live deployment.
"""

try:
    from predictrisk.web_app import main
    
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print(f"Package import error: {e}")
    print("Note: This requires the predictrisk package to be properly installed.")
    print("For the working version, use app.py instead.")
