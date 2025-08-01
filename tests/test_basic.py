"""
Basic tests for PredictRisk application
Required for JOSS submission
"""
import os
import pandas as pd

def test_data_file_exists():
    """Test that the synthetic dataset file exists."""
    assert os.path.exists("multi_cvd_dataset.csv"), "Dataset file not found"

def test_data_structure():
    """Test that data has expected structure."""
    df = pd.read_csv("multi_cvd_dataset.csv")
    assert len(df) == 500, f"Expected 500 records, found {len(df)}"
    
    # Check key columns exist
    expected_cols = ['age', 'bmi', 'glucose', 'risk_stroke']
    for col in expected_cols:
        assert col in df.columns, f"Column {col} missing"

def test_app_file_exists():
    """Test that main app file exists."""
    assert os.path.exists("app.py"), "Main app file not found"

def test_requirements_exists():
    """Test that requirements file exists."""
    assert os.path.exists("requirements.txt"), "Requirements file not found"

if __name__ == "__main__":
    print("Running basic tests...")
    test_data_file_exists()
    test_data_structure() 
    test_app_file_exists()
    test_requirements_exists()
    print("✅ All tests passed!")
