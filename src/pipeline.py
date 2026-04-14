"""
Main Pipeline Orchestrator
Runs the complete credit-risk modeling pipeline:
1. Data Processing
2. Feature Engineering
3. Feature Selection
4. Model Training
5. Model Evaluation
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data.process import process_data
from features.engineer import feature_engineering
from features.selection import feature_selection
from model.train import train_models
from model.evaluate import evaluate_all_models


def main(params_file="params.yaml"):
    """Run complete pipeline"""
    
    print("\n" + "="*70)
    print("CREDIT RISK MODELING PIPELINE")
    print("="*70)
    
    # Step 1: Data Processing
    print("\n[STEP 1] Data Processing")
    print("-" * 70)
    X_processed, y_processed = process_data(params_file=params_file)
    
    # Step 2: Feature Engineering
    print("\n[STEP 2] Feature Engineering")
    print("-" * 70)
    X_engineered, y_engineered = feature_engineering(params_file=params_file)
    
    # Step 3: Feature Selection
    print("\n[STEP 3] Feature Selection & Scaling")
    print("-" * 70)
    (X_train_selected, X_test_selected, y_train, y_test, 
     selected_features, scaler, rf_model, selector) = feature_selection(params_file=params_file)
    
    # Step 4: Model Training
    print("\n[STEP 4] Model Training")
    print("-" * 70)
    trained_models = train_models(params_file=params_file)
    
    # Step 5: Model Evaluation
    print("\n[STEP 5] Model Evaluation")
    print("-" * 70)
    results_df, class_reports = evaluate_all_models(params_file=params_file)
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print("\nNext Steps:")
    print("  1. Review model performance in models/evaluation/")
    print("  2. Track experiments with MLflow: mlflow ui")
    print("  3. Use DVC to version control pipeline: dvc repro")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
