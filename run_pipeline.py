#!/usr/bin/env python
"""
Quick Start: Run Credit Risk Modeling Pipeline with a Single Command
Usage: python run_pipeline.py
"""
import sys
import os
from pathlib import Path

# Ensure we're in the right directory
os.chdir(Path(__file__).parent)
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data.process import process_data
from features.engineer import feature_engineering
from features.selection import feature_selection
from model.train import train_models
from model.evaluate import evaluate_all_models


def run_pipeline():
    """Execute full pipeline from raw data to model evaluation"""
    
    print("\n" + "="*80)
    print(" "*20 + "CREDIT RISK MODELING PIPELINE")
    print("="*80)
    
    try:
        # Step 1
        print("\n[STEP 1/5] Processing Raw Data...")
        print("-"*80)
        X_processed, y_processed = process_data()
        
        # Step 2
        print("\n[STEP 2/5] Feature Engineering...")
        print("-"*80)
        X_engineered, y_engineered = feature_engineering()
        
        # Step 3
        print("\n[STEP 3/5] Feature Selection & Scaling...")
        print("-"*80)
        (X_train_selected, X_test_selected, y_train, y_test, 
         selected_features, scaler, rf_model, selector) = feature_selection()
        
        # Step 4
        print("\n[STEP 4/5] Training Models...")
        print("-"*80)
        trained_models = train_models()
        
        # Step 5
        print("\n[STEP 5/5] Evaluating Models...")
        print("-"*80)
        results_df, class_reports = evaluate_all_models()
        
        # Summary
        print("\n" + "="*80)
        print(" "*25 + "✓ PIPELINE COMPLETE!")
        print("="*80)
        
        print("\n📊 Model Performance Summary:")
        print("-"*80)
        print(results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'AUC']].to_string(index=False))
        
        print("\n📁 Output Locations:")
        print("-"*80)
        print("  Data Artifacts:")
        print("    - Processed data: data/processed/")
        print("    - Engineered features: data/features/")
        print("    - Selected features: data/features/selected/")
        print("\n  Models:")
        print("    - Trained models: models/")
        print("    - Model registry: models/model_registry.csv")
        print("\n  Evaluation:")
        print("    - Performance metrics: models/evaluation/model_performance.csv")
        print("    - Classification reports: models/evaluation/classification_reports.json")
        
        print("\n💡 Next Steps:")
        print("-"*80)
        print("  1. Review metrics: models/evaluation/model_performance.csv")
        print("  2. Track with DVC: dvc repro")
        print("  3. Track with MLflow: mlflow ui")
        print("  4. Build scorecard from best model")
        print("  5. Monitor with PSI (Population Stability Index)")
        
        print("\n" + "="*80 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error:")
        print(f"   {str(e)}")
        print("\nDebugging tips:")
        print("  - Check params.yaml for correct paths")
        print("  - Ensure data/loan.csv exists")
        print("  - Run individual steps separately: python src/data/process.py")
        print("  - Check that all dependencies are installed: pip install -r requirements_pipeline.txt")
        return False


if __name__ == "__main__":
    success = run_pipeline()
    sys.exit(0 if success else 1)
