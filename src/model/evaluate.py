"""
Model Evaluation Module
Handles evaluation of trained models and metrics calculation
"""
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, accuracy_score, recall_score, fbeta_score, 
    roc_auc_score, classification_report, precision_score, f1_score
)
import joblib
import json


def load_params(params_file="params.yaml"):
    """Load parameters from params.yaml"""
    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)
    return params


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a single model and return metrics
    
    Returns:
        dict: Dictionary containing all metrics
    """
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    f2 = fbeta_score(y_test, y_pred, beta=2)
    auc = roc_auc_score(y_test, y_pred_prob)
    
    # Create classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    metrics = {
        'Model': model_name,
        'Accuracy': float(accuracy),
        'Precision': float(precision),
        'Recall': float(recall),
        'F1': float(f1),
        'F2': float(f2),
        'AUC': float(auc),
        'TN': int(conf_matrix[0, 0]),
        'FP': int(conf_matrix[0, 1]),
        'FN': int(conf_matrix[1, 0]),
        'TP': int(conf_matrix[1, 1])
    }
    
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    print(f"Confusion Matrix:")
    print(f"  TN={metrics['TN']}, FP={metrics['FP']}")
    print(f"  FN={metrics['FN']}, TP={metrics['TP']}")
    print(f"\nMetrics:")
    print(f"  Accuracy:  {metrics['Accuracy']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall:    {metrics['Recall']:.4f}")
    print(f"  F1 Score:  {metrics['F1']:.4f}")
    print(f"  F2 Score:  {metrics['F2']:.4f}")
    print(f"  AUC:       {metrics['AUC']:.4f}")
    
    return metrics, class_report


def evaluate_all_models(input_path=None, models_path=None, output_path=None, params_file="params.yaml"):
    """
    Evaluate all trained models
    
    Args:
        input_path: Path to feature selection outputs
        models_path: Path to trained models
        output_path: Path to save evaluation results
        params_file: Path to params.yaml
    """
    params = load_params(params_file)
    
    if input_path is None:
        input_path = Path(params['data']['features']) / "selected"
    else:
        input_path = Path(input_path)
    
    if models_path is None:
        models_path = Path(params['data']['models_dir'])
    else:
        models_path = Path(models_path)
    
    if output_path is None:
        output_path = models_path / "evaluation"
    else:
        output_path = Path(output_path)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load test data
    X_test_selected = np.load(input_path / "X_test_selected.npy")
    y_test = pd.read_csv(input_path / "y_test.csv").iloc[:, 0]
    
    selected_features = pd.read_csv(input_path / "selected_features.csv")['selected_features'].tolist()
    
    print(f"Test data shape: {X_test_selected.shape}")
    print(f"Selected features: {len(selected_features)}")
    
    # Load trained models
    lr_model = joblib.load(models_path / "logistic_regression.pkl")
    rf_model = joblib.load(models_path / "random_forest.pkl")
    gb_model = joblib.load(models_path / "gradient_boosting.pkl")
    nn_model = joblib.load(models_path / "neural_network.pkl")
    
    # Evaluate all models
    all_results = []
    all_reports = {}
    
    models = {
        'Logistic Regression': lr_model,
        'Random Forest': rf_model,
        'Gradient Boosting': gb_model,
        'Neural Network': nn_model
    }
    
    for model_name, model in models.items():
        metrics, class_report = evaluate_model(model, X_test_selected, y_test, model_name)
        all_results.append(metrics)
        all_reports[model_name] = class_report
    
    # Create results dataframe
    results_df = pd.DataFrame(all_results)
    
    # Save evaluation results
    results_df.to_csv(output_path / "model_performance.csv", index=False)
    
    # Save detailed classification reports
    with open(output_path / "classification_reports.json", 'w') as f:
        json.dump(all_reports, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(results_df.to_string(index=False))
    
    print(f"\n✓ Evaluation complete!")
    print(f"  Results saved to: {output_path}")
    
    return results_df, all_reports


if __name__ == "__main__":
    evaluate_all_models()
