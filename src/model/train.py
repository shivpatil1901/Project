"""
Model Training Module
Handles training of 4 models: Logistic Regression, Random Forest, Gradient Boosting, Neural Network
"""
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import joblib


def load_params(params_file="params.yaml"):
    """Load parameters from params.yaml"""
    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)
    return params


def train_logistic_regression(X_train, y_train, params):
    """Train Logistic Regression model"""
    model_config = params['models']['logistic_regression']
    
    model = LogisticRegression(
        random_state=model_config['random_state'],
        max_iter=1000
    )
    model.fit(X_train, y_train)
    
    print("✓ Logistic Regression trained")
    return model


def train_random_forest(X_train, y_train, params):
    """Train Random Forest model"""
    model_config = params['models']['random_forest']
    
    model = RandomForestClassifier(
        n_estimators=model_config['n_estimators'],
        random_state=model_config['random_state'],
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    print("✓ Random Forest trained")
    return model


def train_gradient_boosting(X_train, y_train, params):
    """Train Gradient Boosting model"""
    model_config = params['models']['gradient_boosting']
    
    model = GradientBoostingClassifier(
        n_estimators=model_config['n_estimators'],
        random_state=model_config['random_state']
    )
    model.fit(X_train, y_train)
    
    print("✓ Gradient Boosting trained")
    return model


def train_neural_network(X_train, y_train, params):
    """Train Neural Network model"""
    model_config = params['models']['neural_network']
    
    model = MLPClassifier(
        hidden_layer_sizes=tuple(model_config['hidden_layer_sizes']),
        max_iter=model_config['max_iter'],
        random_state=model_config['random_state'],
        solver=model_config['solver']
    )
    model.fit(X_train, y_train)
    
    print("✓ Neural Network trained")
    return model


def train_models(input_path=None, output_path=None, params_file="params.yaml"):
    """Main training pipeline"""
    params = load_params(params_file)
    
    if input_path is None:
        input_path = Path(params['data']['features']) / "selected"
    else:
        input_path = Path(input_path)
    
    if output_path is None:
        output_path = Path(params['data']['models_dir'])
    else:
        output_path = Path(output_path)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    X_train_selected = np.load(input_path / "X_train_selected.npy")
    y_train = pd.read_csv(input_path / "y_train.csv").iloc[:, 0]
    
    print(f"Training data shape: {X_train_selected.shape}")
    print(f"Target shape: {y_train.shape}\n")
    
    # Train all models
    print("=" * 50)
    print("Training Models")
    print("=" * 50)
    
    lr_model = train_logistic_regression(X_train_selected, y_train, params)
    rf_model = train_random_forest(X_train_selected, y_train, params)
    gb_model = train_gradient_boosting(X_train_selected, y_train, params)
    nn_model = train_neural_network(X_train_selected, y_train, params)
    
    # Save trained models
    print(f"\nSaving models to {output_path}...")
    joblib.dump(lr_model, output_path / "logistic_regression.pkl")
    joblib.dump(rf_model, output_path / "random_forest.pkl")
    joblib.dump(gb_model, output_path / "gradient_boosting.pkl")
    joblib.dump(nn_model, output_path / "neural_network.pkl")
    
    # Create a model registry
    model_registry = {
        'logistic_regression': 'logistic_regression.pkl',
        'random_forest': 'random_forest.pkl',
        'gradient_boosting': 'gradient_boosting.pkl',
        'neural_network': 'neural_network.pkl'
    }
    
    registry_df = pd.DataFrame(list(model_registry.items()), columns=['model_name', 'file_path'])
    registry_df.to_csv(output_path / "model_registry.csv", index=False)
    
    print(f"\n✓ Model training complete!")
    print(f"  Models saved to: {output_path}")
    print(f"\nTrained Models:")
    print(f"  - Logistic Regression")
    print(f"  - Random Forest")
    print(f"  - Gradient Boosting")
    print(f"  - Neural Network")
    
    return {
        'lr_model': lr_model,
        'rf_model': rf_model,
        'gb_model': gb_model,
        'nn_model': nn_model
    }


if __name__ == "__main__":
    train_models()
