"""
Feature Selection Module
Handles train/test split, scaling, leakage prevention, and Random Forest feature selection
"""
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib


def load_params(params_file="params.yaml"):
    """Load parameters from params.yaml"""
    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)
    return params


def remove_leaky_features(X, leaky_features_list):
    """Remove outcome-dependent features that cause data leakage"""
    clean_feature_mask = ~X.columns.isin(leaky_features_list)
    X_clean = X.loc[:, clean_feature_mask]
    
    print(f"✓ Removed {len(leaky_features_list)} leaky features")
    print(f"  Leaky features: {leaky_features_list}")
    print(f"  Shape before: {X.shape}, after: {X_clean.shape}")
    
    return X_clean, clean_feature_mask


def train_test_split_data(X, y, test_size, random_state):
    """Split data into train and test sets"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"✓ Train/test split (test_size={test_size})")
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """Standardize features using StandardScaler"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✓ Scaled features with StandardScaler")
    
    return X_train_scaled, X_test_scaled, scaler


def select_features_with_rf(X_train_scaled, X_test_scaled, y_train, X_columns, 
                            n_estimators=100, threshold="mean", random_state=42):
    """Select features using Random Forest importance"""
    
    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )
    rf.fit(X_train_scaled, y_train)
    
    # Select features with importance above threshold
    selector = SelectFromModel(rf, prefit=True, threshold=threshold)
    X_train_selected = selector.transform(X_train_scaled)
    X_test_selected = selector.transform(X_test_scaled)
    
    # Get selected feature names
    selected_mask = selector.get_support()
    selected_feature_names = X_columns[selected_mask]
    
    print(f"✓ Random Forest feature selection (threshold={threshold})")
    print(f"  Input features: {len(X_columns)}, Selected: {len(selected_feature_names)}")
    print(f"  Selected features: {list(selected_feature_names)}")
    
    return X_train_selected, X_test_selected, selected_feature_names, rf, selector


def feature_selection(input_path=None, output_path=None, params_file="params.yaml"):
    """Main feature selection pipeline"""
    params = load_params(params_file)
    
    if input_path is None:
        input_path = Path(params['data']['features'])
    else:
        input_path = Path(input_path)
    
    if output_path is None:
        output_path = Path(params['data']['features']) / "selected"
    else:
        output_path = Path(output_path)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load engineered features
    X = pd.read_csv(input_path / "X_engineered.csv")
    y = pd.read_csv(input_path / "y_engineered.csv").iloc[:, 0]
    
    print(f"Input shape: {X.shape}\n")
    
    # Remove leaky features
    leaky_features = params['leakage_prevention']['leaky_features']
    X_clean, _ = remove_leaky_features(X, leaky_features)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split_data(
        X_clean, y,
        test_size=params['feature_selection']['test_size'],
        random_state=params['feature_selection']['random_state']
    )
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Select features with Random Forest
    X_train_selected, X_test_selected, selected_features, rf_model, selector = select_features_with_rf(
        X_train_scaled, X_test_scaled, y_train, X_clean.columns,
        n_estimators=params['feature_selection']['rf_n_estimators'],
        threshold=params['feature_selection']['rf_threshold'],
        random_state=params['feature_selection']['random_state']
    )
    
    # Save outputs
    np.save(output_path / "X_train_selected.npy", X_train_selected)
    np.save(output_path / "X_test_selected.npy", X_test_selected)
    np.save(output_path / "X_train_scaled.npy", X_train_scaled)
    np.save(output_path / "X_test_scaled.npy", X_test_scaled)
    
    y_train.to_csv(output_path / "y_train.csv", index=False)
    y_test.to_csv(output_path / "y_test.csv", index=False)
    
    pd.DataFrame({'selected_features': selected_features}).to_csv(
        output_path / "selected_features.csv", index=False
    )
    
    joblib.dump(scaler, output_path / "scaler.pkl")
    joblib.dump(rf_model, output_path / "rf_selector_model.pkl")
    joblib.dump(selector, output_path / "feature_selector.pkl")
    
    print(f"\n✓ Feature selection complete!")
    print(f"  Final shape - Train: {X_train_selected.shape}, Test: {X_test_selected.shape}")
    print(f"  Saved to: {output_path}")
    
    return (X_train_selected, X_test_selected, y_train, y_test, 
            selected_features, scaler, rf_model, selector)


if __name__ == "__main__":
    feature_selection()
