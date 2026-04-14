# Credit Risk Modeling Pipeline

A modular, DVC-orchestrated machine learning pipeline for building credit risk models with proper data leakage prevention.

## 📁 Project Structure

```
src/
├── data/
│   └── process.py          # Data loading, cleaning, feature extraction
├── features/
│   ├── engineer.py         # Feature engineering & correlation analysis
│   └── selection.py        # Feature selection, train/test split, scaling
├── model/
│   ├── train.py            # Model training (LR, RF, GB, NN)
│   └── evaluate.py         # Model evaluation & metrics calculation
└── pipeline.py             # Main orchestrator

data/
├── loan.csv                # Raw input data
├── processed/              # L1: Processed data
├── features/               # L2: Engineered features
└── features/selected/      # L3: Selected features (train/test)

models/
├── *.pkl                   # Trained model artifacts
├── evaluation/             # Metrics & reports
└── model_registry.csv      # Model tracking

params.yaml                # Configuration & hyperparameters
dvc.yaml                   # DVC pipeline definition
```

## 🚀 Running the Pipeline

### Option 1: Run Complete Pipeline with DVC
```bash
dvc repro
```

### Option 2: Run Individual Steps
```bash
# Step 1: Process raw data
python src/data/process.py

# Step 2: Engineer features
python src/features/engineer.py

# Step 3: Select features & create train/test split
python src/features/selection.py

# Step 4: Train models
python src/model/train.py

# Step 5: Evaluate models
python src/model/evaluate.py
```

### Option 3: Run Full Pipeline Programmatically
```bash
python src/pipeline.py
```

## 🔧 Configuration

All parameters are centralized in `params.yaml`:

- **Data paths**: Raw and output directories
- **Processing thresholds**: Missing data tolerance, correlation thresholds
- **Feature selection**: Random Forest parameters, test/train split ratio
- **Model hyperparameters**: per-model configuration
- **Leakage prevention**: List of outcome-dependent features to remove

Edit `params.yaml` to customize pipeline behavior.

## 📊 Pipeline Stages

### 1. Data Processing (`src/data/process.py`)
- Load raw CSV data
- Create binary target variable (loan_status_binary)
- Remove columns with >51% missing data
- Convert categorical and date columns
- Engineer temporal features (loan_age, time_since_last_payment, etc.)
- One-hot encode categorical variables
- Impute remaining missing values

**Output**: `X_processed.csv`, `y.csv`

### 2. Feature Engineering (`src/features/engineer.py`)
- Calculate Weight of Evidence (WoE) and Information Value (IV) for categorical features
- Perform correlation analysis with target
- Identify and drop highly correlated feature pairs
- Engineer new features (loan_amnt_div_instlmnt ratio)
- Remove bulk features with low predictive power

**Output**: `X_engineered.csv`, `y_engineered.csv`

### 3. Feature Selection (`src/features/selection.py`)
- **Remove leaky features** (outcome-dependent) BEFORE selection
  - total_rec_prncp, total_rec_int, total_rec_late_fee, recoveries, last_pymnt_amnt, out_prncp
- Train/test split (80/20)
- Standardize features with StandardScaler
- Train Random Forest for feature importance
- Select features with importance > mean threshold

**Output**: 
- `X_train_selected.npy`, `X_test_selected.npy` (selected features)
- `X_train_scaled.npy`, `X_test_scaled.npy` (all features, scaled)
- `y_train.csv`, `y_test.csv`
- `selected_features.csv` (feature names)
- `scaler.pkl`, `rf_selector_model.pkl`, `feature_selector.pkl`

### 4. Model Training (`src/model/train.py`)
Trains 4 models on selected features:
- **Logistic Regression**: Interpretable baseline
- **Random Forest**: Ensemble with feature importance
- **Gradient Boosting**: High-performance gradient boosting
- **Neural Network**: Multi-layer perceptron

**Output**: 
- `logistic_regression.pkl`
- `random_forest.pkl`
- `gradient_boosting.pkl`
- `neural_network.pkl`
- `model_registry.csv` (model tracking)

### 5. Model Evaluation (`src/model/evaluate.py`)
- Load trained models and test data
- Calculate comprehensive metrics:
  - Accuracy, Precision, Recall, F1, F2, AUC
  - Confusion Matrix (TN, FP, FN, TP)
  - Classification Reports

**Output**:
- `model_performance.csv` (summary metrics)
- `classification_reports.json` (detailed per-model reports)

## 🛡️ Data Leakage Prevention

The pipeline removes outcome-dependent features **before** Random Forest feature selection:

```python
leaky_features = {
    'total_rec_prncp',      # Only known after loan settlement
    'total_rec_int',        # Only accrues if borrower pays
    'total_rec_late_fee',   # Only if payment missed
    'recoveries',           # Only if default occurred
    'last_pymnt_amnt',      # Evaluation-time snapshot
    'out_prncp'             # Outcome-driven balance
}
```

This ensures Random Forest cannot select features that wouldn't be available at prediction time.

## 📈 Integration with MLflow

Track experiments with MLflow:

```bash
# Start MLflow UI
mlflow ui

# In your scripts, add MLflow logging
import mlflow

with mlflow.start_run():
    mlflow.log_params(params)
    results = evaluate_all_models()
    mlflow.log_metrics(results)
    mlflow.sklearn.log_model(model, "model")
```

## 📋 Key Features

✅ **Modular Architecture**: Each stage is independent and reusable
✅ **DVC Integration**: Full pipeline orchestration and version control
✅ **Leakage Prevention**: Removes outcome-dependent features before feature selection
✅ **Parameterized**: All configs in `params.yaml`, no hardcoding
✅ **Reproducibility**: Fixed random seeds, stratified splits
✅ **Scalability**: Ready for MLflow experiment tracking
✅ **Clean Output**: Metrics saved as CSV/JSON for upstream consumption

## 🔄 Rerunning/Modifying Pipeline

### Change hyperparameters:
```yaml
# params.yaml
models:
  random_forest:
    n_estimators: 200  # was 100
    random_state: 42
```

Then run:
```bash
dvc repro
```

DVC will only rerun affected stages.

### Add new model:
1. Add config to `params.yaml`
2. Implement `train_<model>()` in `src/model/train.py`
3. Add to `train_models()` function
4. Run `dvc repro`

## 📦 Requirements

```
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=1.0.0
pyyaml>=5.4.0
joblib>=1.1.0
```

Install with:
```bash
pip install -r requirements.txt
```

## 🎯 Next Steps

1. **Run the pipeline**: `dvc repro`
2. **Review results**: Check `models/evaluation/model_performance.csv`
3. **Integrate MLflow**: Add experiment tracking to `src/model/train.py`
4. **Create scorecard**: Build WoE-based scorecard from Logistic Regression
5. **Monitor with PSI**: Track Population Stability Index over time

---

**Author**: MLOps Team  
**Last Updated**: 2026-04-10
