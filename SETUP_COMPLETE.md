# 🎯 ML Pipeline Architecture - Complete Setup

## 📋 Summary of Created Files

Your credit risk modeling pipeline is now fully modularized and DVC-ready. Here's what was created:

### ✅ Core Pipeline Modules

| File | Purpose |
|------|---------|
| `src/data/process.py` | Data loading, cleaning, target creation, feature extraction |
| `src/features/engineer.py` | Correlation analysis, WoE/IV, Feature engineering |
| `src/features/selection.py` | Leakage prevention, Train/test split, Scaling, RF feature selection |
| `src/model/train.py` | Model training only (LR, RF, GB, NN) |
| `src/model/evaluate.py` | Model evaluation, metrics calculation |
| `src/pipeline.py` | Main orchestrator (runs all stages) |

### ✅ Configuration Files

| File | Purpose |
|------|---------|
| `params.yaml` | Centralized configuration for all hyperparameters |
| `dvc.yaml` | DVC pipeline definition with stages and dependencies |
| `requirements_pipeline.txt` | Python dependencies |

### ✅ Documentation & Execution

| File | Purpose |
|------|---------|
| `PIPELINE.md` | Complete pipeline documentation |
| `run_pipeline.py` | One-command pipeline execution script |

---

## 🚀 How to Run

### Option 1: One-Command Execution
```bash
python run_pipeline.py
```
This runs the entire pipeline from raw data → model evaluation.

### Option 2: DVC Pipeline Orchestration
```bash
dvc repro
```
Runs pipeline with dependency tracking and caching.

### Option 3: Step-by-Step
```bash
python src/data/process.py
python src/features/engineer.py
python src/features/selection.py
python src/model/train.py
python src/model/evaluate.py
```

### Option 4: Full Programmatic
```bash
python src/pipeline.py
```

---

## 🔑 Key Features

### ✨ What's Different from the Notebook

**Before** (Monolithic Notebook):
- All code in one 1200+ line notebook
- Hard to reuse, test, reproduce
- Mixed concerns (processing, engineering, training, evaluation)
- No configuration file
- All parameters hardcoded

**After** (Modular Pipeline):
- 5 independent, composable modules
- Each stage is testable and reusable
- DVC orchestration with dependency tracking
- All config in `params.yaml` (no hardcoding)
- Clean separation of concerns
- Ready for MLflow experiment tracking

### 🛡️ Data Leakage Prevention

**Implemented at feature selection stage:**
```python
leaky_features = {
    'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee',
    'recoveries', 'last_pymnt_amnt', 'out_prncp'
}
```
These are removed **BEFORE** Random Forest feature selection to prevent them from influencing feature importance.

### 📊 Pipeline Stages

```
Raw Data (loan.csv)
    ↓
[1] Process Data → X_processed.csv, y.csv
    ↓
[2] Feature Engineering → X_engineered.csv, y_engineered.csv
    ↓
[3] Feature Selection → X_train/test_selected.npy (6 features), scalers, selectors
    ↓
[4] Train Models → 4 trained models (.pkl files)
    ↓
[5] Evaluate → model_performance.csv, classification_reports.json
```

### ⚙️ Easy Configuration

Change any parameter in `params.yaml`:

```yaml
# Example: Change RF parameters
feature_selection:
  rf_n_estimators: 200  # was 100
  rf_threshold: "median"  # was "mean"

# Example: Add new model
models:
  xgboost:
    max_depth: 5
    n_estimators: 100
```

No code changes needed!

---

## 📁 Output Structure

After running the pipeline:

```
data/
├── processed/                    # After Step 1
│   ├── X_processed.csv
│   └── y.csv
├── features/                     # After Step 2
│   ├── X_engineered.csv
│   ├── y_engineered.csv
│   └── selected/                 # After Step 3
│       ├── X_train_selected.npy
│       ├── X_test_selected.npy
│       ├── y_train.csv
│       ├── y_test.csv
│       ├── selected_features.csv
│       ├── scaler.pkl
│       ├── rf_selector_model.pkl
│       └── feature_selector.pkl

models/                           # After Step 4
├── logistic_regression.pkl
├── random_forest.pkl
├── gradient_boosting.pkl
├── neural_network.pkl
├── model_registry.csv
└── evaluation/                   # After Step 5
    ├── model_performance.csv     # ← Key metrics here
    └── classification_reports.json
```

---

## 🔄 Integration with MLflow (Next Step)

To track experiments with MLflow:

```python
# In src/model/train.py - add at top:
import mlflow

# In train_models() function - wrap with:
with mlflow.start_run(run_name="rf_100_estimators"):
    # Training code...
    mlflow.log_params(model_config)
    mlflow.sklearn.log_model(model, "model")

# Then: mlflow ui
# Open: http://localhost:5000
```

---

## 🎓 Understanding the Pipeline

### Stage 1: Processing (`src/data/process.py`)
- Loads raw CSV
- Creates binary target (loan_status_binary: 1=no default, 0=default)
- Removes columns with >51% missing
- Converts date strings to datetime
- Engineers temporal features
- One-hot encodes categorical variables
- Outputs: `X_processed.csv` (32 features), `y.csv`

### Stage 2: Feature Engineering (`src/features/engineer.py`)
- Calculates WoE & IV for categorical features
- Analyzes feature correlations with target
- Drops highly correlated pairs (>0.8 correlation)
- Engineers new features (loan_amnt/installment ratio)
- Removes low-signal features
- Outputs: `X_engineered.csv` (~26 features)

### Stage 3: Feature Selection (`src/features/selection.py`)
- **Removes 6 leaky features** (outcome-dependent)
- Splits into train (80%) / test (20%)
- Standardizes with StandardScaler
- Trains Random Forest on clean data
- Selects features with importance > mean
- Outputs: `X_train/test_selected.npy` (6 features), scalers, selectors

### Stage 4: Training (`src/model/train.py`)
- Loads selected train features
- Trains 4 models: LR, RF, GB, NN
- Saves all models as .pkl files
- No evaluation, no plotting
- Ready for MLflow tracking
- Outputs: 4 trained models, model registry

### Stage 5: Evaluation (`src/model/evaluate.py`)
- Loads trained models + test data
- Calculates metrics: Accuracy, Precision, Recall, F1, F2, AUC
- Generates confusion matrices
- Creates classification reports
- Outputs: `model_performance.csv`, `classification_reports.json`

---

## 💾 DVC Workflow

```bash
# Initialize DVC (if not already done)
dvc init

# Run pipeline
dvc repro

# Check what changed
dvc status

# View pipeline graph
dvc dag

# Commit to Git
git add dvc.yaml dvc.lock
git commit -m "Add complete ML pipeline"
```

---

## 🧪 Testing Individual Stages

```bash
# Test just data processing
python src/data/process.py

# Check outputs exist
ls data/processed/

# Test feature engineering
python src/features/engineer.py

# Verify features were created
ls data/features/

# And so on...
```

---

## 📝 Notes for MLflow Integration

When ready to integrate MLflow:

1. Install MLflow:
   ```bash
   pip install mlflow
   ```

2. Modify `src/model/train.py`:
   ```python
   import mlflow
   
   def train_models(input_path=None, output_path=None, params_file="params.yaml"):
       # ... existing code ...
       
       with mlflow.start_run(run_name="credit_risk_v1"):
           mlflow.log_params(params)
           # Train models
           mlflow.sklearn.log_model(lr_model, "logistic_regression")
           mlflow.sklearn.log_model(rf_model, "random_forest")
           # etc.
   ```

3. Run tracking server:
   ```bash
   mlflow ui
   ```

4. Access dashboard at: `http://localhost:5000`

---

## ✅ Checklist

- [x] Separated processing, engineering, training, evaluation logic
- [x] All parameters in `params.yaml` (no hardcoding)
- [x] Training module contains **only training logic** (no evaluation/plotting)
- [x] DVC pipeline configuration (`dvc.yaml`)
- [x] Leakage prevention implemented before feature selection
- [x] Documentation & quick-start guide
- [x] Ready for MLflow integration
- [x] Ready for DVC orchestration

Ready to use! 🚀

---

**Created**: 2026-04-10  
**For**: Credit Risk Modeling MLOps Pipeline
