"""
Data Processing Module
Handles data loading, cleaning, and initial feature engineering
"""
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split


def load_params(params_file="params.yaml"):
    """Load parameters from params.yaml"""
    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)
    return params


def load_data(csv_path):
    """Load raw data from CSV"""
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded data with shape: {df.shape}")
    return df


def remove_missing_columns(df, threshold):
    """Remove columns with missing percentage > threshold"""
    missing_df = pd.DataFrame(df.isna().sum()).reset_index()
    missing_df.columns = ['Column', 'mis_count']
    missing_df['Missing_Percentage'] = missing_df['mis_count'] / len(df) * 100
    
    cols_to_drop = missing_df[missing_df['Missing_Percentage'] > threshold]['Column'].tolist()
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    print(f"✓ Removed {len(cols_to_drop)} columns with >{threshold}% missing data")
    return df


def create_target_variable(df):
    """Create binary target variable from loan_status"""
    loan_status_mapping = {
        'Fully Paid': 1,
        'Current': 1,
        'In Grace Period': 1,
        'Late (16-30 days)': 0,
        'Late (31-120 days)': 0,
        'Charged Off': 0,
        'Default': 0
    }
    
    df['loan_status_binary'] = df['loan_status'].map(loan_status_mapping)
    df = df.drop('loan_status', axis=1)
    
    print(f"✓ Created target variable")
    print(f"  Class distribution:\n{df['loan_status_binary'].value_counts()}")
    return df


def process_categorical_and_dates(df):
    """Process categorical and date columns"""
    # Strip whitespace from object columns
    df = df.apply(lambda col: col.str.strip() if col.dtypes == 'object' else col)
    
    # Convert dates
    df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y', errors='coerce')
    df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], format='%b-%Y', errors='coerce')
    df['last_pymnt_d'] = pd.to_datetime(df['last_pymnt_d'], format='%b-%Y', errors='coerce')
    df['last_credit_pull_d'] = pd.to_datetime(df['last_credit_pull_d'], format='%b-%Y', errors='coerce')
    
    # Convert percentage columns to numeric
    df['int_rate%'] = pd.to_numeric(df['int_rate'].astype(str).str.strip('%'), errors='coerce')
    df['revol_util%'] = pd.to_numeric(df['revol_util'].astype(str).str.strip('%'), errors='coerce')
    
    # Drop unnecessary columns
    columns_to_drop = ['title', 'zip_code', 'pymnt_plan', 'emp_title', 'int_rate', 'revol_util', 'url']
    df = df.drop(columns=columns_to_drop, inplace=False, errors='ignore')
    
    # Process binary flags
    df['debt_settlement_flag'] = np.where(df['debt_settlement_flag'] == 'Y', 1, 0)
    df['term_36_months'] = np.where(df['term'] == '36 months', 1, 0)
    df = df.drop('term', axis=1)
    
    # Process employment length
    df['emp_length'] = df['emp_length'].fillna('')
    df['emp_length'] = pd.to_numeric(
        df['emp_length'].str.replace('<', '', regex=False).str[:2].str.strip(),
        errors='coerce'
    )
    
    # Process hardship flag
    df['hardship_flag'] = df['hardship_flag'].fillna('N')
    df['hardship_flag'] = np.where(df['hardship_flag'] == 'NaN', 'N', df['hardship_flag'])
    
    print(f"✓ Processed categorical and date features")
    return df


def engineer_date_features(df):
    """Engineer features from date columns"""
    today = pd.to_datetime("today")
    
    # Fill missing dates with today
    df['last_pymnt_d'] = df['last_pymnt_d'].fillna(today)
    df['last_credit_pull_d'] = df['last_credit_pull_d'].fillna(today)
    
    # Time differences
    df['loan_age'] = (today - df['issue_d']).dt.days
    df['credit_history_length'] = (df['issue_d'] - df['earliest_cr_line']).dt.days
    df['time_since_last_payment'] = (today - df['last_pymnt_d']).dt.days
    df['time_since_last_credit_pull'] = (today - df['last_credit_pull_d']).dt.days
    
    # Temporal components
    df['issue_year'] = df['issue_d'].dt.year
    df['issue_month'] = df['issue_d'].dt.month
    
    # Categorical flags
    df['recent_payment'] = (df['time_since_last_payment'] <= 30).astype(int)
    df['recent_credit_pull'] = (df['time_since_last_credit_pull'] <= 90).astype(int)
    
    # Drop original date columns
    df = df.drop(['issue_d', 'earliest_cr_line', 'last_pymnt_d', 'last_credit_pull_d'], axis=1)
    
    print(f"✓ Engineered date features")
    return df


def remove_time_related_features(df):
    """Remove time-derived features that can introduce leakage"""
    time_feature_cols = [
        'loan_age',
        'credit_history_length',
        'time_since_last_payment',
        'time_since_last_credit_pull',
        'issue_year',
        'issue_month',
        'recent_payment',
        'recent_credit_pull'
    ]

    existing_cols = [col for col in time_feature_cols if col in df.columns]
    df = df.drop(columns=existing_cols, errors='ignore')

    print(f"✓ Removed {len(existing_cols)} time-related features to prevent leakage")
    if existing_cols:
        print(f"  Removed features: {existing_cols}")

    return df


def prepare_numeric_features(df):
    """Prepare and impute numeric features"""
    from sklearn.impute import SimpleImputer
    
    # Drop sub_grade (redundant with grade)
    df = df.drop('sub_grade', axis=1, errors='ignore')
    
    # Drop columns with low IV from categorical analysis (to be done in feature engineering)
    low_iv_cols = {'application_type', 'initial_list_status', 'addr_state', 'purpose'}
    df = df.drop(low_iv_cols, axis=1, errors='ignore')
    
    # Map home_ownership categories
    home_ownership_mapping = {
        'ANY': 'other', 'MORTGAGE': 'other', 'RENT': 'rent', 
        'OWN': 'own', 'NONE': 'other'
    }
    if 'home_ownership' in df.columns:
        df['home_ownership'] = df['home_ownership'].map(home_ownership_mapping)
    
    # Separate numeric and categorical
    y = df['loan_status_binary'].copy()
    cat_cols = df.select_dtypes(include=['object']).columns
    num_cols = df.select_dtypes(exclude=['object']).drop('loan_status_binary', axis=1).columns
    
    # One-hot encode categorical features
    df_final = pd.concat([df[num_cols], pd.get_dummies(df[cat_cols], drop_first=True)], axis=1)
    
    # Drop FICO and ID columns
    external_scores = ['last_fico_range_high', 'last_fico_range_low', 'fico_range_low', 'fico_range_high']
    df_final = df_final.drop(columns=external_scores, errors='ignore')
    df_final = df_final.drop(columns=['id'], errors='ignore')
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(df_final)
    df_final = pd.DataFrame(X_imputed, columns=df_final.columns)
    
    print(f"✓ Prepared numeric features with shape: {df_final.shape}")
    return df_final, y


def process_data(params_file="params.yaml", output_path=None):
    """Main processing pipeline"""
    params = load_params(params_file)
    
    if output_path is None:
        output_path = Path(params['data']['processed_dir'])
    else:
        output_path = Path(output_path)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load and process
    df = load_data(params['data']['raw_path'])
    df = remove_missing_columns(df, params['processing']['missing_threshold'])
    df = create_target_variable(df)
    df = process_categorical_and_dates(df)
    df = engineer_date_features(df)
    df = remove_time_related_features(df)
    df_final, y = prepare_numeric_features(df)

    # Create reproducible train/test split artifacts from processed data
    X_train, X_test, y_train, y_test = train_test_split(
        df_final,
        y,
        test_size=params['processing']['test_size'],
        random_state=params['processing']['random_state'],
        stratify=y
    )
    
    # Save
    df_final.to_csv(output_path / "X_processed.csv", index=False)
    y.to_csv(output_path / "y.csv", index=False)

    split_path = output_path / "split"
    split_path.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(split_path / "X_train_processed.csv", index=False)
    X_test.to_csv(split_path / "X_test_processed.csv", index=False)
    y_train.to_csv(split_path / "y_train_processed.csv", index=False)
    y_test.to_csv(split_path / "y_test_processed.csv", index=False)
    
    print(f"\n✓ Processing complete!")
    print(f"  X shape: {df_final.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Train/Test split: {X_train.shape} / {X_test.shape}")
    print(f"  Saved to: {output_path}")
    
    return df_final, y


if __name__ == "__main__":
    process_data()
