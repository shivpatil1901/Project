"""
Feature Engineering Module
Handles correlation analysis, WoE/IV calculation, and feature engineering
"""
import numpy as np
import pandas as pd
import yaml
from pathlib import Path


def load_params(params_file="params.yaml"):
    """Load parameters from params.yaml"""
    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)
    return params


def process_categorical_columns(df, categorical_cols, target_col):
    """Calculate WoE and IV for categorical columns"""
    woe_results = []
    iv_summary = []

    for col in categorical_cols:
        # Handle NaN values
        df[col] = df[col].fillna('NaN')
        
        # Group by the column
        stats = df.groupby(col).agg(
            event_count=(target_col, 'sum'),
            total_count=(target_col, 'count')
        ).reset_index()

        # Calculate metrics
        stats['non_event_count'] = stats['total_count'] - stats['event_count']
        total_events = stats['event_count'].sum()
        total_non_events = stats['non_event_count'].sum()

        stats['event_rate'] = stats['event_count'] / (total_events + 1e-6)
        stats['non_event_rate'] = stats['non_event_count'] / (total_non_events + 1e-6)

        # WoE and IV
        stats['woe'] = np.log((stats['non_event_rate'] + 1e-6) / (stats['event_rate'] + 1e-6))
        stats['iv'] = (stats['non_event_rate'] - stats['event_rate']) * stats['woe']
        
        total_iv = stats['iv'].sum()

        # Format for output
        stats['name'] = col
        stats.rename(columns={col: 'sub_name'}, inplace=True)
        
        woe_results.append(stats[['name', 'sub_name', 'event_count', 'total_count', 
                                   'non_event_count', 'event_rate', 'non_event_rate', 'woe', 'iv']])
        iv_summary.append({'Column': col, 'IV': total_iv})

    woe_table = pd.concat(woe_results, ignore_index=True) if woe_results else pd.DataFrame()
    iv_summary_df = pd.DataFrame(iv_summary).sort_values(by='IV', ascending=False)
    
    return woe_table, iv_summary_df


def correlation_analysis(X, y):
    """Analyze feature correlations with target"""
    correlations = X.corrwith(y).abs()
    correlation_df = correlations.sort_values(ascending=False).reset_index()
    correlation_df.columns = ['name', 'correlation']
    
    return correlation_df


def identify_high_corr_pairs(X, threshold=0.8):
    """Identify pairs of highly correlated features"""
    feature_corr = X.corr()
    
    high_corr_pairs = feature_corr.abs().stack().reset_index()
    high_corr_pairs = high_corr_pairs[high_corr_pairs['level_0'] != high_corr_pairs['level_1']]
    high_corr_pairs = high_corr_pairs[high_corr_pairs[0] > threshold]
    high_corr_pairs.columns = ['Feature1', 'Feature2', 'Correlation']
    
    return high_corr_pairs, feature_corr


def drop_low_corr_features(X, y, high_corr_pairs, correlation_df):
    """Drop one feature from each highly correlated pair based on target correlation"""
    features_to_drop = set()

    for _, row in high_corr_pairs.iterrows():
        feature1 = row['Feature1']
        feature2 = row['Feature2']
        
        corr1 = correlation_df[correlation_df['name'] == feature1]['correlation'].values[0]
        corr2 = correlation_df[correlation_df['name'] == feature2]['correlation'].values[0]
        
        if corr1 >= corr2:
            features_to_drop.add(feature2)
        else:
            features_to_drop.add(feature1)

    selected_features = [col for col in X.columns if col not in features_to_drop]
    X = X[selected_features]
    
    print(f"✓ Dropped {len(features_to_drop)} highly correlated features")
    return X, features_to_drop


def engineer_loan_amount_ratio(X):
    """Engineer loan amount to installment ratio"""
    if 'loan_amnt' in X.columns and 'installment' in X.columns:
        X['loan_amnt_div_instlmnt'] = X['loan_amnt'] / X['installment']
        print(f"✓ Created loan_amnt_div_instlmnt feature")
    
    return X


def drop_bulk_features(X, drop_list):
    """Drop specified list of features"""
    X = X.drop(columns=drop_list, errors='ignore')
    print(f"✓ Dropped {len(drop_list)} bulk features")
    return X


def feature_engineering(input_path=None, output_path=None, params_file="params.yaml"):
    """Main feature engineering pipeline"""
    params = load_params(params_file)
    
    if input_path is None:
        input_path = Path(params['data']['processed_dir'])
    else:
        input_path = Path(input_path)
    
    if output_path is None:
        output_path = Path(params['data']['features'])
    else:
        output_path = Path(output_path)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load processed data
    X = pd.read_csv(input_path / "X_processed.csv")
    y = pd.read_csv(input_path / "y.csv").iloc[:, 0]
    
    print(f"Input shape: {X.shape}")
    
    # Correlation analysis
    correlation_df = correlation_analysis(X, y)
    
    # Identify and drop high correlation pairs
    high_corr_pairs, feature_corr = identify_high_corr_pairs(X, params['feature_engineering']['correlation_threshold'])
    X, dropped_features = drop_low_corr_features(X, y, high_corr_pairs, correlation_df)
    
    # Feature engineering
    X = engineer_loan_amount_ratio(X)
    
    # Drop bulk features
    bulk_drop_list = {
        'funded_amnt_inv', 'total_bal_il', 'total_pymnt', 'num_rev_tl_bal_gt_0',
        'issue_year', 'num_sats', 'num_op_rev_tl', 'total_bc_limit',
        'total_il_high_credit_limit', 'credit_history_length', 'num_bc_tl',
        'out_prncp_inv', 'num_tl_30dpd', 'bc_util', 'tot_cur_bal',
        'num_actv_bc_tl', 'open_acc', 'funded_amnt', 'loan_amnt',
        'installment', 'collection_recovery_fee', 'total_pymnt_inv',
        'revol_bal', 'revol_util%', 'num_bc_sats'
    }
    X = drop_bulk_features(X, bulk_drop_list)
    
    # Save engineered features
    X.to_csv(output_path / "X_engineered.csv", index=False)
    y.to_csv(output_path / "y_engineered.csv", index=False)
    
    print(f"\n✓ Feature engineering complete!")
    print(f"  Output shape: {X.shape}")
    print(f"  Saved to: {output_path}")
    
    return X, y


if __name__ == "__main__":
    feature_engineering()
