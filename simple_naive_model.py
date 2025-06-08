import json
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
import warnings
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from analysis_utils import tag_regime, bucket_report

warnings.filterwarnings('ignore')

def load_data_df():
    """Load the public cases data into a pandas DataFrame."""
    with open('public_cases.json', 'r') as f:
        raw = json.load(f)
    
    rows = []
    for i, c in enumerate(raw):
        rows.append({
            'idx'      : c.get('case_index', i),
            'days'     : c['input']['trip_duration_days'],
            'miles'    : c['input']['miles_traveled'],
            'receipts' : c['input']['total_receipts_amount'],
            'actual'   : c['expected_output']
        })
    return pd.DataFrame(rows)

def load_data():
    """Load the public cases data"""
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    X = []
    y = []
    
    for case in data:
        X.append([
            case['input']['trip_duration_days'],
            case['input']['miles_traveled'],
            case['input']['total_receipts_amount']
        ])
        y.append(case['expected_output'])
    
    return np.array(X), np.array(y)

def create_comprehensive_features(X):
    """Create comprehensive feature engineering"""
    days, miles, receipts = X[:, 0], X[:, 1], X[:, 2]
    
    features = []
    feature_names = []
    
    # Base features
    features.extend([days, miles])
    feature_names.extend(['days', 'miles'])
    
    # Add high receipts-per-mile flag and masked receipts
    flag_hi_rpm = (receipts / np.maximum(miles, 1) > 20).astype(int)
    receipts_mask = np.where(flag_hi_rpm, 0.0, receipts)  # force tree to ignore receipts when flag=1
    features.extend([receipts, flag_hi_rpm, receipts_mask])
    feature_names.extend(['receipts', 'flag_hi_rpm', 'receipts_mask'])
    
    # B. Domain ratios (add early for constraint alignment)
    perdiem_ratio = receipts / np.maximum(80 * days, 1)
    mileage_ratio = receipts / np.maximum(0.655 * miles, 1)
    features.extend([perdiem_ratio, np.log1p(perdiem_ratio)])
    feature_names.extend(['perdiem_ratio', 'log_perdiem_ratio'])
    
    features.extend([mileage_ratio, np.log1p(mileage_ratio)])
    feature_names.extend(['mileage_ratio', 'log_mileage_ratio'])
    
    # Key transformations (keep only most important)
    features.extend([np.log(receipts + 1), np.sqrt(receipts)])
    feature_names.extend(['log_receipts', 'sqrt_receipts'])
    
    # Most important interactions
    features.extend([days * miles * receipts, days * receipts])
    feature_names.extend(['days_miles_receipts', 'days_receipts'])
    
    # Per-day ratios
    features.extend([receipts / np.maximum(days, 1), miles / np.maximum(days, 1)])
    feature_names.extend(['receipts_per_day', 'miles_per_day'])
    
    # Binned features as categorical
    day_bins = np.digitize(days, [1, 4, 8, 12, 16, 20, 24, 28]).astype(int)
    mile_bins = np.digitize(miles, [0, 200, 500, 800, 1000, 1200, 1500, 2000]).astype(int)
    receipt_bins = np.digitize(receipts, [0, 300, 500, 700, 1000, 1500, 2000, 2500]).astype(int)
    
    features.extend([day_bins, mile_bins, receipt_bins])
    feature_names.extend(['day_bins', 'mile_bins', 'receipt_bins'])
    
    # Receipts per day bucket for categorical use
    rpd = receipts / np.maximum(days, 1)
    rpd_bucket = np.digitize(rpd, [200, 500]).astype(int)
    features.append(rpd_bucket)
    feature_names.append('rpd_bucket')
    
    # Key price patterns
    r_cents = (receipts * 100).astype(int)
    cents_mod_100 = r_cents % 100
    price_ending_49_99 = np.isin(cents_mod_100, [49, 99]).astype(int)
    features.append(price_ending_49_99)
    feature_names.append('price_ending_49_99')
    
    # Cap analysis features
    cap_est = 80 * days + 0.5 * miles
    over_cap_amt = np.maximum(receipts - cap_est, 0)
    features.extend([cap_est, over_cap_amt])
    feature_names.extend(['cap_est', 'over_cap_amt'])
    
    # Key anomaly flags (keep only the most predictive)
    fly_and_uber = ((miles < 200) & (receipts > 10 * miles)).astype(int)
    cannonball = ((days <= 2) & (miles > 400)).astype(int)
    weekend_flight = ((days <= 2) & (miles > 600)).astype(int)
    
    features.extend([fly_and_uber, cannonball, weekend_flight])
    feature_names.extend(['fly_and_uber', 'cannonball', 'weekend_flight'])
    
    # Travel Impedance Index
    tii = 0.54*days + 0.33*(miles/100) + 0.13*np.log1p(receipts)
    features.append(tii)
    feature_names.append('travel_impedance_index')
    
    # Final Regime Flag
    per_diem_rate = 80.0
    days_safe = np.where(days == 0, 1, days)
    is_perdiem_regime = ((miles < 120) & (receipts > 4 * per_diem_rate * days_safe)).astype(int)
    features.append(is_perdiem_regime)
    feature_names.append('is_perdiem_regime')
    
    # Create feature array preserving integer types for categorical features
    feature_array = np.column_stack(features)
    
    # Convert specific categorical features to integers
    cat_feature_names = ['day_bins', 'mile_bins', 'receipt_bins', 'rpd_bucket', 
                         'flag_hi_rpm', 'price_ending_49_99', 'fly_and_uber', 
                         'cannonball', 'weekend_flight', 'is_perdiem_regime']
    
    for i, fname in enumerate(feature_names):
        if fname in cat_feature_names:
            feature_array[:, i] = feature_array[:, i].astype(int)
    
    return feature_array, feature_names

def main():
    print("Loading data...")
    df = load_data_df()
    df = tag_regime(df)
    
    X = df[['days', 'miles', 'receipts']].to_numpy()
    y = df['actual'].to_numpy()
    print(f"Loaded {len(df)} cases and tagged regimes.")

    print("\nCreating features...")
    X_features, feature_names = create_comprehensive_features(X)
    print(f"Created {X_features.shape[1]} features")
    
    # Calculate row weights
    flag_hi_rpm = (df['receipts'] / np.maximum(df['miles'], 1) > 20).astype(int)
    row_wt = np.where(flag_hi_rpm, 3,
                      np.where(df['days'] == 1, 2, 1))  # the 1-day cannonballs
    print(f"Applied row weights: {np.sum(row_wt == 3)} high-rpm rows (3x), {np.sum(row_wt == 2)} 1-day rows (2x)")

    # Add regime one-hots as categorical features
    regime_map = {'A-perdiem':0, 'B-mileage':1, 'C-mixed':2}
    reg_ints = df['regime'].map(regime_map).values
    regime_features = []
    regime_start_idx = X_features.shape[1]
    for k in range(3):
        regime_features.append(reg_ints == k)
    
    # Combine with existing features
    if regime_features:
        regime_matrix = np.column_stack(regime_features).astype(int)
        X_features = np.column_stack([X_features, regime_matrix])
        feature_names.extend([f'regime_{k}' for k in range(3)])
        print(f"Added 3 regime one-hot features, total: {X_features.shape[1]} features")

    # 5-fold cross-validation with optimized CatBoost and log-target
    print("\nPerforming 5-fold cross-validation with optimized CatBoost...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_maes = []
    all_predictions = np.zeros(len(y))
    
    # Skip categorical features for now to avoid type issues - the key improvement is the router flag
    cat_cols = []
    # cat_cols = [feature_names.index(n) for n in
    #             ['day_bins', 'mile_bins', 'receipt_bins',
    #              'rpd_bucket', 'flag_hi_rpm',
    #              'regime_0', 'regime_1', 'regime_2',
    #              'price_ending_49_99']]
    print(f"Categorical feature indices: {cat_cols} (disabled for now)")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_features)):
        print(f"Fold {fold + 1}/5:")
        
        X_train, X_val = X_features[train_idx], X_features[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # D. Log-target trick
        y_train_log = np.log1p(y_train)
        
        # Updated XGBoost settings
        model = XGBRegressor(
            n_estimators=3000,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            eval_metric="mae",
            tree_method="hist",
            sampling_method="gradient_based",
            random_state=42,
            early_stopping_rounds=50,
            verbosity=0
        )
        
        # Receipt-jitter data augmentation
        rng = np.random.RandomState(fold + 42)  # Different seed per fold
        X_train_aug = X_train.copy()
        receipts_col_idx = feature_names.index("receipts")
        X_train_aug[:, receipts_col_idx] *= 1 + rng.normal(0, 0.005, len(X_train))
        
        y_val_log = np.log1p(y_val)
        model.fit(X_train_aug, y_train_log, 
                  sample_weight=row_wt[train_idx],
                  eval_set=[(X_val, y_val_log)],
                  verbose=False)
        pred_log = model.predict(X_val)
        predictions = np.expm1(pred_log)
        all_predictions[val_idx] = predictions
        
        # Calculate fold MAE
        fold_mae = mean_absolute_error(y_val, predictions)
        cv_maes.append(fold_mae)
        print(f"  MAE: ${fold_mae:.2f}")
    
    # Calculate overall CV performance
    overall_mae = np.mean(cv_maes)
    mae_std = np.std(cv_maes)
    
    print(f"\n5-Fold Cross-Validation Results (Optimized XGBoost + Log-Target):")
    print(f"Individual fold MAEs: {[f'${mae:.2f}' for mae in cv_maes]}")
    print(f"Overall MAE: ${overall_mae:.2f} ± ${mae_std:.2f}")

    # --- Regime-based analysis ---
    print("\n--- Regime Analysis Report (Single Model) ---")
    report = bucket_report(df, all_predictions)
    print(report)

    # --- Visualizations ---
    print("\nGenerating analysis plots...")
    df['pred'] = all_predictions
    df['abs_err'] = np.abs(df['pred'] - df['actual'])
    
    # Error vs. receipts/day
    fig, ax = plt.subplots(figsize=(10, 6))
    regime_colors = df['regime'].map({'A-perdiem':'red', 'B-mileage':'blue', 'C-mixed':'grey'})
    receipts_per_day = df['receipts'] / df['days'].replace(0, 1)
    ax.scatter(receipts_per_day, df['abs_err'], c=regime_colors, alpha=.6)
    ax.axvline(80*4, color='k', ls='--', lw=1, label='Per-diem cap threshold ($320/day)')
    ax.set_title('Absolute Error vs. Receipts per Day by Regime')
    ax.set_xlabel('Receipts per Day ($)')
    ax.set_ylabel('Absolute Error ($)')
    ax.legend()
    plt.savefig('error_vs_receipts_per_day.png')
    print("Saved error_vs_receipts_per_day.png")
    plt.close(fig)

    # Boxplot of errors per regime
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='regime', y='abs_err', data=df, ax=ax, order=['A-perdiem', 'B-mileage', 'C-mixed'])
    ax.set_title('Error Distribution by Regime')
    ax.set_xlabel('Regime')
    ax.set_ylabel('Absolute Error ($)')
    plt.savefig('error_distribution_by_regime.png')
    print("Saved error_distribution_by_regime.png")
    plt.close(fig)

    # --- Per-Regime Cross-Validation ---
    print("\n--- Per-Regime Cross-Validation ---")
    for reg_name, sub_df in df.groupby('regime'):
        print(f"\n----- Regime: {reg_name} ({len(sub_df)} cases) -----")
        
        n_splits = 5 if len(sub_df) >= 10 else 2
        if len(sub_df) < 4:
            print("  Skipping CV for this regime, not enough samples.")
            continue
            
        kf_reg = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        regime_indices = sub_df.index.to_numpy()
        X_regime = X_features[regime_indices]
        y_regime = y[regime_indices]
        
        maes = []
        for train_idx, val_idx in kf_reg.split(X_regime):
            X_train, X_val = X_regime[train_idx], X_regime[val_idx]
            y_train, y_val = y_regime[train_idx], y_regime[val_idx]

            model = XGBRegressor(
                n_estimators=3000,
                max_depth=5,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                eval_metric="mae",
                tree_method="hist",
                sampling_method="gradient_based",
                random_state=42,
                early_stopping_rounds=50,
                verbosity=0
            )
            # Receipt-jitter data augmentation for per-regime CV
            rng = np.random.RandomState(hash((reg_name, tuple(train_idx))) % 2**32)
            X_train_aug = X_train.copy()
            receipts_col_idx = feature_names.index("receipts")
            X_train_aug[:, receipts_col_idx] *= 1 + rng.normal(0, 0.005, len(X_train))
            
            y_train_log = np.log1p(y_train)
            y_val_log = np.log1p(y_val)
            model.fit(X_train_aug, y_train_log,
                      sample_weight=row_wt[regime_indices][train_idx],
                      eval_set=[(X_val, y_val_log)],
                      verbose=False)
            pred_log = model.predict(X_val)
            predictions = np.expm1(pred_log)
            maes.append(mean_absolute_error(y_val, predictions))

        print(f"  CV MAE: ${np.mean(maes):.2f} ± ${np.std(maes):.2f}")

    # --- Blind Hold-out Test ---
    print("\n--- Blind Hold-out Set Analysis ---")
    errors = np.abs(all_predictions - y)
    blind_idx = np.argsort(errors)[-20:]
    train_idx = np.setdiff1d(np.arange(len(df)), blind_idx)

    # Train on non-blind data
    model = XGBRegressor(
        n_estimators=3000,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        eval_metric="mae",
        tree_method="hist",
        sampling_method="gradient_based",
        random_state=42,
        verbosity=0
    )
    model.fit(X_features[train_idx], np.log1p(y[train_idx]),
              sample_weight=row_wt[train_idx])

    # Evaluate on both sets
    train_preds = np.expm1(model.predict(X_features[train_idx]))
    blind_preds = np.expm1(model.predict(X_features[blind_idx]))
    train_mae = mean_absolute_error(y[train_idx], train_preds)
    blind_mae = mean_absolute_error(y[blind_idx], blind_preds)

    print(f"Model trained on {len(train_idx)} cases (excluding 20 worst-error cases).")
    print(f"  MAE on train set: ${train_mae:.2f}")
    print(f"  MAE on blind hold-out set (20 cases): ${blind_mae:.2f}")

    print("\n--- Final Model Training on All Data ---")
    # Train final model on all data for saving
    final_model = XGBRegressor(
        n_estimators=3000,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        eval_metric="mae",
        tree_method="hist",
        sampling_method="gradient_based",
        random_state=42,
        verbosity=0
    )
    # Receipt-jitter data augmentation for final model
    rng = np.random.RandomState(999)
    X_features_aug = X_features.copy()
    receipts_col_idx = feature_names.index("receipts")
    X_features_aug[:, receipts_col_idx] *= 1 + rng.normal(0, 0.005, len(X_features))
    
    final_model.fit(X_features_aug, np.log1p(y), sample_weight=row_wt)
    
    print("\nSaving trained model to naive_model.joblib...")
    joblib.dump(final_model, 'naive_model.joblib')
    print("Model saved.")

    # Feature importance analysis for all features
    print("\n--- All Feature Importances ---")
    importances = final_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("Rank | Feature Name                | Importance")
    print("-" * 50)
    for i, idx in enumerate(indices):
        print(f"{i+1:4d} | {feature_names[idx]:27s} | {importances[idx]:.6f}")

    # Save detailed results and worst cases analysis
    print("\n--- Saving Detailed Results ---")
    errors = np.abs(all_predictions - y)
    relative_errors = errors / y
    
    # Create detailed results
    results = []
    for i in range(len(df)):
        results.append({
            'case_index': i,
            'days': int(df.iloc[i]['days']),
            'miles': int(df.iloc[i]['miles']),
            'receipts': float(df.iloc[i]['receipts']),
            'actual': float(df.iloc[i]['actual']),
            'predicted': float(all_predictions[i]),
            'absolute_error': float(errors[i]),
            'relative_error': float(relative_errors[i]),
            'regime': df.iloc[i]['regime']
        })
    
    # Sort by absolute error (worst first)
    results.sort(key=lambda x: x['absolute_error'], reverse=True)
    
    with open('naive_model_detailed.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Saved detailed results to 'naive_model_detailed.json'")
    
    # Save worst 50 cases
    worst_50_results = results[:50]
    with open('out_of_sample_worst_cases.json', 'w') as f:
        json.dump(worst_50_results, f, indent=2)
    print("Saved 50 worst predictions to 'out_of_sample_worst_cases.json'")
    
    # Print worst 20 for quick review
    print(f"\nWorst 20 Predictions:")
    print("Rank | Days | Miles | Receipts | Actual  | Predicted | Error   | Regime")
    print("-" * 80)
    for rank, case in enumerate(worst_50_results[:20]):
        print(f"{rank+1:4d} | {case['days']:4d} | {case['miles']:5d} | ${case['receipts']:8.2f} | ${case['actual']:7.2f} | ${case['predicted']:9.2f} | ${case['absolute_error']:7.2f} | {case['regime']}")

if __name__ == "__main__":
    main() 