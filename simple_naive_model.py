import json
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
import warnings
import joblib
warnings.filterwarnings('ignore')

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
    features.extend([days, miles, receipts])
    feature_names.extend(['days', 'miles', 'receipts'])
    
    # Polynomial features
    features.extend([days**2, miles**2, receipts**2, days**3, miles**3, receipts**3])
    feature_names.extend(['days_sq', 'miles_sq', 'receipts_sq', 'days_cube', 'miles_cube', 'receipts_cube'])
    
    # Two-way interactions
    features.extend([days * miles, days * receipts, miles * receipts])
    feature_names.extend(['days_miles', 'days_receipts', 'miles_receipts'])
    
    # Three-way interaction
    features.append(days * miles * receipts)
    feature_names.append('days_miles_receipts')
    
    # Log transformations
    features.extend([np.log(days + 1), np.log(miles + 1), np.log(receipts + 1)])
    feature_names.extend(['log_days', 'log_miles', 'log_receipts'])
    
    # Square root transformations
    features.extend([np.sqrt(days), np.sqrt(miles), np.sqrt(receipts)])
    feature_names.extend(['sqrt_days', 'sqrt_miles', 'sqrt_receipts'])
    
    # Per-day ratios
    features.extend([miles / days, receipts / days, (miles + receipts) / days])
    feature_names.extend(['miles_per_day', 'receipts_per_day', 'total_per_day'])
    
    # Cross ratios
    features.extend([receipts / miles, miles / receipts])
    feature_names.extend(['receipts_per_mile', 'miles_per_receipt'])
    
    # Binned features (tier indicators)
    day_bins = np.digitize(days, [1, 4, 8, 12, 16, 20, 24, 28])
    mile_bins = np.digitize(miles, [0, 200, 500, 800, 1000, 1200, 1500, 2000])
    receipt_bins = np.digitize(receipts, [0, 300, 500, 700, 1000, 1500, 2000, 2500])
    
    features.extend([day_bins, mile_bins, receipt_bins])
    feature_names.extend(['day_bins', 'mile_bins', 'receipt_bins'])
    
    # Modular arithmetic (overflow patterns)
    mod_features = [
        days % 8, days % 16, days % 32,
        miles % 256, miles % 512, miles % 1024, miles % 2048, miles % 4096,
        receipts % 256, receipts % 512, receipts % 1024, receipts % 2048
    ]
    features.extend(mod_features)
    feature_names.extend([
        'days_mod_8', 'days_mod_16', 'days_mod_32',
        'miles_mod_256', 'miles_mod_512', 'miles_mod_1024', 'miles_mod_2048', 'miles_mod_4096',
        'receipts_mod_256', 'receipts_mod_512', 'receipts_mod_1024', 'receipts_mod_2048'
    ])
    
    # Bit patterns (for potential bit-level bugs)
    features.extend([
        days.astype(int) & 1, days.astype(int) & 2, days.astype(int) & 4, days.astype(int) & 8, days.astype(int) & 16,
        miles.astype(int) & 1, miles.astype(int) & 2, miles.astype(int) & 4, miles.astype(int) & 8, miles.astype(int) & 16,
        receipts.astype(int) & 1, receipts.astype(int) & 2, receipts.astype(int) & 4, receipts.astype(int) & 8
    ])
    feature_names.extend([
        'days_bit_1', 'days_bit_2', 'days_bit_4', 'days_bit_8', 'days_bit_16',
        'miles_bit_1', 'miles_bit_2', 'miles_bit_4', 'miles_bit_8', 'miles_bit_16',
        'receipts_bit_1', 'receipts_bit_2', 'receipts_bit_4', 'receipts_bit_8'
    ])
    
    # High-value indicators
    features.extend([
        (days >= 8).astype(int), (days >= 16).astype(int), (days >= 24).astype(int),
        (miles >= 1000).astype(int), (miles >= 1500).astype(int),
        (receipts >= 700).astype(int), (receipts >= 1000).astype(int), (receipts >= 2000).astype(int)
    ])
    feature_names.extend([
        'days_ge_8', 'days_ge_16', 'days_ge_24',
        'miles_ge_1000', 'miles_ge_1500',
        'receipts_ge_700', 'receipts_ge_1000', 'receipts_ge_2000'
    ])
    
    # --- Bug-specific features ---
    
    # 1. Power-of-2 Day Overflows
    power_of_2_days = [(days == 2).astype(int), (days == 4).astype(int), (days == 8).astype(int)]
    features.extend(power_of_2_days)
    feature_names.extend(['is_day_2', 'is_day_4', 'is_day_8'])

    # 2. Receipt Cliffs
    cliff_bins = np.digitize(receipts, [710, 720, 1020, 1030, 1530, 1540])
    features.append(cliff_bins)
    feature_names.append('receipt_cliff_bins')
    
    # 3. 4096-Cent Accumulator Overflow
    # Convert receipts to cents for the modulo operation
    receipts_cents = receipts * 100
    cent_overflow_feature = (receipts_cents % 4096)
    features.append(cent_overflow_feature)
    feature_names.append('receipts_cents_mod_4096')

    # 4. Long-Trip, Low-Spend Penalty
    long_trip_low_spend = ((days > 10) & (receipts < 110)).astype(int)
    features.append(long_trip_low_spend)
    feature_names.append('long_trip_low_spend_penalty')
    
    return np.column_stack(features), feature_names

def main():
    print("Loading data...")
    X, y = load_data()
    print(f"Loaded {len(X)} cases")
    
    print("Creating features...")
    X_features, feature_names = create_comprehensive_features(X)
    print(f"Created {X_features.shape[1]} features")
    
    # 5-fold cross-validation
    print("\nPerforming 5-fold cross-validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_maes = []
    cv_r2s = []
    all_predictions = np.zeros(len(y))
    all_actual = np.zeros(len(y))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_features)):
        print(f"\nFold {fold + 1}/5:")
        
        X_train, X_val = X_features[train_idx], X_features[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=200, 
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Make predictions
        predictions = model.predict(X_val)
        
        # Store predictions for later analysis
        all_predictions[val_idx] = predictions
        all_actual[val_idx] = y_val
        
        # Evaluate fold performance
        mae = mean_absolute_error(y_val, predictions)
        r2 = r2_score(y_val, predictions)
        
        cv_maes.append(mae)
        cv_r2s.append(r2)
        
        print(f"  Train size: {len(X_train)}, Val size: {len(X_val)}")
        print(f"  MAE: ${mae:.2f}")
        print(f"  R²: {r2:.4f}")
    
    # Overall CV performance
    overall_mae = np.mean(cv_maes)
    overall_r2 = np.mean(cv_r2s)
    mae_std = np.std(cv_maes)
    r2_std = np.std(cv_r2s)
    
    print(f"\n5-Fold Cross-Validation Results:")
    print(f"MAE: ${overall_mae:.2f} ± ${mae_std:.2f}")
    print(f"R²: {overall_r2:.4f} ± {r2_std:.4f}")
    print(f"Individual fold MAEs: {[f'${mae:.2f}' for mae in cv_maes]}")
    print(f"Individual fold R²s: {[f'{r2:.4f}' for r2 in cv_r2s]}")
    
    # Train final model on all data for feature importance analysis
    print("\nTraining final model on all data for analysis...")
    final_model = RandomForestRegressor(
        n_estimators=200, 
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    final_model.fit(X_features, y)
    
    # Save the trained model to a file
    print("\nSaving trained model to naive_model.joblib...")
    joblib.dump(final_model, 'naive_model.joblib')
    print("Model saved.")

    # Use CV predictions for error analysis
    predictions = all_predictions
    mae = overall_mae
    r2 = overall_r2
    
    # Feature importance analysis
    importances = final_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print(f"\nTop 20 Most Important Features:")
    for i in range(min(20, len(feature_names))):
        idx = indices[i]
        print(f"{i+1:2d}. {feature_names[idx]:20s}: {importances[idx]:.4f}")
    
    # Error analysis
    errors = np.abs(predictions - y)
    relative_errors = errors / y
    
    print(f"\nError Analysis:")
    print(f"Mean Error: ${np.mean(errors):.2f}")
    print(f"Median Error: ${np.median(errors):.2f}")
    print(f"90th percentile Error: ${np.percentile(errors, 90):.2f}")
    print(f"95th percentile Error: ${np.percentile(errors, 95):.2f}")
    print(f"Max Error: ${np.max(errors):.2f}")
    print(f"Mean Relative Error: {np.mean(relative_errors):.1%}")
    
    # Find worst predictions
    worst_indices = np.argsort(errors)[::-1][:20]
    
    print(f"\nWorst 20 Predictions:")
    print("Rank | Days | Miles | Receipts | Actual  | Predicted | Error   | Rel Err")
    print("-" * 75)
    for rank, idx in enumerate(worst_indices):
        days, miles, receipts = X[idx]
        actual, pred = y[idx], predictions[idx]
        error = errors[idx]
        rel_err = relative_errors[idx]
        print(f"{rank+1:4d} | {days:4.0f} | {miles:5.0f} | ${receipts:8.2f} | ${actual:7.2f} | ${pred:9.2f} | ${error:7.2f} | {rel_err:6.1%}")
    
    # Save detailed results
    results = []
    for i in range(len(X)):
        days, miles, receipts = X[i]
        results.append({
            'case_index': i,
            'days': int(days),
            'miles': int(miles),
            'receipts': float(receipts),
            'actual': float(y[i]),
            'predicted': float(predictions[i]),
            'absolute_error': float(errors[i]),
            'relative_error': float(relative_errors[i]),
            'error_rank': int(np.where(worst_indices == i)[0][0]) + 1 if i in worst_indices else None
        })
    
    # Sort by absolute error (worst first)
    results.sort(key=lambda x: x['absolute_error'], reverse=True)
    
    with open('naive_model_detailed.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to 'naive_model_detailed.json'")

    # Save the worst 50 out-of-sample cases to a separate file
    worst_50_results = results[:50]
    with open('out_of_sample_worst_cases.json', 'w') as f:
        json.dump(worst_50_results, f, indent=2)
    
    print(f"Saved the 50 worst out-of-sample predictions to 'out_of_sample_worst_cases.json'")
    
    # Analyze patterns in worst cases
    print(f"\nAnalyzing patterns in worst 50 cases:")
    worst_50 = results[:50]
    
    high_mileage = sum(1 for case in worst_50 if case['miles'] >= 1000)
    high_receipts = sum(1 for case in worst_50 if case['receipts'] >= 1000)
    long_trips = sum(1 for case in worst_50 if case['days'] >= 10)
    
    print(f"High mileage (≥1000): {high_mileage}/50 ({high_mileage*2}%)")
    print(f"High receipts (≥$1000): {high_receipts}/50 ({high_receipts*2}%)")
    print(f"Long trips (≥10 days): {long_trips}/50 ({long_trips*2}%)")
    
    # Day distribution in worst cases
    day_counts = {}
    for case in worst_50:
        days = case['days']
        day_counts[days] = day_counts.get(days, 0) + 1
    
    print(f"\nDay distribution in worst 50 cases:")
    for days in sorted(day_counts.keys()):
        print(f"{days:2d} days: {day_counts[days]:2d} cases")

if __name__ == "__main__":
    main() 