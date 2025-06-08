#!/usr/bin/env python3

import json
import numpy as np
import joblib
import sys

# Copy the exact feature engineering from predict_private_with_trained_model.py
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
    
    return np.column_stack(features), feature_names

def estimate_regimes(X):
    """Estimate regimes for data"""
    days, miles, receipts = X[:, 0], X[:, 1], X[:, 2]
    
    regime_assignments = []
    for i in range(len(X)):
        d, m, r = days[i], miles[i], receipts[i]
        d_safe = max(d, 1)
        per_diem = 80 * d_safe
        
        if (m < 120) and (r > 4 * per_diem):
            regime = 0  # A-perdiem
        elif (r < 0.4 * m) or ((d <= 2) and (m > 400)):
            regime = 1  # B-mileage  
        else:
            regime = 2  # C-mixed
        
        regime_assignments.append(regime)
    
    return np.array(regime_assignments)

def main():
    print("🚀 Quick Model Evaluation (Python-based)")
    print("=" * 45)
    
    # Load model
    try:
        model = joblib.load('naive_model.joblib')
        print("✅ Model loaded")
    except FileNotFoundError:
        print("❌ Error: naive_model.joblib not found")
        return
    
    # Load test data
    try:
        with open('public_cases.json', 'r') as f:
            public_cases = json.load(f)
        print(f"📊 Loaded {len(public_cases)} test cases")
    except FileNotFoundError:
        print("❌ Error: public_cases.json not found")
        return
    
    # Prepare data
    inputs = np.array([[
        case['input']['trip_duration_days'],
        case['input']['miles_traveled'],
        case['input']['total_receipts_amount']
    ] for case in public_cases])
    
    expected_outputs = np.array([case['expected_output'] for case in public_cases])
    
    # Create features
    print("🔧 Creating features...")
    X_features, feature_names = create_comprehensive_features(inputs)
    
    # Add regime one-hots
    regime_assignments = estimate_regimes(inputs)
    regime_features = []
    for k in range(3):
        regime_features.append((regime_assignments == k).astype(int))
    
    regime_matrix = np.column_stack(regime_features)
    X_features = np.column_stack([X_features, regime_matrix])
    
    print(f"✅ Created {X_features.shape[1]} features")
    
    # Make predictions
    print("🎯 Making predictions...")
    predictions_log = model.predict(X_features)
    predictions = np.expm1(predictions_log)
    
    # Calculate metrics
    errors = np.abs(predictions - expected_outputs)
    exact_matches = np.sum(errors < 0.01)
    close_matches = np.sum(errors < 1.0)
    avg_error = np.mean(errors)
    max_error = np.max(errors)
    num_cases = len(public_cases)
    
    # Calculate percentages and score
    exact_pct = (exact_matches / num_cases) * 100
    close_pct = (close_matches / num_cases) * 100
    score = (avg_error * 100) + (num_cases - exact_matches) * 0.1
    
    print("\n📈 Results:")
    print(f"  Total cases: {num_cases}")
    print(f"  Exact matches (±$0.01): {exact_matches} ({exact_pct:.1f}%)")
    print(f"  Close matches (±$1.00): {close_matches} ({close_pct:.1f}%)")
    print(f"  Average error: ${avg_error:.2f}")
    print(f"  Maximum error: ${max_error:.2f}")
    print(f"  Score: {score:.2f} (lower is better)")
    
    # Show worst cases
    worst_indices = np.argsort(errors)[::-1][:5]
    print(f"\n💡 Top 5 High-Error Cases:")
    for i in worst_indices:
        case_inputs = inputs[i]
        print(f"  {int(case_inputs[0])} days, {int(case_inputs[1])} miles, ${case_inputs[2]:.2f}")
        print(f"    Expected: ${expected_outputs[i]:.2f}, Got: ${predictions[i]:.2f}, Error: ${errors[i]:.2f}")

if __name__ == "__main__":
    main() 