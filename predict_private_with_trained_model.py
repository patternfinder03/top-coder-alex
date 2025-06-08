import json
import numpy as np
import joblib
import pandas as pd
import sys
from analysis_utils import tag_regime
import warnings
warnings.filterwarnings('ignore')

def is_prime(n):
    """Simple prime checker using trial division"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def load_private_data():
    """Load private cases data for prediction"""
    with open('private_cases.json', 'r') as f:
        data = json.load(f)
    
    X = []
    for case in data:
        # Handle different structures for private cases
        input_data = case.get('input', case)
        X.append([
            input_data['trip_duration_days'],
            input_data['miles_traveled'],
            input_data['total_receipts_amount']
        ])
    
    return np.array(X)

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

def estimate_regimes_for_private_data(X):
    """Estimate regimes for private data (since we don't have actual regimes)"""
    days, miles, receipts = X[:, 0], X[:, 1], X[:, 2]
    
    # Heuristic regime assignment based on simple_naive_model.py logic
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
    print("🧾 Generating Private Results with Trained Simple Naive Model", file=sys.stderr)
    print("============================================================", file=sys.stderr)
    
    # Load the trained model
    try:
        model = joblib.load('naive_model.joblib')
        print("✅ Loaded trained model from naive_model.joblib", file=sys.stderr)
    except FileNotFoundError:
        print("❌ Error: naive_model.joblib not found. Please run simple_naive_model.py first.", file=sys.stderr)
        return
    
    # Load private data
    try:
        X_private = load_private_data()
        print(f"📊 Loaded {len(X_private)} private test cases", file=sys.stderr)
    except FileNotFoundError:
        print("❌ Error: private_cases.json not found.", file=sys.stderr)
        return
    
    # Create features using the same engineering as training
    print("🔧 Creating features...", file=sys.stderr)
    X_features, feature_names = create_comprehensive_features(X_private)
    
    # Add regime one-hots (estimated for private data)
    print("🏷️ Estimating regimes...", file=sys.stderr)
    regime_assignments = estimate_regimes_for_private_data(X_private)
    regime_features = []
    for k in range(3):
        regime_features.append((regime_assignments == k).astype(int))
    
    if regime_features:
        regime_matrix = np.column_stack(regime_features)
        X_features = np.column_stack([X_features, regime_matrix])
        feature_names.extend([f'regime_{k}' for k in range(3)])
    
    print(f"✅ Created {X_features.shape[1]} features total", file=sys.stderr)
    
    # Make predictions (remember to apply inverse log transform)
    print("🚀 Generating predictions...", file=sys.stderr)
    predictions_log = model.predict(X_features)
    predictions = np.expm1(predictions_log)  # Inverse of log1p used during training
    
    print("📝 Outputting results...", file=sys.stderr)
    # Print each prediction, formatted to two decimal places
    for pred in predictions:
        print(f"{pred:.2f}")
    
    print(f"✅ Generated {len(predictions)} predictions", file=sys.stderr)

if __name__ == "__main__":
    main() 