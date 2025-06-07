import numpy as np
import joblib
import json
import sys

# This is the same feature engineering function from the training script
def create_comprehensive_features(X):
    """Create comprehensive feature engineering for a batch of data"""
    if X.ndim == 1:
        X = X.reshape(1, -1)
        
    days, miles, receipts = X[:, 0].copy(), X[:, 1].copy(), X[:, 2].copy()
    
    # Ensure no division by zero for ratio features
    days[days == 0] = 1
    miles[miles == 0] = 1
    # Create a copy to avoid modifying the original array for receipts
    receipts_safe = receipts.copy()
    receipts_safe[receipts_safe == 0] = 1

    features = []
    
    # Base features
    features.extend([days, miles, receipts])
    
    # Polynomial features
    features.extend([days**2, miles**2, receipts**2, days**3, miles**3, receipts**3])
    
    # Two-way interactions
    features.extend([days * miles, days * receipts, miles * receipts])
    
    # Three-way interaction
    features.append(days * miles * receipts)
    
    # Log transformations
    features.extend([np.log(days + 1), np.log(miles + 1), np.log(receipts + 1)])
    
    # Square root transformations
    features.extend([np.sqrt(days), np.sqrt(miles), np.sqrt(receipts)])
    
    # Per-day ratios
    features.extend([miles / days, receipts_safe / days, (miles + receipts_safe) / days])
    
    # Cross ratios
    features.extend([receipts_safe / miles, miles / receipts_safe])
    
    # Binned features
    day_bins = np.digitize(days, [1, 4, 8, 12, 16, 20, 24, 28])
    mile_bins = np.digitize(miles, [0, 200, 500, 800, 1000, 1200, 1500, 2000])
    receipt_bins = np.digitize(receipts, [0, 300, 500, 700, 1000, 1500, 2000, 2500])
    features.extend([day_bins, mile_bins, receipt_bins])
    
    # Modular arithmetic
    mod_features = [
        days % 8, days % 16, days % 32,
        miles % 256, miles % 512, miles % 1024, miles % 2048, miles % 4096,
        receipts % 256, receipts % 512, receipts % 1024, receipts % 2048
    ]
    features.extend(mod_features)
    
    # Bit patterns
    features.extend([
        days.astype(int) & 1, days.astype(int) & 2, days.astype(int) & 4, days.astype(int) & 8, days.astype(int) & 16,
        miles.astype(int) & 1, miles.astype(int) & 2, miles.astype(int) & 4, miles.astype(int) & 8, miles.astype(int) & 16,
        receipts.astype(int) & 1, receipts.astype(int) & 2, receipts.astype(int) & 4, receipts.astype(int) & 8
    ])
    
    # High-value indicators
    features.extend([
        (days >= 8).astype(int), (days >= 16).astype(int), (days >= 24).astype(int),
        (miles >= 1000).astype(int), (miles >= 1500).astype(int),
        (receipts >= 700).astype(int), (receipts >= 1000).astype(int), (receipts >= 2000).astype(int)
    ])
    
    # --- Bug-specific features ---
    power_of_2_days = [(days == 2).astype(int), (days == 4).astype(int), (days == 8).astype(int)]
    features.extend(power_of_2_days)

    cliff_bins = np.digitize(receipts, [710, 720, 1020, 1030, 1530, 1540])
    features.append(cliff_bins)
    
    receipts_cents = receipts * 100
    cent_overflow_feature = (receipts_cents % 4096)
    features.append(cent_overflow_feature)

    long_trip_low_spend = ((days > 10) & (receipts < 110)).astype(int)
    features.append(long_trip_low_spend)
    
    return np.column_stack(features)

def main():
    print("🧾 Fast Naive Model Evaluation")
    print("===================================")
    
    # Load the pre-trained model
    try:
        model = joblib.load('naive_model.joblib')
        print("✅ Model naive_model.joblib loaded.")
    except FileNotFoundError:
        print("❌ Error: naive_model.joblib not found. Please run simple_naive_model.py to train and save the model first.", file=sys.stderr)
        sys.exit(1)

    # Load public cases
    try:
        with open('public_cases.json', 'r') as f:
            public_cases = json.load(f)
        print(f"📊 Loaded {len(public_cases)} public cases.")
    except FileNotFoundError:
        print("❌ Error: public_cases.json not found.", file=sys.stderr)
        sys.exit(1)
        
    # Prepare data for prediction
    inputs = np.array([[
        case['input']['trip_duration_days'],
        case['input']['miles_traveled'],
        case['input']['total_receipts_amount']
    ] for case in public_cases])
    
    expected_outputs = np.array([case['expected_output'] for case in public_cases])
    
    # Create features and make predictions for all cases at once
    print("🚀 Generating predictions for all cases...")
    features = create_comprehensive_features(inputs)
    predictions = model.predict(features)
    print("✅ Predictions generated.")
    
    # Calculate metrics
    errors = np.abs(predictions - expected_outputs)
    
    exact_matches = np.sum(errors < 0.01)
    close_matches = np.sum(errors < 1.0)
    avg_error = np.mean(errors)
    max_error = np.max(errors)
    num_cases = len(public_cases)

    # Calculate percentages
    exact_pct = (exact_matches / num_cases) * 100
    close_pct = (close_matches / num_cases) * 100
    
    # Calculate final score (same formula as eval.sh)
    score = (avg_error * 100) + (num_cases - exact_matches) * 0.1

    print("\n📈 Results Summary:")
    print(f"  Total test cases: {num_cases}")
    print(f"  Exact matches (±$0.01): {exact_matches} ({exact_pct:.1f}%)")
    print(f"  Close matches (±$1.00): {close_matches} ({close_pct:.1f}%)")
    print(f"  Average error: ${avg_error:.2f}")
    print(f"  Maximum error: ${max_error:.2f}")
    print("")
    print(f"🎯 Your Score: {score:.2f} (lower is better)")
    print("")

    # Display top 5 worst cases
    print("💡 Top 5 High-Error Cases:")
    worst_indices = np.argsort(errors)[::-1][:5]
    for i in worst_indices:
        case_inputs = inputs[i]
        print(f"    Case {i+1}: {int(case_inputs[0])} days, {int(case_inputs[1])} miles, ${case_inputs[2]:.2f} receipts")
        print(f"      Expected: ${expected_outputs[i]:.2f}, Got: ${predictions[i]:.2f}, Error: ${errors[i]:.2f}")

if __name__ == "__main__":
    main() 