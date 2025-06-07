import numpy as np
import joblib
import sys

# This is the same feature engineering function from the training script
def create_comprehensive_features(X):
    """Create comprehensive feature engineering for a single data point"""
    # Ensure X is a 2D array
    if X.ndim == 1:
        X = X.reshape(1, -1)
        
    days, miles, receipts = X[:, 0], X[:, 1], X[:, 2]
    
    # Ensure no division by zero for ratio features
    days[days == 0] = 1
    miles[miles == 0] = 1
    receipts[receipts == 0] = 1

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
    features.extend([miles / days, receipts / days, (miles + receipts) / days])
    
    # Cross ratios
    features.extend([receipts / miles, miles / receipts])
    
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
    if len(sys.argv) != 4:
        # This message goes to stderr, so it won't interfere with the output
        print("Usage: python predict_single_naive.py <days> <miles> <receipts>", file=sys.stderr)
        sys.exit(1)

    # Load the pre-trained model
    try:
        model = joblib.load('naive_model.joblib')
    except FileNotFoundError:
        print("Error: naive_model.joblib not found. Please run the training script first.", file=sys.stderr)
        sys.exit(1)

    # Parse inputs
    try:
        inputs = np.array([[float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])]])
    except ValueError:
        print(f"Error: Invalid number in input: {sys.argv[1:]}", file=sys.stderr)
        sys.exit(1)

    # Create features for the single input
    features = create_comprehensive_features(inputs)

    # Make a prediction
    prediction = model.predict(features)

    # Print the single result, formatted to two decimal places
    print(f"{prediction[0]:.2f}")

if __name__ == "__main__":
    main() 