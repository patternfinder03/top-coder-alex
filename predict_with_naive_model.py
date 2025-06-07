import json
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path='public_cases.json'):
    """Load cases data from a JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    X = []
    y = None
    
    # The public 'training' data has an 'expected_output'
    if 'expected_output' in data[0]:
        y = []

    for case in data:
        # Handle different structures for public vs. private cases
        input_data = case.get('input', case)

        X.append([
            input_data['trip_duration_days'],
            input_data['miles_traveled'],
            input_data['total_receipts_amount']
        ])
        if y is not None:
            y.append(case['expected_output'])
    
    if y is not None:
        return np.array(X), np.array(y)
    else:
        return np.array(X)

def create_comprehensive_features(X):
    """Create comprehensive feature engineering"""
    days, miles, receipts = X[:, 0], X[:, 1], X[:, 2]
    
    # Ensure no division by zero for ratio features
    days[days == 0] = 1
    miles[miles == 0] = 1
    receipts[receipts == 0] = 1

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
    receipts_cents = receipts * 100
    cent_overflow_feature = (receipts_cents % 4096)
    features.append(cent_overflow_feature)
    feature_names.append('receipts_cents_mod_4096')

    # 4. Long-Trip, Low-Spend Penalty
    long_trip_low_spend = ((days > 10) & (receipts < 110)).astype(int)
    features.append(long_trip_low_spend)
    feature_names.append('long_trip_low_spend_penalty')
    
    return np.column_stack(features)

def main():
    # Load training data
    X_train, y_train = load_data('public_cases.json')
    
    # Create features for training data
    X_train_features = create_comprehensive_features(X_train)
    
    # Train the Random Forest model on the entire public dataset
    model = RandomForestRegressor(
        n_estimators=200, 
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_features, y_train)
    
    # Load private data (for prediction)
    X_private = load_data('private_cases.json')
    
    # Create features for private data
    X_private_features = create_comprehensive_features(X_private)
    
    # Make predictions on the private data
    private_predictions = model.predict(X_private_features)
    
    # Print each prediction, formatted to two decimal places
    for pred in private_predictions:
        print(f"{pred:.2f}")

if __name__ == "__main__":
    main() 