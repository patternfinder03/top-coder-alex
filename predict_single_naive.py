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
    
    # --- New anomaly / cap flags ---------------------------------------
    r_cents = (receipts * 100).astype(int)

    rich_receipts_short_trip = ((receipts / days > 300) & (days <= 5)).astype(int)
    long_trip_low_spend2   = ((days >= 10) & (receipts < 15 * days)).astype(int)
    gas_only_drive         = ((miles > 800) & (receipts < 0.20 * miles)).astype(int)

    near_pow2_miles   = ((np.abs(miles - 512) < 64) |
                         (np.abs(miles - 1024) < 128) |
                         (np.abs(miles - 2048) < 256)).astype(int)

    near_pow2_receipts = ((np.abs(receipts - 1024) < 32) |
                          (np.abs(receipts - 2048) < 64) |
                          (np.abs(receipts - 4096) < 128)).astype(int)

    cents_mod_100 = r_cents % 100
    price_ending_49_99 = np.isin(cents_mod_100, [49, 99]).astype(int)

    nice_round_receipt   = ((r_cents % 5000 == 0) | (r_cents % 10000 == 0)).astype(int)

    weekend_flight       = ((days <= 2) & (miles > 600)).astype(int)
    urban_no_mileage     = ((miles < 100) & (receipts > 1000)).astype(int)
    prime_days_flag      = np.isin(days, [5, 7, 13]).astype(int)
    days_mod7_weird      = np.isin(days % 7, [0, 6]).astype(int)

    features.extend([
        rich_receipts_short_trip, long_trip_low_spend2, gas_only_drive,
        near_pow2_miles, near_pow2_receipts, price_ending_49_99,
        nice_round_receipt, weekend_flight, urban_no_mileage,
        prime_days_flag, days_mod7_weird
    ])

    # --- Residual-error patches (v2) -----------------------------------
    cap_est = 80 * days + 0.5 * miles
    over_cap      = (receipts > 1.5 * cap_est).astype(int)
    over_cap_amt  = np.maximum(receipts - cap_est, 0)
    under_cap_amt = np.maximum(cap_est - receipts, 0)

    fly_and_uber      = ((miles < 200) & (receipts > 10 * miles)).astype(int)
    marathon_couch    = ((days >= 10) & (miles < 250) & (receipts < 25 * days)).astype(int)
    cannonball        = ((days <= 2) & (miles > 400)).astype(int)
    receipts_vs_mileage = receipts - 0.5 * miles  # signed, helps on cluster D

    features.extend([
        cap_est, over_cap, over_cap_amt, under_cap_amt,
        fly_and_uber, marathon_couch, cannonball, receipts_vs_mileage
    ])

    # --- Final Regime Flag ---
    # Heuristic for the 'A-perdiem' regime, verified in analysis
    per_diem_rate = 80.0
    days_safe = np.where(days == 0, 1, days) # Avoid zero-day issues
    is_perdiem_regime = ((miles < 120) & (receipts > 4 * per_diem_rate * days_safe)).astype(int)
    features.append(is_perdiem_regime)

    # --- Pattern-memo add-ons --------------------------------

    # 1. log of receipts per mile
    rpm = receipts / np.maximum(miles, 1)      # avoid div-by-zero
    features.append(np.log1p(rpm))

    # 2. generic short-trip flag
    short_trip = (days < 5).astype(int)
    features.append(short_trip)

    # 3. receipts-per-day buckets
    rpd = receipts / np.maximum(days, 1)
    rpd_buckets = np.digitize(rpd, [200, 500])
    features.append(rpd_buckets)

    # 5. 1/days dampener
    features.append(1 / np.maximum(days, 1))

    # 6. Travel Impedance Index (weights from memo)
    tii = 0.54*days + 0.33*(miles/100) + 0.13*np.log1p(receipts)
    features.append(tii)

    # 4. regime one-hots (estimated for single prediction)
    regime_map = {'A-perdiem':0, 'B-mileage':1, 'C-mixed':2}
    # Estimate regime for single prediction
    d_safe = np.maximum(days, 1)
    per_diem = 80 * d_safe
    if (miles < 120) & (receipts > 4 * per_diem):
        regime = 0  # A-perdiem
    elif (receipts < 0.4 * miles) | ((days <= 2) & (miles > 400)):
        regime = 1  # B-mileage
    else:
        regime = 2  # C-mixed
    
    for k in range(3):
        features.append((regime == k).astype(int))

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
    
    # --- NEW PATTERN-BASED FEATURES ---
    
    # Feature 1: Odd-Day Offset Rule
    # This flag identifies A-perdiem trips with an odd number of days 
    # where miles is one more than a multiple of days.
    is_odd_day = (days % 2 != 0)
    is_perdiem_context = (days <= 7) & (miles < 100)  # Context to avoid firing on wrong trip types
    matches_mod_rule = (miles % np.maximum(days, 1) == 1)  # Use np.maximum to prevent division by zero
    flag_odd_day_offset = (is_odd_day & is_perdiem_context & matches_mod_rule).astype(int)
    features.append(flag_odd_day_offset)
    
    # Feature 2: Mirrored Days-in-Miles Prefix
    # This flag is 1 if the mileage number starts with the day number (e.g., d=11, m=1179).
    # This is most meaningful for multi-digit day counts.
    n_samples = len(days)
    flag_mirrored_prefix = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        # We only care about this pattern for multi-digit days
        if days[i] >= 10:
            day_str = str(int(days[i]))
            mile_str = str(int(miles[i]))
            if mile_str.startswith(day_str):
                flag_mirrored_prefix[i] = 1
    
    features.append(flag_mirrored_prefix)
    
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