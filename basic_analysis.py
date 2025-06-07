#!/usr/bin/env python3
"""
Basic Analysis to understand the fundamental reimbursement structure
"""

import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def load_test_data():
    """Load the public test cases"""
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame([
        {
            'days': case['input']['trip_duration_days'],
            'miles': case['input']['miles_traveled'], 
            'receipts': case['input']['total_receipts_amount'],
            'expected': case['expected_output']
        }
        for case in data
    ])
    return df

def basic_linear_analysis(df):
    """Try to understand the basic linear relationship"""
    print("=== Basic Linear Analysis ===")
    
    # Simple linear regression
    X = df[['days', 'miles', 'receipts']].values
    y = df['expected'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    print(f"Linear coefficients:")
    print(f"  Days coefficient: ${model.coef_[0]:.2f}")
    print(f"  Miles coefficient: ${model.coef_[1]:.2f}")
    print(f"  Receipts coefficient: ${model.coef_[2]:.2f}")
    print(f"  Intercept: ${model.intercept_:.2f}")
    
    # Calculate R² score
    score = model.score(X, y)
    print(f"  R² score: {score:.4f}")
    
    # Test a few predictions
    predictions = model.predict(X[:10])
    print(f"\nFirst 10 predictions vs actual:")
    for i in range(10):
        print(f"  Predicted: ${predictions[i]:.2f}, Actual: ${y[i]:.2f}, Error: ${predictions[i]-y[i]:.2f}")

def analyze_per_diem_pattern(df):
    """Analyze the per diem base calculation"""
    print("\n=== Per Diem Pattern Analysis ===")
    
    # Look at cases with minimal miles and receipts to isolate per diem
    minimal_cases = df[(df['miles'] <= 50) & (df['receipts'] <= 10)].copy()
    minimal_cases = minimal_cases.sort_values('days')
    
    print("Cases with minimal miles/receipts (isolating per diem):")
    for _, row in minimal_cases.head(15).iterrows():
        per_day = row['expected'] / row['days']
        print(f"  {row['days']} days, {row['miles']} miles, ${row['receipts']:.2f} receipts -> ${row['expected']:.2f} (${per_day:.2f}/day)")

def analyze_mileage_pattern(df):
    """Analyze mileage reimbursement pattern"""
    print("\n=== Mileage Pattern Analysis ===")
    
    # Look at single-day trips with minimal receipts to isolate mileage
    single_day = df[(df['days'] == 1) & (df['receipts'] <= 25)].copy()
    single_day = single_day.sort_values('miles')
    
    print("Single-day trips with low receipts (isolating mileage):")
    for _, row in single_day.head(15).iterrows():
        base_per_diem_estimate = 100  # From previous analysis
        mileage_contribution = row['expected'] - base_per_diem_estimate
        if row['miles'] > 0:
            per_mile = mileage_contribution / row['miles']
            print(f"  {row['miles']} miles, ${row['receipts']:.2f} receipts -> ${row['expected']:.2f} (est. ${per_mile:.3f}/mile)")

def analyze_receipt_pattern(df):
    """Analyze receipt reimbursement pattern"""
    print("\n=== Receipt Pattern Analysis ===")
    
    # Look at trips with minimal days/miles to isolate receipt impact
    minimal_travel = df[(df['days'] <= 2) & (df['miles'] <= 50)].copy()
    minimal_travel = minimal_travel.sort_values('receipts')
    
    print("Minimal travel cases (isolating receipt impact):")
    for _, row in minimal_travel.head(15).iterrows():
        receipt_ratio = row['expected'] / row['receipts'] if row['receipts'] > 0 else 0
        print(f"  ${row['receipts']:.2f} receipts -> ${row['expected']:.2f} (ratio: {receipt_ratio:.2f})")

def check_for_day_overflow(df):
    """Check for the 8-day overflow pattern more carefully"""
    print("\n=== Day Overflow Pattern Check ===")
    
    # Look at the pattern by days modulo 8
    df['days_mod_8'] = df['days'] % 8
    mod_analysis = df.groupby('days_mod_8').agg({
        'expected': ['count', 'mean'],
        'days': 'mean'
    })
    
    print("Analysis by days modulo 8:")
    for mod in range(8):
        subset = df[df['days_mod_8'] == mod]
        if len(subset) > 0:
            avg_expected = subset['expected'].mean()
            avg_days = subset['days'].mean()
            per_day = avg_expected / avg_days if avg_days > 0 else 0
            print(f"  Days mod 8 = {mod}: {len(subset)} cases, avg ${avg_expected:.2f} total, ${per_day:.2f}/day")

def estimate_base_formula(df):
    """Try to estimate the base formula before bugs"""
    print("\n=== Base Formula Estimation ===")
    
    # Try different base assumptions
    for per_diem in [90, 100, 110, 120]:
        for mileage_rate in [0.50, 0.55, 0.58, 0.60]:
            for receipt_rate in [0.8, 0.9, 1.0, 1.1]:
                
                predictions = []
                for _, row in df.iterrows():
                    pred = row['days'] * per_diem + row['miles'] * mileage_rate + row['receipts'] * receipt_rate
                    predictions.append(pred)
                
                # Calculate mean absolute error
                mae = np.mean(np.abs(np.array(predictions) - df['expected'].values))
                
                if mae < 200:  # Only print promising combinations
                    print(f"  Per diem ${per_diem}, mileage ${mileage_rate:.2f}, receipts {receipt_rate:.1f} -> MAE ${mae:.2f}")

def main():
    """Main analysis function"""
    df = load_test_data()
    
    basic_linear_analysis(df)
    analyze_per_diem_pattern(df)
    analyze_mileage_pattern(df)
    analyze_receipt_pattern(df)
    check_for_day_overflow(df)
    estimate_base_formula(df)

if __name__ == "__main__":
    main() 