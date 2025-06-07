#!/usr/bin/env python3
"""
Find the Worst Cases - Identify where current algorithm fails most spectacularly
Then hunt for patterns in those specific failures
"""

import json
import pandas as pd
import sys
import subprocess

def load_data():
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    return pd.DataFrame([{
        'days': case['input']['trip_duration_days'],
        'miles': case['input']['miles_traveled'], 
        'receipts': case['input']['total_receipts_amount'],
        'expected': case['expected_output']
    } for case in data])

def get_my_predictions(df):
    """Get predictions from my current algorithm"""
    predictions = []
    
    print("Getting predictions from current algorithm...")
    for _, row in df.iterrows():
        try:
            # Run my current algorithm
            result = subprocess.run([
                'python3', 'calculate_reimbursement.py',
                str(row['days']), str(row['miles']), str(row['receipts'])
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                prediction = float(result.stdout.strip())
                predictions.append(prediction)
            else:
                predictions.append(0.0)  # Default for errors
                
        except Exception as e:
            predictions.append(0.0)  # Default for errors
    
    return predictions

def find_worst_cases(df, predictions):
    """Find cases with the largest errors"""
    df['my_prediction'] = predictions
    df['error'] = abs(df['expected'] - df['my_prediction'])
    df['relative_error'] = df['error'] / df['expected']
    
    # Sort by absolute error (worst first)
    worst_cases = df.sort_values('error', ascending=False)
    
    print("🔥 WORST CASES (Highest Absolute Error)")
    print("Days | Miles | Receipts | Expected | My Pred | Error | Rel Error")
    print("-" * 75)
    
    for i, (_, row) in enumerate(worst_cases.head(20).iterrows()):
        print(f"{int(row['days']):4d} | {int(row['miles']):5d} | ${row['receipts']:8.2f} | ${row['expected']:7.2f} | ${row['my_prediction']:7.2f} | ${row['error']:6.2f} | {row['relative_error']:8.2%}")
    
    return worst_cases

def analyze_worst_case_patterns(worst_cases):
    """Look for patterns in the worst-performing cases"""
    print("\n🔍 PATTERN ANALYSIS OF WORST CASES")
    
    top_20_worst = worst_cases.head(20)
    
    # Analyze day patterns in worst cases
    print(f"\nDay distribution in worst 20 cases:")
    day_counts = top_20_worst['days'].value_counts().sort_index()
    for days, count in day_counts.items():
        print(f"  {days} days: {count} cases")
    
    # Look for 8-day multiples
    multiples_of_8 = top_20_worst[top_20_worst['days'] % 8 == 0]
    if len(multiples_of_8) > 0:
        print(f"\n⚡ MULTIPLES OF 8 in worst cases: {len(multiples_of_8)}")
        for _, row in multiples_of_8.iterrows():
            print(f"  {row['days']} days: Expected ${row['expected']:.2f}, Got ${row['my_prediction']:.2f}")
    
    # Analyze receipt patterns
    print(f"\nReceipt ranges in worst 20 cases:")
    receipt_ranges = {
        "$0-$100": len(top_20_worst[top_20_worst['receipts'] <= 100]),
        "$100-$500": len(top_20_worst[(top_20_worst['receipts'] > 100) & (top_20_worst['receipts'] <= 500)]),
        "$500-$1000": len(top_20_worst[(top_20_worst['receipts'] > 500) & (top_20_worst['receipts'] <= 1000)]),
        "$1000-$2000": len(top_20_worst[(top_20_worst['receipts'] > 1000) & (top_20_worst['receipts'] <= 2000)]),
        "$2000+": len(top_20_worst[top_20_worst['receipts'] > 2000]),
    }
    
    for range_name, count in receipt_ranges.items():
        if count > 0:
            print(f"  {range_name}: {count} cases")
    
    # Look for high receipt cases (potential reverse staircase)
    high_receipts = top_20_worst[top_20_worst['receipts'] > 2000]
    if len(high_receipts) > 0:
        print(f"\n⚡ HIGH RECEIPT CASES in worst 20: {len(high_receipts)}")
        for _, row in high_receipts.iterrows():
            print(f"  ${row['receipts']:.2f} receipts: Expected ${row['expected']:.2f}, Got ${row['my_prediction']:.2f}")
    
    # Look for around $700 cliff cases
    cliff_range = top_20_worst[(top_20_worst['receipts'] >= 650) & (top_20_worst['receipts'] <= 800)]
    if len(cliff_range) > 0:
        print(f"\n⚡ $700 CLIFF RANGE in worst 20: {len(cliff_range)}")
        for _, row in cliff_range.iterrows():
            receipt_cents = int(row['receipts'] * 100)
            tier = receipt_cents // 64
            print(f"  ${row['receipts']:.2f} (tier {tier}): Expected ${row['expected']:.2f}, Got ${row['my_prediction']:.2f}")

def examine_specific_bugs(worst_cases):
    """Look at specific bug patterns in the worst cases"""
    print("\n🐛 SPECIFIC BUG EXAMINATION")
    
    top_50_worst = worst_cases.head(50)
    
    # Bug 1: 8-day explosion
    eight_day_cases = top_50_worst[top_50_worst['days'] == 8]
    print(f"\n8-day cases in top 50 worst: {len(eight_day_cases)}")
    if len(eight_day_cases) > 0:
        print("These should show ~40% too much payment:")
        for _, row in eight_day_cases.iterrows():
            # Calculate what a "normal" 8-day should be (estimated)
            normal_estimate = row['days'] * 100 + row['miles'] * 0.5 + row['receipts'] * 0.8
            explosion_factor = row['expected'] / normal_estimate if normal_estimate > 0 else 0
            print(f"  Expected: ${row['expected']:.2f}, Normal est: ${normal_estimate:.2f}, Factor: {explosion_factor:.2f}x")
    
    # Bug 3: Reverse staircase (high receipts leading to lower reimbursements)
    very_high_receipts = top_50_worst[top_50_worst['receipts'] > 2400]
    print(f"\nVery high receipt cases (>$2400) in top 50 worst: {len(very_high_receipts)}")
    if len(very_high_receipts) > 0:
        print("These might show reverse staircase effect:")
        for _, row in very_high_receipts.head(10).iterrows():
            expected_cents = int(row['expected'] * 100)
            mod_4096 = expected_cents % 4096
            print(f"  ${row['receipts']:.2f} -> ${row['expected']:.2f} (mod 4096: {mod_4096})")

def main():
    print("🔥 WORST CASE HUNTER")
    print("Finding cases where current algorithm fails most spectacularly")
    print("=" * 60)
    
    df = load_data()
    print(f"Loaded {len(df)} test cases")
    
    # Get predictions from current algorithm
    predictions = get_my_predictions(df)
    
    # Find worst cases
    worst_cases = find_worst_cases(df, predictions)
    
    # Analyze patterns in worst cases
    analyze_worst_case_patterns(worst_cases)
    
    # Examine specific bug patterns
    examine_specific_bugs(worst_cases)
    
    print("\n" + "=" * 60)
    print("🎯 Worst case analysis complete!")
    print("Focus on implementing fixes for these specific failure patterns")

if __name__ == "__main__":
    main() 