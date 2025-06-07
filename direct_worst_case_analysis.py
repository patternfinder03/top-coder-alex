#!/usr/bin/env python3
"""
Direct analysis of worst cases - look at the specific problematic cases
"""

import json
import pandas as pd
from calculate_reimbursement import calculate_reimbursement

def load_data():
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    return pd.DataFrame([{
        'days': case['input']['trip_duration_days'],
        'miles': case['input']['miles_traveled'], 
        'receipts': case['input']['total_receipts_amount'],
        'expected': case['expected_output']
    } for case in data])

def analyze_worst_cases():
    """Find the worst performing cases and analyze them"""
    df = load_data()
    
    # Get my predictions directly
    df['my_prediction'] = df.apply(lambda row: calculate_reimbursement(row['days'], row['miles'], row['receipts']), axis=1)
    df['error'] = abs(df['expected'] - df['my_prediction'])
    df['relative_error'] = df['error'] / df['expected']
    
    # Sort by absolute error (worst first)
    worst_cases = df.sort_values('error', ascending=False)
    
    print("🔥 TOP 20 WORST CASES (Real Analysis)")
    print("Days | Miles | Receipts | Expected | My Pred | Error | Factor")
    print("-" * 70)
    
    for i, (_, row) in enumerate(worst_cases.head(20).iterrows()):
        factor = row['expected'] / row['my_prediction'] if row['my_prediction'] > 0 else float('inf')
        print(f"{int(row['days']):4d} | {int(row['miles']):5d} | ${row['receipts']:8.2f} | ${row['expected']:7.2f} | ${row['my_prediction']:7.2f} | ${row['error']:6.2f} | {factor:6.1f}x")
    
    return worst_cases

def analyze_patterns_in_worst(worst_cases):
    """Look for specific patterns in the worst cases"""
    print("\n🔍 PATTERN ANALYSIS")
    
    top_20 = worst_cases.head(20)
    
    # Look for 8-day multiples
    multiples_of_8 = top_20[top_20['days'] % 8 == 0]
    print(f"\nMultiples of 8 days in worst 20: {len(multiples_of_8)}")
    for _, row in multiples_of_8.iterrows():
        wrapped_days = row['days'] & 0b111  # 3-bit wrap
        print(f"  {row['days']} days (wraps to {wrapped_days}): Expected ${row['expected']:.2f}, Got ${row['my_prediction']:.2f}")
    
    # Look for $700 cliff cases
    cliff_cases = top_20[(top_20['receipts'] >= 650) & (top_20['receipts'] <= 800)]
    print(f"\n$700 cliff range cases in worst 20: {len(cliff_cases)}")
    for _, row in cliff_cases.iterrows():
        receipt_cents = int(row['receipts'] * 100)
        tier = receipt_cents // 64
        print(f"  ${row['receipts']:.2f} (tier {tier}): Expected ${row['expected']:.2f}, Got ${row['my_prediction']:.2f}")
    
    # Look for high receipt cases (reverse staircase)
    high_receipts = top_20[top_20['receipts'] > 2000]
    print(f"\nHigh receipt cases (>$2000) in worst 20: {len(high_receipts)}")
    for _, row in high_receipts.iterrows():
        expected_cents = int(row['expected'] * 100)
        mod_4096 = expected_cents % 4096
        print(f"  ${row['receipts']:.2f} -> ${row['expected']:.2f} (mod 4096: {mod_4096})")
    
    # Day distribution analysis
    print(f"\nDay distribution in worst 20:")
    day_counts = top_20['days'].value_counts().sort_index()
    for days, count in day_counts.items():
        avg_error = top_20[top_20['days'] == days]['error'].mean()
        print(f"  {days} days: {count} cases, avg error ${avg_error:.2f}")

def look_for_specific_bug_evidence(worst_cases):
    """Look for evidence of the specific bugs mentioned in the challenge"""
    print("\n🐛 SPECIFIC BUG EVIDENCE")
    
    all_cases = worst_cases  # Look at all cases for patterns
    
    # Evidence of 8-day explosion
    print("\n1. Eight-Day Trip Explosion Evidence:")
    eight_day_cases = all_cases[all_cases['days'] == 8]
    seven_day_cases = all_cases[all_cases['days'] == 7]
    nine_day_cases = all_cases[all_cases['days'] == 9]
    
    if len(eight_day_cases) > 0 and len(seven_day_cases) > 0:
        eight_avg = eight_day_cases['expected'].mean()
        seven_avg = seven_day_cases['expected'].mean() 
        eight_per_day = eight_avg / 8
        seven_per_day = seven_avg / 7
        
        print(f"  8-day average: ${eight_avg:.2f} (${eight_per_day:.2f}/day)")
        print(f"  7-day average: ${seven_avg:.2f} (${seven_per_day:.2f}/day)")
        
        if eight_per_day > seven_per_day * 1.3:  # More than 30% higher
            print("  ⚡ POSSIBLE 8-DAY EXPLOSION DETECTED!")
        else:
            print(f"  ❌ No 8-day explosion (ratio: {eight_per_day/seven_per_day:.2f})")
    
    # Evidence of reverse staircase
    print("\n3. Reverse Staircase Evidence:")
    high_receipt_cases = all_cases[all_cases['receipts'] > 2200].sort_values('receipts')
    
    if len(high_receipt_cases) > 5:
        print("  High receipt cases (sorted by receipt amount):")
        decreasing_count = 0
        total_pairs = 0
        
        prev_expected = None
        for _, row in high_receipt_cases.head(10).iterrows():
            trend = ""
            if prev_expected is not None:
                if row['expected'] < prev_expected:
                    trend = " ⬇️"
                    decreasing_count += 1
                else:
                    trend = " ⬆️"
                total_pairs += 1
            
            print(f"    ${row['receipts']:8.2f} -> ${row['expected']:8.2f}{trend}")
            prev_expected = row['expected']
        
        if total_pairs > 0:
            decrease_rate = decreasing_count / total_pairs
            print(f"  Decrease rate: {decrease_rate:.1%}")
            if decrease_rate > 0.4:
                print("  ⚡ REVERSE STAIRCASE DETECTED!")

def main():
    print("🔍 DIRECT WORST CASE ANALYSIS")
    print("=" * 50)
    
    worst_cases = analyze_worst_cases()
    analyze_patterns_in_worst(worst_cases)
    look_for_specific_bug_evidence(worst_cases)
    
    print("\n" + "=" * 50)
    print("🎯 Now we know exactly where the algorithm fails!")

if __name__ == "__main__":
    main() 