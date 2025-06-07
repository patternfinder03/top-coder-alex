#!/usr/bin/env python3
"""
Focused Bug Hunt - Now that I see the patterns, let me hunt more systematically
"""

import json
import pandas as pd
import numpy as np

def load_data():
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    return pd.DataFrame([{
        'days': case['input']['trip_duration_days'],
        'miles': case['input']['miles_traveled'], 
        'receipts': case['input']['total_receipts_amount'],
        'expected': case['expected_output']
    } for case in data])

def hunt_receipt_cliff_precisely(df):
    """Look for the exact cliff pattern mentioned in the challenge"""
    print("🔍 PRECISE HUNT: Receipt Cliff Analysis")
    
    # The challenge mentions ~$716.80 causes a cliff due to (receipt_cents >> 6)
    # Let's examine the exact boundary where the high bit flips
    
    df['receipt_cents'] = (df['receipts'] * 100).round().astype(int)
    df['tier_index'] = df['receipt_cents'] // (2**6)  # Right shift by 6
    df['tier_high_bit'] = (df['tier_index'] // (2**7)) & 1  # Extract bit 7 (the 0x80 bit)
    
    # Group by tier and look for the cliff
    print("\nTier analysis:")
    print("Tier | High Bit | Count | Avg Expected | Avg per Receipt")
    print("-" * 60)
    
    tier_analysis = df.groupby(['tier_index', 'tier_high_bit']).agg({
        'expected': ['count', 'mean'],
        'receipts': 'mean'
    }).round(2)
    
    # Look specifically around the boundary where high bit flips
    boundary_tiers = df[(df['tier_index'] >= 1110) & (df['tier_index'] <= 1130)]
    
    print("\nCritical boundary analysis (tiers 1110-1130):")
    for _, row in boundary_tiers.sort_values('receipt_cents').iterrows():
        high_bit = (int(row['tier_index']) >> 7) & 1
        marker = " ⚡ HIGH BIT SET!" if high_bit else ""
        print(f"${row['receipts']:7.2f} | Tier {int(row['tier_index']):4d} | High bit: {high_bit} | Expected: ${row['expected']:7.2f}{marker}")

def hunt_reverse_staircase_precisely(df):
    """Look for the signed 12-bit overflow causing reverse staircase"""
    print("\n🔍 PRECISE HUNT: Reverse Staircase Analysis")
    
    # The challenge mentions reimbursements DECREASE as receipts grow above ~$2407
    high_receipts = df[df['receipts'] > 2200].copy()  # Focus on high receipt cases
    high_receipts = high_receipts.sort_values('receipts')
    
    print("\nHigh receipt cases - looking for decreasing reimbursement trend:")
    print("Receipts | Expected | Trend")
    print("-" * 40)
    
    prev_expected = None
    decrease_count = 0
    total_count = 0
    
    for _, row in high_receipts.iterrows():
        trend = ""
        if prev_expected is not None:
            if row['expected'] < prev_expected:
                trend = " ⬇️ DECREASE"
                decrease_count += 1
            else:
                trend = " ⬆️ INCREASE"
            total_count += 1
        
        print(f"${row['receipts']:8.2f} | ${row['expected']:8.2f} |{trend}")
        prev_expected = row['expected']
    
    if total_count > 0:
        decrease_rate = decrease_count / total_count
        print(f"\nDecrease rate: {decrease_count}/{total_count} = {decrease_rate:.2%}")
        if decrease_rate > 0.4:  # If >40% of consecutive pairs decrease
            print("⚡ REVERSE STAIRCASE DETECTED!")
    
    # Check for the 4096-cent periodicity mentioned in challenge
    print(f"\n🎯 Checking for 4096-cent periodicity...")
    df['expected_cents'] = (df['expected'] * 100).round().astype(int)
    df['mod_4096'] = df['expected_cents'] % 4096
    
    # Look for the saw-tooth pattern
    high_expected = df[df['expected'] > 2000]
    print(f"High reimbursement cases (>${2000}):")
    print("Expected | Cents | Mod 4096 | Wrapped 12-bit")
    print("-" * 50)
    
    for _, row in high_expected.sort_values('expected').head(20).iterrows():
        expected_cents = row['expected_cents']
        mod_4096 = row['mod_4096']
        
        # Simulate 12-bit signed arithmetic
        if expected_cents > 2047:
            wrapped_12bit = ((expected_cents - 2048) % 4096) - 2048
        else:
            wrapped_12bit = expected_cents
        
        print(f"${row['expected']:8.2f} | {expected_cents:6d} | {mod_4096:8d} | {wrapped_12bit:11d}")

def hunt_day_overflow_cleverly(df):
    """Look for day overflow in a different way - maybe it's not 8-day explosion but something else"""
    print("\n🔍 CLEVER HUNT: Day Overflow Patterns")
    
    # Maybe the overflow isn't causing higher payments, but different patterns
    # Let's look at the 3-bit pattern more carefully
    df['days_wrapped'] = df['days'] & 0b111  # 3-bit wrap
    df['long_trip_flag'] = (df['days'] > 7).astype(int)
    
    print("\nDay wrap analysis:")
    print("Original | Wrapped | Long Flag | Count | Avg Expected")
    print("-" * 55)
    
    for original_days in sorted(df['days'].unique()):
        wrapped = original_days & 0b111
        long_flag = 1 if original_days > 7 else 0
        subset = df[df['days'] == original_days]
        avg_expected = subset['expected'].mean()
        
        marker = " ⚡ WRAPPED!" if wrapped != original_days else ""
        
        print(f"{original_days:8d} | {wrapped:7d} | {long_flag:9d} | {len(subset):5d} | ${avg_expected:10.2f}{marker}")
    
    # Look for the pattern described: when days wraps to 0, but long-trip bonus also fires
    wrapped_zero_cases = df[(df['days_wrapped'] == 0) & (df['long_trip_flag'] == 1)]
    if len(wrapped_zero_cases) > 0:
        print(f"\n⚡ WRAPPED TO ZERO + LONG TRIP BONUS cases: {len(wrapped_zero_cases)}")
        print("These should show the '40% too much' explosion:")
        for _, row in wrapped_zero_cases.iterrows():
            print(f"  {row['days']} days -> ${row['expected']:.2f}")

def hunt_luck_bit_cleverly(df):
    """Hunt for the parity-based rounding more systematically"""
    print("\n🔍 CLEVER HUNT: Luck Bit Parity Analysis")
    
    # Since we don't have submission dates, the system must use input parameters
    # to create a pseudo-date for the parity calculation
    
    print("Testing different parity formulas...")
    
    # Try different combinations to create the "epoch_day" value
    formulas = [
        ("days * 7 + miles // 10 + receipts", lambda row: int(row['days'] * 7 + row['miles'] // 10 + row['receipts'])),
        ("days + miles + receipts", lambda row: int(row['days'] + row['miles'] + row['receipts'])),
        ("days * miles + receipts", lambda row: int(row['days'] * row['miles'] + row['receipts'])),
        ("hash-like: sum of inputs", lambda row: int(row['days'] + row['miles'] + row['receipts'] * 100)),
    ]
    
    for formula_name, formula_func in formulas:
        print(f"\nTesting formula: {formula_name}")
        
        df['pseudo_epoch'] = df.apply(formula_func, axis=1)
        df['epoch_mod_16'] = df['pseudo_epoch'] % 16
        df['parity'] = df['epoch_mod_16'] % 2
        df['is_fractional'] = ((df['expected'] * 100) % 1 != 0).astype(int)
        
        # Check correlation between parity and fractional cents
        parity_vs_fractional = df.groupby('parity')['is_fractional'].agg(['count', 'sum', 'mean'])
        
        print("Parity | Count | Fractional | Rate")
        print("-" * 35)
        for parity in [0, 1]:
            if parity in parity_vs_fractional.index:
                count = parity_vs_fractional.loc[parity, 'count']
                fractional = parity_vs_fractional.loc[parity, 'sum']
                rate = parity_vs_fractional.loc[parity, 'mean']
                print(f"{parity:6d} | {count:5d} | {fractional:10d} | {rate:.3f}")
        
        # Calculate how well this predicts fractional vs whole cents
        even_parity_fractional_rate = parity_vs_fractional.loc[0, 'mean'] if 0 in parity_vs_fractional.index else 0
        odd_parity_fractional_rate = parity_vs_fractional.loc[1, 'mean'] if 1 in parity_vs_fractional.index else 0
        
        if abs(even_parity_fractional_rate - odd_parity_fractional_rate) > 0.02:  # >2% difference
            print(f"⚡ LUCK BIT CANDIDATE! Difference: {abs(even_parity_fractional_rate - odd_parity_fractional_rate):.3f}")

def main():
    print("🔍 FOCUSED BUG HUNTER - Targeted Analysis")
    print("=" * 50)
    
    df = load_data()
    
    hunt_receipt_cliff_precisely(df)
    hunt_reverse_staircase_precisely(df)
    hunt_day_overflow_cleverly(df)
    hunt_luck_bit_cleverly(df)
    
    print("\n" + "=" * 50)
    print("🎯 Focused bug hunting complete!")

if __name__ == "__main__":
    main() 