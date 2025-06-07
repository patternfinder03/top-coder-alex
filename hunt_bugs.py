#!/usr/bin/env python3
"""
Bug Hunter: Systematic search for the four specific legacy bugs
1. Eight-Day Trip Explosion (3-bit overflow at 8, 16, 24...)
2. The "$700" Cliff (4-bit tier index boundary)
3. >$2400 Reverse Staircase (signed 12-bit overflow)
4. Weekday/Lunar "Luck" Bit (parity rounding)
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    return pd.DataFrame([{
        'days': case['input']['trip_duration_days'],
        'miles': case['input']['miles_traveled'], 
        'receipts': case['input']['total_receipts_amount'],
        'expected': case['expected_output']
    } for case in data])

def hunt_eight_day_explosion(df):
    """Hunt for Bug #1: Eight-Day Trip Explosion"""
    print("🔍 HUNTING BUG #1: Eight-Day Trip Explosion")
    print("Looking for 3-bit overflow causing 8, 16, 24-day trips to pay ~40% too much...")
    
    # Create synthetic grid as suggested in challenge
    days_range = range(1, 21)
    
    print("\nReimbursement per day analysis:")
    print("Days | Count | Avg Total | Per Day | Expected vs Reality")
    print("-" * 60)
    
    for days in days_range:
        subset = df[df['days'] == days]
        if len(subset) > 0:
            avg_total = subset['expected'].mean()
            per_day = avg_total / days
            
            # Check if this is a multiple of 8 (potential overflow)
            is_overflow = (days % 8 == 0) and (days > 0)
            overflow_marker = " ⚡ OVERFLOW!" if is_overflow else ""
            
            print(f"{days:4d} | {len(subset):5d} | ${avg_total:8.2f} | ${per_day:7.2f} |{overflow_marker}")
    
    # Specific analysis of 8-day vs 7-day and 9-day
    print("\n🎯 FOCUSED ANALYSIS: 7-day vs 8-day vs 9-day trips")
    for target_days in [7, 8, 9]:
        subset = df[df['days'] == target_days]
        if len(subset) > 0:
            avg_total = subset['expected'].mean()
            per_day = avg_total / target_days
            print(f"{target_days} days: ${avg_total:.2f} total, ${per_day:.2f} per day (n={len(subset)})")
    
    # Check if 8-day trips are anomalously high compared to neighbors
    days_7 = df[df['days'] == 7]['expected'].mean() / 7 if len(df[df['days'] == 7]) > 0 else 0
    days_8 = df[df['days'] == 8]['expected'].mean() / 8 if len(df[df['days'] == 8]) > 0 else 0  
    days_9 = df[df['days'] == 9]['expected'].mean() / 9 if len(df[df['days'] == 9]) > 0 else 0
    
    if days_8 > days_7 * 1.2 or days_8 > days_9 * 1.2:
        print("⚠️  ANOMALY DETECTED: 8-day trips have unusually high per-day rate!")
    else:
        print("❌ No clear 8-day explosion found in this dataset")

def hunt_700_cliff(df):
    """Hunt for Bug #2: The '$700' Cliff"""
    print("\n🔍 HUNTING BUG #2: The '$700' Cliff")  
    print("Looking for 4-bit tier index causing cliff around $716.80...")
    
    # Sort by receipts and look for sudden drops around $700
    receipt_sorted = df.sort_values('receipts')
    
    print("\nReceipt analysis around $700 boundary:")
    print("Receipts | Expected | Cents | Tier (>>6) | Rate")
    print("-" * 50)
    
    # Focus on the critical range around $700
    critical_range = df[(df['receipts'] >= 650) & (df['receipts'] <= 750)]
    critical_range = critical_range.sort_values('receipts')
    
    for _, row in critical_range.iterrows():
        receipt_cents = int(row['receipts'] * 100)
        tier_index = receipt_cents >> 6  # Right shift by 6 bits
        
        # Rough rate calculation (very approximate)
        base_estimate = row['days'] * 100 + row['miles'] * 0.5  # Rough base
        receipt_contribution = max(0, row['expected'] - base_estimate)
        rate = receipt_contribution / row['receipts'] if row['receipts'] > 0 else 0
        
        print(f"${row['receipts']:7.2f} | ${row['expected']:8.2f} | {receipt_cents:5d} | {tier_index:8d} | {rate:.3f}")
    
    # Look for the specific boundary at ~71680 cents where tier jumps by 0x80
    print(f"\n🎯 CHECKING CRITICAL BOUNDARY around $716.80 (71680 cents)")
    boundary_receipts = 716.80
    near_boundary = df[abs(df['receipts'] - boundary_receipts) < 20].sort_values('receipts')
    
    if len(near_boundary) > 0:
        print("Cases near the critical boundary:")
        for _, row in near_boundary.iterrows():
            receipt_cents = int(row['receipts'] * 100)
            tier_index = receipt_cents >> 6
            print(f"  ${row['receipts']:.2f} -> tier {tier_index} -> ${row['expected']:.2f}")

def hunt_reverse_staircase(df):
    """Hunt for Bug #3: >$2400 Reverse Staircase"""
    print("\n🔍 HUNTING BUG #3: >$2400 Reverse Staircase")
    print("Looking for signed 12-bit overflow causing reimbursements to decrease above $2407...")
    
    # Look at high reimbursement cases
    high_reimbursement = df[df['expected'] > 2000].sort_values('expected')
    
    print("\nHigh reimbursement analysis:")
    print("Expected | Cents | Mod 4096 | Signed12 | Abs(Signed12)")
    print("-" * 60)
    
    for _, row in high_reimbursement.iterrows():
        expected_cents = int(row['expected'] * 100)
        mod_4096 = expected_cents % 4096
        
        # Simulate signed 12-bit arithmetic (range: -2048 to +2047)
        if expected_cents > 2047:
            signed_12_bit = ((expected_cents - 2048) % 4096) - 2048
        else:
            signed_12_bit = expected_cents
            
        abs_signed = abs(signed_12_bit)
        
        print(f"${row['expected']:8.2f} | {expected_cents:5d} | {mod_4096:8d} | {signed_12_bit:9d} | {abs_signed:11d}")
    
    # Check for the telltale saw-tooth pattern
    print(f"\n🎯 CHECKING FOR SAW-TOOTH PATTERN (4096-cent period)")
    df['expected_cents'] = (df['expected'] * 100).astype(int)
    df['mod_4096'] = df['expected_cents'] % 4096
    
    # Plot distribution of mod 4096 values
    mod_counts = df['mod_4096'].value_counts()
    print(f"Reimbursements span {len(mod_counts)} different values mod 4096")
    
    # Look for cases where high receipts lead to lower reimbursements
    high_receipts = df[df['receipts'] > 2000].sort_values('receipts')
    if len(high_receipts) > 1:
        print("\nHigh receipt cases (looking for reverse relationship):")
        for _, row in high_receipts.iterrows():
            print(f"  ${row['receipts']:.2f} receipts -> ${row['expected']:.2f} reimbursement")

def hunt_luck_bit(df):
    """Hunt for Bug #4: Weekday/Lunar 'Luck' Bit"""
    print("\n🔍 HUNTING BUG #4: Weekday/Lunar 'Luck' Bit")  
    print("Looking for parity-based rounding causing pennies differences...")
    
    # Since we don't have submission dates, look for rounding patterns
    df['cents_part'] = ((df['expected'] * 100) % 1).round(2)
    
    print("\nRounding pattern analysis:")
    print("Most results should end in .00 (whole cents)")
    print("But some should show .01 differences due to luck bit")
    
    # Count fractional vs whole cent results
    whole_cents = (df['cents_part'] == 0).sum()
    fractional_cents = (df['cents_part'] != 0).sum()
    
    print(f"Whole cent results (.00): {whole_cents}")
    print(f"Fractional cent results: {fractional_cents}")
    
    if fractional_cents > 0:
        print("\nFractional cent breakdown:")
        frac_counts = df['cents_part'].value_counts().sort_index()
        for frac, count in frac_counts.items():
            if frac != 0:
                print(f"  .{frac:02.0f}: {count} cases")
    
    # Look for pairs of similar inputs with slightly different outputs
    print(f"\n🎯 LOOKING FOR LUCK BIT EVIDENCE")
    print("Searching for similar inputs with rounding differences...")
    
    # Group by similar characteristics and look for small variations
    df['rounded_receipts'] = (df['receipts'] / 10).round() * 10  # Round to nearest $10
    df['rounded_expected'] = (df['expected']).round(0)  # Round to nearest dollar
    
    similar_groups = df.groupby(['days', 'rounded_receipts']).filter(lambda x: len(x) > 1)
    
    if len(similar_groups) > 0:
        print("Found groups with similar inputs:")
        for (days, receipts), group in similar_groups.groupby(['days', 'rounded_receipts']):
            if len(group) > 1:
                expected_values = group['expected'].values
                if len(set(expected_values)) > 1:  # Different outputs for similar inputs
                    print(f"  {days} days, ~${receipts:.0f} receipts:")
                    for _, row in group.iterrows():
                        print(f"    {row['miles']} miles, ${row['receipts']:.2f} -> ${row['expected']:.2f}")

def main():
    print("🐛 BLACK-BOX BUG HUNTER")
    print("Systematically hunting for the four legacy algorithm bugs...")
    print("=" * 60)
    
    df = load_data()
    print(f"Loaded {len(df)} test cases for analysis\n")
    
    hunt_eight_day_explosion(df)
    hunt_700_cliff(df)  
    hunt_reverse_staircase(df)
    hunt_luck_bit(df)
    
    print("\n" + "=" * 60)
    print("🎯 Bug hunting complete! Look for patterns marked with ⚡ and ⚠️")

if __name__ == "__main__":
    main() 