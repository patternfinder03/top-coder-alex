#!/usr/bin/env python3
"""
Pattern Analysis Script for Black Box Reimbursement System
Systematically analyze the test data to identify the four bugs:
1. Eight-Day Trip Explosion
2. The "$700" Cliff  
3. >$2400 Reverse Staircase
4. Weekday/Lunar "Luck" Bit
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

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

def analyze_eight_day_explosion(df):
    """Analyze Bug #1: Eight-Day Trip Explosion"""
    print("=== Bug #1: Eight-Day Trip Explosion Analysis ===")
    
    # Group by trip days and analyze average reimbursement per day
    day_analysis = df.groupby('days').agg({
        'expected': ['count', 'mean', 'std'],
        'miles': 'mean',
        'receipts': 'mean'
    }).round(2)
    
    # Calculate reimbursement per day to spot the explosion
    per_day_reimbursement = df.groupby('days')['expected'].mean() / df.groupby('days')['days'].first()
    
    print("Average reimbursement per day by trip length:")
    for days in sorted(df['days'].unique()):
        if days <= 20:  # Focus on relevant range
            subset = df[df['days'] == days]
            avg_reimbursement = subset['expected'].mean()
            per_day = avg_reimbursement / days
            print(f"  {days:2d} days: ${avg_reimbursement:7.2f} total, ${per_day:6.2f} per day (n={len(subset)})")
    
    # Look for spikes at 8, 16, 24...
    print("\nLooking for overflow spikes at 8, 16, 24...")
    multiples_of_8 = df[df['days'].isin([8, 16, 24])]
    if len(multiples_of_8) > 0:
        for days in [8, 16, 24]:
            subset = df[df['days'] == days]
            if len(subset) > 0:
                avg = subset['expected'].mean()
                per_day = avg / days
                print(f"  {days} days: ${avg:.2f} total, ${per_day:.2f} per day")

def analyze_receipt_cliff(df):
    """Analyze Bug #2: The "$700" Cliff"""
    print("\n=== Bug #2: The '$700' Cliff Analysis ===")
    
    # Focus on receipts around $700
    receipt_range = df[(df['receipts'] >= 600) & (df['receipts'] <= 800)].copy()
    receipt_range = receipt_range.sort_values('receipts')
    
    print("Receipt amounts near $700 and their reimbursements:")
    for _, row in receipt_range.head(20).iterrows():
        receipt_cents = int(row['receipts'] * 100)
        tier_index = receipt_cents >> 6
        print(f"  ${row['receipts']:7.2f} -> ${row['expected']:7.2f} (cents={receipt_cents}, tier={tier_index})")
    
    # Look for the exact boundary where tier jumps by 0x80
    print(f"\nLooking for tier boundary around ${716.80:.2f} (45,872 cents)...")
    boundary_cents = 45872  # Approximate boundary
    for cents in range(boundary_cents - 100, boundary_cents + 100, 10):
        tier = cents >> 6
        dollars = cents / 100
        subset = df[abs(df['receipts'] - dollars) < 0.5]
        if len(subset) > 0:
            avg_reimbursement = subset['expected'].mean()
            print(f"  ${dollars:7.2f} -> tier {tier:3d} -> avg reimbursement ${avg_reimbursement:.2f}")

def analyze_reverse_staircase(df):
    """Analyze Bug #3: >$2400 Reverse Staircase"""
    print("\n=== Bug #3: >$2400 Reverse Staircase Analysis ===")
    
    # Look for high-value reimbursements that decrease
    high_value = df[df['expected'] > 2000].copy()
    high_value = high_value.sort_values('expected')
    
    print("High-value reimbursements (looking for reverse staircase):")
    for _, row in high_value.head(15).iterrows():
        modulo_4096 = int(row['expected'] * 100) % 4096
        print(f"  ${row['expected']:7.2f} (mod 4096 = {modulo_4096})")
    
    # Check for 4096-cent period pattern
    print("\nChecking for 4096-cent periodicity...")
    df['reimbursement_cents'] = (df['expected'] * 100).astype(int)
    df['mod_4096'] = df['reimbursement_cents'] % 4096
    
    # Look for saw-tooth pattern
    mod_analysis = df.groupby('mod_4096')['expected'].count()
    print(f"Reimbursements distributed across mod 4096 range: {len(mod_analysis)} unique values")

def analyze_luck_bit(df):
    """Analyze Bug #4: Weekday/Lunar 'Luck' Bit"""
    print("\n=== Bug #4: Weekday/Lunar 'Luck' Bit Analysis ===")
    
    # Since we don't have submission dates, we'll need to reverse-engineer
    # the pattern from the data itself. Look for cases where similar inputs
    # give slightly different outputs due to rounding differences
    
    print("Looking for rounding patterns...")
    df['cents_fraction'] = (df['expected'] * 100) % 1
    
    # Count how many results end in .00 vs other fractions
    whole_cents = sum(df['cents_fraction'] == 0)
    fractional_cents = sum(df['cents_fraction'] != 0)
    
    print(f"Results ending in whole cents: {whole_cents}")
    print(f"Results with fractional cents: {fractional_cents}")
    
    # Look at the distribution of fractional parts
    print("\nFractional cent distribution:")
    for frac in sorted(df['cents_fraction'].unique()):
        count = sum(df['cents_fraction'] == frac)
        if count > 5:  # Only show common patterns
            print(f"  .{frac:02.0f}: {count} cases")

def main():
    """Main analysis function"""
    print("Loading test data...")
    df = load_test_data()
    print(f"Loaded {len(df)} test cases")
    print(f"Days range: {df['days'].min()} to {df['days'].max()}")
    print(f"Miles range: {df['miles'].min()} to {df['miles'].max()}")
    print(f"Receipts range: ${df['receipts'].min():.2f} to ${df['receipts'].max():.2f}")
    print(f"Expected output range: ${df['expected'].min():.2f} to ${df['expected'].max():.2f}")
    print()
    
    analyze_eight_day_explosion(df)
    analyze_receipt_cliff(df)
    analyze_reverse_staircase(df)
    analyze_luck_bit(df)

if __name__ == "__main__":
    main() 