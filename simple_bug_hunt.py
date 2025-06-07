#!/usr/bin/env python3
"""
Simple Bug Hunt - Clean implementation to find the four bugs
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

def analyze_day_patterns(df):
    """Analyze day-based patterns for the 3-bit overflow"""
    print("🔍 Day Pattern Analysis")
    print("Days | Count | Avg Expected | Per Day | Days % 8 | Long Trip")
    print("-" * 65)
    
    for days in sorted(df['days'].unique()):
        subset = df[df['days'] == days]
        avg_expected = subset['expected'].mean()
        per_day = avg_expected / days
        days_mod_8 = days % 8
        is_long_trip = "Yes" if days > 7 else "No"
        
        # Mark potential 3-bit overflow cases
        marker = " ⚡" if days % 8 == 0 and days > 0 else ""
        
        print(f"{days:4d} | {len(subset):5d} | ${avg_expected:11.2f} | ${per_day:7.2f} | {days_mod_8:7d} | {is_long_trip:9s}{marker}")

def analyze_receipt_cliff(df):
    """Look for the $700 cliff in receipt processing"""
    print("\n🔍 Receipt Cliff Analysis")
    
    # Focus on receipts around $700 boundary
    receipt_range = df[(df['receipts'] >= 650) & (df['receipts'] <= 800)].copy()
    receipt_range = receipt_range.sort_values('receipts')
    
    print("\nReceipts around $700:")
    print("Receipts | Expected | Cents | Tier Index | Cliff Check")
    print("-" * 60)
    
    for _, row in receipt_range.head(20).iterrows():
        receipt_cents = int(row['receipts'] * 100)
        tier_index = receipt_cents // 64  # Right shift by 6 bits
        
        # Check if we cross the 0x80 boundary (128 in decimal)
        crosses_boundary = tier_index >= 1120  # Around where $716.80 would be
        marker = " ⚡ CLIFF" if crosses_boundary else ""
        
        print(f"${row['receipts']:8.2f} | ${row['expected']:8.2f} | {receipt_cents:5d} | {tier_index:10d} |{marker}")

def analyze_reverse_staircase(df):
    """Look for decreasing reimbursements at high values"""
    print("\n🔍 Reverse Staircase Analysis")
    
    # Look at high receipt cases
    high_receipts = df[df['receipts'] > 2200].sort_values('receipts')
    
    print("\nHigh receipts (looking for reverse trend):")
    print("Receipts | Expected | Mod 4096 | Trend")
    print("-" * 45)
    
    prev_expected = None
    for _, row in high_receipts.head(20).iterrows():
        expected_cents = int(row['expected'] * 100)
        mod_4096 = expected_cents % 4096
        
        trend = ""
        if prev_expected is not None:
            if row['expected'] < prev_expected:
                trend = " ⬇️"
            else:
                trend = " ⬆️"
        
        print(f"${row['receipts']:8.2f} | ${row['expected']:8.2f} | {mod_4096:8d} |{trend}")
        prev_expected = row['expected']

def analyze_luck_bit(df):
    """Look for the parity-based rounding pattern"""
    print("\n🔍 Luck Bit Analysis")
    
    # Count fractional vs whole cent endings
    df['is_whole_cents'] = ((df['expected'] * 100) % 1 == 0)
    
    whole_count = df['is_whole_cents'].sum()
    fractional_count = len(df) - whole_count
    
    print(f"Whole cent results: {whole_count}")
    print(f"Fractional cent results: {fractional_count}")
    print(f"Fractional rate: {fractional_count/len(df):.3f}")
    
    # Try to identify parity pattern using input parameters
    print("\nTesting parity formulas:")
    
    # Simple test: use sum of all inputs modulo 16 for parity
    df['test_parity'] = (df['days'] + df['miles'] + (df['receipts'] * 100).astype(int)) % 16 % 2
    
    # Group by parity and check fractional rate
    parity_analysis = df.groupby('test_parity')['is_whole_cents'].agg(['count', 'mean'])
    
    print("Parity | Count | Whole Cent Rate")
    print("-" * 35)
    for parity in [0, 1]:
        if parity in parity_analysis.index:
            count = parity_analysis.loc[parity, 'count']
            whole_rate = parity_analysis.loc[parity, 'mean']
            print(f"{parity:6d} | {count:5d} | {whole_rate:.3f}")
    
    # Check if there's a significant difference
    if len(parity_analysis) == 2:
        diff = abs(parity_analysis.loc[0, 'mean'] - parity_analysis.loc[1, 'mean'])
        if diff > 0.05:  # >5% difference
            print(f"⚡ LUCK BIT DETECTED! Difference: {diff:.3f}")

def find_basic_formula(df):
    """Try to understand the basic structure before bugs"""
    print("\n🔍 Basic Formula Analysis")
    
    # Look at simple cases to understand base structure
    simple_cases = df[(df['days'] <= 3) & (df['miles'] <= 100) & (df['receipts'] <= 50)]
    
    print("\nSimple cases (low complexity):")
    print("Days | Miles | Receipts | Expected | Days*100 | Miles*0.6 | Est Total")
    print("-" * 75)
    
    for _, row in simple_cases.head(10).iterrows():
        days_contrib = row['days'] * 100
        miles_contrib = row['miles'] * 0.6
        est_total = days_contrib + miles_contrib + row['receipts']
        
        print(f"{row['days']:4d} | {row['miles']:5.0f} | ${row['receipts']:8.2f} | ${row['expected']:8.2f} | {days_contrib:8.0f} | {miles_contrib:9.1f} | ${est_total:8.2f}")

def main():
    print("🔍 SIMPLE BUG HUNTER")
    print("=" * 50)
    
    df = load_data()
    print(f"Analyzing {len(df)} test cases\n")
    
    analyze_day_patterns(df)
    analyze_receipt_cliff(df)
    analyze_reverse_staircase(df)
    analyze_luck_bit(df)
    find_basic_formula(df)
    
    print("\n" + "=" * 50)
    print("🎯 Analysis complete!")

if __name__ == "__main__":
    main() 