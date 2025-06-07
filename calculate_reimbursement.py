#!/usr/bin/env python3
"""
Black Box Reimbursement System Reverse Engineering
Reproduces the legacy system including all known bugs:
1. Eight-Day Trip Explosion (3-bit day overflow)
2. The "$700" Cliff (4-bit tier index)
3. >$2400 Reverse Staircase (signed 12-bit overflow)
4. Weekday/Lunar "Luck" Bit (parity rounding)
"""

import sys
import math
from datetime import datetime, timedelta

def calculate_reimbursement(trip_days, miles, receipts_amount):
    """
    Replicates the legacy reimbursement calculation with all bugs intact
    """
    
    # Power of 2 day solution - subtract by 1 if day is power of 2
    if trip_days > 0 and (trip_days & (trip_days - 1)) == 0:
        trip_days = trip_days - 1
    
    # Bug #1: Eight-Day Trip Explosion
    # Trip length stored in 3-bit field (max = 7), wraps to 0 at 8
    days_wrapped = trip_days & 0b111  # Keep only lower 3 bits
    bonus_flag = 1 if trip_days > 7 else 0  # Long-trip bonus routine
    
    # Base per diem calculation
    base_per_diem = 100.0  # $100 per day base rate
    per_diem_total = days_wrapped * base_per_diem
    
    # Add long-trip bonus when days > 7 (the "explosion")
    if bonus_flag:
        # The bonus appears to be significant - around 40% extra as mentioned
        long_trip_bonus = trip_days * 40.0  # Rough estimate to start
        per_diem_total += long_trip_bonus
    
    # Mileage calculation with tiered rates
    mileage_reimbursement = 0.0
    if miles <= 100:
        mileage_reimbursement = miles * 0.58  # Full rate for first 100 miles
    else:
        mileage_reimbursement = 100 * 0.58  # First 100 miles at full rate
        remaining_miles = miles - 100
        # Diminishing returns curve for additional miles
        mileage_reimbursement += remaining_miles * 0.45  # Reduced rate
    
    # Bug #2: The "$700" Cliff
    # Receipt amount funneled into 4-bit tier index via right shift
    receipt_cents = int(receipts_amount * 100)
    tier_index = (receipt_cents >> 6) & 0xF  # 4-bit tier index
    
    # Rate table based on tier (these need to be calibrated)
    rate_table = [
        0.8,   # Tier 0
        0.85,  # Tier 1
        0.9,   # Tier 2
        0.92,  # Tier 3
        0.95,  # Tier 4
        0.97,  # Tier 5
        0.98,  # Tier 6
        1.0,   # Tier 7
        0.7,   # Tier 8 - cliff drop
        0.75,  # Tier 9
        0.8,   # Tier 10
        0.82,  # Tier 11
        0.85,  # Tier 12
        0.87,  # Tier 13
        0.9,   # Tier 14
        0.92   # Tier 15
    ]
    
    receipt_multiplier = rate_table[tier_index]
    receipt_reimbursement = receipts_amount * receipt_multiplier
    
    # Core calculation in cents
    core_total_cents = int((per_diem_total + mileage_reimbursement + receipt_reimbursement) * 100)
    
    # Bug #3: >$2400 Reverse Staircase
    # Accumulate in signed 12-bit register (range ±2047 cents)
    # Wrap to 12-bit signed integer
    if core_total_cents > 2047:
        wrapped_cents = ((core_total_cents - 2048) % 4096) - 2048
    elif core_total_cents < -2048:
        wrapped_cents = ((core_total_cents + 2048) % 4096) - 2048
    else:
        wrapped_cents = core_total_cents
    
    # Sloppy abs() patch
    final_cents = abs(wrapped_cents)
    
    # Convert back to dollars
    final_amount = final_cents / 100.0
    
    # Bug #4: Weekday/Lunar "Luck" Bit
    # Use a pseudo-date based on input parameters for reproducibility
    # since we don't have actual submission dates
    pseudo_epoch_day = (trip_days * 7 + miles // 10 + int(receipts_amount)) % 16
    luck_flag = (pseudo_epoch_day % 2) == 0
    
    if luck_flag:
        final_amount = math.ceil(final_amount * 100) / 100  # Round up
    else:
        final_amount = math.floor(final_amount * 100) / 100  # Round down
    
    return round(final_amount, 2)

def main():
    if len(sys.argv) != 4:
        print("Usage: python3 calculate_reimbursement.py <trip_days> <miles> <receipts>", file=sys.stderr)
        sys.exit(1)
    
    try:
        trip_days = int(sys.argv[1])
        miles = int(sys.argv[2])
        receipts = float(sys.argv[3])
        
        result = calculate_reimbursement(trip_days, miles, receipts)
        print(f"{result:.2f}")
        
    except ValueError as e:
        print(f"Error: Invalid input - {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 