#!/usr/bin/env python3
"""
Generate JSON list of top 100 worst performing cases
"""

import json
import sys

def load_data():
    with open('public_cases.json', 'r') as f:
        return json.load(f)

def calculate_simple_prediction(days, miles, receipts):
    """Simple version of my algorithm to avoid type issues"""
    # Basic calculation matching my current algorithm structure
    days_wrapped = int(days) & 0b111  # 3-bit wrap
    bonus_flag = 1 if int(days) > 7 else 0
    
    base_per_diem = 100.0
    per_diem_total = days_wrapped * base_per_diem
    
    if bonus_flag:
        long_trip_bonus = int(days) * 40.0
        per_diem_total += long_trip_bonus
    
    # Mileage calculation
    if miles <= 100:
        mileage_reimbursement = miles * 0.58
    else:
        mileage_reimbursement = 100 * 0.58 + (miles - 100) * 0.45
    
    # Receipt calculation with tier
    receipt_cents = int(receipts * 100)
    tier_index = receipt_cents // 64
    
    # Simple rate table
    if tier_index <= 7:
        receipt_multiplier = 0.8 + tier_index * 0.025
    else:
        receipt_multiplier = 0.7  # Cliff drop
    
    receipt_reimbursement = receipts * receipt_multiplier
    
    # Core calculation
    core_total_cents = int((per_diem_total + mileage_reimbursement + receipt_reimbursement) * 100)
    
    # 12-bit overflow simulation
    if core_total_cents > 2047:
        wrapped_cents = ((core_total_cents - 2048) % 4096) - 2048
    else:
        wrapped_cents = core_total_cents
    
    final_cents = abs(wrapped_cents)
    final_amount = final_cents / 100.0
    
    # Luck bit simulation
    pseudo_epoch_day = (int(days) * 7 + int(miles) // 10 + int(receipts)) % 16
    luck_flag = (pseudo_epoch_day % 2) == 0
    
    if luck_flag:
        import math
        final_amount = math.ceil(final_amount * 100) / 100
    else:
        import math
        final_amount = math.floor(final_amount * 100) / 100
    
    return round(final_amount, 2)

def main():
    # Load all test cases
    cases = load_data()
    
    # Calculate errors for each case
    cases_with_errors = []
    
    for i, case in enumerate(cases):
        inp = case['input']
        expected = case['expected_output']
        
        days = inp['trip_duration_days']
        miles = inp['miles_traveled'] 
        receipts = inp['total_receipts_amount']
        
        try:
            my_prediction = calculate_simple_prediction(days, miles, receipts)
            error = abs(expected - my_prediction)
            
            cases_with_errors.append({
                'case_index': i,
                'input': inp,
                'expected_output': expected,
                'my_prediction': my_prediction,
                'absolute_error': error,
                'relative_error': error / expected if expected > 0 else 0,
                'error_factor': expected / my_prediction if my_prediction > 0 else float('inf')
            })
        except Exception as e:
            # Skip cases that cause errors
            continue
    
    # Sort by absolute error (worst first)
    cases_with_errors.sort(key=lambda x: x['absolute_error'], reverse=True)
    
    # Take top 100
    top_100_worst = cases_with_errors[:100]
    
    # Output as JSON
    print(json.dumps(top_100_worst, indent=2))

if __name__ == "__main__":
    main() 