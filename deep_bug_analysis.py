import json
import numpy as np

def deep_analysis():
    """Deep analysis of the worst performing cases to identify exact bug patterns"""
    
    # Load data
    with open('naive_model_detailed.json', 'r') as f:
        results = json.load(f)
    
    # Focus on the worst 13 cases
    worst_13 = results[:13]
    
    print("=== DEEP BUG ANALYSIS ===")
    print("Analyzing the 13 worst cases to identify exact bug mechanisms...\n")
    
    # Group by over/under prediction
    over_predictions = [case for case in worst_13 if case['predicted'] > case['actual']]
    under_predictions = [case for case in worst_13 if case['predicted'] < case['actual']]
    
    print(f"Over-predictions (bugs that REDUCE reimbursement): {len(over_predictions)}")
    print(f"Under-predictions (bugs that INCREASE reimbursement): {len(under_predictions)}")
    
    print("\n=== OVER-PREDICTION ANALYSIS (Bugs that reduce reimbursement) ===")
    
    for i, case in enumerate(over_predictions):
        days, miles, receipts = case['days'], case['miles'], case['receipts']
        actual, pred = case['actual'], case['predicted']
        reduction = pred - actual
        reduction_pct = (reduction / pred) * 100
        
        print(f"\nCase {i+1}: {days}d {miles}mi ${receipts:.2f}")
        print(f"  Predicted: ${pred:.2f}")
        print(f"  Actual:    ${actual:.2f}")
        print(f"  Reduction: ${reduction:.2f} ({reduction_pct:.1f}% less than expected)")
        
        # Look for mathematical patterns
        # Check if it's close to 50% reduction (potential bit truncation)
        if 40 <= reduction_pct <= 60:
            print(f"  *** POTENTIAL BIT TRUNCATION: ~50% reduction suggests MSB loss ***")
        
        # Check for modular patterns (overflow)
        if miles > 1000:
            mile_mod_1024 = miles % 1024
            mile_mod_2048 = miles % 2048
            print(f"  Mile overflow analysis: {miles} % 1024 = {mile_mod_1024}, {miles} % 2048 = {mile_mod_2048}")
        
        if receipts > 1000:
            receipt_mod_1024 = int(receipts) % 1024
            receipt_mod_2048 = int(receipts) % 2048
            print(f"  Receipt overflow analysis: {int(receipts)} % 1024 = {receipt_mod_1024}, {int(receipts)} % 2048 = {receipt_mod_2048}")
        
        # Check for day-based patterns
        if days in [8, 16, 24]:
            print(f"  *** POTENTIAL DAY OVERFLOW: {days} days is 8/16/24 boundary ***")
        
        # Check ratios
        receipt_per_mile = receipts / miles if miles > 0 else 0
        print(f"  Ratios: ${receipt_per_mile:.2f} per mile")
        
    print("\n=== UNDER-PREDICTION ANALYSIS (Bugs that increase reimbursement) ===")
    
    for i, case in enumerate(under_predictions):
        days, miles, receipts = case['days'], case['miles'], case['receipts']
        actual, pred = case['actual'], case['predicted']
        increase = actual - pred
        increase_pct = (increase / pred) * 100
        
        print(f"\nCase {i+1}: {days}d {miles}mi ${receipts:.2f}")
        print(f"  Predicted: ${pred:.2f}")
        print(f"  Actual:    ${actual:.2f}")
        print(f"  Increase:  ${increase:.2f} ({increase_pct:.1f}% more than expected)")
        
        # Check for special conditions
        if miles < 100 and receipts < 200:
            print(f"  *** POTENTIAL MINIMUM REIMBURSEMENT BUG: Very low inputs but high output ***")
        
        if miles > 1000:
            print(f"  *** HIGH MILEAGE CASE: {miles} miles could trigger special calculation ***")
        
        if days >= 10:
            print(f"  *** LONG TRIP: {days} days might trigger different logic ***")
    
    print("\n=== PATTERN DETECTION ===")
    
    # Look for specific mathematical relationships
    print("\n1. Checking for 50% reduction pattern (bit truncation):")
    truncation_cases = []
    for case in over_predictions:
        reduction_pct = ((case['predicted'] - case['actual']) / case['predicted']) * 100
        if 40 <= reduction_pct <= 60:
            truncation_cases.append(case)
    
    print(f"   Found {len(truncation_cases)} cases with ~50% reduction")
    
    print("\n2. Checking for receipt amount patterns:")
    receipt_patterns = {}
    for case in worst_13:
        receipts = case['receipts']
        # Check for specific receipt ranges that might trigger bugs
        if 600 <= receipts < 700:
            receipt_patterns['600-700'] = receipt_patterns.get('600-700', 0) + 1
        elif 700 <= receipts < 800:
            receipt_patterns['700-800'] = receipt_patterns.get('700-800', 0) + 1
        elif 800 <= receipts < 1000:
            receipt_patterns['800-1000'] = receipt_patterns.get('800-1000', 0) + 1
        elif 1000 <= receipts < 1500:
            receipt_patterns['1000-1500'] = receipt_patterns.get('1000-1500', 0) + 1
        elif 1500 <= receipts:
            receipt_patterns['1500+'] = receipt_patterns.get('1500+', 0) + 1
    
    for range_name, count in receipt_patterns.items():
        print(f"   Receipt range ${range_name}: {count} cases")
    
    print("\n3. Checking for day count patterns:")
    day_patterns = {}
    for case in worst_13:
        days = case['days']
        day_patterns[days] = day_patterns.get(days, 0) + 1
    
    for day_count in sorted(day_patterns.keys()):
        print(f"   {day_count} days: {day_patterns[day_count]} cases")
    
    print("\n4. Checking for mile patterns:")
    mile_patterns = {}
    for case in worst_13:
        miles = case['miles']
        if miles < 200:
            mile_patterns['<200'] = mile_patterns.get('<200', 0) + 1
        elif 200 <= miles < 500:
            mile_patterns['200-500'] = mile_patterns.get('200-500', 0) + 1
        elif 500 <= miles < 1000:
            mile_patterns['500-1000'] = mile_patterns.get('500-1000', 0) + 1
        elif miles >= 1000:
            mile_patterns['1000+'] = mile_patterns.get('1000+', 0) + 1
    
    for range_name, count in mile_patterns.items():
        print(f"   Mile range {range_name}: {count} cases")
    
    print("\n=== HYPOTHESIS GENERATION ===")
    
    print("\nBased on the analysis, here are the most likely bug patterns:")
    
    print("\n1. RECEIPT AMOUNT BUG:")
    high_receipt_over = [case for case in over_predictions if case['receipts'] > 1000]
    print(f"   - {len(high_receipt_over)}/10 over-predictions have receipts >$1000")
    print(f"   - These cases get much LOWER reimbursement than expected")
    print(f"   - HYPOTHESIS: High receipt amounts trigger a bug that reduces reimbursement")
    print(f"   - Possible causes: Integer overflow, incorrect tier calculation, bit truncation")
    
    print("\n2. LOW INPUT/HIGH OUTPUT BUG:")
    low_input_under = [case for case in under_predictions if case['miles'] < 200 or case['receipts'] < 200]
    print(f"   - {len(low_input_under)}/3 under-predictions have very low inputs")
    print(f"   - These cases get much HIGHER reimbursement than expected")
    print(f"   - HYPOTHESIS: Very low inputs trigger minimum reimbursement logic or default values")
    
    print("\n3. SHORT TRIP BUG:")
    short_trips = [case for case in over_predictions if case['days'] <= 5]
    print(f"   - {len(short_trips)}/10 over-predictions are ≤5 days")
    print(f"   - HYPOTHESIS: Short trips with high receipts/miles trigger calculation errors")
    
    print("\nNext steps:")
    print("1. Focus on cases with receipts >$1000 that get reduced reimbursement")
    print("2. Examine the mathematical relationship between inputs and the 'reduction factor'")
    print("3. Look for bit-level patterns (powers of 2, modular arithmetic)")
    print("4. Test specific hypotheses about overflow and truncation")

if __name__ == "__main__":
    deep_analysis() 