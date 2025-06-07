import json
import numpy as np

def analyze_bug_patterns():
    """Analyze the naive model results to identify specific bug patterns"""
    
    # Load naive model results
    with open('naive_model_detailed.json', 'r') as f:
        results = json.load(f)
    
    print("=== BUG PATTERN ANALYSIS ===")
    print("Analyzing cases where naive model fails most dramatically...")
    print("(These are likely cases where bugs create unexpected reimbursement values)")
    
    # Focus on worst cases where model significantly over or under-predicts
    worst_cases = []
    
    for case in results:
        # Look for cases with large absolute errors
        if case['absolute_error'] > 200:
            worst_cases.append(case)
    
    print(f"\nFound {len(worst_cases)} cases with >$200 error")
    
    # Analyze patterns
    print("\n=== PATTERN 1: Cases where naive model OVER-predicts (predicts too high) ===")
    over_predict = [case for case in worst_cases if case['predicted'] > case['actual']]
    print(f"Found {len(over_predict)} cases where naive model over-predicts")
    
    print("\nTop 10 over-predictions:")
    over_predict.sort(key=lambda x: x['predicted'] - x['actual'], reverse=True)
    for i, case in enumerate(over_predict[:10]):
        days, miles, receipts = case['days'], case['miles'], case['receipts']
        actual, pred = case['actual'], case['predicted']
        error = pred - actual
        print(f"{i+1:2d}. {days:2d}d {miles:4d}mi ${receipts:7.2f} → Actual:${actual:7.2f} Pred:${pred:7.2f} Over:${error:6.2f}")
    
    print("\n=== PATTERN 2: Cases where naive model UNDER-predicts (predicts too low) ===")
    under_predict = [case for case in worst_cases if case['predicted'] < case['actual']]
    print(f"Found {len(under_predict)} cases where naive model under-predicts")
    
    print("\nTop 10 under-predictions:")
    under_predict.sort(key=lambda x: x['actual'] - x['predicted'], reverse=True)
    for i, case in enumerate(under_predict[:10]):
        days, miles, receipts = case['days'], case['miles'], case['receipts']
        actual, pred = case['actual'], case['predicted']
        error = actual - pred
        print(f"{i+1:2d}. {days:2d}d {miles:4d}mi ${receipts:7.2f} → Actual:${actual:7.2f} Pred:${pred:7.2f} Under:${error:6.2f}")
    
    # Look for specific patterns that match the described bugs
    print("\n=== BUG HUNTING: Specific Pattern Analysis ===")
    
    # Bug 1: Eight-Day Trip Explosion - looking for 8,16,24 day anomalies
    print("\n1. Eight-Day Trip Explosion Analysis:")
    day_8_cases = [case for case in results if case['days'] == 8]
    day_16_cases = [case for case in results if case['days'] == 16]
    day_24_cases = [case for case in results if case['days'] == 24]
    
    print(f"   8-day trips: {len(day_8_cases)} cases")
    print(f"  16-day trips: {len(day_16_cases)} cases") 
    print(f"  24-day trips: {len(day_24_cases)} cases")
    
    # Look for cases where actual > predicted for these day counts
    day_8_over = [case for case in day_8_cases if case['actual'] > case['predicted'] * 1.2]
    day_16_over = [case for case in day_16_cases if case['actual'] > case['predicted'] * 1.2]
    day_24_over = [case for case in day_24_cases if case['actual'] > case['predicted'] * 1.2]
    
    print(f"   8-day anomalies (actual >> predicted): {len(day_8_over)}")
    print(f"  16-day anomalies (actual >> predicted): {len(day_16_over)}")
    print(f"  24-day anomalies (actual >> predicted): {len(day_24_over)}")
    
    # Bug 2: The "$700" Cliff - looking for receipts around $700
    print("\n2. '$700' Cliff Analysis:")
    cliff_candidates = []
    for case in results:
        receipts = case['receipts']
        if 650 <= receipts <= 800:  # Around the $700 mark
            cliff_candidates.append(case)
    
    print(f"   Cases with receipts $650-$800: {len(cliff_candidates)}")
    
    # Look for anomalies around $716.80 specifically
    precise_cliff = []
    for case in cliff_candidates:
        receipts = case['receipts']
        if 710 <= receipts <= 725:  # Very close to $716.80
            if abs(case['actual'] - case['predicted']) > 50:  # Significant deviation
                precise_cliff.append(case)
    
    print(f"   Anomalies near $716.80: {len(precise_cliff)}")
    if precise_cliff:
        print("   Top cliff anomalies:")
        precise_cliff.sort(key=lambda x: x['absolute_error'], reverse=True)
        for i, case in enumerate(precise_cliff[:5]):
            days, miles, receipts = case['days'], case['miles'], case['receipts']
            actual, pred = case['actual'], case['predicted']
            print(f"      {i+1}. {days:2d}d {miles:4d}mi ${receipts:7.2f} → Actual:${actual:7.2f} Pred:${pred:7.2f}")
    
    # Bug 3: >$2400 Reverse Staircase - looking for high reimbursement anomalies
    print("\n3. '>$2400 Reverse Staircase' Analysis:")
    high_reimbursement = [case for case in results if case['actual'] > 2000]
    print(f"   Cases with actual reimbursement >$2000: {len(high_reimbursement)}")
    
    # Look for cases where model predicts high but actual is much lower
    reverse_stair = []
    for case in high_reimbursement:
        if case['predicted'] > case['actual'] * 1.5:  # Model predicts much higher
            reverse_stair.append(case)
    
    print(f"   Potential reverse staircase cases: {len(reverse_stair)}")
    if reverse_stair:
        print("   Top reverse staircase anomalies:")
        reverse_stair.sort(key=lambda x: x['predicted'] - x['actual'], reverse=True)
        for i, case in enumerate(reverse_stair[:5]):
            days, miles, receipts = case['days'], case['miles'], case['receipts']
            actual, pred = case['actual'], case['predicted']
            over = pred - actual
            print(f"      {i+1}. {days:2d}d {miles:4d}mi ${receipts:7.2f} → Actual:${actual:7.2f} Pred:${pred:7.2f} Over:${over:6.2f}")
    
    # Bug 4: Weekday/Lunar "Luck" Bit - looking for small systematic differences
    print("\n4. Weekday/Lunar 'Luck' Bit Analysis:")
    
    # Load original public cases to get case indices
    with open('public_cases.json', 'r') as f:
        public_cases = json.load(f)
    
    luck_candidates = []
    for i, result in enumerate(results):
        case_idx = result['case_index']
        
        # Check if this is a case with small but systematic error
        error = abs(result['actual'] - result['predicted'])
        relative_error = result['relative_error']
        
        # Look for cases with small absolute error but decent relative error (indicating systematic bias)
        if 10 <= error <= 50 and 0.05 <= relative_error <= 0.2:
            # Calculate epoch_day pattern
            epoch_day = case_idx  # Using case index as proxy for epoch day
            luck_bit = epoch_day % 16
            
            luck_candidates.append({
                'case': result,
                'epoch_day': epoch_day,
                'luck_bit': luck_bit,
                'error': error
            })
    
    print(f"   Potential luck bit candidates: {len(luck_candidates)}")
    
    if luck_candidates:
        # Analyze luck bit patterns
        luck_bit_errors = {}
        for candidate in luck_candidates:
            bit = candidate['luck_bit']
            if bit not in luck_bit_errors:
                luck_bit_errors[bit] = []
            luck_bit_errors[bit].append(candidate['error'])
        
        print("   Average error by luck bit:")
        for bit in sorted(luck_bit_errors.keys()):
            avg_error = np.mean(luck_bit_errors[bit])
            count = len(luck_bit_errors[bit])
            print(f"      Bit {bit:2d}: {avg_error:5.2f} avg error ({count:2d} cases)")
    
    # Summary of findings
    print("\n=== SUMMARY OF FINDINGS ===")
    print(f"Total cases analyzed: {len(results)}")
    print(f"Cases with >$200 error: {len(worst_cases)}")
    print(f"Over-predictions: {len(over_predict)}")
    print(f"Under-predictions: {len(under_predict)}")
    print(f"Potential 8/16/24-day anomalies: {len(day_8_over) + len(day_16_over) + len(day_24_over)}")
    print(f"Potential $700 cliff anomalies: {len(precise_cliff)}")
    print(f"Potential >$2400 reverse staircase: {len(reverse_stair)}")
    print(f"Potential luck bit candidates: {len(luck_candidates)}")
    
    print("\nNext steps:")
    print("1. Focus on the over-prediction cases - these may reveal where bugs inflate reimbursements")
    print("2. Examine under-prediction cases - these may show where bugs reduce reimbursements")
    print("3. Look for mathematical patterns in the specific day/mile/receipt combinations")

if __name__ == "__main__":
    analyze_bug_patterns() 