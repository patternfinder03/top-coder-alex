import json
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    """Load public cases and naive model results"""
    with open('public_cases.json', 'r') as f:
        public_cases = json.load(f)
    
    with open('naive_model_detailed.json', 'r') as f:
        naive_results = json.load(f)
    
    return public_cases, naive_results

def test_8_day_overflow(public_cases, naive_results):
    """Test Hypothesis 1: 8-day overflow (3-bit wrap)"""
    print("=== HYPOTHESIS 1: 8-DAY OVERFLOW TEST ===")
    print("Testing if days wrap to 3 bits (days & 7)")
    
    # Find cases with 8-day trips
    eight_day_cases = []
    for i, case in enumerate(public_cases):
        if case['input']['trip_duration_days'] == 8:
            naive_case = next((r for r in naive_results if r['case_index'] == i), None)
            if naive_case:
                eight_day_cases.append({
                    'case_index': i,
                    'days': case['input']['trip_duration_days'],
                    'miles': case['input']['miles_traveled'],
                    'receipts': case['input']['total_receipts_amount'],
                    'actual': case['expected_output'],
                    'naive_pred': naive_case['predicted'],
                    'error': naive_case['absolute_error']
                })
    
    print(f"Found {len(eight_day_cases)} cases with 8 days")
    print("Top 10 worst 8-day cases:")
    eight_day_cases.sort(key=lambda x: x['error'], reverse=True)
    
    print("Case | Miles | Receipts  | Actual   | Predicted | Error    | Days&7")
    print("-" * 65)
    for case in eight_day_cases[:10]:
        days_wrapped = case['days'] & 7  # 3-bit wrap
        print(f"{case['case_index']:4d} | {case['miles']:5d} | ${case['receipts']:8.2f} | ${case['actual']:8.2f} | ${case['naive_pred']:9.2f} | ${case['error']:8.2f} | {days_wrapped}")
    
    # Calculate if wrapping explains the error
    print(f"\nHypothesis: If days wrap to 3 bits, 8 becomes 0")
    print(f"Expected behavior: Algorithm treats 8-day trips as 0-day trips")
    
    return eight_day_cases

def test_700_cliff(public_cases, naive_results):
    """Test Hypothesis 2: $700 cliff (receipt tier bit shifting)"""
    print("\n=== HYPOTHESIS 2: $700 CLIFF TEST ===")
    print("Testing receipt tier calculation: receipt_cents >> 6")
    
    # Calculate receipt tiers for all cases
    cliff_analysis = []
    for i, case in enumerate(public_cases):
        receipts = case['input']['total_receipts_amount']
        receipt_cents = int(receipts * 100)
        tier = receipt_cents >> 6  # Bit shift by 6
        
        naive_case = next((r for r in naive_results if r['case_index'] == i), None)
        if naive_case:
            cliff_analysis.append({
                'case_index': i,
                'receipts': receipts,
                'receipt_cents': receipt_cents,
                'tier': tier,
                'actual': case['expected_output'],
                'naive_pred': naive_case['predicted'],
                'error': naive_case['absolute_error']
            })
    
    # Look for tier boundary effects
    tier_boundaries = {}
    for case in cliff_analysis:
        tier = case['tier']
        if tier not in tier_boundaries:
            tier_boundaries[tier] = []
        tier_boundaries[tier].append(case)
    
    print("Receipt tier analysis (receipt_cents >> 6):")
    print("Tier | Range ($)  | Cases | Avg Error | Max Error")
    print("-" * 50)
    
    critical_tiers = []
    for tier in sorted(tier_boundaries.keys())[:20]:  # Show first 20 tiers
        cases = tier_boundaries[tier]
        min_receipts = min(c['receipts'] for c in cases)
        max_receipts = max(c['receipts'] for c in cases)
        avg_error = np.mean([c['error'] for c in cases])
        max_error = max(c['error'] for c in cases)
        
        # Look for tiers with high average error
        if avg_error > 100:
            critical_tiers.append((tier, avg_error, max_error))
        
        print(f"{tier:4d} | ${min_receipts:6.2f}-${max_receipts:6.2f} | {len(cases):5d} | ${avg_error:8.2f} | ${max_error:8.2f}")
    
    # Focus on the cliff around $716.80 (tier 11)
    print(f"\nCritical analysis around $716.80 cliff:")
    cliff_cases = [c for c in cliff_analysis if 700 <= c['receipts'] <= 750]
    cliff_cases.sort(key=lambda x: x['receipts'])
    
    print("Case | Receipts | Tier | Actual   | Predicted | Error")
    print("-" * 55)
    for case in cliff_cases[:15]:
        print(f"{case['case_index']:4d} | ${case['receipts']:8.2f} | {case['tier']:4d} | ${case['actual']:8.2f} | ${case['naive_pred']:9.2f} | ${case['error']:6.2f}")
    
    return cliff_analysis, critical_tiers

def test_2400_wrap(public_cases, naive_results):
    """Test Hypothesis 3: ~$2400 accumulator wrap (12-bit overflow)"""
    print("\n=== HYPOTHESIS 3: $2400 ACCUMULATOR WRAP TEST ===")
    print("Testing 12-bit overflow: receipt_cents % 4096")
    
    wrap_analysis = []
    for i, case in enumerate(public_cases):
        receipts = case['input']['total_receipts_amount']
        receipt_cents = int(receipts * 100)
        wrap_value = receipt_cents % 4096  # 12-bit wrap
        
        naive_case = next((r for r in naive_results if r['case_index'] == i), None)
        if naive_case:
            wrap_analysis.append({
                'case_index': i,
                'receipts': receipts,
                'receipt_cents': receipt_cents,
                'wrap_value': wrap_value,
                'actual': case['expected_output'],
                'naive_pred': naive_case['predicted'],
                'error': naive_case['absolute_error']
            })
    
    # Look for high-receipt cases with wrap effects
    high_receipt_cases = [c for c in wrap_analysis if c['receipts'] >= 2000]
    high_receipt_cases.sort(key=lambda x: x['error'], reverse=True)
    
    print(f"High receipt cases (≥$2000) showing potential 12-bit wrap:")
    print("Case | Receipts | Cents | Wrap Value | Actual   | Predicted | Error")
    print("-" * 70)
    
    for case in high_receipt_cases[:15]:
        print(f"{case['case_index']:4d} | ${case['receipts']:8.2f} | {case['receipt_cents']:5d} | {case['wrap_value']:10d} | ${case['actual']:8.2f} | ${case['naive_pred']:9.2f} | ${case['error']:6.2f}")
    
    # Plot wrap value vs error to look for sawtooth pattern
    wrap_values = [c['wrap_value'] for c in wrap_analysis]
    errors = [c['error'] for c in wrap_analysis]
    
    plt.figure(figsize=(12, 6))
    plt.scatter(wrap_values, errors, alpha=0.6, s=20)
    plt.xlabel('Receipt Cents % 4096 (12-bit wrap value)')
    plt.ylabel('Prediction Error ($)')
    plt.title('12-bit Overflow Pattern: Error vs Wrap Value')
    plt.grid(True, alpha=0.3)
    plt.savefig('wrap_pattern.png', dpi=300, bbox_inches='tight')
    print("Saved wrap pattern plot as 'wrap_pattern.png'")
    
    return wrap_analysis

def test_long_trip_penalty(public_cases, naive_results):
    """Test Hypothesis 4: Long-trip low-spend penalty"""
    print("\n=== HYPOTHESIS 4: LONG-TRIP LOW-SPEND PENALTY TEST ===")
    print("Testing receipt/per-diem ratio for long trips with low spending")
    
    penalty_analysis = []
    for i, case in enumerate(public_cases):
        days = case['input']['trip_duration_days']
        receipts = case['input']['total_receipts_amount']
        
        # Estimate per-diem (rough calculation)
        estimated_per_diem = 50  # Rough estimate based on earlier analysis
        expected_receipts = days * estimated_per_diem
        receipt_ratio = receipts / expected_receipts if expected_receipts > 0 else 0
        
        naive_case = next((r for r in naive_results if r['case_index'] == i), None)
        if naive_case:
            penalty_analysis.append({
                'case_index': i,
                'days': days,
                'receipts': receipts,
                'expected_receipts': expected_receipts,
                'receipt_ratio': receipt_ratio,
                'actual': case['expected_output'],
                'naive_pred': naive_case['predicted'],
                'error': naive_case['absolute_error']
            })
    
    # Look for long trips with low spending ratios
    long_low_cases = [c for c in penalty_analysis if c['days'] >= 10 and c['receipt_ratio'] < 0.5]
    long_low_cases.sort(key=lambda x: x['error'], reverse=True)
    
    print(f"Long trips (≥10 days) with low spending (<0.5 ratio):")
    print("Case | Days | Receipts | Expected | Ratio | Actual   | Predicted | Error")
    print("-" * 75)
    
    for case in long_low_cases[:15]:
        print(f"{case['case_index']:4d} | {case['days']:4d} | ${case['receipts']:8.2f} | ${case['expected_receipts']:8.2f} | {case['receipt_ratio']:5.2f} | ${case['actual']:8.2f} | ${case['naive_pred']:9.2f} | ${case['error']:6.2f}")
    
    return penalty_analysis, long_low_cases

def test_luck_nibble(public_cases, naive_results):
    """Test Hypothesis 5: Odd/even "luck" nibble (epoch day modulo)"""
    print("\n=== HYPOTHESIS 5: LUCK NIBBLE TEST ===")
    print("Testing epoch_day % patterns for systematic bias")
    
    luck_analysis = []
    for i, case in enumerate(public_cases):
        # Use case index as proxy for epoch day
        epoch_day = i
        luck_bit_2 = epoch_day % 2
        luck_bit_4 = epoch_day % 4
        luck_bit_16 = epoch_day % 16
        
        naive_case = next((r for r in naive_results if r['case_index'] == i), None)
        if naive_case:
            # Calculate signed error (predicted - actual)
            signed_error = naive_case['predicted'] - case['expected_output']
            
            luck_analysis.append({
                'case_index': i,
                'epoch_day': epoch_day,
                'luck_bit_2': luck_bit_2,
                'luck_bit_4': luck_bit_4,
                'luck_bit_16': luck_bit_16,
                'actual': case['expected_output'],
                'naive_pred': naive_case['predicted'],
                'signed_error': signed_error,
                'abs_error': naive_case['absolute_error']
            })
    
    # Analyze patterns by different modulo values
    print("Luck bit analysis:")
    
    # Modulo 2 analysis
    mod2_groups = {0: [], 1: []}
    for case in luck_analysis:
        mod2_groups[case['luck_bit_2']].append(case['signed_error'])
    
    print(f"\nEpoch day % 2 analysis:")
    for bit in [0, 1]:
        errors = mod2_groups[bit]
        print(f"  Bit {bit}: {len(errors)} cases, mean signed error: ${np.mean(errors):6.2f}, std: ${np.std(errors):6.2f}")
    
    # Modulo 16 analysis  
    mod16_groups = {i: [] for i in range(16)}
    for case in luck_analysis:
        mod16_groups[case['luck_bit_16']].append(case['signed_error'])
    
    print(f"\nEpoch day % 16 analysis:")
    print("Bit | Cases | Mean Error | Std Error")
    print("-" * 40)
    
    luck_patterns = []
    for bit in range(16):
        errors = mod16_groups[bit]
        if len(errors) > 0:
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            print(f"{bit:3d} | {len(errors):5d} | ${mean_error:9.2f} | ${std_error:8.2f}")
            
            if abs(mean_error) > 20:  # Significant bias
                luck_patterns.append((bit, mean_error, len(errors)))
    
    # Look for 5-day cases with similar parameters but different outcomes
    five_day_cases = [c for c in luck_analysis if 
                     any(case['input']['trip_duration_days'] == 5 for case in public_cases 
                         if public_cases.index(case) == c['case_index'])]
    
    print(f"\nFound {len(five_day_cases)} cases with 5-day trips for luck comparison")
    
    return luck_analysis, luck_patterns

def generate_summary_report(eight_day_cases, critical_tiers, wrap_analysis, long_low_cases, luck_patterns):
    """Generate summary report of all hypothesis tests"""
    print("\n" + "="*60)
    print("SUMMARY REPORT: BUG HYPOTHESIS TEST RESULTS")
    print("="*60)
    
    print(f"\n1. 8-DAY OVERFLOW:")
    print(f"   - Found {len(eight_day_cases)} cases with 8-day trips")
    high_error_8day = len([c for c in eight_day_cases if c['error'] > 200])
    print(f"   - {high_error_8day} have errors >$200")
    print(f"   - VERDICT: {'LIKELY BUG' if high_error_8day > 3 else 'NEEDS MORE INVESTIGATION'}")
    
    print(f"\n2. $700 CLIFF:")
    print(f"   - Found {len(critical_tiers)} receipt tiers with high average error")
    print(f"   - Critical tiers: {[f'Tier {t[0]} (${t[1]:.0f} avg error)' for t in critical_tiers[:3]]}")
    print(f"   - VERDICT: {'LIKELY BUG' if len(critical_tiers) > 2 else 'NEEDS MORE INVESTIGATION'}")
    
    print(f"\n3. $2400 ACCUMULATOR WRAP:")
    high_receipt_errors = len([c for c in wrap_analysis if c['receipts'] >= 2000 and c['error'] > 200])
    print(f"   - {high_receipt_errors} high-receipt cases (≥$2000) with errors >$200")
    print(f"   - Check wrap_pattern.png for sawtooth pattern")
    print(f"   - VERDICT: {'LIKELY BUG' if high_receipt_errors > 5 else 'NEEDS MORE INVESTIGATION'}")
    
    print(f"\n4. LONG-TRIP LOW-SPEND PENALTY:")
    print(f"   - Found {len(long_low_cases)} long trips with low spending ratios")
    high_error_penalty = len([c for c in long_low_cases if c['error'] > 100])
    print(f"   - {high_error_penalty} have errors >$100")
    print(f"   - VERDICT: {'LIKELY BUG' if high_error_penalty > 2 else 'NEEDS MORE INVESTIGATION'}")
    
    print(f"\n5. LUCK NIBBLE:")
    print(f"   - Found {len(luck_patterns)} epoch_day%16 values with significant bias (>$20)")
    strong_patterns = len([p for p in luck_patterns if abs(p[1]) > 50])
    print(f"   - {strong_patterns} patterns with very strong bias (>$50)")
    print(f"   - VERDICT: {'LIKELY BUG' if strong_patterns > 2 else 'NEEDS MORE INVESTIGATION'}")
    
    print(f"\nRECOMMENDED NEXT STEPS:")
    print(f"1. Focus on 8-day overflow - implement days & 7 in algorithm")
    print(f"2. Investigate receipt tier boundaries - test receipt_cents >> 6")
    print(f"3. Check for 12-bit accumulator overflow in high-receipt cases")
    print(f"4. Examine per-diem penalty logic for under-spending")
    print(f"5. Look for date-based rounding bias in algorithm")

def main():
    """Run all hypothesis tests"""
    print("Loading data...")
    public_cases, naive_results = load_data()
    
    # Run all hypothesis tests
    eight_day_cases = test_8_day_overflow(public_cases, naive_results)
    cliff_analysis, critical_tiers = test_700_cliff(public_cases, naive_results)
    wrap_analysis = test_2400_wrap(public_cases, naive_results)
    penalty_analysis, long_low_cases = test_long_trip_penalty(public_cases, naive_results)
    luck_analysis, luck_patterns = test_luck_nibble(public_cases, naive_results)
    
    # Generate summary
    generate_summary_report(eight_day_cases, critical_tiers, wrap_analysis, long_low_cases, luck_patterns)
    
    # Save detailed results
    results = {
        'eight_day_cases': eight_day_cases,
        'cliff_analysis': cliff_analysis[:100],  # Limit size
        'wrap_analysis': [c for c in wrap_analysis if c['receipts'] >= 2000],
        'long_low_cases': long_low_cases,
        'luck_patterns': luck_patterns
    }
    
    with open('hypothesis_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to 'hypothesis_test_results.json'")

if __name__ == "__main__":
    main() 