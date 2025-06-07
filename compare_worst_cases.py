import json

def compare_worst_cases():
    """Compare worst cases from naive model vs original simple algorithm"""
    
    # Load naive model worst cases
    with open('worst_100_naive_with_original.json', 'r') as f:
        naive_worst = json.load(f)
    
    # Load original algorithm worst cases
    with open('worst_100_cases.json', 'r') as f:
        original_worst = json.load(f)
    
    # Extract case indices
    naive_case_indices = set(case['case_index'] for case in naive_worst)
    original_case_indices = set(case['case_index'] for case in original_worst)
    
    # Find overlaps
    overlap_indices = naive_case_indices.intersection(original_case_indices)
    naive_only = naive_case_indices - original_case_indices
    original_only = original_case_indices - naive_case_indices
    
    print("=== COMPARISON OF WORST 100 CASES ===")
    print(f"Naive model worst cases: {len(naive_worst)}")
    print(f"Original algorithm worst cases: {len(original_worst)}")
    print(f"Overlapping cases: {len(overlap_indices)}")
    print(f"Only in naive worst: {len(naive_only)}")
    print(f"Only in original worst: {len(original_only)}")
    print(f"Overlap percentage: {len(overlap_indices)/100*100:.1f}%")
    
    # Create dictionaries for easy lookup
    naive_dict = {case['case_index']: case for case in naive_worst}
    original_dict = {case['case_index']: case for case in original_worst}
    
    # Analyze overlapping cases
    if overlap_indices:
        print(f"\n=== TOP 20 OVERLAPPING CASES ===")
        print("(Cases that appear in both worst 100 lists)")
        print("Case | Days | Miles | Receipts  | Actual   | Naive Pred | Orig Pred | Naive Err | Orig Err")
        print("-" * 95)
        
        # Sort overlapping cases by naive model error
        overlap_cases = []
        for case_idx in overlap_indices:
            naive_case = naive_dict[case_idx]
            original_case = original_dict[case_idx]
            overlap_cases.append((case_idx, naive_case, original_case))
        
        # Sort by naive model absolute error
        overlap_cases.sort(key=lambda x: x[1]['absolute_error'], reverse=True)
        
        for i, (case_idx, naive_case, original_case) in enumerate(overlap_cases[:20]):
            days = naive_case['original_case']['input']['trip_duration_days']
            miles = naive_case['original_case']['input']['miles_traveled']
            receipts = naive_case['original_case']['input']['total_receipts_amount']
            actual = naive_case['actual_output']
            naive_pred = naive_case['naive_prediction']
            orig_pred = original_case['my_prediction']
            naive_err = naive_case['absolute_error']
            orig_err = original_case['absolute_error']
            
            print(f"{case_idx:4d} | {int(days):4d} | {int(miles):5d} | ${receipts:8.2f} | ${actual:8.2f} | ${naive_pred:9.2f} | ${orig_pred:8.2f} | ${naive_err:8.2f} | ${orig_err:7.2f}")
    
    # Analyze cases only in naive worst
    print(f"\n=== TOP 10 CASES ONLY IN NAIVE WORST ===")
    print("(Cases where naive model fails but original algorithm works better)")
    print("Case | Days | Miles | Receipts  | Actual   | Naive Pred | Naive Err")
    print("-" * 70)
    
    naive_only_cases = [(case['case_index'], case) for case in naive_worst if case['case_index'] in naive_only]
    naive_only_cases.sort(key=lambda x: x[1]['absolute_error'], reverse=True)
    
    for i, (case_idx, case) in enumerate(naive_only_cases[:10]):
        days = case['original_case']['input']['trip_duration_days']
        miles = case['original_case']['input']['miles_traveled']
        receipts = case['original_case']['input']['total_receipts_amount']
        actual = case['actual_output']
        naive_pred = case['naive_prediction']
        naive_err = case['absolute_error']
        
        print(f"{case_idx:4d} | {int(days):4d} | {int(miles):5d} | ${receipts:8.2f} | ${actual:8.2f} | ${naive_pred:9.2f} | ${naive_err:8.2f}")
    
    # Analyze cases only in original worst
    print(f"\n=== TOP 10 CASES ONLY IN ORIGINAL WORST ===")
    print("(Cases where original algorithm fails but naive model works better)")
    print("Case | Days | Miles | Receipts  | Actual   | Orig Pred | Orig Err")
    print("-" * 68)
    
    original_only_cases = [(case['case_index'], case) for case in original_worst if case['case_index'] in original_only]
    original_only_cases.sort(key=lambda x: x[1]['absolute_error'], reverse=True)
    
    for i, (case_idx, case) in enumerate(original_only_cases[:10]):
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        actual = case['expected_output']
        orig_pred = case['my_prediction']
        orig_err = case['absolute_error']
        
        print(f"{case_idx:4d} | {int(days):4d} | {int(miles):5d} | ${receipts:8.2f} | ${actual:8.2f} | ${orig_pred:8.2f} | ${orig_err:7.2f}")
    
    # Summary analysis
    print(f"\n=== SUMMARY ANALYSIS ===")
    
    if overlap_indices:
        # Calculate average errors for overlapping cases
        naive_avg_error = sum(naive_dict[idx]['absolute_error'] for idx in overlap_indices) / len(overlap_indices)
        original_avg_error = sum(original_dict[idx]['absolute_error'] for idx in overlap_indices) / len(overlap_indices)
        
        print(f"For overlapping cases:")
        print(f"  Average naive model error: ${naive_avg_error:.2f}")
        print(f"  Average original algorithm error: ${original_avg_error:.2f}")
        print(f"  Ratio: {original_avg_error/naive_avg_error:.2f}x")
    
    # Look for patterns in non-overlapping cases
    print(f"\nPattern analysis:")
    
    # High mileage analysis
    naive_high_miles = sum(1 for case in naive_worst if case['case_index'] in naive_only 
                          and case['original_case']['input']['miles_traveled'] >= 1000)
    original_high_miles = sum(1 for case in original_worst if case['case_index'] in original_only 
                             and case['input']['miles_traveled'] >= 1000)
    
    print(f"  High mileage (≥1000) in naive-only: {naive_high_miles}/{len(naive_only)}")
    print(f"  High mileage (≥1000) in original-only: {original_high_miles}/{len(original_only)}")
    
    # High receipts analysis
    naive_high_receipts = sum(1 for case in naive_worst if case['case_index'] in naive_only 
                             and case['original_case']['input']['total_receipts_amount'] >= 1000)
    original_high_receipts = sum(1 for case in original_worst if case['case_index'] in original_only 
                                and case['input']['total_receipts_amount'] >= 1000)
    
    print(f"  High receipts (≥$1000) in naive-only: {naive_high_receipts}/{len(naive_only)}")
    print(f"  High receipts (≥$1000) in original-only: {original_high_receipts}/{len(original_only)}")

if __name__ == "__main__":
    compare_worst_cases() 