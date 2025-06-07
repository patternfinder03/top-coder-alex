import json

def get_worst_100_with_original():
    """Get top 100 worst naive model predictions with original public case data"""
    
    # Load naive model results
    with open('naive_model_detailed.json', 'r') as f:
        naive_results = json.load(f)
    
    # Load original public cases
    with open('public_cases.json', 'r') as f:
        public_cases = json.load(f)
    
    # Get top 100 worst cases
    worst_100 = naive_results[:100]
    
    # Create combined data
    combined_results = []
    
    for naive_case in worst_100:
        case_index = naive_case['case_index']
        
        # Get the original public case
        original_case = public_cases[case_index]
        
        # Combine the data
        combined_case = {
            'rank': len(combined_results) + 1,
            'case_index': case_index,
            'original_case': original_case,
            'naive_prediction': naive_case['predicted'],
            'actual_output': naive_case['actual'],
            'absolute_error': naive_case['absolute_error'],
            'relative_error': naive_case['relative_error'],
            'error_factor': naive_case['predicted'] / naive_case['actual'] if naive_case['actual'] > 0 else float('inf')
        }
        
        combined_results.append(combined_case)
    
    # Save to JSON
    with open('worst_100_naive_with_original.json', 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"Created worst_100_naive_with_original.json with {len(combined_results)} cases")
    
    # Print summary
    print(f"\nTop 10 worst cases:")
    print("Rank | Case | Days | Miles | Receipts | Actual  | Predicted | Error   | Factor")
    print("-" * 80)
    
    for i, case in enumerate(combined_results[:10]):
        original = case['original_case']
        days = original['input']['trip_duration_days']
        miles = original['input']['miles_traveled']
        receipts = original['input']['total_receipts_amount']
        actual = case['actual_output']
        predicted = case['naive_prediction']
        error = case['absolute_error']
        factor = case['error_factor']
        
        print(f"{i+1:4d} | {case['case_index']:4d} | {int(days):4d} | {int(miles):5d} | ${receipts:8.2f} | ${actual:7.2f} | ${predicted:9.2f} | ${error:7.2f} | {factor:6.2f}x")

if __name__ == "__main__":
    get_worst_100_with_original() 