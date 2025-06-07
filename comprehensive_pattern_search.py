#!/usr/bin/env python3
import json
import re
import struct

def convert_value_to_various_formats(value):
    """Convert a value to various representations for comprehensive search"""
    formats = {}
    
    if isinstance(value, (int, float)):
        # Original value as string
        formats['string'] = str(value)
        
        # Hexadecimal (integer part)
        if isinstance(value, int):
            formats['hex'] = hex(value)[2:].upper()
        else:
            formats['hex'] = hex(int(value))[2:].upper()
        
        # Binary representation as hex
        if isinstance(value, float):
            formats['float_hex'] = struct.pack('!f', value).hex().upper()
            formats['double_hex'] = struct.pack('!d', float(value)).hex().upper()
        else:
            formats['int_hex'] = struct.pack('!i', int(value)).hex().upper()
        
        # ASCII encoding of string representation
        formats['ascii_hex'] = str(value).encode('ascii').hex().upper()
        
    return formats

def search_all_representations(case_data, patterns):
    """Search for patterns in all representations of the case data"""
    found_matches = {}
    
    # Get all values from the case
    values = [
        case_data['input']['trip_duration_days'],
        case_data['input']['miles_traveled'],
        case_data['input']['total_receipts_amount'],
        case_data.get('expected_output', 0)
    ]
    
    all_text = ""
    
    for value in values:
        formats = convert_value_to_various_formats(value)
        for format_name, format_value in formats.items():
            all_text += format_value + " "
    
    # Also check the raw JSON string representation
    json_str = json.dumps(case_data).upper()
    all_text += json_str
    
    # Search for each pattern
    for pattern in patterns:
        pattern_upper = pattern.upper()
        if pattern_upper in all_text:
            found_matches[pattern] = True
        
        # Also search for pattern as ASCII hex
        ascii_hex_pattern = pattern.encode('ascii').hex().upper()
        if ascii_hex_pattern in all_text:
            found_matches[f"{pattern}_as_ascii_hex"] = True
    
    return found_matches

def main():
    # Load public cases
    with open('public_cases.json', 'r') as f:
        cases = json.load(f)
    
    target_patterns = ['8090', 'ALLIN', 'CHAMATH', 'ACME']
    all_matches = []
    
    print("Comprehensive pattern search in public cases...")
    print(f"Total cases to analyze: {len(cases)}")
    print(f"Searching for patterns: {target_patterns}")
    print()
    
    # Also search in the raw JSON text itself
    raw_json_text = json.dumps(cases).upper()
    print("=== SEARCHING IN RAW JSON TEXT ===")
    for pattern in target_patterns:
        if pattern.upper() in raw_json_text:
            print(f"Found '{pattern}' in raw JSON text!")
        else:
            print(f"'{pattern}' not found in raw JSON text")
    
    print("\n=== SEARCHING IN INDIVIDUAL CASES ===")
    
    for i, case in enumerate(cases):        
        matches = search_all_representations(case, target_patterns)
        
        if matches:
            all_matches.append({
                'case_index': i,
                'matches': matches,
                'case_data': case
            })
    
    if all_matches:
        print(f"Found {len(all_matches)} cases with target patterns:")
        for match in all_matches:
            print(f"Case {match['case_index']}:", match['matches'])
            print(f"  Data: {match['case_data']}")
            print()
    else:
        print("No cases found with target patterns in any representation")
    
    # Check if patterns might be hidden in calculated values or relationships
    print("\n=== CHECKING FOR PATTERNS IN CALCULATED VALUES ===")
    
    special_cases = []
    for i, case in enumerate(cases):
        input_data = case['input']
        output = case.get('expected_output', 0)
        
        # Check various calculations
        calculations = {
            'sum': input_data['trip_duration_days'] + input_data['miles_traveled'] + input_data['total_receipts_amount'],
            'product_int': int(input_data['trip_duration_days'] * input_data['miles_traveled']),
            'ratio1': int(output / input_data['trip_duration_days']) if input_data['trip_duration_days'] != 0 else 0,
            'ratio2': int(output / input_data['miles_traveled']) if input_data['miles_traveled'] != 0 else 0,
        }
        
        for calc_name, calc_value in calculations.items():
            calc_str = str(calc_value)
            for pattern in target_patterns:
                if pattern in calc_str:
                    special_cases.append({
                        'case_index': i,
                        'pattern': pattern,
                        'found_in': calc_name,
                        'value': calc_value,
                        'case_data': case
                    })
    
    if special_cases:
        print(f"Found {len(special_cases)} cases with patterns in calculated values:")
        for case in special_cases:
            print(f"Case {case['case_index']}: Pattern '{case['pattern']}' found in {case['found_in']} = {case['value']}")
            print(f"  Data: {case['case_data']}")
            print()
    else:
        print("No patterns found in calculated values")
    
    # Save comprehensive results
    with open('comprehensive_pattern_search_results.json', 'w') as f:
        json.dump({
            'direct_matches': all_matches,
            'calculated_value_matches': special_cases,
            'summary': {
                'total_cases_analyzed': len(cases),
                'direct_matches_count': len(all_matches),
                'calculated_matches_count': len(special_cases)
            }
        }, f, indent=2)
    
    print(f"\nComprehensive results saved to 'comprehensive_pattern_search_results.json'")

if __name__ == "__main__":
    main() 