#!/usr/bin/env python3
import json
import re

def convert_to_hex(value):
    """Convert a number to hexadecimal representation"""
    if isinstance(value, float):
        # Convert float to bytes and then to hex
        import struct
        return struct.pack('!f', value).hex().upper()
    elif isinstance(value, int):
        return hex(value)[2:].upper()
    else:
        return str(value).encode().hex().upper()

def search_patterns_in_hex(hex_string, patterns):
    """Search for specific patterns in hex string"""
    found_patterns = []
    for pattern in patterns:
        if pattern in hex_string:
            found_patterns.append(pattern)
    return found_patterns

def analyze_hex_for_strange_patterns(hex_string):
    """Look for strange patterns in hex"""
    strange_patterns = []
    
    # Look for repeating patterns (same hex digit repeated 4+ times)
    repeating = re.findall(r'([0-9A-F])\1{3,}', hex_string)
    if repeating:
        strange_patterns.append(f"Repeating digits: {set(repeating)}")
    
    # Look for sequential patterns
    sequences = ['0123456789ABCDEF', 'FEDCBA9876543210']
    for seq in sequences:
        for i in range(len(seq) - 3):
            if seq[i:i+4] in hex_string:
                strange_patterns.append(f"Sequential pattern: {seq[i:i+4]}")
    
    # Look for palindromes (4+ chars)
    for i in range(len(hex_string) - 3):
        for j in range(i + 4, min(i + 9, len(hex_string) + 1)):
            substring = hex_string[i:j]
            if substring == substring[::-1] and len(substring) >= 4:
                strange_patterns.append(f"Palindrome: {substring}")
    
    return strange_patterns

def main():
    # Load public cases
    with open('public_cases.json', 'r') as f:
        cases = json.load(f)
    
    target_patterns = ['8090', 'ALLIN', 'CHAMATH', 'ACME']
    pattern_matches = []
    strange_hex_cases = []
    
    print("Analyzing public cases for hex patterns...")
    print(f"Total cases to analyze: {len(cases)}")
    print()
    
    for i, case in enumerate(cases):
        # Convert all values to hex
        input_data = case['input']
        output_data = case.get('expected_output', 0)
        
        # Convert each field to hex
        trip_days_hex = convert_to_hex(input_data['trip_duration_days'])
        miles_hex = convert_to_hex(input_data['miles_traveled'])
        receipts_hex = convert_to_hex(input_data['total_receipts_amount'])
        output_hex = convert_to_hex(output_data)
        
        # Combine all hex values
        combined_hex = trip_days_hex + miles_hex + receipts_hex + output_hex
        
        # Search for target patterns
        found_patterns = search_patterns_in_hex(combined_hex, target_patterns)
        if found_patterns:
            pattern_matches.append({
                'case_index': i,
                'patterns_found': found_patterns,
                'input': input_data,
                'output': output_data,
                'hex_data': {
                    'trip_days': trip_days_hex,
                    'miles': miles_hex,
                    'receipts': receipts_hex,
                    'output': output_hex,
                    'combined': combined_hex
                }
            })
        
        # Look for strange patterns
        strange_patterns = analyze_hex_for_strange_patterns(combined_hex)
        if strange_patterns:
            strange_hex_cases.append({
                'case_index': i,
                'strange_patterns': strange_patterns,
                'input': input_data,
                'output': output_data,
                'hex_data': {
                    'trip_days': trip_days_hex,
                    'miles': miles_hex,
                    'receipts': receipts_hex,
                    'output': output_hex,
                    'combined': combined_hex
                }
            })
    
    # Report findings
    print("=== TARGET PATTERN SEARCH RESULTS ===")
    print(f"Searching for patterns: {target_patterns}")
    print()
    
    if pattern_matches:
        print(f"Found {len(pattern_matches)} cases with target patterns:")
        for match in pattern_matches:
            print(f"Case {match['case_index']}:")
            print(f"  Patterns found: {match['patterns_found']}")
            print(f"  Input: {match['input']}")
            print(f"  Output: {match['output']}")
            print(f"  Hex data:")
            for key, value in match['hex_data'].items():
                print(f"    {key}: {value}")
            print()
    else:
        print("No cases found with target patterns (8090, ALLIN, CHAMATH, ACME)")
    
    print("\n=== STRANGE HEX PATTERN ANALYSIS ===")
    
    if strange_hex_cases:
        print(f"Found {len(strange_hex_cases)} cases with strange hex patterns:")
        # Show first 10 to avoid overwhelming output
        for match in strange_hex_cases[:10]:
            print(f"Case {match['case_index']}:")
            print(f"  Strange patterns: {match['strange_patterns']}")
            print(f"  Input: {match['input']}")
            print(f"  Output: {match['output']}")
            print(f"  Combined hex: {match['hex_data']['combined']}")
            print()
        
        if len(strange_hex_cases) > 10:
            print(f"... and {len(strange_hex_cases) - 10} more cases with strange patterns")
    else:
        print("No cases found with obviously strange hex patterns")
    
    # Save detailed results
    with open('hex_analysis_results.json', 'w') as f:
        json.dump({
            'target_pattern_matches': pattern_matches,
            'strange_hex_cases': strange_hex_cases,
            'summary': {
                'total_cases_analyzed': len(cases),
                'target_pattern_matches_count': len(pattern_matches),
                'strange_hex_cases_count': len(strange_hex_cases)
            }
        }, f, indent=2)
    
    print(f"\nDetailed results saved to 'hex_analysis_results.json'")

if __name__ == "__main__":
    main() 