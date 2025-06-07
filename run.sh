#!/bin/bash

# Black Box Challenge - Reverse Engineering Implementation
# This script replicates the legacy reimbursement system bugs and all
# Usage: ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>

python3 predict_single_naive.py "$1" "$2" "$3" 