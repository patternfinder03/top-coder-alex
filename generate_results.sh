#!/bin/bash

# Black Box Challenge - Generating Private Results
# This script uses the trained naive ML model to generate predictions.

echo "🧾 Black Box Challenge - Generating Private Results with Naive ML Model"
echo "===================================================================="
echo ""
echo "📊 Processing test cases and generating results..."
echo "📝 Output will be saved to private_results.txt"
echo ""

# Run the prediction script and redirect output to the results file
python3 predict_with_naive_model.py > private_results.txt

echo ""
echo "✅ Done! Results saved to private_results.txt" 