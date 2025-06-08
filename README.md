# Top Coder Challenge: Black Box Legacy Reimbursement System - 8th place code


# Alex notes:

I used a lot of experimentation automation to try and find results combined with my initial intuitons on how to solve the problem. I have a second repo where I primarily tried focusing on recreating the legacy formula (which did not work). I did a lot of experimenting on trying to recreate the bugs the legacy code had and found some promise with bitwise information / flips (), but was getting stuck on cases where the receipts-per-miles was very high.

Overall, I think I could've squeezed out a second place. On 5-fold CV I had roughly a 6.25K score and most certainly overfit on the final set. In the final model there were 35 features (some scrambled together last second) which most likely was not the best move. I had GPT generate a summary of the top 20 features by importantance which you can see here. I found some interesting ones in addition to some of the more basic ones that I think most people found. if I tinkered with/explored further you might find some results.

If you want to tinker with/run mine its in the predict_single_naive.py. The rest of the code is from the experimentation and can be largely ignored (sorry for the mess lol). 

If you wanna talk about the comp just hit me on twitter @alexmaxxing

(GPT SUMMARY)
## Top 20 Feature Engineering Breakdown
Your feature engineering captures the complex business logic of this 60-year-old legacy system. Here's what each feature represents:
1. Core Foundation Features (Ranks 1, 5, 11)
- days (#1): Raw trip duration - the most important input
- miles (#5): Raw miles traveled
- receipts (#11): Raw receipt amount

These are your base inputs, with days being most critical because it drives per-diem calculations.

2. Business Logic Reconstruction (Ranks 2, 14)
- cap_est (#2): Your estimate of the system's internal cap formula: 80 * days + 0.5 * miles
- over_cap_amt (#14): How much receipts exceed this estimated cap

This reveals you discovered the legacy system has an internal reimbursement ceiling based on both trip length and distance.

3. Non-Linear Receipt Processing (Ranks 3, 4, 6, 12)
- log_receipts (#3): Captures diminishing returns on high receipt amounts
- price_ending_49_99 (#4): Binary flag for receipts ending in .49/.99 - suggests legacy pricing patterns
- receipt_bins (#6): Categorical receipt tiers (thresholds at $300, $500, $700, $1000, $1500, $2000, $2500)
- sqrt_receipts (#12): Another non-linear transformation for receipt processing

The high importance of price_ending_49_99 suggests the old system had special handling for common retail price endings.

4. High-Value Trip Detection (Rank 10)
- receipts_mask (#10): Receipts with high receipts-per-mile cases masked to 0

This is your "router flag" - when receipts/mile > 20, you force the model to ignore the raw receipt amount, suggesting these cases follow completely different logic.

5. Complex Interaction Terms (Ranks 7, 8)
- days_miles_receipts (#7): Three-way interaction capturing complex interdependencies
- days_receipts (#8): Two-way interaction between trip length and spending

These capture the non-additive nature of the legacy calculations.

6. Custom Travel Metrics (Ranks 9, 15)
- travel_impedance_index (#9): Your custom composite score: 0.54*days + 0.33*(miles/100) + 0.13*log(receipts+1)
- miles_per_day (#15): Travel intensity metric

The Travel Impedance Index is particularly clever - it weights trip characteristics in a way that apparently mirrors the legacy system's internal scoring.

7. Regime Classification (Ranks 16-20)
- log_mileage_ratio (#16): Log of receipts/(0.655 * miles) - mileage reimbursement rate
- log_perdiem_ratio (#17): Log of receipts/(80 * days) - per-diem rate
- mileage_ratio (#18): Raw receipts-to-mileage ratio
- perdiem_ratio (#19): Raw receipts-to-per-diem ratio
- regime_0 (#20): One-hot encoding for A-perdiem regime

These reveal you identified the system processes different trip types through separate calculation paths:
A-perdiem: Low mileage, high receipts (luxury accommodations)
B-mileage: High mileage, low receipts (road trips)
C-mixed: Everything else

8. Categorical Encoding (Rank 13)
- day_bins (#13): Trip duration categories with thresholds at [1, 4, 8, 12, 16, 20, 24, 28] days
This suggests the legacy system treats trip lengths categorically rather than continuously.



**Reverse-engineer a 60-year-old travel reimbursement system using only historical data and employee interviews.**

ACME Corp's legacy reimbursement system has been running for 60 years. No one knows how it works, but it's still used daily.

8090 has built them a new system, but ACME Corp is confused by the differences in results. Your mission is to figure out the original business logic so we can explain why ours is different and better.

Your job: create a perfect replica of the legacy system by reverse-engineering its behavior from 1,000 historical input/output examples and employee interviews.

## What You Have

### Input Parameters

The system takes three inputs:

- `trip_duration_days` - Number of days spent traveling (integer)
- `miles_traveled` - Total miles traveled (integer)
- `total_receipts_amount` - Total dollar amount of receipts (float)

## Documentation

- A PRD (Product Requirements Document)
- Employee interviews with system hints

### Output

- Single numeric reimbursement amount (float, rounded to 2 decimal places)

### Historical Data

- `public_cases.json` - 1,000 historical input/output examples

## Getting Started

1. **Analyze the data**: 
   - Look at `public_cases.json` to understand patterns
   - Look at `PRD.md` to understand the business problem
   - Look at `INTERVIEWS.md` to understand the business logic
2. **Create your implementation**:
   - Copy `run.sh.template` to `run.sh`
   - Implement your calculation logic
   - Make sure it outputs just the reimbursement amount
3. **Test your solution**: 
   - Run `./eval.sh` to see how you're doing
   - Use the feedback to improve your algorithm
4. **Submit**:
   - Run `./generate_results.sh` to get your final results.
   - Add `arjun-krishna1` to your repo.
   - Complete [the submission form](https://forms.gle/sKFBV2sFo2ADMcRt8).

## Implementation Requirements

Your `run.sh` script must:

- Take exactly 3 parameters: `trip_duration_days`, `miles_traveled`, `total_receipts_amount`
- Output a single number (the reimbursement amount)
- Run in under 5 seconds per test case
- Work without external dependencies (no network calls, databases, etc.)

Example:

```bash
./run.sh 5 250 150.75
# Should output something like: 487.25
```

## Evaluation

Run `./eval.sh` to test your solution against all 1,000 cases. The script will show:

- **Exact matches**: Cases within ±$0.01 of the expected output
- **Close matches**: Cases within ±$1.00 of the expected output
- **Average error**: Mean absolute difference from expected outputs
- **Score**: Lower is better (combines accuracy and precision)

Your submission will be tested against `private_cases.json` which does not include the outputs.

## Submission

When you're ready to submit:

1. Push your solution to a GitHub repository
2. Add `arjun-krishna1` to your repository
3. Submit via the [submission form](https://forms.gle/sKFBV2sFo2ADMcRt8).
4. When you submit the form you will submit your `private_results.txt` which will be used for your final score.

---

**Good luck and Bon Voyage!**
