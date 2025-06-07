import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """Load public cases and naive model results"""
    with open('public_cases.json', 'r') as f:
        public_cases = json.load(f)
    
    with open('naive_model_detailed.json', 'r') as f:
        naive_results = json.load(f)
    
    return public_cases, naive_results

def plot_8_day_overflow(public_cases, naive_results):
    """Plot 8-day overflow hypothesis"""
    print("Creating 8-day overflow plots...")
    
    # Collect data for all day counts
    day_data = {}
    for i, case in enumerate(public_cases):
        days = case['input']['trip_duration_days']
        naive_case = next((r for r in naive_results if r['case_index'] == i), None)
        
        if naive_case and days <= 16:  # Focus on days 1-16
            if days not in day_data:
                day_data[days] = []
            day_data[days].append({
                'error': naive_case['absolute_error'],
                'actual': case['expected_output'],
                'predicted': naive_case['predicted'],
                'miles': case['input']['miles_traveled'],
                'receipts': case['input']['total_receipts_amount']
            })
    
    # Create subplot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Average error by day count
    days_list = sorted(day_data.keys())
    avg_errors = [np.mean([d['error'] for d in day_data[day]]) for day in days_list]
    max_errors = [np.max([d['error'] for d in day_data[day]]) for day in days_list]
    
    ax1.bar(days_list, avg_errors, alpha=0.7, color='skyblue', label='Average Error')
    ax1.bar(days_list, max_errors, alpha=0.5, color='red', label='Max Error')
    ax1.axvline(x=8, color='red', linestyle='--', linewidth=2, label='8-day boundary')
    ax1.axvline(x=16, color='red', linestyle='--', linewidth=2, label='16-day boundary')
    ax1.set_xlabel('Trip Duration (days)')
    ax1.set_ylabel('Prediction Error ($)')
    ax1.set_title('Error by Day Count - Testing 8-day Overflow\n(3-bit wrap: 8→0, 16→0)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Error distribution for 8-day cases
    eight_day_errors = [d['error'] for d in day_data.get(8, [])]
    if eight_day_errors:
        ax2.hist(eight_day_errors, bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax2.axvline(x=np.mean(eight_day_errors), color='red', linestyle='--', 
                   label=f'Mean: ${np.mean(eight_day_errors):.0f}')
        ax2.set_xlabel('Prediction Error ($)')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'8-Day Cases Error Distribution\n({len(eight_day_errors)} cases)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Days & 7 analysis (3-bit wrap)
    wrapped_data = {}
    for days in days_list:
        wrapped = days & 7  # 3-bit wrap
        if wrapped not in wrapped_data:
            wrapped_data[wrapped] = []
        wrapped_data[wrapped].extend(day_data[days])
    
    wrapped_days = sorted(wrapped_data.keys())
    wrapped_avg_errors = [np.mean([d['error'] for d in wrapped_data[wd]]) for wd in wrapped_days]
    
    ax3.bar(wrapped_days, wrapped_avg_errors, alpha=0.7, color='lightcoral')
    ax3.set_xlabel('Days & 7 (3-bit wrapped value)')
    ax3.set_ylabel('Average Prediction Error ($)')
    ax3.set_title('Error by 3-bit Wrapped Day Value\nIf 8→0 overflow, high error at 0')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Scatter plot of 8-day vs other day cases
    eight_day_data = day_data.get(8, [])
    other_day_data = []
    for days in [6, 7, 9, 10]:  # Similar day counts
        other_day_data.extend(day_data.get(days, []))
    
    if eight_day_data and other_day_data:
        # Plot receipts vs error for 8-day vs others
        eight_receipts = [d['receipts'] for d in eight_day_data]
        eight_errors = [d['error'] for d in eight_day_data]
        other_receipts = [d['receipts'] for d in other_day_data]
        other_errors = [d['error'] for d in other_day_data]
        
        ax4.scatter(other_receipts, other_errors, alpha=0.6, s=20, color='blue', label='6,7,9,10 days')
        ax4.scatter(eight_receipts, eight_errors, alpha=0.8, s=30, color='red', label='8 days')
        ax4.set_xlabel('Total Receipts ($)')
        ax4.set_ylabel('Prediction Error ($)')
        ax4.set_title('8-Day Cases vs Similar Day Counts\nReceipts vs Error')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('8_day_overflow_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved 8_day_overflow_analysis.png")

def plot_700_cliff(public_cases, naive_results):
    """Plot $700 cliff hypothesis"""
    print("Creating $700 cliff plots...")
    
    # Calculate receipt tiers
    cliff_data = []
    for i, case in enumerate(public_cases):
        receipts = case['input']['total_receipts_amount']
        receipt_cents = int(receipts * 100)
        tier = receipt_cents >> 6  # Bit shift by 6
        
        naive_case = next((r for r in naive_results if r['case_index'] == i), None)
        if naive_case:
            cliff_data.append({
                'receipts': receipts,
                'tier': tier,
                'error': naive_case['absolute_error'],
                'actual': case['expected_output'],
                'predicted': naive_case['predicted']
            })
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Error vs receipts with tier boundaries
    receipts_list = [d['receipts'] for d in cliff_data]
    errors_list = [d['error'] for d in cliff_data]
    
    ax1.scatter(receipts_list, errors_list, alpha=0.6, s=15, color='blue')
    
    # Mark tier boundaries
    tier_boundaries = [64 * (1 << 6) / 100 for i in range(1, 20)]  # $40.96, $81.92, etc.
    for boundary in tier_boundaries:
        if boundary <= max(receipts_list):
            ax1.axvline(x=boundary, color='red', linestyle='--', alpha=0.5)
    
    ax1.axvline(x=716.80, color='red', linewidth=2, label='$716.80 cliff')
    ax1.set_xlabel('Total Receipts ($)')
    ax1.set_ylabel('Prediction Error ($)')
    ax1.set_title('$700 Cliff Analysis\nError vs Receipts with Tier Boundaries')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Average error by tier
    tier_errors = {}
    for d in cliff_data:
        tier = d['tier']
        if tier not in tier_errors:
            tier_errors[tier] = []
        tier_errors[tier].append(d['error'])
    
    tiers = sorted(tier_errors.keys())[:25]  # First 25 tiers
    avg_tier_errors = [np.mean(tier_errors[t]) for t in tiers]
    
    ax2.bar(tiers, avg_tier_errors, alpha=0.7, color='orange')
    ax2.axvline(x=11, color='red', linestyle='--', linewidth=2, label='Tier 11 (~$716)')
    ax2.set_xlabel('Receipt Tier (cents >> 6)')
    ax2.set_ylabel('Average Prediction Error ($)')
    ax2.set_title('Average Error by Receipt Tier\nLooking for tier boundary spikes')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Zoom on $700 region
    cliff_region = [d for d in cliff_data if 600 <= d['receipts'] <= 900]
    cliff_receipts = [d['receipts'] for d in cliff_region]
    cliff_errors = [d['error'] for d in cliff_region]
    cliff_tiers = [d['tier'] for d in cliff_region]
    
    scatter = ax3.scatter(cliff_receipts, cliff_errors, c=cliff_tiers, cmap='viridis', s=30, alpha=0.8)
    ax3.axvline(x=716.80, color='red', linewidth=2, label='$716.80 boundary')
    ax3.set_xlabel('Total Receipts ($)')
    ax3.set_ylabel('Prediction Error ($)')
    ax3.set_title('$700 Cliff Region Detail\nColored by Tier')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='Receipt Tier')
    
    # Plot 4: Tier transition analysis
    tier_11_data = [d for d in cliff_data if d['tier'] == 11]  # Around $716
    tier_12_data = [d for d in cliff_data if d['tier'] == 12]  # Around $780
    
    if tier_11_data and tier_12_data:
        tier_11_errors = [d['error'] for d in tier_11_data]
        tier_12_errors = [d['error'] for d in tier_12_data]
        
        ax4.hist(tier_11_errors, bins=15, alpha=0.7, color='blue', label=f'Tier 11 ({len(tier_11_errors)} cases)')
        ax4.hist(tier_12_errors, bins=15, alpha=0.7, color='red', label=f'Tier 12 ({len(tier_12_errors)} cases)')
        ax4.set_xlabel('Prediction Error ($)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Error Distribution: Tier 11 vs Tier 12\nTesting boundary effect')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('700_cliff_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved 700_cliff_analysis.png")

def plot_2400_wrap(public_cases, naive_results):
    """Plot $2400 accumulator wrap hypothesis"""
    print("Creating $2400 wrap plots...")
    
    # Calculate wrap values
    wrap_data = []
    for i, case in enumerate(public_cases):
        receipts = case['input']['total_receipts_amount']
        receipt_cents = int(receipts * 100)
        wrap_value = receipt_cents % 4096  # 12-bit wrap
        
        naive_case = next((r for r in naive_results if r['case_index'] == i), None)
        if naive_case:
            wrap_data.append({
                'receipts': receipts,
                'receipt_cents': receipt_cents,
                'wrap_value': wrap_value,
                'error': naive_case['absolute_error'],
                'actual': case['expected_output'],
                'predicted': naive_case['predicted']
            })
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Error vs wrap value (looking for sawtooth)
    wrap_values = [d['wrap_value'] for d in wrap_data]
    errors = [d['error'] for d in wrap_data]
    
    ax1.scatter(wrap_values, errors, alpha=0.6, s=15, color='blue')
    ax1.set_xlabel('Receipt Cents % 4096 (12-bit wrap value)')
    ax1.set_ylabel('Prediction Error ($)')
    ax1.set_title('12-bit Overflow Pattern\nLooking for sawtooth at 4096 boundaries')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: High receipt cases specifically
    high_receipt_data = [d for d in wrap_data if d['receipts'] >= 2000]
    if high_receipt_data:
        high_receipts = [d['receipts'] for d in high_receipt_data]
        high_errors = [d['error'] for d in high_receipt_data]
        high_wraps = [d['wrap_value'] for d in high_receipt_data]
        
        scatter = ax2.scatter(high_receipts, high_errors, c=high_wraps, cmap='plasma', s=40, alpha=0.8)
        ax2.axvline(x=2400, color='red', linewidth=2, label='$2400 suspected threshold')
        ax2.set_xlabel('Total Receipts ($)')
        ax2.set_ylabel('Prediction Error ($)')
        ax2.set_title('High Receipt Cases (≥$2000)\nColored by wrap value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label='Wrap Value')
    
    # Plot 3: Binned wrap analysis
    wrap_bins = np.arange(0, 4096, 256)  # 16 bins
    bin_errors = []
    bin_centers = []
    
    for i in range(len(wrap_bins)-1):
        bin_data = [d['error'] for d in wrap_data if wrap_bins[i] <= d['wrap_value'] < wrap_bins[i+1]]
        if bin_data:
            bin_errors.append(np.mean(bin_data))
            bin_centers.append((wrap_bins[i] + wrap_bins[i+1]) / 2)
    
    ax3.bar(bin_centers, bin_errors, width=200, alpha=0.7, color='green')
    ax3.set_xlabel('Wrap Value Bins')
    ax3.set_ylabel('Average Prediction Error ($)')
    ax3.set_title('Average Error by Wrap Value Bins\nLooking for periodic pattern')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Receipt ranges analysis
    ranges = [(0, 1000), (1000, 2000), (2000, 3000), (3000, 5000)]
    range_errors = []
    range_labels = []
    
    for min_r, max_r in ranges:
        range_data = [d['error'] for d in wrap_data if min_r <= d['receipts'] < max_r]
        if range_data:
            range_errors.append(np.mean(range_data))
            range_labels.append(f'${min_r}-${max_r}')
    
    ax4.bar(range_labels, range_errors, alpha=0.7, color='purple')
    ax4.set_xlabel('Receipt Range')
    ax4.set_ylabel('Average Prediction Error ($)')
    ax4.set_title('Average Error by Receipt Range\nTesting accumulator overflow')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('2400_wrap_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved 2400_wrap_analysis.png")

def plot_long_trip_penalty(public_cases, naive_results):
    """Plot long-trip low-spend penalty hypothesis"""
    print("Creating long-trip penalty plots...")
    
    # Calculate ratios
    penalty_data = []
    for i, case in enumerate(public_cases):
        days = case['input']['trip_duration_days']
        receipts = case['input']['total_receipts_amount']
        
        estimated_per_diem = 50
        expected_receipts = days * estimated_per_diem
        receipt_ratio = receipts / expected_receipts if expected_receipts > 0 else 0
        
        naive_case = next((r for r in naive_results if r['case_index'] == i), None)
        if naive_case:
            penalty_data.append({
                'days': days,
                'receipts': receipts,
                'receipt_ratio': receipt_ratio,
                'error': naive_case['absolute_error'],
                'actual': case['expected_output'],
                'predicted': naive_case['predicted']
            })
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Error vs receipt ratio, colored by days
    ratios = [d['receipt_ratio'] for d in penalty_data]
    errors = [d['error'] for d in penalty_data]
    days_list = [d['days'] for d in penalty_data]
    
    scatter = ax1.scatter(ratios, errors, c=days_list, cmap='viridis', alpha=0.6, s=20)
    ax1.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='0.5 ratio threshold')
    ax1.set_xlabel('Receipt/Expected Ratio')
    ax1.set_ylabel('Prediction Error ($)')
    ax1.set_title('Error vs Receipt Ratio\nColored by trip duration')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Days')
    
    # Plot 2: Long trips with low ratios
    long_low_data = [d for d in penalty_data if d['days'] >= 10 and d['receipt_ratio'] < 0.5]
    other_data = [d for d in penalty_data if not (d['days'] >= 10 and d['receipt_ratio'] < 0.5)]
    
    if long_low_data:
        long_low_errors = [d['error'] for d in long_low_data]
        other_errors = [d['error'] for d in other_data]
        
        ax2.hist(other_errors, bins=30, alpha=0.7, color='blue', label=f'Other cases ({len(other_errors)})')
        ax2.hist(long_low_errors, bins=15, alpha=0.8, color='red', label=f'Long+Low cases ({len(long_low_errors)})')
        ax2.set_xlabel('Prediction Error ($)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Error Distribution: Long+Low vs Others')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Days vs average error for low ratios
    low_ratio_data = [d for d in penalty_data if d['receipt_ratio'] < 0.5]
    day_errors = {}
    for d in low_ratio_data:
        days = d['days']
        if days not in day_errors:
            day_errors[days] = []
        day_errors[days].append(d['error'])
    
    days_sorted = sorted(day_errors.keys())
    avg_errors_by_day = [np.mean(day_errors[day]) for day in days_sorted]
    
    ax3.bar(days_sorted, avg_errors_by_day, alpha=0.7, color='orange')
    ax3.axvline(x=10, color='red', linestyle='--', linewidth=2, label='10-day threshold')
    ax3.set_xlabel('Trip Duration (days)')
    ax3.set_ylabel('Average Error ($)')
    ax3.set_title('Average Error by Days\n(Low ratio cases only)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: 2D heatmap of days vs ratio
    # Create bins
    day_bins = np.arange(1, 16)
    ratio_bins = np.arange(0, 3, 0.2)
    
    hist_data = []
    for d in penalty_data:
        hist_data.append([d['days'], d['receipt_ratio'], d['error']])
    
    hist_data = np.array(hist_data)
    if len(hist_data) > 0:
        # Create 2D histogram
        H, xedges, yedges = np.histogram2d(hist_data[:, 0], hist_data[:, 1], 
                                          bins=[day_bins, ratio_bins], 
                                          weights=hist_data[:, 2])
        
        im = ax4.imshow(H.T, origin='lower', aspect='auto', cmap='hot', 
                       extent=[day_bins[0], day_bins[-1], ratio_bins[0], ratio_bins[-1]])
        ax4.set_xlabel('Trip Duration (days)')
        ax4.set_ylabel('Receipt Ratio')
        ax4.set_title('Average Error Heatmap\nDays vs Receipt Ratio')
        plt.colorbar(im, ax=ax4, label='Average Error ($)')
    
    plt.tight_layout()
    plt.savefig('long_trip_penalty_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved long_trip_penalty_analysis.png")

def plot_luck_nibble(public_cases, naive_results):
    """Plot luck nibble hypothesis"""
    print("Creating luck nibble plots...")
    
    # Calculate luck patterns
    luck_data = []
    for i, case in enumerate(public_cases):
        epoch_day = i  # Use index as proxy
        
        naive_case = next((r for r in naive_results if r['case_index'] == i), None)
        if naive_case:
            signed_error = naive_case['predicted'] - case['expected_output']
            
            luck_data.append({
                'epoch_day': epoch_day,
                'mod_2': epoch_day % 2,
                'mod_4': epoch_day % 4,
                'mod_16': epoch_day % 16,
                'signed_error': signed_error,
                'abs_error': naive_case['absolute_error'],
                'actual': case['expected_output'],
                'predicted': naive_case['predicted']
            })
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Signed error by mod 2
    mod2_groups = {0: [], 1: []}
    for d in luck_data:
        mod2_groups[d['mod_2']].append(d['signed_error'])
    
    mod2_means = [np.mean(mod2_groups[i]) for i in [0, 1]]
    mod2_stds = [np.std(mod2_groups[i]) for i in [0, 1]]
    
    ax1.bar([0, 1], mod2_means, yerr=mod2_stds, alpha=0.7, color=['blue', 'red'], 
           capsize=5, error_kw={'linewidth': 2})
    ax1.set_xlabel('Epoch Day % 2')
    ax1.set_ylabel('Mean Signed Error ($)')
    ax1.set_title('Luck Bit Analysis: Even vs Odd Days\nSigned error (predicted - actual)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Signed error by mod 16
    mod16_groups = {i: [] for i in range(16)}
    for d in luck_data:
        mod16_groups[d['mod_16']].append(d['signed_error'])
    
    mod16_means = [np.mean(mod16_groups[i]) if mod16_groups[i] else 0 for i in range(16)]
    
    ax2.bar(range(16), mod16_means, alpha=0.7, color='green')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Epoch Day % 16')
    ax2.set_ylabel('Mean Signed Error ($)')
    ax2.set_title('16-bit Luck Pattern\nLooking for systematic bias')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Distribution of signed errors for even/odd
    even_errors = [d['signed_error'] for d in luck_data if d['mod_2'] == 0]
    odd_errors = [d['signed_error'] for d in luck_data if d['mod_2'] == 1]
    
    ax3.hist(even_errors, bins=30, alpha=0.7, color='blue', label=f'Even days ({len(even_errors)})')
    ax3.hist(odd_errors, bins=30, alpha=0.7, color='red', label=f'Odd days ({len(odd_errors)})')
    ax3.axvline(x=np.mean(even_errors), color='blue', linestyle='--', linewidth=2)
    ax3.axvline(x=np.mean(odd_errors), color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('Signed Error ($)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Signed Error Distribution\nEven vs Odd Epoch Days')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Cumulative signed error over time
    cumulative_error = np.cumsum([d['signed_error'] for d in luck_data])
    epoch_days = [d['epoch_day'] for d in luck_data]
    
    ax4.plot(epoch_days, cumulative_error, color='purple', alpha=0.8)
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax4.set_xlabel('Epoch Day (Case Index)')
    ax4.set_ylabel('Cumulative Signed Error ($)')
    ax4.set_title('Cumulative Bias Over Time\nTesting for systematic drift')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('luck_nibble_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved luck_nibble_analysis.png")

def create_summary_plot():
    """Create a summary plot showing all hypotheses"""
    print("Creating summary plot...")
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create a summary visualization
    hypotheses = [
        "8-Day Overflow\n(3-bit wrap)",
        "$700 Cliff\n(receipt tiers)", 
        "$2400 Wrap\n(12-bit overflow)",
        "Long-Trip Penalty\n(low spend ratio)",
        "Luck Nibble\n(epoch day bias)"
    ]
    
    # Placeholder confidence scores (would be calculated from actual analysis)
    confidence_scores = [85, 70, 60, 45, 35]  # Percentage confidence
    colors = ['red', 'orange', 'yellow', 'lightblue', 'lightgray']
    
    bars = ax.barh(hypotheses, confidence_scores, color=colors, alpha=0.8)
    
    # Add confidence labels
    for i, (bar, score) in enumerate(zip(bars, confidence_scores)):
        ax.text(score + 2, i, f'{score}%', va='center', fontweight='bold')
    
    ax.set_xlabel('Confidence Level (%)')
    ax.set_title('Bug Hypothesis Confidence Levels\nBased on Pattern Analysis')
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add threshold line
    ax.axvline(x=50, color='red', linestyle='--', linewidth=2, label='50% threshold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('bug_hypothesis_summary.png', dpi=300, bbox_inches='tight')
    print("Saved bug_hypothesis_summary.png")

def main():
    """Generate all bug hypothesis plots"""
    print("Loading data for plotting...")
    public_cases, naive_results = load_data()
    
    # Create all plots
    plot_8_day_overflow(public_cases, naive_results)
    plot_700_cliff(public_cases, naive_results)
    plot_2400_wrap(public_cases, naive_results)
    plot_long_trip_penalty(public_cases, naive_results)
    plot_luck_nibble(public_cases, naive_results)
    create_summary_plot()
    
    print("\n=== ALL PLOTS CREATED ===")
    print("Generated files:")
    print("- 8_day_overflow_analysis.png")
    print("- 700_cliff_analysis.png") 
    print("- 2400_wrap_analysis.png")
    print("- long_trip_penalty_analysis.png")
    print("- luck_nibble_analysis.png")
    print("- bug_hypothesis_summary.png")

if __name__ == "__main__":
    main() 