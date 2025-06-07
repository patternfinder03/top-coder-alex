import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import seaborn as sns

def load_and_prepare_data():
    """Load public cases and prepare data for clustering"""
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    # Extract features
    inputs = []
    outputs = []
    case_indices = []
    
    for i, case in enumerate(data):
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        reimbursement = case['expected_output']
        
        inputs.append([days, miles, receipts])
        outputs.append(reimbursement)
        case_indices.append(i)
    
    return np.array(inputs), np.array(outputs), case_indices

def cluster_analysis():
    """Perform clustering analysis on reimbursement data"""
    
    print("Loading data...")
    X_inputs, y_outputs, case_indices = load_and_prepare_data()
    
    # Create combined feature matrix (inputs + normalized output)
    # Normalize output to similar scale as inputs
    output_normalized = y_outputs / 100  # Scale down to hundreds
    X_combined = np.column_stack([X_inputs, output_normalized])
    
    print(f"Data shape: {X_combined.shape}")
    print(f"Features: days, miles, receipts, reimbursement/100")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)
    
    # Try different clustering methods
    print("\nTesting clustering methods...")
    
    # K-means with different k values
    k_scores = []
    k_range = range(3, 15)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        k_scores.append(score)
        print(f"K-means k={k}: silhouette score = {score:.3f}")
    
    # Find optimal k
    best_k = k_range[np.argmax(k_scores)]
    print(f"\nBest k={best_k} with silhouette score {max(k_scores):.3f}")
    
    # Perform clustering with best k
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    
    # Also try DBSCAN for outlier detection
    dbscan = DBSCAN(eps=0.8, min_samples=10)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_outliers_dbscan = list(dbscan_labels).count(-1)
    
    print(f"\nDBSCAN: {n_clusters_dbscan} clusters, {n_outliers_dbscan} outliers")
    
    # PCA for visualization
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"\nPCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.3f}")
    
    # Create comprehensive plots
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: 3D scatter with K-means clusters
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    scatter1 = ax1.scatter(X_inputs[:, 0], X_inputs[:, 1], y_outputs, 
                          c=kmeans_labels, cmap='tab10', alpha=0.6, s=20)
    ax1.set_xlabel('Trip Duration (days)')
    ax1.set_ylabel('Miles Traveled')
    ax1.set_zlabel('Reimbursement ($)')
    ax1.set_title(f'K-means Clustering (k={best_k})\nInput-Output Space')
    
    # Plot 2: 2D scatter days vs reimbursement
    ax2 = fig.add_subplot(2, 3, 2)
    scatter2 = ax2.scatter(X_inputs[:, 0], y_outputs, c=kmeans_labels, cmap='tab10', alpha=0.6, s=20)
    ax2.set_xlabel('Trip Duration (days)')
    ax2.set_ylabel('Reimbursement ($)')
    ax2.set_title('Days vs Reimbursement\nColored by Cluster')
    
    # Plot 3: 2D scatter miles vs reimbursement
    ax3 = fig.add_subplot(2, 3, 3)
    scatter3 = ax3.scatter(X_inputs[:, 1], y_outputs, c=kmeans_labels, cmap='tab10', alpha=0.6, s=20)
    ax3.set_xlabel('Miles Traveled')
    ax3.set_ylabel('Reimbursement ($)')
    ax3.set_title('Miles vs Reimbursement\nColored by Cluster')
    
    # Plot 4: 2D scatter receipts vs reimbursement
    ax4 = fig.add_subplot(2, 3, 4)
    scatter4 = ax4.scatter(X_inputs[:, 2], y_outputs, c=kmeans_labels, cmap='tab10', alpha=0.6, s=20)
    ax4.set_xlabel('Total Receipts ($)')
    ax4.set_ylabel('Reimbursement ($)')
    ax4.set_title('Receipts vs Reimbursement\nColored by Cluster')
    
    # Plot 5: PCA visualization
    ax5 = fig.add_subplot(2, 3, 5)
    scatter5 = ax5.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='tab10', alpha=0.6, s=20)
    ax5.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax5.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax5.set_title('PCA Visualization\nFirst 2 Principal Components')
    
    # Plot 6: DBSCAN outliers
    ax6 = fig.add_subplot(2, 3, 6)
    outlier_mask = dbscan_labels == -1
    # Plot normal points
    ax6.scatter(X_pca[~outlier_mask, 0], X_pca[~outlier_mask, 1], 
               c=dbscan_labels[~outlier_mask], cmap='tab10', alpha=0.6, s=20, label='Normal')
    # Plot outliers
    if np.any(outlier_mask):
        ax6.scatter(X_pca[outlier_mask, 0], X_pca[outlier_mask, 1], 
                   c='red', s=50, marker='x', label='Outliers', alpha=0.8)
    ax6.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax6.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax6.set_title(f'DBSCAN Outlier Detection\n{n_outliers_dbscan} outliers found')
    ax6.legend()
    
    plt.tight_layout()
    plt.savefig('cluster_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved plot as 'cluster_analysis.png'")
    
    # Identify and analyze outliers
    print("\n=== OUTLIER ANALYSIS ===")
    
    # Method 1: DBSCAN outliers
    if n_outliers_dbscan > 0:
        outlier_indices = np.where(dbscan_labels == -1)[0]
        print(f"\nDBSCAN identified {len(outlier_indices)} outliers:")
        print("Case | Days | Miles | Receipts | Reimbursement")
        print("-" * 50)
        
        for idx in outlier_indices[:20]:  # Show top 20
            days, miles, receipts = X_inputs[idx]
            reimbursement = y_outputs[idx]
            case_idx = case_indices[idx]
            print(f"{case_idx:4d} | {int(days):4d} | {int(miles):5d} | ${receipts:8.2f} | ${reimbursement:8.2f}")
    
    # Method 2: Distance from cluster centers (K-means)
    cluster_centers = kmeans.cluster_centers_
    distances = []
    for i, point in enumerate(X_scaled):
        cluster_id = kmeans_labels[i]
        center = cluster_centers[cluster_id]
        dist = np.linalg.norm(point - center)
        distances.append((i, dist))
    
    # Sort by distance (farthest from center = most outlier-like)
    distances.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 20 points farthest from K-means cluster centers:")
    print("Case | Days | Miles | Receipts | Reimbursement | Distance")
    print("-" * 60)
    
    for i, (idx, dist) in enumerate(distances[:20]):
        days, miles, receipts = X_inputs[idx]
        reimbursement = y_outputs[idx]
        case_idx = case_indices[idx]
        print(f"{case_idx:4d} | {int(days):4d} | {int(miles):5d} | ${receipts:8.2f} | ${reimbursement:8.2f} | {dist:6.3f}")
    
    # Analyze cluster characteristics
    print(f"\n=== CLUSTER ANALYSIS ===")
    for cluster_id in range(best_k):
        mask = kmeans_labels == cluster_id
        cluster_points = X_inputs[mask]
        cluster_outputs = y_outputs[mask]
        
        print(f"\nCluster {cluster_id} ({np.sum(mask)} points):")
        print(f"  Days: {np.mean(cluster_points[:, 0]):.1f} ± {np.std(cluster_points[:, 0]):.1f}")
        print(f"  Miles: {np.mean(cluster_points[:, 1]):.0f} ± {np.std(cluster_points[:, 1]):.0f}")
        print(f"  Receipts: ${np.mean(cluster_points[:, 2]):.2f} ± ${np.std(cluster_points[:, 2]):.2f}")
        print(f"  Reimbursement: ${np.mean(cluster_outputs):.2f} ± ${np.std(cluster_outputs):.2f}")
    
    # Save outlier data
    outlier_data = []
    
    # Add DBSCAN outliers
    if n_outliers_dbscan > 0:
        for idx in outlier_indices:
            days, miles, receipts = X_inputs[idx]
            reimbursement = y_outputs[idx]
            case_idx = case_indices[idx]
            outlier_data.append({
                'case_index': int(case_idx),
                'method': 'DBSCAN',
                'days': int(days),
                'miles': int(miles), 
                'receipts': float(receipts),
                'reimbursement': float(reimbursement),
                'outlier_score': 1.0  # Binary for DBSCAN
            })
    
    # Add K-means distant points
    for i, (idx, dist) in enumerate(distances[:50]):  # Top 50 most distant
        days, miles, receipts = X_inputs[idx]
        reimbursement = y_outputs[idx]
        case_idx = case_indices[idx]
        
        # Skip if already added as DBSCAN outlier
        if any(od['case_index'] == case_idx and od['method'] == 'DBSCAN' for od in outlier_data):
            continue
            
        outlier_data.append({
            'case_index': int(case_idx),
            'method': 'K-means_distance',
            'days': int(days),
            'miles': int(miles),
            'receipts': float(receipts),
            'reimbursement': float(reimbursement),
            'outlier_score': float(dist)
        })
    
    # Save outliers to JSON
    with open('clustering_outliers.json', 'w') as f:
        json.dump(outlier_data, f, indent=2)
    
    print(f"\nSaved {len(outlier_data)} outliers to 'clustering_outliers.json'")
    
    return outlier_data

if __name__ == "__main__":
    outliers = cluster_analysis() 