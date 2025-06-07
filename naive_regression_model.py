import json
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd

def load_data():
    """Load the public cases data"""
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    inputs = []
    outputs = []
    
    for case in data:
        inputs.append([
            case['input']['trip_duration_days'],
            case['input']['miles_traveled'],
            case['input']['total_receipts_amount']
        ])
        outputs.append(case['expected_output'])
    
    return np.array(inputs), np.array(outputs)

def create_features(X):
    """Create comprehensive feature engineering"""
    days, miles, receipts = X[:, 0], X[:, 1], X[:, 2]
    
    features = []
    feature_names = []
    
    # Original features
    features.extend([days, miles, receipts])
    feature_names.extend(['days', 'miles', 'receipts'])
    
    # Polynomial features
    features.extend([days**2, miles**2, receipts**2])
    feature_names.extend(['days_sq', 'miles_sq', 'receipts_sq'])
    
    # Interactions
    features.extend([days * miles, days * receipts, miles * receipts])
    feature_names.extend(['days_miles', 'days_receipts', 'miles_receipts'])
    
    # Three-way interaction
    features.append(days * miles * receipts)
    feature_names.append('days_miles_receipts')
    
    # Log transformations (adding small constant to avoid log(0))
    features.extend([
        np.log(days + 1),
        np.log(miles + 1),
        np.log(receipts + 1)
    ])
    feature_names.extend(['log_days', 'log_miles', 'log_receipts'])
    
    # Square root transformations
    features.extend([
        np.sqrt(days),
        np.sqrt(miles),
        np.sqrt(receipts)
    ])
    feature_names.extend(['sqrt_days', 'sqrt_miles', 'sqrt_receipts'])
    
    # Per-day features
    features.extend([
        miles / days,
        receipts / days
    ])
    feature_names.extend(['miles_per_day', 'receipts_per_day'])
    
    # Ratio features
    features.extend([
        receipts / miles,
        miles / receipts,
        (receipts + miles) / days
    ])
    feature_names.extend(['receipts_per_mile', 'miles_per_receipt', 'total_per_day'])
    
    # Binned features (potential tier indicators)
    # Days bins
    day_bins = np.digitize(days, [1, 4, 8, 12, 16, 20, 24, 28])
    features.append(day_bins)
    feature_names.append('day_bins')
    
    # Miles bins
    mile_bins = np.digitize(miles, [0, 200, 500, 800, 1000, 1200, 1500])
    features.append(mile_bins)
    feature_names.append('mile_bins')
    
    # Receipt bins
    receipt_bins = np.digitize(receipts, [0, 500, 700, 1000, 1500, 2000, 2500])
    features.append(receipt_bins)
    feature_names.append('receipt_bins')
    
    # Modular arithmetic features (to catch potential overflow patterns)
    features.extend([
        days % 8,
        days % 16,
        days % 32,
        miles % 1024,
        miles % 2048,
        miles % 4096,
        receipts % 512,
        receipts % 1024,
        receipts % 2048
    ])
    feature_names.extend([
        'days_mod_8', 'days_mod_16', 'days_mod_32',
        'miles_mod_1024', 'miles_mod_2048', 'miles_mod_4096',
        'receipts_mod_512', 'receipts_mod_1024', 'receipts_mod_2048'
    ])
    
    return np.column_stack(features), feature_names

class NaiveReimbursementModel:
    def __init__(self):
        self.models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boost': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'polynomial': Pipeline([
                ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                ('scaler', StandardScaler()),
                ('linear', LinearRegression())
            ])
        }
        self.best_model = None
        self.best_score = -np.inf
        self.feature_names = None
        self.scaler = StandardScaler()
        
    def fit(self, X, y):
        """Fit multiple models and select the best one"""
        X_features, self.feature_names = create_features(X)
        X_scaled = self.scaler.fit_transform(X_features)
        
        print("Training multiple models...")
        scores = {}
        
        for name, model in self.models.items():
            try:
                # Cross-validation score
                cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_absolute_error')
                scores[name] = -cv_scores.mean()
                print(f"{name}: MAE = {scores[name]:.2f} (+/- {cv_scores.std() * 2:.2f})")
                
                # Fit the full model
                model.fit(X_scaled, y)
                
                # Check if this is the best model
                if -cv_scores.mean() < scores.get(self.best_model or 'dummy', np.inf):
                    self.best_model = name
                    self.best_score = -cv_scores.mean()
                    
            except Exception as e:
                print(f"Error training {name}: {e}")
                
        print(f"\nBest model: {self.best_model} with MAE: {self.best_score:.2f}")
        
        # Analyze feature importance for tree-based models
        if self.best_model in ['random_forest', 'gradient_boost']:
            self.analyze_feature_importance(X_scaled, y)
            
        return self
    
    def analyze_feature_importance(self, X, y):
        """Analyze feature importance for the best model"""
        if self.best_model in ['random_forest', 'gradient_boost']:
            model = self.models[self.best_model]
            importances = model.feature_importances_
            
            # Get top 15 most important features
            indices = np.argsort(importances)[::-1][:15]
            
            print(f"\nTop 15 Feature Importances ({self.best_model}):")
            for i, idx in enumerate(indices):
                print(f"{i+1:2d}. {self.feature_names[idx]:20s}: {importances[idx]:.4f}")
    
    def predict(self, X):
        """Make predictions using the best model"""
        X_features, _ = create_features(X)
        X_scaled = self.scaler.transform(X_features)
        return self.models[self.best_model].predict(X_scaled)
    
    def evaluate(self, X, y):
        """Evaluate the model performance"""
        predictions = self.predict(X)
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)
        
        print(f"\nModel Evaluation:")
        print(f"MAE: {mae:.2f}")
        print(f"R²: {r2:.4f}")
        
        # Find worst predictions
        errors = np.abs(predictions - y)
        worst_indices = np.argsort(errors)[::-1][:10]
        
        print(f"\nWorst 10 Predictions:")
        for i, idx in enumerate(worst_indices):
            days, miles, receipts = X[idx]
            actual, pred = y[idx], predictions[idx]
            error = errors[idx]
            print(f"{i+1:2d}. Days:{days:2.0f} Miles:{miles:4.0f} Receipts:${receipts:7.2f} | "
                  f"Actual:${actual:7.2f} Pred:${pred:7.2f} Error:${error:7.2f}")
        
        return mae, r2, predictions

def main():
    # Load data
    print("Loading data...")
    X, y = load_data()
    print(f"Loaded {len(X)} cases")
    
    # Create and train model
    model = NaiveReimbursementModel()
    model.fit(X, y)
    
    # Evaluate
    mae, r2, predictions = model.evaluate(X, y)
    
    # Create detailed analysis of errors
    errors = np.abs(predictions - y)
    relative_errors = errors / y
    
    print(f"\nError Analysis:")
    print(f"Mean Error: ${np.mean(errors):.2f}")
    print(f"Median Error: ${np.median(errors):.2f}")
    print(f"Max Error: ${np.max(errors):.2f}")
    print(f"90th percentile Error: ${np.percentile(errors, 90):.2f}")
    print(f"Mean Relative Error: {np.mean(relative_errors):.1%}")
    
    # Save predictions for further analysis
    results = []
    for i in range(len(X)):
        days, miles, receipts = X[i]
        results.append({
            'case_index': i,
            'days': int(days),
            'miles': int(miles),
            'receipts': float(receipts),
            'actual': float(y[i]),
            'predicted': float(predictions[i]),
            'absolute_error': float(errors[i]),
            'relative_error': float(relative_errors[i])
        })
    
    # Sort by absolute error (worst first)
    results.sort(key=lambda x: x['absolute_error'], reverse=True)
    
    with open('naive_model_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to 'naive_model_results.json'")
    print("This naive model can now be compared against the actual algorithm to find bug patterns!")

if __name__ == "__main__":
    main() 