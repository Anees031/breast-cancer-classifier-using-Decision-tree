"""
Model Comparison Script
=======================
Compare Decision Tree with different configurations and other algorithms.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')


def compare_models():
    """Compare different machine learning models."""
    print("\n" + "="*70)
    print("MODEL COMPARISON FOR BREAST CANCER CLASSIFICATION")
    print("="*70)
    
    # Load data
    print("\n✓ Loading data...")
    df = pd.read_csv('data.csv')
    
    # Preprocess
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    
    le = LabelEncoder()
    df['diagnosis'] = le.fit_transform(df['diagnosis'])
    
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("✓ Data preprocessed")
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Test samples: {len(X_test)}")
    
    # Define models
    models = {
        'Decision Tree (depth=5)': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Decision Tree (depth=10)': DecisionTreeClassifier(max_depth=10, random_state=42),
        'Decision Tree (optimized)': DecisionTreeClassifier(
            max_depth=7, min_samples_split=10, min_samples_leaf=5, random_state=42
        ),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
    }
    
    print("\n" + "="*70)
    print("TRAINING AND EVALUATING MODELS")
    print("="*70)
    
    results = []
    
    for name, model in models.items():
        print(f"\n✓ Training {name}...")
        
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        cv_mean = cv_scores.mean()
        
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'CV Score': cv_mean
        })
        
        print(f"  - Accuracy: {accuracy:.4f}")
        print(f"  - CV Score: {cv_mean:.4f}")
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Accuracy', ascending=False)
    
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print("\n" + results_df.to_string(index=False))
    
    # Find best model
    best_model = results_df.iloc[0]
    print("\n" + "="*70)
    print("BEST MODEL")
    print("="*70)
    print(f"\n✓ Model: {best_model['Model']}")
    print(f"✓ Accuracy: {best_model['Accuracy']:.4f} ({best_model['Accuracy']*100:.2f}%)")
    print(f"✓ Precision: {best_model['Precision']:.4f}")
    print(f"✓ Recall: {best_model['Recall']:.4f}")
    print(f"✓ F1-Score: {best_model['F1-Score']:.4f}")
    print(f"✓ CV Score: {best_model['CV Score']:.4f}")
    
    print("\n" + "="*70 + "\n")
    
    return results_df


if __name__ == "__main__":
    compare_models()
