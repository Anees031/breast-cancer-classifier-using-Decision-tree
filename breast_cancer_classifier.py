"""
Breast Cancer Classification using Decision Tree
==================================================
A professional machine learning project for classifying breast cancer tumors
as Malignant (M) or Benign (B) using Decision Tree algorithm.

Author: ML Project
Date: 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


class BreastCancerClassifier:
    """
    A comprehensive breast cancer classification system using Decision Tree.
    """
    
    def __init__(self, data_path='data.csv'):
        """Initialize the classifier with data path."""
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.best_model = None
        
    def load_data(self):
        """Load and perform initial data exploration."""
        print("=" * 70)
        print("LOADING BREAST CANCER DATASET")
        print("=" * 70)
        
        self.df = pd.read_csv(self.data_path)
        
        print(f"\n✓ Dataset loaded successfully!")
        print(f"  - Total samples: {len(self.df)}")
        print(f"  - Total features: {len(self.df.columns)}")
        
        # Display basic info
        print("\n" + "-" * 70)
        print("DATASET OVERVIEW")
        print("-" * 70)
        print(self.df.head())
        
        print("\n" + "-" * 70)
        print("DATASET INFORMATION")
        print("-" * 70)
        print(self.df.info())
        
        return self
    
    def explore_data(self):
        """Perform exploratory data analysis."""
        print("\n" + "=" * 70)
        print("EXPLORATORY DATA ANALYSIS")
        print("=" * 70)
        
        # Check for missing values
        print("\n✓ Missing Values:")
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("  No missing values found!")
        else:
            print(missing[missing > 0])
        
        # Check diagnosis distribution
        print("\n✓ Diagnosis Distribution:")
        diagnosis_counts = self.df['diagnosis'].value_counts()
        print(diagnosis_counts)
        print(f"\n  Malignant (M): {diagnosis_counts.get('M', 0)} ({diagnosis_counts.get('M', 0)/len(self.df)*100:.2f}%)")
        print(f"  Benign (B): {diagnosis_counts.get('B', 0)} ({diagnosis_counts.get('B', 0)/len(self.df)*100:.2f}%)")
        
        # Statistical summary
        print("\n✓ Statistical Summary:")
        print(self.df.describe())
        
        return self
    
    def preprocess_data(self, test_size=0.2, random_state=42):
        """
        Preprocess the data: handle missing values, encode labels, 
        split data, and scale features.
        """
        print("\n" + "=" * 70)
        print("DATA PREPROCESSING")
        print("=" * 70)
        
        # Drop ID column as it's not useful for prediction
        if 'id' in self.df.columns:
            self.df = self.df.drop('id', axis=1)
            print("\n✓ Dropped 'id' column")
        
        # Encode diagnosis labels (M=1, B=0)
        self.df['diagnosis'] = self.label_encoder.fit_transform(self.df['diagnosis'])
        print("✓ Encoded diagnosis labels (M=1, B=0)")
        
        # Separate features and target
        X = self.df.drop('diagnosis', axis=1)
        y = self.df['diagnosis']
        
        print(f"\n✓ Features shape: {X.shape}")
        print(f"✓ Target shape: {y.shape}")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\n✓ Data split completed:")
        print(f"  - Training set: {len(self.X_train)} samples ({(1-test_size)*100:.0f}%)")
        print(f"  - Test set: {len(self.X_test)} samples ({test_size*100:.0f}%)")
        
        # Scale features for better performance
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print("✓ Features scaled using StandardScaler")
        
        return self
    
    def train_model(self, max_depth=5, min_samples_split=10, min_samples_leaf=5):
        """Train the Decision Tree model with specified parameters."""
        print("\n" + "=" * 70)
        print("MODEL TRAINING")
        print("=" * 70)
        
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            criterion='gini'
        )
        
        print(f"\n✓ Training Decision Tree Classifier...")
        print(f"  - Max depth: {max_depth}")
        print(f"  - Min samples split: {min_samples_split}")
        print(f"  - Min samples leaf: {min_samples_leaf}")
        
        self.model.fit(self.X_train, self.y_train)
        print("\n✓ Model training completed!")
        
        return self
    
    def optimize_model(self):
        """Optimize model using GridSearchCV for best hyperparameters."""
        print("\n" + "=" * 70)
        print("HYPERPARAMETER OPTIMIZATION")
        print("=" * 70)
        
        param_grid = {
            'max_depth': [3, 5, 7, 10, 15, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 5, 10],
            'criterion': ['gini', 'entropy']
        }
        
        print("\n✓ Performing Grid Search with Cross-Validation...")
        print(f"  - Total combinations: {len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf']) * len(param_grid['criterion'])}")
        
        grid_search = GridSearchCV(
            DecisionTreeClassifier(random_state=42),
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        self.best_model = grid_search.best_estimator_
        
        print("\n✓ Optimization completed!")
        print(f"\n  Best Parameters:")
        for param, value in grid_search.best_params_.items():
            print(f"    - {param}: {value}")
        print(f"\n  Best Cross-Validation Score: {grid_search.best_score_:.4f}")
        
        return self
    
    def evaluate_model(self, use_best_model=True):
        """Evaluate the model performance on test data."""
        print("\n" + "=" * 70)
        print("MODEL EVALUATION")
        print("=" * 70)
        
        model = self.best_model if use_best_model and self.best_model else self.model
        
        # Predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        print("\n✓ Performance Metrics:")
        print(f"  - Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  - Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"  - Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"  - F1-Score:  {f1:.4f}")
        print(f"  - ROC-AUC:   {roc_auc:.4f}")
        
        # Cross-validation score
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
        print(f"\n✓ Cross-Validation Scores:")
        print(f"  - Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Classification report
        print("\n✓ Detailed Classification Report:")
        print(classification_report(self.y_test, y_pred, 
                                   target_names=['Benign (B)', 'Malignant (M)']))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print("✓ Confusion Matrix:")
        print(f"  True Negatives:  {cm[0][0]}")
        print(f"  False Positives: {cm[0][1]}")
        print(f"  False Negatives: {cm[1][0]}")
        print(f"  True Positives:  {cm[1][1]}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def visualize_results(self, results):
        """Create comprehensive visualizations of the results."""
        print("\n" + "=" * 70)
        print("GENERATING VISUALIZATIONS")
        print("=" * 70)
        
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Confusion Matrix
        ax1 = plt.subplot(2, 3, 1)
        sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Benign', 'Malignant'],
                   yticklabels=['Benign', 'Malignant'])
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # 2. ROC Curve
        ax2 = plt.subplot(2, 3, 2)
        fpr, tpr, _ = roc_curve(self.y_test, results['y_pred_proba'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # 3. Performance Metrics Bar Chart
        ax3 = plt.subplot(2, 3, 3)
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        values = [results['accuracy'], results['precision'], results['recall'], 
                 results['f1_score'], results['roc_auc']]
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
        bars = plt.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
        plt.ylim([0, 1.1])
        plt.title('Performance Metrics', fontsize=14, fontweight='bold')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontweight='bold')
        
        # 4. Feature Importance
        ax4 = plt.subplot(2, 3, 4)
        model = self.best_model if self.best_model else self.model
        feature_names = [col for col in self.df.columns if col != 'diagnosis']
        importances = model.feature_importances_
        indices = np.argsort(importances)[-10:]  # Top 10 features
        
        plt.barh(range(len(indices)), importances[indices], color='skyblue', edgecolor='black')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title('Top 10 Most Important Features', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        
        # 5. Diagnosis Distribution
        ax5 = plt.subplot(2, 3, 5)
        diagnosis_counts = pd.Series(self.y_train).value_counts()
        labels = ['Benign (B)', 'Malignant (M)']
        colors_pie = ['#3498db', '#e74c3c']
        explode = (0.05, 0.05)
        
        plt.pie(diagnosis_counts.values, labels=labels, autopct='%1.1f%%',
               colors=colors_pie, explode=explode, shadow=True, startangle=90)
        plt.title('Training Data Distribution', fontsize=14, fontweight='bold')
        
        # 6. Decision Tree Visualization (simplified)
        ax6 = plt.subplot(2, 3, 6)
        plot_tree(model, max_depth=3, filled=True, feature_names=feature_names,
                 class_names=['Benign', 'Malignant'], fontsize=8)
        plt.title('Decision Tree Structure (Depth=3)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('breast_cancer_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualizations saved as 'breast_cancer_analysis.png'")
        plt.show()
        
        return self
    
    def predict_new_sample(self, sample_data):
        """Make prediction on new sample data."""
        model = self.best_model if self.best_model else self.model
        
        # Scale the sample
        sample_scaled = self.scaler.transform([sample_data])
        
        # Predict
        prediction = model.predict(sample_scaled)[0]
        probability = model.predict_proba(sample_scaled)[0]
        
        diagnosis = 'Malignant (M)' if prediction == 1 else 'Benign (B)'
        confidence = probability[prediction] * 100
        
        print(f"\n✓ Prediction: {diagnosis}")
        print(f"✓ Confidence: {confidence:.2f}%")
        print(f"✓ Probabilities: Benign={probability[0]:.4f}, Malignant={probability[1]:.4f}")
        
        return diagnosis, confidence


def main():
    """Main execution function."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 10 + "BREAST CANCER CLASSIFICATION PROJECT" + " " * 21 + "║")
    print("║" + " " * 15 + "Using Decision Tree Algorithm" + " " * 24 + "║")
    print("╚" + "═" * 68 + "╝")
    
    # Initialize classifier
    classifier = BreastCancerClassifier('data.csv')
    
    # Execute pipeline
    classifier.load_data()
    classifier.explore_data()
    classifier.preprocess_data(test_size=0.2, random_state=42)
    
    # Train basic model
    classifier.train_model(max_depth=5, min_samples_split=10, min_samples_leaf=5)
    
    # Optimize model
    classifier.optimize_model()
    
    # Evaluate optimized model
    results = classifier.evaluate_model(use_best_model=True)
    
    # Visualize results
    classifier.visualize_results(results)
    
    print("\n" + "=" * 70)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\n✓ Model trained and optimized")
    print("✓ Evaluation metrics calculated")
    print("✓ Visualizations generated")
    print(f"\n✓ Final Model Accuracy: {results['accuracy']*100:.2f}%")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
