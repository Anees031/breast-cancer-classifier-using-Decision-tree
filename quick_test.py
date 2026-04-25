"""
Quick Test Script for Breast Cancer Classifier
===============================================
A simplified script to quickly test the classifier without full optimization.
"""

from breast_cancer_classifier import BreastCancerClassifier

def quick_test():
    """Run a quick test of the classifier."""
    print("\n" + "="*70)
    print("QUICK TEST - BREAST CANCER CLASSIFIER")
    print("="*70)
    
    # Initialize and run basic pipeline
    classifier = BreastCancerClassifier('data.csv')
    
    # Load and preprocess
    classifier.load_data()
    classifier.preprocess_data(test_size=0.2, random_state=42)
    
    # Train with good default parameters
    classifier.train_model(max_depth=7, min_samples_split=10, min_samples_leaf=5)
    
    # Evaluate
    results = classifier.evaluate_model(use_best_model=False)
    
    print("\n" + "="*70)
    print("QUICK TEST COMPLETED!")
    print("="*70)
    print(f"\nModel Accuracy: {results['accuracy']*100:.2f}%")
    print("\nFor full optimization and visualizations, run: python breast_cancer_classifier.py")
    print("="*70 + "\n")

if __name__ == "__main__":
    quick_test()
