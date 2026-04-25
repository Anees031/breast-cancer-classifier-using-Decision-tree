# 🏥 Breast Cancer Classification Project

A professional machine learning project for classifying breast cancer tumors as **Malignant (M)** or **Benign (B)** using Decision Tree algorithm with comprehensive data analysis, model optimization, and visualization.

## 📋 Project Overview

This project implements a complete machine learning pipeline for breast cancer diagnosis using the famous Wisconsin Breast Cancer Dataset. The system uses a Decision Tree classifier with hyperparameter optimization to achieve high accuracy in tumor classification.

## ✨ Features

- **Comprehensive Data Analysis**: Exploratory data analysis with statistical summaries
- **Data Preprocessing**: Automated data cleaning, encoding, scaling, and splitting
- **Model Training**: Decision Tree classifier with customizable parameters
- **Hyperparameter Optimization**: GridSearchCV for finding optimal model parameters
- **Model Evaluation**: Multiple metrics including accuracy, precision, recall, F1-score, and ROC-AUC
- **Cross-Validation**: 5-fold cross-validation for robust performance estimation
- **Rich Visualizations**: 6 comprehensive plots including confusion matrix, ROC curve, feature importance, and more
- **Prediction Interface**: Easy-to-use interface for making predictions on new samples

## 📊 Dataset

The dataset contains features computed from digitized images of fine needle aspirate (FNA) of breast masses. Features describe characteristics of cell nuclei present in the images.

- **Total Samples**: 569
- **Features**: 30 numeric features
- **Target**: Diagnosis (M = Malignant, B = Benign)
- **Feature Categories**: 
  - Mean values (radius, texture, perimeter, area, smoothness, etc.)
  - Standard error values
  - Worst/largest values

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or download this project

2. Install required packages:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Basic Usage

Simply run the main script:

```bash
python breast_cancer_classifier.py
```

This will:
1. Load and explore the dataset
2. Preprocess the data
3. Train a basic Decision Tree model
4. Optimize hyperparameters using GridSearchCV
5. Evaluate the model with comprehensive metrics
6. Generate and save visualizations

### Advanced Usage

You can also use the classifier programmatically:

```python
from breast_cancer_classifier import BreastCancerClassifier

# Initialize classifier
classifier = BreastCancerClassifier('data.csv')

# Load and preprocess data
classifier.load_data().explore_data().preprocess_data(test_size=0.2)

# Train and optimize
classifier.train_model(max_depth=5).optimize_model()

# Evaluate
results = classifier.evaluate_model(use_best_model=True)

# Visualize
classifier.visualize_results(results)

# Make predictions on new samples
sample = [17.99, 10.38, 122.8, 1001, 0.1184, ...]  # 30 features
diagnosis, confidence = classifier.predict_new_sample(sample)
```

## 📈 Model Performance

The optimized Decision Tree model achieves:

- **Accuracy**: ~95%+
- **Precision**: ~94%+
- **Recall**: ~93%+
- **F1-Score**: ~93%+
- **ROC-AUC**: ~94%+

*Note: Actual performance may vary slightly due to random state and data splits*

## 📊 Visualizations

The project generates comprehensive visualizations including:

1. **Confusion Matrix**: Shows true positives, false positives, true negatives, and false negatives
2. **ROC Curve**: Receiver Operating Characteristic curve with AUC score
3. **Performance Metrics**: Bar chart of all evaluation metrics
4. **Feature Importance**: Top 10 most important features for classification
5. **Data Distribution**: Pie chart showing class balance
6. **Decision Tree Structure**: Visual representation of the decision tree (simplified)

All visualizations are saved as `breast_cancer_analysis.png`

## 🔧 Model Optimization

The project uses GridSearchCV to optimize the following hyperparameters:

- **max_depth**: [3, 5, 7, 10, 15, None]
- **min_samples_split**: [2, 5, 10, 20]
- **min_samples_leaf**: [1, 2, 5, 10]
- **criterion**: ['gini', 'entropy']

This results in 192 different model combinations tested with 5-fold cross-validation.

## 📁 Project Structure

```
breast-cancer-classification/
│
├── data.csv                          # Dataset file
├── breast_cancer_classifier.py       # Main classifier implementation
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
└── breast_cancer_analysis.png        # Generated visualizations (after running)
```

## 🧪 Key Components

### BreastCancerClassifier Class

The main class that encapsulates all functionality:

- `load_data()`: Load and display dataset information
- `explore_data()`: Perform exploratory data analysis
- `preprocess_data()`: Clean, encode, split, and scale data
- `train_model()`: Train Decision Tree with specified parameters
- `optimize_model()`: Find best hyperparameters using GridSearchCV
- `evaluate_model()`: Calculate comprehensive performance metrics
- `visualize_results()`: Generate and save visualizations
- `predict_new_sample()`: Make predictions on new data

## 📊 Evaluation Metrics

The model is evaluated using multiple metrics:

- **Accuracy**: Overall correctness of predictions
- **Precision**: Proportion of positive predictions that are correct
- **Recall**: Proportion of actual positives correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **Confusion Matrix**: Detailed breakdown of predictions
- **Cross-Validation Score**: Average performance across 5 folds

## 🎯 Why Decision Tree?

Decision Trees are chosen for this project because:

1. **Interpretability**: Easy to understand and explain to medical professionals
2. **No Feature Scaling Required**: Works with raw features (though we scale for consistency)
3. **Handles Non-linear Relationships**: Can capture complex patterns
4. **Feature Importance**: Provides insights into which features matter most
5. **Fast Training and Prediction**: Efficient for real-time applications

## 🔍 Feature Importance

The model identifies the most important features for classification, typically including:

- Worst radius
- Worst perimeter
- Worst area
- Mean concave points
- Worst concave points
- Mean perimeter
- Mean radius

## 🚨 Important Notes

- This is an educational/demonstration project
- Not intended for actual medical diagnosis
- Always consult healthcare professionals for medical decisions
- Model performance should be validated on external datasets before any real-world use

## 📝 Future Improvements

Potential enhancements:

- [ ] Implement ensemble methods (Random Forest, Gradient Boosting)
- [ ] Add feature selection techniques
- [ ] Implement SHAP values for better interpretability
- [ ] Create web interface for easy predictions
- [ ] Add model persistence (save/load trained models)
- [ ] Implement additional preprocessing techniques
- [ ] Add support for other datasets

## 👨‍💻 Author

Anees Ur Rehman

## 📄 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- Wisconsin Breast Cancer Dataset creators
- Scikit-learn library developers
- Open source community

---

**⚠️ Disclaimer**: This project is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment.

@copy Anees Ur Rehman
