# Iris Flower Classification and Visualization

## Overview
This project focuses on the classification of Iris flowers using machine learning techniques, with a special emphasis on data visualization. The implementation includes various visualization techniques to understand the underlying patterns in the Iris dataset, followed by a classification model to predict the Iris species.

## Features
- Comprehensive data visualization including:
  - Scatter plots for feature relationships
  - Box plots for feature distribution
  - Violin plots for density estimation
  - Correlation heatmaps
- Machine learning classification using Logistic Regression
- Data preprocessing and scaling
- Model evaluation and performance metrics

## Prerequisites
Make sure you have Python 3.7+ installed. The following Python packages are required:

```bash
pandas==2.0.0
numpy==1.24.0
scikit-learn==1.2.0
seaborn==0.12.0
matplotlib==3.7.0
```

## Installation
1. Clone the repository:
```bash
git clone https://github.com/NiharSanoria1/iris_flower_classification.git
cd iris_flower_classification
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Project Structure
```
iris-classification/
│
├── iris.py
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

### Visualization Functions
The project includes several visualization functions:

1. Feature Relationships:
```python
# Create scatter plots
visualize_iris_data(df)
```

2. Distribution Analysis:
```python
# Create violin plots
create_violin_plots(df)
```

### Model Training
```python
# Split and preprocess data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model, scaler = train_evaluate_model(df)
```

## Visualization Explanations

### Violin Plots
The violin plots in this project show:
- Distribution shape of each feature
- Density estimation at different values
- Median and quartile ranges
- Overall data spread

### Correlation Heatmap
The correlation heatmap displays:
- Feature relationships
- Positive and negative correlations
- Correlation strength through color intensity

## Model Performance
The current implementation uses Logistic Regression and typically achieves:
- Accuracy: ~95-98%
- Good separation especially for Setosa species
- Reliable predictions for Versicolor and Virginica

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



## Acknowledgments
- Iris dataset from UCI Machine Learning Repository
- Inspired by various data visualization techniques in the scientific Python ecosystem