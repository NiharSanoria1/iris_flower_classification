import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)


def iris_visualization(df):
    
    
    
     # figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Pairplot for all features
    plt.subplot(2, 2, 1)
    sns.scatterplot(data=df, x='sepal length (cm)', y='sepal width (cm)', 
                    hue='species', style='species')
    plt.title('Sepal Length vs Width')
    
    # 2. Petal length vs width
    plt.subplot(2, 2, 2)
    sns.scatterplot(data=df, x='petal length (cm)', y='petal width (cm)', 
                    hue='species', style='species')
    plt.title('Petal Length vs Width')
    
    # 3. Box plot for feature distribution
    plt.subplot(2, 2, 3)
    df_melted = df.melt(id_vars=['species'], 
                        value_vars=[col for col in df.columns if col != 'species'])
    sns.boxplot(data=df_melted, x='variable', y='value', hue='species')
    plt.xticks(rotation=45)
    plt.title('Feature Distribution by Species')
    
    # 4. Violin plot for density estimation
    plt.subplot(2, 2, 4)
    sns.violinplot(data=df_melted, x='variable', y='value', hue='species')
    plt.xticks(rotation=45)
    plt.title('Density Distribution by Species')
    
    plt.tight_layout()
    plt.show()
    
    # Correlation heatmap
    plt.figure(figsize=(8, 6))
    numeric_cols = [col for col in df.columns if col != 'species']
    correlation = df[numeric_cols].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.show()
    
# Function for model training and evaluation
def train_evaluate_model(df):
    # Prepare features and target
    X = df.drop('species', axis=1)
    y = pd.Categorical(df['species']).codes
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = LogisticRegression(multi_class='multinomial', max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    
    return model, scaler

iris_visualization(df)


model, scaler = train_evaluate_model(df)
