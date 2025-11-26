import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def main():
    # --- 1. Data Loading ---
    print("Loading Breast Cancer Wisconsin Dataset...")
    data = load_breast_cancer()
    
    # Create DataFrame (30 features!)
    df = pd.DataFrame(data=data.data, columns=data.feature_names)
    df['target'] = data.target
    
    # Target: 0 = Malignant (Bad), 1 = Benign (Safe)
    # Let's map it for clarity
    df['diagnosis'] = df['target'].map({0: 'Malignant', 1: 'Benign'})
    
    print(f"Dataset Shape: {df.shape}")
    print(df.head())

    # --- 2. Data Visualization ---
    # Visualize the count of Malignant vs Benign
    plt.figure(figsize=(6, 4))
    sns.countplot(x='diagnosis', data=df, palette='magma')
    plt.title("Distribution of Malignant vs Benign Samples")
    plt.show()

    # Visualize correlation (heatmap) - Looks great in a portfolio
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.iloc[:, 0:10].corr(), annot=True, fmt=".1f", cmap='coolwarm')
    plt.title("Correlation of First 10 Features")
    plt.show()

    # --- 3. Preprocessing ---
    X = data.data
    y = data.target

    # Split: 80% Train, 20% Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling is MANDATORY for SVM (Unlike Random Forest)
    print("\nScaling Features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model Training with SVM for high dimensional data
    print("Training Support Vector Machine (SVM)...")
    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(X_train_scaled, y_train)

    # Evaluation ---
    print("Evaluating Model...")
    y_pred = svm_model.predict(X_test_scaled)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {acc * 100:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))

if __name__ == "__main__":
    main()