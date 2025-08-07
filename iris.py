# Iris Flower Classification with Machine Learning
# This script demonstrates basic classification concepts using the famous Iris dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score  
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_data():
    """Load the Iris dataset from a CSV file and perform initial exploration"""
    print("=== LOADING AND EXPLORING THE IRIS DATASET (CSV) ===")

    # Load the dataset from the CSV file
    df = pd.read_csv("Iris.csv")  

    if 'Id' in df.columns:
        df.drop(columns='Id', inplace=True)

    if 'Species' in df.columns:
        df.rename(columns={'Species': 'species_name'}, inplace=True)

    # Map species to numeric values
    species_map = {name: idx for idx, name in enumerate(df['species_name'].unique())}
    df['species'] = df['species_name'].map(species_map)

    print(f"Dataset shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nFirst 5 rows:")
    print(df.head())

    print(f"\nDataset info:")
    print(df.info())

    print(f"\nSpecies distribution:")
    print(df['species_name'].value_counts())

    print(f"\nStatistical summary:")
    print(df.describe())

    return df, species_map

def visualize_data(df):
    """Create visualizations to understand the data better"""
    print("\n=== DATA VISUALIZATION ===")
    
    feature_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Sepal Length vs Sepal Width
    plt.subplot(2, 2, 1)
    for species in df['species_name'].unique():
        species_data = df[df['species_name'] == species]
        plt.scatter(species_data['SepalLengthCm'], species_data['SepalWidthCm'], 
                    label=species, alpha=0.7)
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Sepal Width (cm)')
    plt.title('Sepal Length vs Sepal Width')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Petal Length vs Petal Width
    plt.subplot(2, 2, 2)
    for species in df['species_name'].unique():
        species_data = df[df['species_name'] == species]
        plt.scatter(species_data['PetalLengthCm'], species_data['PetalWidthCm'], 
                    label=species, alpha=0.7)
    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Petal Width (cm)')
    plt.title('Petal Length vs Petal Width')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Box plot for feature distributions
    plt.subplot(2, 2, 3)
    df_melted = df.melt(id_vars=['species_name'], value_vars=feature_cols, 
                        var_name='feature', value_name='value')
def prepare_data(df):
    """Prepare the data for machine learning"""
    print("\n=== DATA PREPARATION ===")

    # Extract features and target
    feature_cols = df.columns[:4]  
    X = df[feature_cols].values
    y = df['species'].values

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    print(f"Features: {X_train.shape[1]}")

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler

def train_multiple_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled):
    """Train and compare multiple classification algorithms"""
    print("\n=== TRAINING MULTIPLE MODELS ===")
    
    # Define models to compare
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=3)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Use scaled data for models that benefit from it
        if name in ['Logistic Regression', 'SVM', 'K-Nearest Neighbors']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            # Cross-validation with scaled data
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Cross-validation with original data
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'predictions': y_pred
        }
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Cross-validation: {cv_mean:.4f} (±{cv_std:.4f})")
    
    return results

def evaluate_best_model(results, y_test, iris):
    """Detailed evaluation of the best performing model"""
    print("\n=== MODEL EVALUATION ===")
    
    # Find the best model based on cross-validation score
    best_model_name = max(results.keys(), key=lambda k: results[k]['cv_mean'])
    best_result = results[best_model_name]
    
    print(f"Best Model: {best_model_name}")
    print(f"Test Accuracy: {best_result['accuracy']:.4f}")
    print(f"Cross-validation Score: {best_result['cv_mean']:.4f} (±{best_result['cv_std']:.4f})")
    
    # Detailed classification report
    print(f"\nDetailed Classification Report for {best_model_name}:")
    print(classification_report(y_test, best_result['predictions'], 
                              target_names=iris.target_names))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, best_result['predictions'])
    
    plt.figure(figsize=(10, 4))
    
    # Plot confusion matrix
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=iris.target_names, 
                yticklabels=iris.target_names)
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Plot model comparison
    plt.subplot(1, 2, 2)
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    cv_scores = [results[name]['cv_mean'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.bar(x - width/2, accuracies, width, label='Test Accuracy', alpha=0.8)
    plt.bar(x + width/2, cv_scores, width, label='CV Score', alpha=0.8)
    
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Model Comparison')
    plt.xticks(x, model_names, rotation=45)
    plt.legend()
    plt.ylim(0.8, 1.0)
    
    plt.tight_layout()
    plt.show()
    
    return best_model_name, best_result['model']

def make_predictions_example(best_model, scaler, iris, best_model_name):
    """Demonstrate how to make predictions on new data"""
    print("\n=== MAKING PREDICTIONS ON NEW DATA ===")
    
    # Example new flower measurements
    new_flowers = np.array([
        [5.1, 3.5, 1.4, 0.2],  # Should be Setosa
        [6.2, 2.8, 4.8, 1.8],  # Should be Versicolor  
        [7.2, 3.0, 5.8, 2.2]   # Should be Virginica
    ])
    
    print("New flower measurements:")
    feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    for i, flower in enumerate(new_flowers):
        print(f"Flower {i+1}: {dict(zip(feature_names, flower))}")
    
    # Scale the new data if needed
    if best_model_name in ['Logistic Regression', 'SVM', 'K-Nearest Neighbors']:
        new_flowers_scaled = scaler.transform(new_flowers)
        predictions = best_model.predict(new_flowers_scaled)
        probabilities = best_model.predict_proba(new_flowers_scaled)
    else:
        predictions = best_model.predict(new_flowers)
        probabilities = best_model.predict_proba(new_flowers)
    
    print(f"\nPredictions using {best_model_name}:")
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        predicted_species = iris.target_names[pred]
        confidence = np.max(prob)
        print(f"Flower {i+1}: {predicted_species} (Confidence: {confidence:.3f})")
        
        # Show probabilities for all classes
        print("  Probabilities:")
        for j, species in enumerate(iris.target_names):
            print(f"    {species}: {prob[j]:.3f}")

def explain_concepts():
    """Explain key machine learning concepts demonstrated"""
    print("\n=== KEY MACHINE LEARNING CONCEPTS EXPLAINED ===")
    
    concepts = {
        "Classification": "Predicting discrete categories (species) based on input features (measurements)",
        
        "Feature Scaling": "Normalizing features to have similar scales, important for distance-based algorithms",
        
        "Train-Test Split": "Dividing data to train models on one part and evaluate on unseen data",
        
        "Cross-Validation": "Testing model performance across multiple data splits to ensure robustness",
        
        "Overfitting": "When a model memorizes training data but performs poorly on new data",
        
        "Confusion Matrix": "Table showing correct vs incorrect predictions for each class",
        
        "Accuracy": "Percentage of correct predictions (good for balanced datasets like Iris)",
        
        "Model Comparison": "Testing multiple algorithms to find the best one for your specific problem"
    }
    
    for concept, explanation in concepts.items():
        print(f"\n• {concept}: {explanation}")

def main():
    print("IRIS FLOWER CLASSIFICATION WITH MACHINE LEARNING (CSV)")
    print("="*60)

    # Load from CSV
    df, species_map = load_and_explore_data()

    # Visualize
    visualize_data(df)

    # Prepare
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = prepare_data(df)

    # Train models
    results = train_multiple_models(X_train, X_test, y_train, y_test,
                                    X_train_scaled, X_test_scaled)

    # Evaluate
    best_model_name, best_model = evaluate_best_model(results, y_test, iris=datasets.load_iris())

    # Predict
    make_predictions_example(best_model, scaler, datasets.load_iris(), best_model_name)

    # Explain
    explain_concepts()

    print("\n" + "="*60)
    print(" ANALYSIS COMPLETE! ")

if __name__ == "__main__":
    main()