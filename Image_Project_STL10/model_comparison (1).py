# model_comparison.py - FINAL HOG VERSION 

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# =========================================================================
# --- CONSTANTS ---
# IMPORTANT: Updated to read the HOG feature file
DATA_INPUT_FILE = 'processed_stl10_data_hog.pkl'
# =========================================================================

def load_results():
    """Loads results from Logistic Regression and K-Means models."""
    try:
        with open('lr_results.pkl', 'rb') as f:
            lr_results = pickle.load(f)
        
        with open('kmeans_results.pkl', 'rb') as f:
            kmeans_results = pickle.load(f)
            
        return lr_results, kmeans_results
    except FileNotFoundError:
        print("Error: Results files (lr_results.pkl or kmeans_results.pkl) not found.")
        print("Please run logistic_regression_model.py and kmeans_model.py first.")
        return None, None

def plot_accuracy_comparison(lr_results, kmeans_results):
    """Plots a bar chart comparing the accuracy of the two models."""
    print("\n--- 1. Accuracy Comparison ---")
    
    models = [lr_results['model_name'], kmeans_results['model_name']]
    accuracies = [lr_results['accuracy'], kmeans_results['accuracy']]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies, color=['skyblue', 'lightcoral'])
    
    # Add accuracy values on top of the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom')
        
    plt.ylim(0, 1) # Accuracy scale
    plt.title('Comparison of Model Accuracy (HOG Features)')
    plt.xlabel('Model')
    plt.ylabel('Test Accuracy')
    plt.grid(axis='y', linestyle='--')
    plt.show()
    


def plot_confusion_matrices(lr_results, kmeans_results):
    """Plots confusion matrices for both models side-by-side."""
    print("\n--- 2. Confusion Matrices Comparison ---")
    
    conf_matrices = [lr_results['conf_matrix'], kmeans_results['conf_matrix']]
    model_names = [lr_results['model_name'], kmeans_results['model_name']]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    class_names = [f'Class {i}' for i in range(5)]
    
    for i, conf_matrix in enumerate(conf_matrices):
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False,
                    xticklabels=class_names, yticklabels=class_names)
        axes[i].set_title(f'Confusion Matrix - {model_names[i]}')
        axes[i].set_xlabel('Predicted Label')
        axes[i].set_ylabel('True Label')
        
    plt.tight_layout()
    plt.show()
    


def display_final_summary(lr_results, kmeans_results):
    """Prints a final summary table of key metrics."""
    print("\n--- 3. Final Performance Summary ---")
    
    print(f"{'Metric':<30} | {'Logistic Regression':<25} | {'K-Means':<25}")
    print("-" * 85)
    
    print(f"{'Test Accuracy':<30} | {lr_results['accuracy']:<25.4f} | {kmeans_results['accuracy']:<25.4f}")
    
    lr_auc_avg = np.mean(list(lr_results['roc_auc_scores'].values())) if 'roc_auc_scores' in lr_results else 'N/A'
    if isinstance(lr_auc_avg, float):
        auc_str = f"{lr_auc_avg:<25.4f}"
    else:
        auc_str = f"{lr_auc_avg:<25}"

    print(f"{'Average ROC AUC (Classification)':<30} | {auc_str} | {'N/A':<25}")

    print(f"{'Features Used':<30} | {'HOG + PCA (30)':<25} | {'HOG + PCA (30)':<25}")
    print(f"{'Hyperparameter (C / K)':<30} | {lr_results['best_hyperparameter_C']:<25} | {kmeans_results['k_chosen']:<25}")
    

if __name__ == '__main__':
    
    lr_results, kmeans_results = load_results()
    
    if lr_results and kmeans_results:
        plot_accuracy_comparison(lr_results, kmeans_results)
        plot_confusion_matrices(lr_results, kmeans_results)
        display_final_summary(lr_results, kmeans_results)