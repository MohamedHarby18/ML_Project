# kmeans_model.py - FINAL HOG VERSION 

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.optimize import linear_sum_assignment 

# =========================================================================
# --- CONSTANTS ---
K_CLUSTERS = 5 
# IMPORTANT: Updated to read the HOG feature file
DATA_INPUT_FILE = 'processed_stl10_data_hog.pkl'
# =========================================================================


def load_data(filename):
    """Loads the preprocessed data."""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['X_train'], data['X_test'], data['y_train'], data['y_test']

def assign_labels_to_clusters(kmeans, X_test, y_test):
    """Assigns cluster IDs to true labels for evaluation using the Hungarian algorithm."""
    labels = np.zeros_like(y_test)
    y_predicted = kmeans.predict(X_test)
    
    mapping_matrix = np.zeros((K_CLUSTERS, K_CLUSTERS), dtype=np.int64)
    for i in range(len(y_test)):
        mapping_matrix[y_predicted[i], y_test[i]] += 1

    row_ind, col_ind = linear_sum_assignment(mapping_matrix.max() - mapping_matrix)
    
    cluster_to_label = {cluster: label for cluster, label in zip(row_ind, col_ind)}
    
    for i in range(len(y_predicted)):
        labels[i] = cluster_to_label[y_predicted[i]]
        
    return labels

def display_sample_predictions_kmeans(X_test, y_test, y_pred_adjusted, num_samples=10):
    """Takes random samples and displays model predictions."""
    print("\n--- 3. Sample Predictions Check (K-Means) ---")
    
    if X_test.shape[0] < num_samples:
        num_samples = X_test.shape[0]
        
    print(f"Total number of images in the test set: {X_test.shape[0]}")
    
    np.random.seed(42) 
    sample_indices = np.random.choice(X_test.shape[0], num_samples, replace=False)
    
    class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']
    
    for i, idx in enumerate(sample_indices):
        true_label_idx = y_test[idx]
        pred_label_idx = y_pred_adjusted[idx]
        
        true_name = class_names[true_label_idx]
        pred_name = class_names[pred_label_idx]
        
        if true_label_idx == pred_label_idx:
            status = "✅ Successfully Classified"
        else:
            status = f"❌ Misclassified (True Class: {true_name})"
        
        print(f"Sample {i+1}/{num_samples}: Predicted: {pred_name}, Status: {status}")


def train_and_evaluate_kmeans(X_train, X_test, y_train, y_test):
    
    print("--- 1. TRAINING K-MEANS CLUSTERING ---")
    
    kmeans = KMeans(n_clusters=K_CLUSTERS, random_state=42, n_init=10) 
    kmeans.fit(X_train)
    
    y_pred_adjusted = assign_labels_to_clusters(kmeans, X_test, y_test)
    
    print("\n--- 2. RESULTS DETAILS (K-Means as Classifier) ---")
    
    # Adjusted Accuracy
    accuracy = accuracy_score(y_test, y_pred_adjusted)
    print(f"Adjusted Accuracy on Testing Data: {accuracy:.4f}")

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred_adjusted)
    
    # Loss Curve Alternative: Inertia Plot (Elbow Method)
    print("\n--- Loss Curve Alternative: Inertia Plot (Elbow Method) ---")
    inertias = []
    K_range = range(2, K_CLUSTERS + 3)
    for k in K_range:
        kmeans_k = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_train)
        inertias.append(kmeans_k.inertia_) 

    plt.figure(figsize=(8, 6))
    plt.plot(K_range, inertias, marker='o')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia (Within-cluster sum of squares)')
    plt.title('Elbow Method (K-Means Performance)')
    plt.axvline(x=K_CLUSTERS, color='r', linestyle='--', label=f'Chosen K={K_CLUSTERS}')
    plt.legend()
    plt.grid(True)
    plt.show()
    

    # Display 10 random sample predictions
    display_sample_predictions_kmeans(X_test, y_test, y_pred_adjusted, num_samples=10)

    # 4. Save results for comparison
    kmeans_results = {
        'accuracy': accuracy,
        'conf_matrix': conf_matrix,
        'k_chosen': K_CLUSTERS,
        'model_name': 'K-Means (Clustering with HOG features)'
    }
    
    with open('kmeans_results.pkl', 'wb') as f:
        pickle.dump(kmeans_results, f)
    print("\nK-Means results saved to kmeans_results.pkl")


if __name__ == '__main__':
    
    try:
        X_train, X_test, y_train, y_test = load_data(DATA_INPUT_FILE)
        train_and_evaluate_kmeans(X_train, X_test, y_train, y_test) 
    except FileNotFoundError:
        print(f"Error: Data file {DATA_INPUT_FILE} not found. Please run data_processor.py first.")