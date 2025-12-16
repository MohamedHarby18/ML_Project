# logistic_regression_model.py - FINAL HOG VERSION 

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold, GridSearchCV 
import seaborn as sns

# =========================================================================
# --- CONSTANTS ---
N_FOLDS = 5 
C_VALUES = [0.01, 0.1, 1.0, 10.0] 
# IMPORTANT: Updated to read the HOG feature file
DATA_INPUT_FILE = 'processed_stl10_data_hog.pkl' 
# =========================================================================

def load_data(filename):
    """Loads the preprocessed data."""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['X_train'], data['X_test'], data['y_train'], data['y_test']

def display_sample_predictions_lr(X_test, y_test, y_pred, num_samples=10):
    """Takes random samples and displays model predictions."""
    print("\n--- 3. Sample Predictions Check (Logistic Regression) ---")
    
    if X_test.shape[0] < num_samples:
        num_samples = X_test.shape[0]
        
    print(f"Total number of images in the test set: {X_test.shape[0]}")
    
    np.random.seed(42) 
    sample_indices = np.random.choice(X_test.shape[0], num_samples, replace=False)
    
    class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']
    
    for i, idx in enumerate(sample_indices):
        true_label_idx = y_test[idx]
        pred_label_idx = y_pred[idx]
        
        true_name = class_names[true_label_idx]
        pred_name = class_names[pred_label_idx]
        
        if true_label_idx == pred_label_idx:
            status = "✅ Successfully Classified"
        else:
            status = f"❌ Misclassified (True Class: {true_name})"
        
        print(f"Sample {i+1}/{num_samples}: Predicted: {pred_name}, Status: {status}")


def train_and_evaluate_lr(X_train, X_test, y_train, y_test):
    
    print("--- 1. HYPERPARAMETER TUNING USING CROSS-VALIDATION (K=5) ---")

    lr = LogisticRegression(
        penalty='l2',
        solver='saga',
        max_iter=2000,
        random_state=42,
        tol=0.01
    )

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    param_grid = {'C': C_VALUES}
    grid_search = GridSearchCV(lr, param_grid, cv=cv, scoring='accuracy', verbose=1, n_jobs=-1) 
    grid_search.fit(X_train, y_train)
    
    best_c = grid_search.best_params_['C']
    print(f"Best C found using {N_FOLDS}-Fold CV: {best_c}")
    
    model = grid_search.best_estimator_
    
    # OVERFITTING CHECK: Predict on training data
    y_train_pred = model.predict(X_train)
    training_accuracy = accuracy_score(y_train, y_train_pred)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    print("\n--- 2. RESULTS DETAILS (Logistic Regression) ---")
    
    # a. Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on Training Data: {training_accuracy:.4f}") 
    print(f"Accuracy on Testing Data: {accuracy:.4f}")

    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # b. Loss Curve Alternative: CV Accuracy vs. C
    print("\n--- Loss Curve Alternative: CV Accuracy vs. C ---")
    mean_scores = grid_search.cv_results_['mean_test_score']
    
    plt.figure(figsize=(8, 6))
    plt.plot(C_VALUES, mean_scores, marker='o', label='Mean CV Accuracy')
    plt.xscale('log') 
    plt.axvline(x=best_c, color='r', linestyle='--', label=f'Best C: {best_c}')
    plt.title(f'Loss Curve Alternative: CV Accuracy vs. Regularization Strength (C)')
    plt.xlabel('C (Inverse of Regularization Strength, Log Scale)')
    plt.ylabel(f'Mean CV Accuracy ({N_FOLDS}-Fold)')
    plt.legend()
    plt.grid(True)
    plt.show() 
    

    # c. ROC Curve and AUC
    print("\n--- ROC Curve ---")
    n_classes = len(np.unique(y_test))
    y_test_binarized = label_binarize(y_test, classes=range(n_classes))
    
    plt.figure(figsize=(8, 6))
    roc_auc_scores = {}
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        roc_auc_scores[f'Class {i}'] = roc_auc
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Logistic Regression')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
    

    # Display 10 random sample predictions
    display_sample_predictions_lr(X_test, y_test, y_pred, num_samples=10)

    # 4. Save results for comparison
    lr_results = {
        'accuracy': accuracy,
        'conf_matrix': conf_matrix,
        'roc_auc_scores': roc_auc_scores,
        'cross_validation_folds': N_FOLDS, 
        'best_hyperparameter_C': best_c,
        'model_name': 'Logistic Regression (HOG features)'
    }
    
    with open('lr_results.pkl', 'wb') as f:
        pickle.dump(lr_results, f)
    print("\nLogistic Regression results saved to lr_results.pkl")


if __name__ == '__main__':
    
    try:
        X_train, X_test, y_train, y_test = load_data(DATA_INPUT_FILE) 
        train_and_evaluate_lr(X_train, X_test, y_train, y_test)
    except FileNotFoundError:
        print(f"Error: Data file {DATA_INPUT_FILE} not found. Please run data_processor.py first.")