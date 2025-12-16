# data_processor.py - HOG + PCA VERSION 

import numpy as np
import pickle
import cv2 # Used internally by skimage
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
from skimage.feature import hog

# =========================================================================
# --- CONSTANTS AND CONFIGURATION ---
# IMPORTANT: We are sticking to 30 components for a realistic accuracy (80-90%).  ******************From 30 to 128*************************
N_COMPONENTS = 128
DATA_SAVE_FILE = 'processed_stl10_data_hog.pkl' 
N_CLASSES = 5 
# Smaller size for HOG extraction (HOG is sensitive to image size)
IMAGE_SIZE = 64 
# =========================================================================

# Custom dataset class to filter STL-10 to the first N_CLASSES
class FilteredSTL10(Dataset):
    def __init__(self, stl10_dataset, n_classes):
        self.dataset = stl10_dataset
        self.indices = [i for i, (_, label) in enumerate(stl10_dataset) if label < n_classes]
        self.data_subset = Subset(self.dataset, self.indices)
        
    def __len__(self):
        return len(self.data_subset)
    
    def __getitem__(self, idx):
        return self.data_subset[idx]

def load_and_extract_hog_features():
    """Loads STL-10, applies HOG feature extraction, and filters."""
    print("1. Loading STL-10 dataset (test split) and extracting HOG features...")
    
    # Transformations: Resize and convert to PIL Image for HOG (skimage/HOG typically needs NumPy array)
    # NOTE: We use transforms.ToTensor() initially, then convert to NumPy array for HOG
    transform_to_numpy = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(), # C x H x W tensor, scales to [0, 1]
    ])
    
    # We load the dataset without filtering yet
    stl10_test = datasets.STL10(root='./data', split='test', download=True, transform=transform_to_numpy)
    
    # Filter to only include the first N_CLASSES (e.g., 5 classes)
    filtered_dataset = FilteredSTL10(stl10_test, N_CLASSES)
    
    X = []
    y = []
    
    # HOG Parameters (can be tuned later if needed)

    hog_params = {
        'orientations': 9,
        'pixels_per_cell': (4, 4),   # ************************From (8, 8) to (4,4)*******************************
        'cells_per_block': (2, 2),
        'visualize': False,
        'channel_axis': 0 # For RGB images C x H x W format
    }
    
    for item in filtered_dataset.data_subset:
        # Convert tensor (C x H x W) to NumPy array (H x W x C)
        image_np = item[0].permute(1, 2, 0).numpy()

        # Convert to grayscale for stable HOG
        gray = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        # HOG extraction on grayscale image
        features = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=False,
            block_norm='L2-Hys'
        )

        X.append(features)
        y.append(item[1])

    X_features = np.array(X)
    y_labels = np.array(y)
    
    print(f"Total filtered samples loaded: {len(y_labels)}")
    print(f"Features extracted (HOG dimension): {X_features.shape[1]}") # This will be the HOG feature size (e.g., 2916)
    return X_features, y_labels

def apply_pca_and_split(X_features, y_labels):
    """Applies PCA and splits data into train/test sets."""
    print(f"2. Applying PCA to reduce dimension to {N_COMPONENTS} components...")
    
    # Apply PCA with the controlled number of components (30)
    pca = PCA(n_components=N_COMPONENTS)
    X_pca = pca.fit_transform(X_features)
    
    print(f"Features after PCA (final dimension): {X_pca.shape}")
    
    # Split the dataset into Training (80%) and Testing (20%)
    print("3. Splitting data into 80% Train and 20% Test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )
    
    print(f"Train set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test

def save_processed_data(X_train, X_test, y_train, y_test):
    """Saves the processed data to a pickle file."""
    data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    with open(DATA_SAVE_FILE, 'wb') as f:
        pickle.dump(data, f)
    print(f"\nSUCCESS: Processed data saved to {DATA_SAVE_FILE}. Ready for model training.")


if __name__ == '__main__':
    # Full Processing Pipeline
    X_features, y_labels = load_and_extract_hog_features()
    
    X_train, X_test, y_train, y_test = apply_pca_and_split(X_features, y_labels)
    
    save_processed_data(X_train, X_test, y_train, y_test)