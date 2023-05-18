#train 39209 test 12630
import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical

def load_data(data_dir, file):
    image_paths = []
    labels = []
    rois = []
    data = pd.read_csv(os.path.join(data_dir, file))

    for idx, row in data.iterrows():
        image_paths.append(os.path.join(data_dir, row['Path']))
        labels.append(row['ClassId'])
        rois.append((row['Roi.X1'], row['Roi.Y1'], row['Roi.X2'], row['Roi.Y2']))

    return image_paths, labels, rois

def preprocess_images(image_paths, rois, img_size=(32, 32)):
    images = []
    for path, roi in zip(image_paths, rois):
        img = cv2.imread(path)
        img = img[roi[1]:roi[3], roi[0]:roi[2]]  # Crop image using ROI coordinates
        img = cv2.resize(img, img_size)
        img = img / 255.0  # Normalize pixel values between 0 and 1
        images.append(img)

    return np.array(images)

def save_data(X, y, file='preprocessed_data_cropped.npz'):
    np.savez(file, X=X, y=y)
    print(f'Data saved to {file}')

def load_data_from_file(file='preprocessed_data_cropped.npz'):
    data = np.load(file)
    return data['X'], data['y']

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir)

train_file = 'Train.csv'
test_file = 'Test.csv'

train_image_paths, train_labels, train_rois = load_data(data_dir, train_file)
test_image_paths, test_labels, test_rois = load_data(data_dir, test_file)

# Preprocess images
X_train = preprocess_images(train_image_paths, train_rois)
X_test = preprocess_images(test_image_paths, test_rois)

# Merge train and test data
X = np.concatenate((X_train, X_test), axis=0)
y = to_categorical(train_labels + test_labels)

# Save preprocessed data
save_data(X, y)
