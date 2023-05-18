import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical


def load_data(data_dir, file):
    image_paths = []
    labels = []
    data = pd.read_csv(os.path.join(data_dir, file))

    for idx, row in data.iterrows():
        image_paths.append(os.path.join(data_dir, row['Path']))
        labels.append(row['ClassId'])

    return image_paths, labels


def preprocess_images(image_paths, img_size=(32, 32)):
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        img = cv2.resize(img, img_size)
        img = img / 255.0  # Normalize pixel values between 0 and 1
        images.append(img)

    return np.array(images)


def save_data(X, y, file='preprocessed_data.npz'):
    np.savez(file, X=X, y=y)
    print(f'Data saved to {file}')


def load_data_from_file(file='preprocessed_data.npz'):
    data = np.load(file)
    return data['X'], data['y']


current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir)

train_file = 'Train.csv'
test_file = 'Test.csv'

train_image_paths, train_labels = load_data(data_dir, train_file)
test_image_paths, test_labels = load_data(data_dir, test_file)

# Merge train and test data
image_paths = train_image_paths + test_image_paths
labels = train_labels + test_labels

# Preprocess images
X = preprocess_images(image_paths)
y = to_categorical(labels)

# Save preprocessed data to a file
save_data(X, y)

# Load preprocessed data from a file
#X, y = load_data_from_file()
