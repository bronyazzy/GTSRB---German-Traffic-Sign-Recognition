README.txt

Project Title: Traffic Sign Recognition using Machine Learning and Deep Learning Models

Description:
This project aims to develop a traffic sign recognition system using machine learning and deep learning models, such as logistic regression, MLP, SVM, CNN, and ResNet. The project uses various preprocessing techniques like cropping images and PCA (Principal Component Analysis) to achieve better performance.

Contents:

Preprocessing Scripts：dataProcess.py dataProcessCrop.py
Model Scripts

Dataset
Requirements:

Python 3.x
NumPy
scikit-learn
TensorFlow
Keras
Matplotlib
Instructions:

Preprocessing:
dataProcess.py dataProcessCrop.py ，result have been store in preprocessed_data_cropped.npz preprocessed_data.npz

Model Training:
Run the individual model scripts (Logistic Regression, MLP, SVM, CNN, and ResNet) to train the models on the preprocessed data.

Model Evaluation:
After training the models, run the evaluation scripts to calculate accuracy, F1-score, and other performance metrics on the test dataset.



Notes:

Ensure all dependencies are installed before running the scripts.
The dataset used in this project is the German Traffic Sign Recognition Benchmark (GTSRB) dataset.