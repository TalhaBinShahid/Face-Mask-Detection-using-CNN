# Face Mask Detection using CNN
This project focuses on detecting face masks using a Convolutional Neural Network (CNN). The model is trained to classify images into two categories: with mask and without mask.

## Table of Contents
- Overview
- Dataset
- Data Preprocessing
- Model Building
- Model Evaluation
- Results
- Dependencies
- Conclusion

## Overview
The project involves building a deep learning model using a CNN to detect whether a person in an image is wearing a face mask. The model was trained on a dataset of images labeled as either "with mask" or "without mask."

## Dataset
The dataset used for this project consists of images categorized into two classes:
- **With Mask**: Images of people wearing face masks.
- **Without Mask**: Images of people not wearing face masks.

## Data Preprocessing
Several preprocessing steps were performed to prepare the data for model training:
- Images were resized to a consistent dimension to match the input requirements of the CNN.
- The dataset was split into training and validation sets to evaluate model performance.
- Data augmentation techniques such as rotation, zoom, and flipping were applied to the training images to increase the diversity of the training set and improve model generalization.

## Model Building
A Convolutional Neural Network (CNN) was constructed using the following layers:
- **Convolutional Layers**: To extract features from the images.
- **Max Pooling Layers**: To reduce the spatial dimensions of the feature maps.
- **Fully Connected Layers**: To classify the images based on the features extracted by the convolutional layers.

The model was compiled with:
- **Loss Function**: Binary Crossentropy, suitable for binary classification.
- **Optimizer**: Adam, chosen for its efficient handling of large datasets and good convergence properties.
- **Metrics**: Accuracy, to evaluate the model's performance.

## Model Evaluation
The model was evaluated on the validation set using accuracy as the primary metric. The training and validation accuracy were monitored to assess the model's learning and to detect any signs of overfitting.

## Results
The CNN model achieved high accuracy on the validation set, demonstrating its effectiveness in detecting face masks.

## Dependencies
- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib

## Conclusion
The DL Face Mask Detection model successfully identifies whether a person is wearing a face mask based on image input. The model can be further improved by fine-tuning hyperparameters, experimenting with different CNN architectures, or using a larger and more diverse dataset.
