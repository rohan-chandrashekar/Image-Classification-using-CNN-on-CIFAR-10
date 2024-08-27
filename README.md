# Convolutional Neural Network for Image Classification on CIFAR-10

This project implements a Convolutional Neural Network (CNN) for classifying images from the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. This repository includes the code for building, training, and evaluating the CNN model using TensorFlow and Keras.

## Dataset Description

The CIFAR-10 dataset contains:

- **Training Data**: 50,000 images
- **Test Data**: 10,000 images
- **Image Dimensions**: 32x32 pixels
- **Classes**: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

## Project Structure

- **CNN.ipynb**: This Jupyter Notebook contains the implementation of the CNN model. The model architecture consists of multiple layers including convolutional, batch normalization, ReLU activation, max-pooling, and dense layers. The model is trained for 20 epochs using the Adam optimizer.

- **Images Augmentation**: Image augmentation techniques such as random flips and rotations have already been applied to enhance the diversity of training data.

## Model Architecture

The CNN model consists of the following layers:
1. **Convolutional Layer**: Extracts features from the input image using multiple learnable filters (kernels). Each convolution operation performs a dot product between the filter and parts of the input image.
2. **Batch Normalization Layer**: Normalizes the output of each feature map to stabilize and speed up training.
3. **ReLU Activation**: Applies a non-linear transformation, replacing negative pixel values with zero.
4. **Max Pooling Layer**: Reduces the size of feature maps by summarizing regions through the maximum value within a sliding window.
5. **Dense Layer**: Fully connected layers for classification, where the output of the convolutional base is flattened and passed through these layers to predict the class.

### Detailed Layer Summary

The architecture comprises several layers as shown below:

- 3 Convolutional layers followed by Batch Normalization, ReLU, and Max Pooling
- Final layers include flattening the tensor, followed by 3 Dense layers for classification
- The output layer has 10 nodes (one for each class)

Use `model.summary()` to view the complete architecture.

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/rohan-chandrashekar/cnn-cifar10.git
    ```
2. Run the Jupyter notebook:
    ```bash
    jupyter notebook CNN.ipynb
    ```

## Training

The model is trained on the CIFAR-10 dataset for 20 epochs using the Adam optimizer. The learning rate and other parameters can be configured in the notebook. The training process involves:
- Splitting the data into training and testing sets
- Compiling the model with the loss function (`sparse_categorical_crossentropy`) and optimizer (`Adam`)
- Training the model using the `fit()` method

## Evaluation

After training, the model can be evaluated using the test set to compute the accuracy and other metrics. You can also save the model using `model.save()` and load it later for inference.

## Instructions

- **Conv2D Layer**: No bias is used in any convolution operation. You must ensure the strides and padding are set appropriately to maintain the input and output dimensions.
- **Max Pooling**: Use a 2x2 filter size with a stride of 2 for each max-pooling operation.
- **Batch Normalization**: Apply batch normalization after each Conv2D operation, normalizing across the entire feature map.
- **Model Training**: The model is pre-configured to train on 20 epochs, and you should use the provided boilerplate code for testing.

## Results

You can view the model's performance and the number of parameters at each stage using `model.summary()`. The grading is based on the number of parameters at each stage, and a correct implementation should match the layer structure.

## Conclusion

This project demonstrates how to build a CNN for image classification using the CIFAR-10 dataset. It covers the key concepts such as convolution, pooling, and batch normalization, which are crucial for building deep learning models for image recognition.
