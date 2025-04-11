# MNIST Object Localization with Bounding Box Regression using TensorFlow/Keras

This repository contains a Jupyter Notebook (`object_Localization.ipynb`) that demonstrates how to build and train a Convolutional Neural Network (CNN) to perform both **classification** and **object localization** (predicting bounding boxes) on a modified MNIST dataset.

## Description

The notebook tackles a synthetic object localization task. Standard MNIST digits (28x28 pixels) are randomly placed onto a larger black canvas (75x75 pixels). The goal is to train a single model that can simultaneously:

1.  **Classify** the digit (0-9).
2.  **Localize** the digit by predicting the coordinates of its bounding box within the 75x75 canvas.



## Dataset

The dataset used is derived from the standard **MNIST** dataset (handwritten digits 0-9).
-   Original MNIST images (28x28) are loaded using `tensorflow_datasets`.
-   Each image is then randomly padded onto a 75x75 black background.
-   The corresponding label is the digit class (0-9).
-   A bounding box label `[xmin, ymin, xmax, ymax]` (normalized coordinates relative to the 75x75 canvas) is generated based on the random placement of the digit.

## Model Architecture

A custom Convolutional Neural Network (CNN) is built using TensorFlow/Keras. It features a **multi-output** structure:

-   **Input:** Takes the 75x75 grayscale images.
-   **Feature Extractor:** A series of `Conv2D` and `AveragePooling2D` layers to learn spatial features from the input image.
-   **Dense Layers:** A `Flatten` layer followed by a `Dense` layer (128 units, ReLU activation) to process the extracted features.
-   **Output Heads:** The output of the dense layer feeds into two separate heads:
    1.  **Classification Head:** A `Dense` layer with 10 units and `softmax` activation to predict the probability distribution over the 10 digit classes.
    2.  **Bounding Box Regression Head:** A `Dense` layer with 4 units (linear activation implied) to predict the bounding box coordinates (`xmin`, `ymin`, `xmax`, `ymax`).

## Loss Function & Metrics

Since the model has two distinct outputs performing different tasks (classification and regression), a composite loss function is used during compilation:

-   **Classification Loss:** `categorical_crossentropy` (as labels are one-hot encoded).
-   **Bounding Box Loss:** `mse` (Mean Squared Error), suitable for regression tasks.
-   **Metrics:** Accuracy is tracked for classification, and MSE is tracked for the bounding box prediction. Additionally, Intersection over Union (IoU) is calculated post-training to evaluate localization quality.

```python
# Model Compilation Snippet
model.compile(optimizer='adam',
              loss = {'classification' : 'categorical_crossentropy',
                      'bounding_box' : 'mse' },
              metrics = {'classification' : 'accuracy',
                         'bounding_box' : 'mse' })
