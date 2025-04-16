# Autism Detection Using CNN - Updated Implementation

## Overview

This project implements an enhanced Convolutional Neural Network (CNN) to detect autism from image data. The updated code leverages advanced techniques such as data augmentation, a robust CNN architecture with Batch Normalization and Dropout layers, and optimized training parameters to improve model accuracy. While achieving 98% accuracy is highly dependent on data quality and dataset specifics, these improvements aim to significantly boost performance from the baseline.

## Directory Structure

```plaintext
.
├── data
│   ├── train
│   │   ├── class1
│   │   └── class2
│   └── validation
│       ├── class1
│       └── class2
├── main.py         # Main code file containing the updated CNN implementation
└── README.md       # This file
```

*Note:* Ensure your training and validation images are organized in subdirectories by class.

## Requirements

- **Python:** 3.x
- **TensorFlow:** 2.x (which includes Keras)
- **Additional Libraries:** 
  - numpy
  - (Optional) matplotlib for plotting training metrics

Install the required packages with:

```bash
pip install tensorflow numpy
```

## Code Description

### Data Preprocessing
- **Data Augmentation:**  
  The `ImageDataGenerator` is used to enhance the training set by applying rotations, shifts, shearing, zooming, and horizontal flipping. This creates variations of your images and helps the model generalize better.
- **Rescaling:**  
  Both training and validation images are rescaled to a [0, 1] range by dividing by 255.

### Model Architecture
- **Convolutional Blocks:**  
  The model consists of multiple convolutional blocks:
  - **Block 1:** Convolution (32 filters), Batch Normalization, MaxPooling, and Dropout.
  - **Block 2:** Convolution (64 filters), Batch Normalization, MaxPooling, and Dropout.
  - **Block 3:** Convolution (128 filters), Batch Normalization, MaxPooling, and Dropout.
- **Fully Connected Layers:**  
  After flattening, a Dense layer with 256 neurons is used followed by Batch Normalization and a Dropout layer. The final output layer is a single neuron with a sigmoid activation for binary classification.

### Compilation and Training
- **Optimizer & Loss Function:**  
  The model uses the Adam optimizer with a tuned learning rate of 1e-4 and binary crossentropy loss.
- **Callbacks:**  
  EarlyStopping is implemented to halt training when the validation loss stops improving, and ModelCheckpoint is used to save the best model.

## Running the Code

1. **Prepare Your Data:**  
   Place your images into `data/train` and `data/validation` directories, organized by class (each class in its own subfolder).

2. **Configure Parameters:**  
   - Adjust image dimensions (`img_width`, `img_height`) if your images require a different resolution.
   - Ensure file paths in the code match your directory structure.

3. **Execute the Script:**  
   Run the main code file from the command line:
   ```bash
   python main.py
   ```
   During training, the model will output progress with epoch-wise accuracy and loss metrics. The best model is saved as `best_autism_detection_model.h5`, and the final model is saved as `final_autism_detection_model.h5`.
