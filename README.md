# Evaluating Deep Learning Architectures for Unbalanced SAR Ship Classification: A Focus on Class-wise Performance

This repository contains the code for training deep learning models for classifying ship types in Synthetic Aperture Radar (SAR) images. The focus is on evaluating performance metrics for unbalanced datasets, where some ship classes may be under-represented compared to others.

**Models:**

The code implements five different deep learning architectures:

* **CNN**: A custom convolutional neural network (CNN) architecture with batch normalization for improved training stability.
* **VGG**: The VGG16 model, a pre-trained architecture fine-tuned for SAR ship classification.
* **ResNet**: The ResNet50 model, another pre-trained architecture fine-tuned for the task.
* **Fine-Tuned VGG**: VGG16 with its pre-trained layers frozen and a new classifier head added.
* **Fine-Tuned ResNet**: ResNet50 with its pre-trained layers frozen and a new classifier head added.

**Code Structure:**

* `train.py`: Defines the different model architectures using PyTorch. It also contains the training loop for each model with functions for data loading and loss calculation.


**Training Details:**

* Loss function: Cross-entropy loss (lf)
* Learning rate: lr = 0.001
* Epochs: epochs = 10

**Note:**

This is a basic example, and hyperparameters like learning rate and number of epochs might need further tuning based on your specific dataset and desired performance.

**Getting Started:**

1. Install required libraries (PyTorch, torchvision etc.)
2. Download and prepare your SAR image dataset. Ensure proper handling of class imbalance. We have used Fusar and opensarship, then we created a new dataset after mixing them.
3. Modify the `path variables` in the script to point to your data directory and adjust hyperparameters if needed.
4. Run the script: `python train.py`

**Further Exploration:**

* Experiment with different hyperparameter settings.
* Try data augmentation techniques to address class imbalance.
* Implement additional evaluation metrics beyond basic accuracy.

**Disclaimer:**

This code is provided for educational purposes only. You might need to adapt it for your specific use case and dataset.
