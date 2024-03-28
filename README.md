# Evaluating Deep Learning Architectures for Unbalanced SAR Ship Classification: A Focus on Class-wise Performance

This repository contains the code for training deep learning models for classifying ship types in Synthetic Aperture Radar (SAR) images. The focus is on evaluating performance metrics for unbalanced datasets, where some ship classes may be under-represented compared to others.

**Models:**

The code implements five different deep learning architectures:

* **CNN**: A custom convolutional neural network (CNN) architecture with batch normalization for improved training stability.
* **VGG**: The VGG16 model, a pre-trained architecture fine-tuned for SAR ship classification.
* **ResNet**: The ResNet50 model, another pre-trained architecture fine-tuned for the task.
* **Fine-Tuned VGG**: VGG16 with its pre-trained layers frozen and a new classifier head added.
* **Fine-Tuned ResNet**: ResNet50 with its pre-trained layers frozen and a new classifier head added.

**Replication Instructions**

To replicate the experiments and obtain the results, follow these steps:

1. **Download Datasets:**
   - Download the OpenSARShip and Fusar datasets from their respective sources. These datasets are crucial for training and testing the deep learning models.
   - FUSAR: https://drive.google.com/file/d/1SOEMud9oUq69gxbfcBkOvtUkZ3LWEpZJ/view?usp=sharing
   - OpenSARShip: https://emwlab.fudan.edu.cn/67/05/c20899a222981/page.htm

2. **Install Dependencies:**
   - Open a terminal or command prompt and navigate to your project directory.
   - Install the required Python libraries using `pip`:

     ```bash
     pip install -r requirements.txt
     ```

     The `requirements.txt` file specifies all the necessary libraries for running the code.

3. **Prepare Datasets:**
   - Run the `dataset_prep.py` script to process and prepare the downloaded datasets for use with the deep learning models. This script involve tasks like data cleaning, normalization, and splitting into training and testing sets, creation of merged dataset.

     ```bash
     python dataset_prep.py
     ```

4. **Train and Test Models:**
   - Run the `test_models.py` script to train the deep learning models on the prepared datasets and evaluate their performance on the test sets. This script will perform the training, testing, and save the results.

     ```bash
     python test_models.py
     ```

5. **Analyze Results:**
   - The script will generate two result files:
     - `results.txt`: This file contains detailed logs of the training process, including loss values, accuracy metrics, and hyperparameter configurations.
     - `results.csv`: This file summarizes the performance of each model on different test sets. It typically contains accuracy scores for each ship class and potentially other relevant metrics.


**Further Exploration:**

* Experiment with different hyperparameter settings.
* Try data augmentation techniques to address class imbalance.
* Implement additional evaluation metrics beyond basic accuracy.

**Disclaimer:**

This code is provided for educational purposes only. You might need to adapt it for your specific use case and dataset.
