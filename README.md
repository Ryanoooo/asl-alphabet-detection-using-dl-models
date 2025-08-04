# ASL Alphabet Detection using Deep Learning Models

This repository contains a Jupyter Notebook (`asl-alphabet-detection-using-dl-models.ipynb`) that demonstrates the process of building and evaluating deep learning models for American Sign Language (ASL) alphabet detection.

## Table of Contents

  * [Introduction](https://www.google.com/search?q=%23introduction)
  * [Dataset](https://www.google.com/search?q=%23dataset)
  * [Installation](https://www.google.com/search?q=%23installation)
  * [Usage](https://www.google.com/search?q=%23usage)
  * [Models](https://www.google.com/search?q=%23models)
  * [Results](https://www.google.com/search?q=%23results)
  * [Contributing](https://www.google.com/search?q=%23contributing)
  * [License](https://www.google.com/search?q=%23license)

## Introduction

American Sign Language (ASL) is a natural language that uses handshapes, positions, and movements to convey meaning. This project aims to develop an image classification system using deep learning to recognize ASL alphabet signs from images. The goal is to provide a foundation for more complex sign language recognition systems.

## Dataset

The dataset used in this project is the "ASL (American Sign Language) Alphabet Dataset". It contains images of various hand gestures representing letters of the ASL alphabet. The dataset is split into training and testing sets.

  * **Training Data Path**: `../input/aslamerican-sign-language-aplhabet-dataset/ASL_Alphabet_Dataset/asl_alphabet_train`
  * **Testing Data Path**: `../input/aslamerican-sign-language-aplhabet-dataset/ASL_Alphabet_Dataset/asl_alphabet_test`

The notebook includes a function `process(file_path)` to load and prepare the dataset by extracting labels from file paths and creating a Pandas DataFrame.

## Installation

To run this notebook, you will need the following libraries. You can install them using pip:

```bash
pip install numpy pandas seaborn matplotlib scikit-learn tensorflow
```

The specific libraries imported in the notebook are:

```python
import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import Callback, EarlyStopping
```

## Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```
2.  **Download the dataset:** Ensure the "ASL (American Sign Language) Alphabet Dataset" is downloaded and placed in the appropriate directory as specified by `path_train` and `path_test` variables in the notebook.
3.  **Open the Jupyter Notebook:**
    ```bash
    jupyter notebook asl-alphabet-detection-using-dl-models.ipynb
    ```
4.  **Run all cells:** Execute the cells in the notebook sequentially to import libraries, load data, preprocess images, build models, train them, and evaluate their performance.

## Models

The notebook explores the use of deep learning models for image classification. While the specific architectures are not fully detailed in the provided snippet, the imports suggest the use of `tensorflow.keras.models.Sequential` and `tensorflow.keras.models.Model`, indicating the potential for various CNN architectures. `ImageDataGenerator` is used for data augmentation.

## Results

The notebook includes sections for evaluating model performance using metrics such as confusion matrices and classification reports, generated with `sklearn.metrics.confusion_matrix` and `sklearn.metrics.classification_report`.

## Contributing

Feel free to fork this repository, make improvements, and submit pull requests.


