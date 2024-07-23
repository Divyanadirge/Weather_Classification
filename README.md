# Weather_Classification
This project focuses on weather classification using the ResNet152V2 model, achieving an accuracy of 91%. The dataset is visualized and analyzed, revealing an imbalance among the classes. Data augmentation techniques are employed to enhance model training, followed by thorough evaluation to ensure model performance.



# Weather Classification using ResNet152V2

This project focuses on classifying weather conditions using the ResNet152V2 deep learning model, achieving an accuracy of 91%. The workflow includes data exploration, preprocessing, model training, evaluation, and prediction on new data.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Prediction](#prediction)
- [Conclusion](#conclusion)
- [Requirements](#requirements)
- [Usage](#usage)

## Introduction
This project aims to classify weather conditions from images using a deep learning approach. We utilize the ResNet152V2 architecture, which is known for its robust performance in image classification tasks. The model is trained on a dataset with various weather conditions and achieves a notable accuracy of 91%.

## Dataset
The dataset is stored in the `carla_trafficdata 1/train` directory. It contains images categorized into different weather conditions. During exploration, we identified an imbalance in class distribution, which is addressed through data augmentation.

## Data Preprocessing
Data augmentation techniques such as rescaling, horizontal flipping, and rotation are employed using Keras' `ImageDataGenerator`. These techniques help in generating a more diverse training set, enhancing the model's ability to generalize.

## Model Architecture
Several models, including Inception and Xception, were considered. Ultimately, ResNet152V2 was chosen for its superior performance. The model is initialized with weights pre-trained on ImageNet and fine-tuned on our dataset.

## Training
The model is trained using the Keras `fit` method, with specified epochs and callbacks to monitor performance. The training process is documented, highlighting adjustments and optimizations made to improve accuracy.

## Evaluation
Post-training, the model is evaluated on a validation dataset. The evaluation metrics demonstrate the model's ability to accurately classify weather conditions. The notebook includes code for loading the trained model and making predictions on new data.

## Prediction
The model's prediction capabilities are tested on unseen data, confirming its effectiveness in real-world scenarios. Sample predictions are provided to illustrate the model's performance.

## Conclusion
This project showcases a structured approach to weather classification using deep learning. With an accuracy of 91%, the ResNet152V2 model proves effective for this task. The project emphasizes the importance of data visualization, augmentation, and thorough evaluation in developing a reliable model.

## Requirements
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Plotly
- Seaborn
- Pandas

## Usage
To run the notebook:
1. Clone this repository.
2. Install the required libraries using `pip install -r requirements.txt`.
3. Open the notebook and run the cells sequentially.

---

This README provides a comprehensive overview of the project, guiding users through each phase from data preprocessing to model evaluation and prediction. Feel free to adjust any sections to better fit your project's specifics.
