# ECG-Based Heart Disease Classification

## Project Overview

This project aims to develop a machine learning model for classifying heart diseases based on 12-lead ECG data and patient information. 

## Motivation

Electrocardiograms (ECGs) are a crucial tool in diagnosing various heart conditions, including arrhythmias, myocardial ischemia, and myocardial injury. By automating the analysis of ECG data, we can potentially improve the speed and accuracy of heart disease diagnosis, leading to better patient outcomes and more efficient healthcare delivery.

## Dataset

The dataset used in this project includes:

- 12-lead ECG recordings
- Patient information (age, gender, height, weight, etc.)
- Labels indicating the presence and type of heart disease

## Methodology

The project utilizes the following approach:

1. Data Preprocessing:
   - Handling missing values
   - Encoding categorical variables
   - Scaling numerical features

2. Feature Engineering:
   - Extracting relevant features from ECG signals
   - Combining ECG features with patient information

3. Model Architecture:
   - Hand craft Naive Bayes Classifier and PCA 
   - Custom loss function to handle multi-label classification


## Key Features

- Multi-label classification for various heart diseases
- Integration of ECG signal data with patient metadata
- Custom loss function to handle class imbalance
- Early stopping mechanism to optimize model performance


## Acknowledgements

This project was developed as part of a machine learning class focused on ECG-based heart disease classification. 
