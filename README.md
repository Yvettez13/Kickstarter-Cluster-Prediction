## Kickstarter Project Analysis
This repository hosts the analysis of a Kickstarter dataset, focusing on developing machine learning models for classification and clustering. The goal is to categorize creative projects, aiding funders in making informed decisions and enhancing project success rates.

# Project Structure
# Project Structure

## data/
Contains the raw dataset used for the Kickstarter project analysis. This directory may include various data files in formats such as CSV, JSON, or SQL dumps, which are essential for running the models and conducting the analysis.

## models/
This directory houses the trained machine learning models and their serialization files. It includes files like `.pkl` for the Gradient Boosting Classifier and other models used in the project. These files are used for predictions and can be loaded directly for further analysis or modification.

## notebooks/
Jupyter notebooks with detailed analysis are stored here. These notebooks provide a step-by-step walkthrough of the data analysis process, including data cleaning, exploration, model training, and evaluation. They are interactive, allowing users to see the code, run it, and view the results in real-time.

## src/
The source code directory contains scripts for data preprocessing, model training, and evaluation. It includes various Python files that define functions and classes used across the project. This directory is the backbone of the project, containing the logic for data manipulation, model building, and performance evaluation.

### src/ structure:
- `preprocessing.py`: Code for cleaning and preparing the data.
- `train.py`: Scripts for training machine learning models.
- `evaluate.py`: Evaluation scripts to assess model performance.
- `utils.py`: Utility functions used across the project.


# Classification Model

The classification model, based on 45 variables, predicts project outcomes as 'successful' or 'failed'. After rigorous data preprocessing, the Gradient Boosting Classifier was selected for its superior performance, exhibiting a 79.68% accuracy.

The model is robust in handling diverse project types and scales, making it a versatile tool for potential Kickstarter creators and investors.

# Clustering Model

The clustering model segments projects into six distinct groups using PCA and K-Means. Each cluster is identified based on features like backer count, pledged amount, and timeline metrics.

These clusters help in understanding the different types of projects that thrive on Kickstarter, offering tailored strategies for each category.

