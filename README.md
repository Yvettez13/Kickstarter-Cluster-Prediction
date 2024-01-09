## Kickstarter Project Analysis
# Introduction
This repository hosts the analysis of a Kickstarter dataset, focusing on developing machine learning models for classification and clustering. The goal is to categorize creative projects, aiding funders in making informed decisions and enhancing project success rates.

# Project Structure
data/ - Dataset used for analysis.
models/ - Trained machine learning models and their serialization files.
notebooks/ - Jupyter notebooks with detailed analysis.
src/ - Source code for data preprocessing, model training, and evaluation.

# Classification Model

The classification model, based on 45 variables, predicts project outcomes as 'successful' or 'failed'. After rigorous data preprocessing, the Gradient Boosting Classifier was selected for its superior performance, exhibiting a 79.68% accuracy.

The model is robust in handling diverse project types and scales, making it a versatile tool for potential Kickstarter creators and investors.

# Clustering Model

The clustering model segments projects into six distinct groups using PCA and K-Means. Each cluster is identified based on features like backer count, pledged amount, and timeline metrics.

These clusters help in understanding the different types of projects that thrive on Kickstarter, offering tailored strategies for each category.

