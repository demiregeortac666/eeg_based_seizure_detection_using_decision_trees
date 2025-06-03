# EEG Seizure Detection GUI

A graphical interface for analyzing EEG seizure detection data and building classification models.

## Features

- Data visualization and exploration
- Feature distribution analysis
- Model training and evaluation
- Results visualization (confusion matrix, ROC curve, etc.)

## Installation

1. Ensure you have Python 3.7+ installed
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the application:

```bash
python eeg_seizure_gui.py
```

### Data Loading

1. The application will try to load "output/selected_features.csv" by default if it exists
2. Otherwise, click "Browse" to select a CSV file containing EEG features data
3. Click "Load Data" to load the selected file

### Data Exploration

The "Data Overview" tab displays:
- Summary statistics about the dataset
- A preview of the data

### Data Visualization

The "Data Visualization" tab offers several visualization options:
- Feature Distribution: Visualize the distribution of individual features, split by class
- Feature Correlation: View the correlation matrix of features
- Class Distribution: See the balance between seizure and non-seizure samples
- Feature Importance: After training a model, view the most important features

### Model Training

The "Modeling" tab allows you to:
1. Select a machine learning algorithm (Random Forest, SVM, K-NN)
2. Configure training parameters (test size, random state)
3. Train the model on the loaded data
4. View model performance metrics and the confusion matrix

### Results Analysis

The "Results" tab provides detailed visualization of model performance:
- Confusion Matrix: View model classification errors
- ROC Curve: Analyze the trade-off between sensitivity and specificity
- Precision-Recall Curve: Examine model precision and recall
- Feature Importance: Identify the most predictive features

## Project Structure

- `eeg_seizure_gui.py`: Main application file
- `requirements.txt`: Dependencies
- `output/selected_features.csv`: Default data file path

## Note

This GUI is designed to work with the output of the EEG seizure detection pipeline. The expected data format includes:
- EEG feature columns
- A 'label' column (0 for non-seizure, 1 for seizure)
- Optional 'patient' and 'file' columns for metadata 