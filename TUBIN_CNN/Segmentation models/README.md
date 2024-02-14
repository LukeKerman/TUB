# Thermal Infrared Image Classifier

This repository contains the code and models for a machine learning project focused on classifying thermal infrared (IR) images. The models aim to identify and segment regions of interest within IR imagery.

## Overview
This repository contains the necessary files for training and evaluating an infrared cloud classification and mask prediction model. The model is designed to analyze infrared images and predict cloud masks, comparing them to ground truth data for evaluation.## Models

## Contents
The repository includes several trained models:

- `best_IR_clf_256x16_wc_2.keras` - Another variation of the model with a different configuration or training checkpoint.
- `best_IR_clf_256x16_wc_2.hdf5`
- `best_IR_clf_256x16_wc.keras` - The first model file.
- `best_IR_img2classmap_128x32_wc.keras` - A model trained to 128x128 images and class labels.

## Code

- `TUBIN_IR_classifier_model.ipynb` - A Jupyter notebook containing the code for training the models, evaluating them, and possibly visualizing the results.

## Results

![Evaluation Results](eval_plot_selected.png)

The examples above show the evaluation results of the model. It includes test images, predicted masks, ground truth masks, and the differences between predictions and ground truth.

![Full Frame Result](full_frame_pred.png)

Output layers with different thresholds and fire pixel voting.

## Usage

To use these models for your own IR image classification and segmentation, you'll need to set up an environment with the necessary dependencies, load the models, and run the inference on your images.

### Prerequisites

- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib (for visualization)