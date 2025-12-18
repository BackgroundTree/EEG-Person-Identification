# EEG Person Identification via Hybrid CNN-BiLSTM

This repository contains a deep learning pipeline for identifying individuals based on their EEG (Electroencephalography) brainwave patterns. It utilizes a hybrid architecture combining **Convolutional Neural Networks (CNNs)** for spatial/frequency feature extraction and **Bidirectional LSTMs** for temporal dynamics.

## Project Overview
Biometric identification using EEG signals offers a high-security alternative to traditional methods as brainwaves are difficult to spoof. This project implements a robust pipeline that transforms raw EEG signals into Time-Frequency spectrograms and classifies them using a deep neural network.

## Dataset
This project uses the **EEG Motor Movement/Imagery Dataset** provided by PhysioNet.
* **Source:** [PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/)
* **Subjects:** 109 volunteers.
* **Structure:** 64-channel EEG recordings sampled at 160 Hz.
* **Split:**
    * **Training:** Session A (Runs 1–7)
    * **Testing:** Session B (Runs 8–14)

## The Pipeline

### 1. Preprocessing (MNE & SciPy)
Raw EDF files are processed to remove noise and standardize inputs:
* **Filtering:** FIR Bandpass filter (1–40 Hz) to isolate relevant brain rhythms (Delta through Beta).
* **Epoching:** Signals are segmented into 2.0-second windows with a 0.5-second step.
* **Feature Extraction:** Short-Time Fourier Transform (STFT) is applied to generate spectrograms, converting 1D signals into 2D images (Frequency × Time).

### 2. Model Architecture
The model (`CNNBiLSTM`) is designed to capture spatiotemporal features:
1.  **CNN Encoder:**
    * Stack of 2D Convolutional layers with Batch Normalization and ReLU.
    * **Asymmetric Pooling:** Max pooling is applied primarily to the Frequency dimension, preserving the Time dimension to allow the LSTM to analyze temporal changes.
2.  **Temporal Processing:**
    * A **Bidirectional LSTM** processes the sequence of feature vectors extracted by the CNN.
3.  **Classification:**
    * A dense classification head maps the final LSTM states to subject identities.

## Requirements
The project relies on the following major libraries:
* `Python 3.11+`
* `PyTorch` (Model training)
* `MNE` (EEG signal loading and filtering)
* `NumPy` & `Pandas` (Data manipulation)
* `SciPy` (Signal processing/STFT)
* `Scikit-Learn` (Metrics and Label Encoding)

## Usage
The pipeline is contained within `eeg-person-id.ipynb`.

1.  **Configure Paths:** Update the `Config` class in the notebook to point to your local dataset path.
    ```python
    Data_Root = Path("/path/to/eeg-motor-movementimagery-dataset")
    ```
2.  **Run Pipeline:** Set `RUN_PIPELINE = True` to process raw data into `.npy` arrays.
3.  **Train:** Run the training cells to train the CNN-BiLSTM model.
4.  **Evaluate:** The notebook automatically generates a classification report, confusion matrix, and t-SNE plot.

## Results
* **Visualization:** The notebook includes t-SNE visualization to project the high-dimensional latent embeddings into 2D, demonstrating how the model clusters signals by subject identity.
* **Metrics:** Standard accuracy, Precision, Recall, and F1-Score are calculated per subject.
