![DEEPFRUIT](logo.png)

# General-Purpose CNN-VAE Anomaly Detection Framework

This project implements a Convolutional Variational Autoencoder (CNN-VAE) for unsupervised anomaly detection on grayscale images.  
Originally built around the Fruits-360 dataset, the system has been generalized to support any folder-organized image dataset.

The VAE learns to reconstruct normal images; anomalies are identified based on reconstruction error thresholds.


## Features

### Modular Preprocessing
- Automatic image discovery (.jpg/.jpeg/.png)
- RGB → grayscale conversion
- Min-max normalization
- Optional visualization for debugging and inspection

### Flexible Dataset Loader
Loads any directory structure of the form:
```
dataset/
    Training/
        Class1/
        Class2/
    Test/
        Class1/
        Class2/
```

Supports:
- Per-class image limits  
- Total image limits  
- Filtering out unwanted classes or patterns

### CNN-Based Variational Autoencoder
- Encoder: convolutional downsampling + latent vector sampling
- Decoder: transposed convolutions for upsampling
- Uses KL divergence + reconstruction loss
- Designed for 100×100 grayscale images but easily configurable

### Anomaly Detection Pipeline
- Computes per-image reconstruction error (MSE)
- Learns threshold from training set (default: 95th percentile)
- Folder-level classification:
  - Number of anomalies
  - Percent anomalies
  - Clean or Anomalous classification
- Summaries printed to terminal


## Repository Structure

```
project_root/
├── main.py                Full training + anomaly detection pipeline
├── train_vae.py           Simple VAE training script for clean datasets
├── cnn_vae.py             VAE architecture
├── dataset_loader.py      Dataset reader + preprocessing
├── pre_processing.py      Image normalization utilities
└── README.md
```


## Dataset Requirements

Prepare your dataset like this:

```
dataset_root/
    Training/
        CategoryA/
        CategoryB/
    Test/
        CategoryA/
        CategoryB/
```

You may rename “Training” and “Test” by editing the configuration in main.py.

Supported formats: .jpg, .jpeg, .png


## Getting Started

### 1. Install Dependencies

Run:
```
pip install tensorflow numpy scikit-image matplotlib scikit-learn
```

### 2. Place Your Dataset

For example:
```
dataset/
    Training/
        NormalClass1/
        NormalClass2/
    Test/
        MixedClass1/
        MixedClass2/
```

### 3. Run the Full Pipeline

```
python main.py
```

This:
- Loads & preprocesses training data
- Trains the VAE
- Computes reconstruction errors
- Applies anomaly threshold
- Prints anomaly summary by folder

### 4. Train Only the VAE

```
python train_vae.py
```

Useful for pretraining on clean datasets.


## Model Architecture Overview

### Encoder
- Convolution: 100×100 → 50×50
- Convolution: 50×50 → 25×25
- Convolution (same size)
- Flatten + Dense
- Outputs:
  - z_mean
  - z_log_var
  - sampled z

### Decoder
- Dense → reshape to 25×25×128
- Transposed convolutions to upsample back to 100×100
- Sigmoid output for normalized grayscale reconstruction

### Loss
- Reconstruction loss (binary cross entropy)
- KL divergence
- Total loss = recon_loss + KL


## Anomaly Detection Method

1. Compute reconstruction error for each training image  
2. Determine threshold as the 95th percentile (configurable)  
3. For each test image:

```
error = mean( (x - reconstructed_x)^2 )
```

4. Errors above threshold → anomaly  
5. Compute folder-level anomaly percentage  
6. Mark folder as Clean or Anomalous based on configured percent threshold


## Configuration

Edit these in main.py:
```
DATASET_BASE = "dataset"
EPOCHS = 60
BATCH_SIZE = 64
TRAIN_SPLIT_NAME = "Training"
TEST_SPLIT_NAME = "Test"
ANOMALY_THRESH_PERCENT = 10
LIMIT_DATA = False
NUM_TRAIN_IMAGES = 12000
NUM_TEST_IMAGES = 3200
```


## Use Cases

This framework can be applied to:

- Manufacturing defect detection  
- Medical imaging anomaly detection  
- Dataset quality control  
- Out-of-distribution detection  
- Security anomaly detection  
- Industrial inspection workflows  


## Credits

This project originally used the Fruits-360 dataset.  
Please credit the authors if your work incorporates it:

Fruits-360 Dataset (Mureșan & Oltean)  
https://www.kaggle.com/datasets/moltean/fruits

If used in academic work, cite using the authors' recommended format.
