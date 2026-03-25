# Computer Vision — Multi-Layer Neural Networks

Implemented a multi-layer neural network from scratch using **NumPy** to reconstruct images from **2D pixel coordinates**.  
This project explores how different **input feature mappings** affect reconstruction quality, especially when comparing standard coordinate inputs against **Fourier-based encodings**.

## Overview

This repository contains a coordinate-based image reconstruction model where each input point `(x, y)` is mapped to an output color `(r, g, b)`.  
The neural network is implemented manually, including:

- forward propagation
- backpropagation
- ReLU activations for hidden layers
- sigmoid output activation
- mean squared error loss
- parameter updates with **SGD**
- optional **Adam** optimizer support

The main goal of the project is to study how feature mappings influence the network’s ability to capture image detail at both **low** and **high** resolutions.

## Key Idea

Instead of feeding raw image pixels directly into a model, this project trains a neural network to learn a function:

\[
(x, y) \rightarrow (r, g, b)
\]

For each coordinate in an image, the network predicts the corresponding RGB value.

This allows the project to test how different coordinate encodings change the model’s ability to represent fine visual structure.

## Feature Mapping Strategies

The notebook compares several input representations:

### 1. No Mapping
Raw coordinate input:
\[
\gamma(\mathbf{v}) = \mathbf{v}
\]

### 2. Basic Mapping
Applies sine and cosine transforms to the input coordinates:
\[
\gamma(\mathbf{v}) = [\cos(2\pi \mathbf{v}), \sin(2\pi \mathbf{v})]^T
\]

### 3. Gaussian Fourier Feature Mapping
Uses a sampled Gaussian matrix \( \mathbf{B} \) to transform coordinates before applying sine/cosine:
\[
\gamma(\mathbf{v}) = [\cos(2\pi \mathbf{Bv}), \sin(2\pi \mathbf{Bv})]^T
\]

Experiments include Fourier mappings with different scales such as:

- `gauss_1.0`
- `gauss_10.0`

These mappings help the network represent higher-frequency image detail more effectively.

## Repository Structure

```bash
.
├── neural_net.py              # Neural network implementation from scratch
├── neural_network.ipynb       # Main notebook for training and experiments
├── vinayek2_mp2_output.pdf    # Output visualizations/results
├── vinayek2_mp2_report.pdf    # Written report and analysis
└── README.md
