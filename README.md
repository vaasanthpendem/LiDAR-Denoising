## LiDAR Signal Denoising
> Applying 1D signal processing algorithms — SVD, VMD, and Wavelet Transform — to real-world LiDAR point cloud data from the KITTI dataset.

---

## Overview

This project bridges classroom signal processing theory with a practical robotics perception challenge: denoising LiDAR intensity signals. Starting from raw `.bin` point cloud files, we build a full pipeline from data preprocessing through denoising and quantitative evaluation.

---

## Pipeline

```
Raw .bin (KITTI) → Preprocessing → Azimuthal 1D Signal → Denoising → Evaluation (SNR / RMSE)
```

### 1. Preprocessing
- Loaded raw point cloud data from the **KITTI dataset** (`.bin` format)
- Applied **Voxel Downsampling** to reduce data density uniformly
- Applied **Statistical Outlier Removal (SOR)** to eliminate gross noise points

### 2. Dimensionality Reduction — Azimuthal Representation
- Converted the 3D point cloud into a **1D intensity signal** by computing the **maximum intensity per azimuthal bin**
- This representation preserves the most salient intensity variation along each angular sweep

### 3. Denoising Methods
Three algorithms were implemented and compared:

| Method | Type | Key Strength |
|---|---|---|
| **SVD** (Singular Value Decomposition) | Linear algebraic | Fast, interpretable rank reduction |
| **VMD** (Variational Mode Decomposition) | Adaptive signal decomposition | Handles non-stationary signals well |
| **Wavelet Transform** | Multi-resolution analysis | Effective at localizing transient features |

### 4. Evaluation
- Quantified performance using **Signal-to-Noise Ratio (SNR)** and **Root Mean Square Error (RMSE)**
- Results visualized through comparative plots across all three methods

---

## Results

**VMD outperformed SVD and Wavelet Transform** on the KITTI dataset across both SNR and RMSE metrics, demonstrating superior adaptability to the non-stationary nature of LiDAR intensity signals.

---

## Technical Choices

- **Language**: Python — chosen for its rich scientific ecosystem (`numpy`, `scipy`, `open3d`, `PyWavelets`, `vmdpy`)
- **Dataset**: [KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/)
- **Scope**: 1D analysis (hardware constraints limited full 2D/3D denoising experimentation)

---

## Project Context

Developed as a hands-on application of graduate-level signal processing coursework to a real-world autonomous driving perception problem. The project incorporates AI-assisted development workflows alongside traditional engineering methodology.

---

## Repository Structure

```
LiDAR_denoising/
├── data/               # Raw .bin KITTI files
├── preprocessing/      # Voxel downsampling, SOR
├── denoising/          # SVD, VMD, Wavelet implementations
├── evaluation/         # SNR, RMSE computation and plots
└── results/            # Output figures and metrics
```

