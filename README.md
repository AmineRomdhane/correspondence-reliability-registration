# Correspondence Reliability Prediction for Point Cloud Registration

This repository contains a clean reproducible version of the code and experiment outputs for the study:

**Reliability Weight Prediction for Candidate Correspondences in 3D Point Cloud Registration**

## Context

Rigid point cloud registration is a key component in simultaneous localization and mapping (SLAM) and lifelong mapping. In practical CAD-to-scan and scan-to-scan situations, candidate correspondences can contain false matches caused by noise, partial overlap, different point densities, repeated structures, and scene changes.

This project studies whether a lightweight machine-learning model can predict reliability weights for candidate correspondences, and whether these weights improve one-shot rigid registration.

## Method Summary

The pipeline is:

1. Generate synthetic point-cloud pairs with controlled transformations, noise, partial overlap, and outliers.
2. Curate real CAD/scan registration cases used as pseudo-ground truth.
3. Extract candidate correspondences after an initial alignment.
4. Compute geometric and descriptor-based features.
5. Train a multilayer perceptron (MLP) to predict correspondence reliability.
6. Use predicted weights in a one-shot weighted Singular Value Decomposition (SVD)-based registration solver.
7. Compare against an unweighted one-shot SVD baseline.

## Main Tools

- Python
- Open3D
- NumPy
- Pandas
- scikit-learn
- PyTorch
- Matplotlib

## Repository Structure

- `src/`: source scripts for data generation, training, and registration comparison
- `configs/`: dataset configuration files
- `results/`: generated tables and figures
- `papers/`: article files and references
- `docs/`: notes and supplementary documentation
- `data/`: data instructions only; large raw data is not tracked in Git

## Main Result

In the leave-one-real-case-out v3 experiment, the MLP-weighted SVD method achieved similar nearest-neighbor RMSE to the unweighted SVD baseline, but improved transform-related metrics in most held-out real cases.

The MLP-weighted method outperformed the unweighted one-shot SVD baseline in:

- 66.7% of cases for translation error
- 75.0% of cases for rotation error
- 75.0% of cases for point-action RMSE

This suggests that learned correspondence reliability weights can reduce transform drift, although one-shot weighted SVD is not sufficient for fully robust real-case registration.

## Data Availability

Large raw point clouds, CAD files, and ROS bags are not tracked directly in this repository. They should be stored separately or added through Git LFS or an external archive.
