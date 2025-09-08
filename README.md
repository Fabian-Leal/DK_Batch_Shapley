# Interpreting Deepkriging for Spatial Interpolation in Geostatistics

## Overview
*Written by Fabian Leal-Villaseca, Edward Cripps, Mark Jessell, and Mark Lindsay.*

This repository contains the code and data associated with the paper "[Interpreting Deepkriging for Spatial Interpolation in Geostatistics](https://www.sciencedirect.com/science/article/pii/S0098300424003169)".


The paper produces an interpretation method for Deepkriging, a deep learning model tailored for geostatistical interpolation problems. The study demonstrates the feasibility of interpreting Deepkriging results using an adaptation of a well established feature attribution method, addressing key challenges of deep neural networks in capturing spatial dependencies and model interpretability.

## Repository Structure

- **Data**: Directory containing the datasets used in the study.
- **Figures**: Directory containing figures generated during the analysis.
- **src**: Jupyter notebooks and scripts used for various stages of the analysis:
  - `3d_plots.ipynb`: Notebook for generating 3D plots of the results of Deepkriging, and the benchmark methods, and the original data.
  -  `3d_uncert_quant.ipynb`: Notebook for generating the full grid interpolation plots and the uncertainty estimates for each interpolation method in the comparison.
  - `Basis_function_generation_and_comparison.ipynb`: Notebook for generating and comparing Deepkriging's performance with different levels of basis functions.
  - `Data_pre-processing.ipynb`: Notebook for data preprocessing steps, mainly variable selection, missing value treatment and normalisation.
  - `Deepkriging_benchmarking.ipynb`: Notebook for benchmarking the Deepkriging model against other methods using accuracy metrics in a train/test format.
  - `Explanations.ipynb`: Notebook for generating explanations and feature importance using the proposed methodology. The Batched Shapley algorithm is contained in this notebook.
  - `Spatial_clustering.ipynb`: Notebook for performing spatial clustering analyses to select a data dense area to conduct our case study.
- **Scripts**: Python scripts:
  - `dk_model.py`: Script containing the Deepkriging model implementation.

## Key Features

- **Deepkriging**: A deep neural network designed for spatial interpolation in geostatistics. This model expands the spatial coordinate feature space using basis functions, enabling the network to learn spatial dependencies effectively. It was developed by Chen et al. (2020). See https://arxiv.org/abs/2007.11972.
- **Batched Shapley**: An adaptation of Shapley values, a game-theoretic approach used for feature importance, that we developed specifically for interpreting Deepkriging. This method provides insights into the importance of features for each prediction made by a predictive model. In this case, these are density spatial interpolation predictions.
- **Benchmarking**: Comparison of Deepkriging with traditional kriging methods, demonstrating superior performance in both purely spatial cases and scenarios involving additional variables.

## References
- Chen, W., Li, Y., Reich, B.J., Sun, Y., 2020. Deepkriging: Spatially dependent deep neural networks for spatial prediction. arXiv preprint461
arXiv:2007.11972 .

## License
This project is licensed under the [MIT License](LICENSE).
