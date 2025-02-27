<h1 style="font-family: 'Arial', sans-serif;">GBM_LCCM</h1>

This repository contains the code for estimating Gaussian Bernoulli Mixture – Latent Class Choice Models (GBM-LCCM) using the Expectation Maximization (EM) algorithm. The model combines a Gaussian Mixture Model (GMM) for continuous variables and a Bernoulli (dummy) mixture for discrete variables to determine latent classes. Each latent class also has its own weighted multinomial logit model for modeling choice behavior.

## Overview
The GBM_LCCM code: 
- **Processes Data:** Constructs sparse matrices for choice sets, decision-makers, and observations from long-format choice data.
- **Estimates Class Membership:** Uses a Gaussian mixture approach (with support for multiple covariance structures such as full, tied, diag, and spherical) and a Bernoulli model for discrete/dummy variables.
- **Estimates Class-Specific Models:** Implements a weighted multinomial logit model for each latent class.
- **Optimizes via EM:** Alternates between calculating latent class responsibilities (E-step) and updating model parameters (M-step) until convergence.
- **Outputs Results:** Displays model fit statistics (log-likelihood, AIC, BIC), parameter estimates (with standard errors, t-statistics, and p-values), and optionally exports prediction enumerations.

## Requirements
- Python 3.7+ (tested with Python 3.12)

## Usage 
For an example of how to use this code, refer to the Jupyter Notebook: GBM_LCCM_Example.ipynb.

## Configuration
- **Covariance Types:** Choose from 'full', 'tied', 'diag', or 'spherical'.
- **EM Algorithm Settings:** Adjust tol (tolerance) and max_iter (maximum iterations) to control convergence.
- **Initialization:** Choose between 'random' or 'kmeans' initialization for the GMM parameters (GMM_Initialization).

## Contributing
Contributions, bug fixes, and feature suggestions are welcome. Please open an issue, submit a pull request, or feel free to contact me directly.

## Acknowledgements
- **Author:** Georges Sfeir.
- **Advising:** Filipe Rodrigues.
- **Based On:** The latent class choice model (lccm) package, methodologies from Bishop's Pattern Recognition and Machine Learning, and some functions from the GaussianMixture class of sklearn.mixture.

## More Information
For more information about the GBM_LCCM model see the following paper:
 
Sfeir, G., Abou-Zeid, M., Rodrigues, F., Pereira, F.C., Kaysi, I. (2021). “Latent class choice model with a flexible class membership component: A mixture model approach.” Journal of Chocie Modelling, 41, 100320. https://doi.org/10.1016/j.jocm.2021.100320

## Citation
If you find this model useful in your research or work, please cite it by citing the paper above.
