"""
@name:            Gaussian Bernoulli Mixture - Latent Class Choice Model
@author:          Georges Sfeir
@summary:         Contains functions necessary for estimating Gaussiam Bernoulli Mixture latent class choice models
                  using the Expectation Maximization algorithm
@acknowledgement: Filipe Rodrigues for advising during the development of this code.

            
General References
------------------
This code is based on the latent class choice model (lccm) package which can be downloaded from:
    https://github.com/ferasz/LCCM
This code also relies on some function from the GaussianMixture class of sklearn.mixture:
   https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture
"""


import numpy as np

# pylogit imports the following: from collections import Iterable. However, in Python 3, the abstract base classes like Iterable have been moved to the collections.abc module. 
# To fix this issue, either locate the files in pylogit where Iterable is imported and change the line to: from collections.abc import Iterable
# Or add the following before importing pylogit in the code.
import collections.abc
import collections
collections.Iterable = collections.abc.Iterable

import pylogit
from scipy.sparse import coo_matrix
from scipy.optimize import minimize
import scipy.stats
from datetime import datetime
import warnings
from scipy.special import logsumexp
from scipy import linalg
from sklearn.cluster import KMeans
from sklearn.utils.extmath import row_norms


# Global variables
emTol = 1e-04
llTol = 1e-06
grTol = 1e-06
maxIters = 10000


def processClassSpecificPanel(dms, dmID, obsID, altID, choice):
    """
    Method that constructs a tuple and three sparse matrices containing information 
    on available observations, and available and chosen alternative
    
    Parameters
    ----------
    dms : 1D numpy array of size nDms.
        Each element identifies a unique decision-maker in the dataset.
    dmID : 1D numpy array of size nRows.
        Identifies which rows in the dataset correspond to which decision-maker.
    obsID : 1D numpy array of size nRows.
        Identifies which rows in the dataset correspond to which observation.
    altID : 1D numpy array of size nRows.
        Identifies which rows in the dataset correspond to which alternative.
    choice : 1D numpy array of size nRows.
        Identifies which rows in the dataset correspond to alternatives that were 
        chosen during the corresponding observation.
    
    Returns
    -------
    altAvTuple : a tuple containing arrays of size nRows.
        The first array denotes which row in the data file corresponds to which 
        row in the data file (redundant but retained for conceptual elegance) and 
        the second array denotes which row in the data file corresponds to which 
        observation in the data file.    
    altChosen : sparse matrix of size (nRows x nObs).
        The (i, j)th element of the returned matrix  is 1 if the alternative corresponding 
        to the ith row in the data file was chosen during observation j, and 0 otherwise.   
    obsAv : sprase matrix of size (nObs x nDms).
        The (i, j)th element of the returned matrix is 1 if observation i corresponds to 
        decision-maker j, and 0 otherwise.
    rowAv : sparse matrix of size (nRows x (nAlts * nDms)).
        The (i, ((n - 1) * nAlts) + j)th element of the returned matrix is 1 if the ith row 
        in the data file corresponds to the jth alternative and the nth decision-maker, 
        and 0 otherwise.   
    """
    
    nRows = choice.shape[0]
    alts = np.unique(altID)
    nAlts = alts.shape[0]
    obs = np.unique(obsID)
    nObs = obs.shape[0]
    nDms = dms.shape[0]
    
    xAlt, yAlt = np.zeros((nRows)), np.zeros((nRows))
    xChosen, yChosen = np.zeros((nObs)), np.zeros((nObs))
    xObs, yObs = np.zeros((nObs)), np.zeros((nObs))
    xRow, yRow = np.zeros((nRows)), np.zeros((nRows))

    currentRow, currentObs, currentDM = 0, 0, 0    
    for n in dms:
        obs = np.unique(np.extract(dmID == n, obsID))
        for k in obs:      
            xObs[currentObs], yObs[currentObs] = currentObs, currentDM
            cAlts = np.extract((dmID == n) & (obsID == k), altID)        
            for j in cAlts:
                xAlt[currentRow], yAlt[currentRow] = currentRow, currentObs  
                xRow[currentRow], yRow[currentRow] = currentRow, (np.where(dms == n)[0][0] * nAlts) + np.where(alts == j)[0][0]
                if np.extract((dmID == n) & (obsID == k) & (altID == j), choice) == 1:                
                    xChosen[currentObs], yChosen[currentObs] = currentRow, currentObs
                currentRow += 1
            currentObs += 1
        currentDM += 1
            
    altAvTuple = (xAlt, yAlt)
    altChosen = coo_matrix((np.ones((nObs)), (xChosen, yChosen)), shape = (nRows, nObs))
    obsAv = coo_matrix((np.ones((nObs)), (xObs, yObs)), shape = (nObs, nDms))
    rowAv = coo_matrix((np.ones((nRows)), (xRow, yRow)), shape = (nRows, nDms * nAlts))
    
    return altAvTuple, altChosen, obsAv, rowAv
    
    
def imposeCSConstraints(altID, availAlts):
    """
    Method that constrains the choice set for each of the decision-makers across the different
    latent classes following the imposed choice-set by the analyst to each class. 
    Usually, when the data is in longformat, this would not be necessary, since the 
    file would contain rows for only those alternatives that are available. However, 
    in an LCCM, the analyst may wish to impose additional constraints to introduce 
    choice-set heterogeneity.
    
    Parameters
    ----------
    altID : 1D numpy array of size nRows.
        Identifies which rows in the data correspond to which alternative.
    availAlts : List of size nClasses.
        Determines choice set constraints for each of the classes. The sth element is an 
        array containing identifiers for the alternatives that are available to decision-makers 
        belonging to the sth class.
    
    Returns
    -------
    altAvVec : 1D numpy array of size nRows.
        An element is 1 if the alternative corresponding to that row in the data 
        file is available, and 0 otherwise.
    """   
    altAvVec = np.zeros(altID.shape[0]) != 0   
    for availAlt in availAlts:
        altAvVec = altAvVec | (altID == availAlt)
    return altAvVec.astype(int)






def estimate_gmm_covariances(covariance_type, resp, X, nk, means, reg_covar, nClasses):
    """Estimate the gmm covariance matrix.
    Parameters
    ----------
    covariance_type: {'full', 'tied', 'diag', 'spherical'}
        The type of Gaussian Mixture Model (GMM) covariance matrices.

    resp: array-like, shape (n_samples, n_components)

    X: array-like, shape (n_samples, n_features)
        data (continuous variables) entering the class membership model in wide format

    nk: array-like, shape (n_components,)

    means: array-like, shape (n_components, n_features)
        The centers of the current components of the Gaussian Mixture Model (GMM)
        This is for continuous variables.

    reg_covar: float

    nClasses: Integer
        Number of classes/components.
        
    Returns: the covariance matrix of the current components
        The shape depends on the covariance type.
        full: (n_components, n_features, n_features)
        tied: (n_features, n_features)
        diag: (n_components, n_features)
        spherical: (n_components,)
    """
    n_samples, n_features = X.shape
    
    if covariance_type == 'full':
        covariances = np.empty((nClasses, n_features, n_features))
        for k in range(nClasses):
            diff = X - means[k]
            covariances[k] = np.dot(resp[:, k] * diff.T, diff) / nk[k]
            covariances[k].flat[::n_features + 1] += reg_covar
    elif covariance_type == 'tied':
        avg_X2 = np.dot(X.T, X)
        avg_means2 = np.dot(nk * means.T, means)
        covariances = avg_X2 - avg_means2
        covariances /= nk.sum()
        covariances.flat[::len(covariances) + 1] += reg_covar
    elif covariance_type == 'diag':
        avg_X2 = np.dot(resp.T, X * X) / nk[:, np.newaxis]
        avg_means2 = means ** 2
        avg_X_means = means * np.dot(resp.T, X) / nk[:, np.newaxis]
        covariances = avg_X2 - 2 * avg_X_means + avg_means2 + reg_covar
    elif covariance_type == 'spherical':
        avg_X2 = np.dot(resp.T, X * X) / nk[:, np.newaxis]
        avg_means2 = means ** 2
        avg_X_means = means * np.dot(resp.T, X) / nk[:, np.newaxis]
        covariances = (avg_X2 - 2 * avg_X_means + avg_means2 + reg_covar).mean(1)

    return covariances



def compute_precisions_choleskey(covariance_type, covariances):
    """Compute the Cholesky decomposition of the precisions.

    Parameters
    ----------
    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        The type of Gaussian Mixture Model (GMM) covariance matrices.
        
    covariances : array-like
        The covariance matrix of the current components.
        The shape depends on the covariance_type.

    Returns
    -------
    precisions_cholesky : array-like
        The cholesky decomposition of sample precisions of the current
        components. The shape depends of the covariance_type.
    """
    estimate_precision_error_message = (
        "Fitting the mixture model failed because some components have "
        "ill-defined empirical covariance (for instance caused by singleton "
        "or collapsed samples). Try to decrease the number of components, "
        "or increase reg_covar.")

    if covariance_type in 'full':
        n_components, n_features, _ = covariances.shape
        precisions_chol = np.empty((n_components, n_features, n_features))
        for k, covariance in enumerate(covariances):
            try:
                cov_chol = linalg.cholesky(covariance, lower=True)
            except linalg.LinAlgError:
                raise ValueError(estimate_precision_error_message)
            precisions_chol[k] = linalg.solve_triangular(cov_chol,np.eye(n_features),lower=True).T

    elif covariance_type == 'tied':
        _, n_features = covariances.shape
        try:
            cov_chol = linalg.cholesky(covariances, lower=True)
        except linalg.LinAlgError:
            raise ValueError(estimate_precision_error_message)
        precisions_chol = linalg.solve_triangular(cov_chol, np.eye(n_features),lower=True).T

    else:
        if np.any(np.less_equal(covariances, 0.0)):
            raise ValueError(estimate_precision_error_message)
        precisions_chol = 1. / np.sqrt(covariances)

    return precisions_chol


def compute_log_det_cholesky(matrix_chol, covariance_type, n_features):
    """Compute the log-det of the cholesky decomposition of matrices.

    Parameters
    ----------
    matrix_chol : array-like
        Cholesky decompositions of the matrices.
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)

    covariance_type : {'full', 'tied', 'diag', 'spherical'}

    n_features : int
        Number of continuous features.

    Returns
    -------
    log_det_precisions_chol : array-like, shape (n_components,)
        The determinant of the precision matrix for each component.
    """
    if covariance_type == 'full':
        n_components, _, _ = matrix_chol.shape
        log_det_chol = (np.sum(np.log(matrix_chol.reshape(n_components, -1)[:, ::n_features + 1]), 1))

    elif covariance_type == 'tied':
        log_det_chol = (np.sum(np.log(np.diag(matrix_chol))))

    elif covariance_type == 'diag':
        log_det_chol = (np.sum(np.log(matrix_chol), axis=1))

    else:
        log_det_chol = n_features * (np.log(matrix_chol))

    return log_det_chol



def calClassMemProb(nClasses, means, means_dummy, mixing_coefficients, precisions_chol, covariance_type, X, X_Dummy):
    """
    Function that calculates the class membership probabilities for each individual in the
    dataset based on a GMM model.
    
    Parameters
    ----------
    nClasses : Integer.
        Number of classes/components.
    means : The centers of the current components of the Gaussian Mixture Model (GMM)
        This is for continuous variables.
    means_dummy: The centers of the current components of the Bernoulli Mixture Model (BMM)
        This is for dummy/discrete variables.
    mixing_coefficients : array-like, shape (nClasses,)
        The proportions of components of each mixture.
    precisions_chol : array-like
        The cholesky decomposition of sample precisions of the current
        components. The shape depends of the covariance_type.
    covariance_type: {'full', 'tied', 'diag', 'spherical'}
    X : data (continuous variables) entering the class membership model in wide format
    X_Dummy : data (discrete variables) entering the class membership model in wide format

    Returns
    -------
    p : Identifies the GMM-class membership probabilities for each individual and 
        each available latent class/component.
    pclass_d: Identifies the BMM-class membership probabilities for each individual and 
        each available latent class/component.
    """
    
    n_samples, n_features = X.shape
    _, n_dummyfeatures = X_Dummy.shape
    log_det = compute_log_det_cholesky(precisions_chol, covariance_type, n_features)
    
    """Estimate log gaussina probability = log N(x,mu,cov) = log p(x|z)"""
    if covariance_type == 'full':
        log_prob = np.empty((n_samples, nClasses))
        for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
            y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
            log_prob[:, k] = np.sum(np.square(y), axis=1)

    elif covariance_type == 'tied':
        log_prob = np.empty((n_samples, nClasses))
        for k, mu in enumerate(means):
            y = np.dot(X, precisions_chol) - np.dot(mu, precisions_chol)
            log_prob[:, k] = np.sum(np.square(y), axis=1)

    elif covariance_type == 'diag':
        precisions = precisions_chol ** 2
        log_prob = (np.sum((means ** 2 * precisions), 1) - 2. * np.dot(X, (means * precisions).T) + np.dot(X ** 2, precisions.T))

    elif covariance_type == 'spherical':
        precisions = precisions_chol ** 2
        log_prob = (np.sum(means ** 2, 1) * precisions - 2 * np.dot(X, means.T * precisions) + np.outer(row_norms(X, squared=True), precisions))

    """log p(x|z): equation 9.11 from Patern Recogniiton and Machine Learning (Bishop, 2006)"""
    log_prob_N = -.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det

    log_mixing_coefficients = np.log(mixing_coefficients)

    """ log(phi_k*N(x|mu,cov))"""
    weighted_log_prob = log_prob_N + log_mixing_coefficients

    """pclass_d: equation 9.44 from bishop """
    z = np.zeros([n_samples, nClasses])
    pclass_d = np.zeros([n_samples, nClasses])
    for n in range(n_samples):
        sumz=0
        for k in range(nClasses):
            resp_d = 1
            for i in range(n_dummyfeatures):
                if X_Dummy[n][i]==1:
                    resp_d*=means_dummy[k][i]
                else:
                    resp_d*=1-means_dummy[k][i]
            pclass_d[n][k]=resp_d


    p = np.exp(weighted_log_prob)

    return p, pclass_d


def calClassSpecificProbPanel(param, expVars, altAvMat, altChosen, obsAv):
    """
    Function that calculates the class specific probabilities for each decision-maker in the
    dataset
    
    Parameters
    ----------
    param : 1D numpy array of size nExpVars.
        Contains parameter values.
    expVars : 2D numpy array of size (nExpVars x (nRows)).
        Contains explanatory variables.
    altAvMat : sparse matrix of size (nRows x nObs).
        The (i, j)th element equals 1 if the alternative corresponding to the ith 
        column in expVars is available to the decision-maker corresponding to the 
        jth observation, and 0 otherwise.
    altChosen : sparse matrix of size (nRows x nObs).
        The (i, j)th element equals 1 if the alternative corresponding to the ith 
        column in expVars was chosen by the decision-maker corresponding to the 
        jth observation, and 0 otherwise.
    obsAv : sparse matrix of size (nObs x nInds).
        The (i, j)th element equals 1 if the ith observation in the dataset corresponds 
        to the jth decision-maker, and 0 otherwise.
    
    Returns
    -------
    np.exp(lPInd) : 2D numpy array of size 1 x nInds. (k x N)
        Identifies the class specific probabilities for each individual in the 
        dataset.
    """
    v = np.dot(param[None, :], expVars)       # v is 1 x nRows
    ev = np.exp(v)                            # ev is 1 x nRows
    ev[np.isinf(ev)] = 1e+20                  # As precaution when exp(v) is too large for machine
    ev[ev < 1e-200] = 1e-200                  # As precaution when exp(v) is too close to zero
    nev = ev * altAvMat                       # nev is 1 x nObs
    nnev = altAvMat * np.transpose(nev)       # nnev is nRows x 1
    p = np.divide(ev, np.transpose(nnev))     # p is 1 x nRows 
    p[np.isinf(p)] = 1e-200                   # When none of the alternatives are available
    pObs = p * altChosen                      # pObs is 1 x nObs
    lPObs = np.log(pObs)                      # lPObs is 1 x nObs
    lPInd = lPObs * obsAv                     # lPInd is 1 x nInds
    return np.exp(lPInd)                      # prob is 1 x nInds


def calClassSpecificProbScenarios(param, expVars, altAvMat, altChosen, obsAv):
    """
    Function that calculates the class specific probabilities for each decision-maker in the
    dataset
    
    Parameters
    ----------
    param : 1D numpy array of size nExpVars.
        Contains parameter values.
    expVars : 2D numpy array of size (nExpVars x (nRows)).
        Contains explanatory variables.
    altAvMat : sparse matrix of size (nRows x nObs).
        The (i, j)th element equals 1 if the alternative corresponding to the ith 
        column in expVars is available to the decision-maker corresponding to the 
        jth observation, and 0 otherwise.
    altChosen : sparse matrix of size (nRows x nObs).
        The (i, j)th element equals 1 if the alternative corresponding to the ith 
        column in expVars was chosen by the decision-maker corresponding to the 
        jth observation, and 0 otherwise.
    obsAv : sparse matrix of size (nObs x nInds).
        The (i, j)th element equals 1 if the ith observation in the dataset corresponds 
        to the jth decision-maker, and 0 otherwise.
    
    Returns
    -------
    np.exp(lPInd) : 2D numpy array of size 1 x nInds. (k x N)
        Identifies the class specific probabilities for each individual in the 
        dataset.
    """
    v = np.dot(param[None, :], expVars)       # v is 1 x nRows
    ev = np.exp(v)                            # ev is 1 x nRows
    ev[np.isinf(ev)] = 1e+20                  # As precaution when exp(v) is too large for machine
    ev[ev < 1e-200] = 1e-200                  # As precaution when exp(v) is too close to zero
    nev = ev * altAvMat                       # nev is 1 x nObs
    nnev = altAvMat * np.transpose(nev)       # nnev is nRows x 1
    p = np.divide(ev, np.transpose(nnev))     # p is 1 x nRows 
    p[np.isinf(p)] = 1e-200                   # When none of the alternatives are available
    pObs = p * altChosen                      # pObs is 1 x nObs
    lPObs = np.log(pObs)                      # lPObs is 1 x nObs
#    lPInd = lPObs * obsAv                     # lPInd is 1 x nInds
    return np.exp(lPObs)                      # prob is 1 x nInds


def initialize_parameters(X, X_Dummy, tol, reg_covar, max_iter, covariance_type, nClasses, GMM_Initialization):
    """Initialize the parameters of the Class membership model

    Parameters
    ----------
    X : data (continuous variables) entering the class membership model in wide format
    X_Dummy : data (discrete variables) entering the class membership model in wide format
    tol: float
    reg_covar: float
    max_iter: float
    covariance_type: {'full', 'tied', 'diag', 'spherical'}
        The type of precision matrices.
    nClasses: Integer
    GMM_Initialization: 'random' to initialize the clusters of the Gaussian Bernoulli Mixture models randomly or
                        'kmeans' to initinalize the clusters using the k-means clustering algorithms

    Returns
    -------
    resp: array-like, shape (n_samples, n_components)

    nk: array-like, shape (n_components,)

    means: array-like, shape (n_components, n_features)
    mixing_coefficients, means, means_dummy, covariances, precisions_chol, log_det

    covariances: array-like
        The covariance matrix of the current components.
        The shape depends on the covariance_type.

    precisions_chol: array-like
        The cholesky decomposition of sample precisions of the current
        components.
        The shape depends on the covariance_type.
            'full' : shape of (n_components, n_features, n_features)
            'tied' : shape of (n_features, n_features)
            'diag' : shape of (n_components, n_features)
            'spherical' : shape of (n_components,)

    log_det: array-like, shape (n_components,)
        The determinant of the precision matrix for each component.
    """

    n_samples, n_features = X.shape


    """Responsabilities are initialized using kmeans or randomly"""
    if GMM_Initialization == 'kmeans':
        resp = np.zeros((n_samples, nClasses))
        label = KMeans(n_clusters=nClasses, n_init=1).fit(X).labels_
        resp[np.arange(n_samples), label] = 1
    else:
        resp = np.random.rand(n_samples, nClasses)
        resp /= resp.sum(axis=1)[:, np.newaxis]

    """The effective number of points assigned to cluster k: equation 9.18 from (Bishop, 2006)"""
    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps

    """Weights of the gaussians: equation 9.22 from (Bishop, 2006)"""
    mixing_coefficients = nk/n_samples

    """Means: equation 9.17 from (Bishop, 2006)"""
    means = np.dot(resp.T, X) / nk[:, np.newaxis]

    """Means of dummy variables: equation 9.58 from (Bishop, 2006)"""
    means_dummy = np.dot(resp.T, X_Dummy) / nk[:, np.newaxis]
    
    covariances = estimate_gmm_covariances(covariance_type, resp, X, nk, means, reg_covar, nClasses)

    precisions_chol = compute_precisions_choleskey(covariance_type, covariances)

    log_det = compute_log_det_cholesky(precisions_chol, covariance_type, n_features)

    return resp, nk, mixing_coefficients, means, means_dummy, covariances, precisions_chol, log_det

    

def wtLogitPanel(param, expVars, altAv, weightsProb, weightsGr, altChosen, obsAv, choice):
    """
    Function that calculates the log-likelihood function and the gradient for a weighted
    multinomial logit model with panel data. 
    
    Parameters
    ----------
    param : 1D numpy array of size nExpVars.
        Contains parameter values.
    expVars : 2D numpy array of size (nExpVars x (nRows)).
        Contains explanatory variables.
    altAv : sparse matrix of size (nRows x nObs).
        The (i, j)th element equals 1 if the alternative corresponding to the ith 
        column in expVars is available to the decision-maker corresponding to the 
        jth observation, and 0 otherwise. 
    weightsProb : 1D numpy array of size nInds.
        The jth element is the weight to be used for the jth decision-maker.
    weightsGr : 1D numpy array of size nRows.
        The jth element is the weight to be used for the jth row in the dataset. 
        The weights will be the same for all rows in the dataset corresponding to 
        the same decision-maker. However, passing it as a separate parameter speeds up the optimization.   
    altChosen : sparse matrix of size (nRows x nObs).
        The (i, j)th element equals 1 if the alternative corresponding to the ith column 
        in expVars was chosen by the decision-maker corresponding to the jth observation,
        and 0 otherwise.
    obsAv : sparse matrix of size (nObs x nInds).
        The (i, j)th element equals 1 if the ith observation in the dataset corresponds 
        to the jth decision-maker, and 0 otherwise.
    choice : 1D numpy array of size nRows.
        The jth element equals 1 if the alternative corresponding to the jth column 
        in expVars was chosen by the decision-maker corresponding to that observation, and 0 otherwise.
        
    Returns
    -------
    ll : a scalar.
        Log-likelihood value for the weighted multinomidal logit model.
    np.asarray(gr).flatten() : 1D numpy array of size nExpVars.
        Gradient for the weighted multinomial logit model.
    
    """       
    v = np.dot(param[None, :], expVars)         # v is 1 x nRows
    ev = np.exp(v)                              # ev is 1 x nRows
    ev[np.isinf(ev)] = 1e+20                    # As precaution when exp(v) is too large for machine
    ev[ev < 1e-200] = 1e-200                    # As precaution when exp(v) is too close to zero
    nev = ev * altAv                            # nev is 1 x nObs
    nnev = altAv * np.transpose(nev)            # nnev is nRows x 1
    p = np.divide(ev, np.transpose(nnev))       # p is 1 x nRows 
    p[np.isinf(p)] = 1e-200                     # When none of the alternatives are available
    p[p < 1e-200] = 1e-200                      # As precaution when p is too close to zero
    tgr = choice - np.transpose(p)              # ttgr is nRows x 1
    ttgr = -np.multiply(weightsGr, tgr)         # tgr is nRows x 1
    gr = np.dot(expVars, ttgr)                  # gr is nExpVars x 1
    pObs = p * altChosen                        # pObs is 1 x nObs
    lPObs = np.log(pObs)                        # lPObs is 1 x nObs
    lPInd = lPObs * obsAv                       # lPInd is 1 x nInds
    wtLPInd = np.multiply(lPInd, weightsProb)   # wtLPInd is 1 x nInds
    ll = -np.sum(wtLPInd)                       # ll is a scalar
    
    return ll, np.asarray(gr).flatten()
    

def calStdErrWtLogitPanel(param, expVars, altAv, weightsProb, weightsGr, altChosen, obsAv, choice):
    """
    Function that calculates the standard errors for a weighted multinomial logit model 
    with panel data.
    
    Parameters
    ----------
    param : 1D numpy array of size nExpVars.
        Contains parameter values.
    expVars : 2D numpy array of size (nExpVars x (nRows)).
        Contains explanatory variables.
    altAv : sparse matrix of size (nRows x nObs).
        The (i, j)th element equals 1 if the alternative corresponding to the ith 
        column in expVars is available to the decision-maker corresponding to the 
        jth observation, and 0 otherwise. 
    weightsProb : 1D numpy array of size nInds.
        The jth element is the weight to be used for the jth decision-maker.
    weightsGr : 1D numpy array of size nRows.
        The jth element is the weight to be used for the jth row in the dataset. 
        The weights will be the same for all rows in the dataset corresponding to 
        the same decision-maker. However, passing it as a separate parameter speeds up the optimization.   
    altChosen : sparse matrix of size (nRows x nObs).
        The (i, j)th element equals 1 if the alternative corresponding to the ith column 
        in expVars was chosen by the decision-maker corresponding to the jth observation,
        and 0 otherwise.
    obsAv : sparse matrix of size (nObs x nInds).
        The (i, j)th element equals 1 if the ith observation in the dataset corresponds 
        to the jth decision-maker, and 0 otherwise.
    choice : 1D numpy array of size nRows.
        The jth element equals 1 if the alternative corresponding to the jth column 
        in expVars was chosen by the decision-maker corresponding to that observation, and 0 otherwise.
        
    Returns
    -------
    se : 2D numpy array of size (nExpVars x 1).
        Standard error for the weighted multinomidal logit model.
    
    """ 
    v = np.dot(param[None, :], expVars)         # v is 1 x nRows
    ev = np.exp(v)                              # ev is 1 x nRows
    ev[np.isinf(ev)] = 1e+20                    # As precaution when exp(v) is too large for machine
    ev[ev < 1e-200] = 1e-200                    # As precaution when exp(v) is too close to zero
    nev = ev * altAv                            # nev is 1 x nObs
    nnev = altAv * np.transpose(nev)            # nnev is nRows x 1
    p = np.divide(ev, np.transpose(nnev))       # p is 1 x nRows 
    p[np.isinf(p)] = 1e-200                     # When none of the alternatives are available
    p[p < 1e-200] = 1e-200                      # As precaution when p is too close to zero
    tgr = choice - np.transpose(p)              # ttgr is nRows x 1
    ttgr = np.multiply(weightsGr, tgr)          # tgr is nRows x 1
    gr = np.tile(ttgr, (1, expVars.shape[0]))   # gr is nRows x nExpVars 
    sgr = np.multiply(np.transpose(expVars),gr) # sgr is nRows x nExpVars 
    hess = np.dot(np.transpose(sgr), sgr)       # hess is nExpVars x nExpVars 
    try:                                        # iHess is nExpVars x nExpVars 
        iHess = np.linalg.inv(hess)             # If hess is non-singular
    except:
        iHess = np.identity(expVars.shape[0])   # If hess is singular
    se = np.sqrt(np.diagonal(iHess))            # se is nExpVars x 1

    return se



def displayOutput(outputFile, startTime, llEstimation, llNull, llTestNormalized, prediction_test, nClasses, 
        namesExpVarsClassSpec, paramClassSpec, stdErrClassSpec, obsID, X, X_Dummy, means, means_dummy, covariances, covariance_type, mixing_coefficients, pIndClass, pChoice, pChoiceTest): 
    """
    Function that displays the estimation results and model's stastical fit results. 
    
    Parameters
    ----------
    outputFile : File.
        A file object to which the output on the display screen is concurrently written.
    startTime : Datetime.
        A datetime object to indicate the starting time for estimation of the algorithm.
    llEstiamtion : a scalar.
        Log-likelihood value for the weighted multinomidal logit model at convergence.
    llNull : a scalar.
        Log-likelihood value for the weighted multinomidal logit model when all 
        parameters are set to zero.
    llTestNormalized
    prediction_test
    nClasses : Integer.
        Number of classes to be estimated by the model.
     namesExpVarsClassSpec : List of size nClasses.
        The jth element is a list containing the names of the explanatory variables
        entering the class-specific utilities for the jth latent class.
    paramClassSpec : List of size nClasses.
        The jth element is a 1D numpy array containing the parameter estimates associated with 
        the explanatory variables entering the class-specific utilities for the jth latent class.
    stdErrClassSpec : List of size nClasses.
        The jth element is a 1D numpy array containing standard errors for parameters of the class
        specific choice model for the jth class.
    obsID : 1D numpy array of size nRows.
        Identifies which rows in the dataset correspond to which observation. 
    X, X_Dummy, means, means_dummy, covariances, covariance_type, mixing_coefficients, pIndClass, pChoice, pChoiceTest

    -------
    """
    
    num_class_specific_model = 0
    for i in range(0, nClasses):
        num_class_specific_model = num_class_specific_model + paramClassSpec[i].shape[0]

    n_samples, n_features = X.shape
    _, n_dummyfeatures = X_Dummy.shape

    if covariance_type == 'full':
        cov_params = nClasses * n_features * (n_features + 1) / 2.
    elif covariance_type == 'diag':
        cov_params = nClasses * n_features
    elif covariance_type == 'tied':
        cov_params = n_features * (n_features + 1) / 2.
    elif covariance_type == 'spherical':
        cov_params = nClasses

    means_params = n_features * nClasses
    means_paramsdummy = n_dummyfeatures*nClasses
    
    num_class_membership_model = int(cov_params + means_params + means_paramsdummy + nClasses - 1)
    num_parameters_total = num_class_specific_model + num_class_membership_model

    #Full Model
    AIC = -2*llEstimation + 2*num_parameters_total
    BIC = -2*llEstimation  + num_parameters_total*np.log(np.unique(obsID).shape[0])

    #Normalized Model
    pIndClassNormalized = np.divide(pIndClass.T, np.tile(np.sum(pIndClass.T, axis = 0), (nClasses, 1)))
    a=np.multiply(pChoice, pIndClassNormalized)
    llNormalized = np.sum(np.log(np.sum(a, axis = 0)))
    AIC_Normalized = -2*llNormalized + 2*num_parameters_total
    BIC_Normalized = -2*llNormalized  + num_parameters_total*np.log(np.unique(obsID).shape[0])
    
    #Membership and Class-Specific Models
    timeElapsed = datetime.now() - startTime
    timeElapsed = (timeElapsed.days * 24.0 * 60.0) + (timeElapsed.seconds/60.0)

    
        
    print("\n")
    print("Number of Parameters:".ljust(45,' '), str(num_parameters_total).rjust(10,' '))
    print("Number of Observations:".ljust(45, ' '), str(np.unique(obsID).shape[0]).rjust(10,' '))   
    print("Null Log-Likelihood:".ljust(45, ' '), str(round(llNull,2)).rjust(10,' '))   
    print("Joint Log-Likelihood:".ljust(45, ' '), str(round(llEstimation,2)).rjust(10,' '))    
    print("AIC-Joint:".ljust(45, ' '), str(round(AIC,2)).rjust(10,' ')) 
    print("BIC-Joint:".ljust(45, ' '), str(round(BIC)).rjust(10,' '))
    print("Estimation time (minutes):".ljust(45, ' '), str(round(timeElapsed,2)).rjust(10,' ')) 
    print("\n")

    print("Marginal Log-Likelihood:".ljust(45, ' '), str(round(llNormalized,2)).rjust(10,' '))    
    print("AIC:".ljust(45, ' '), str(round(AIC_Normalized,2)).rjust(10,' ')) 
    print("BIC:".ljust(45, ' '), str(round(BIC_Normalized)).rjust(10,' '))
    print("\n")
    
    # Display screen

    print()
    print('Class-Specific Choice Model:')
    print('-----------------------------------------------------------------------------------------')
    print("Number of Parameters:".ljust(45,' ')), (str(num_class_specific_model).rjust(10,' '))
    
    for s in range(0, nClasses):
        print
        print('Class %d Model: ' %(s + 1))
        print('-----------------------------------------------------------------------------------------')
        print('Variables                                     parameters    std_err     t_stat    p_value')
        print('-----------------------------------------------------------------------------------------')
        for k in range(0, len(namesExpVarsClassSpec[s])):
            print('%-45s %10.4f %10.4f %10.4f %10.4f' %(namesExpVarsClassSpec[s][k], paramClassSpec[s][k], 
                    stdErrClassSpec[s][k], paramClassSpec[s][k]/stdErrClassSpec[s][k], scipy.stats.norm.sf(abs(paramClassSpec[s][k]/stdErrClassSpec[s][k]))*2 ))
        print('-----------------------------------------------------------------------------------------')

        
    print("\n")
    print('Class Membership Model:')
    print('-----------------------------------------------------------------------------------------')
    print("Number of Parameters:".ljust(45,' '), str(num_class_membership_model).rjust(10,' '))
    print()
    print('-----------------------------------------------------------------------------------------')
    print('Mixing Coefficients')
    print('-----------------------------------------------------------------------------------------')
    for k in range(0, nClasses):
        ClassN = 'Class %d' %(k + 1)
        print('%-45s' %ClassN, end=' ')
        print('%10.4f' %mixing_coefficients[k])
    print()
    print('-----------------------------------------------------------------------------------------')
    printMeans = 'Means (Continuous)'
    print('%-25s' %printMeans , end=' ')
    for n in range(0, n_features):
        Xn = 'X %d' %(n + 1)
        print('%-8.4s' %Xn , end=' ')
    print()
    print('-----------------------------------------------------------------------------------------')
    for k in range(0, nClasses):
        ClassN = 'Class %d' %(k + 1)
        print('%-20s' %ClassN, end=' ')
        for n in range(0, n_features):
            print('%8.4f' %means[k,n], end=' ')
        print()
    print()
    print('-----------------------------------------------------------------------------------------')
    printMeans = 'Means (Dummy)'
    print('%-25s' %printMeans, end=' ')
    for n in range(0, n_dummyfeatures):
        Xn = 'X %d' %(n + 1)
        print('%-8.4s' %Xn, end=' ')
    print()
    print('-----------------------------------------------------------------------------------------')
    for k in range(0, nClasses):
        ClassN = 'Class %d' %(k + 1)
        print('%-20s' %ClassN, end=' ')
        for n in range(0, n_dummyfeatures):
            print('%8.4f' %means_dummy[k,n], end=' ')
        print()
    print()
    print('-----------------------------------------------------------------------------------------')
    
    if covariance_type == 'spherical':
        print('Covariances')
        print('-----------------------------------------------------------------------------------------')
        for k in range(0, nClasses):
            ClassN = 'Class %d' %(k + 1)
            print('%-45s' %ClassN, end=' ')
            print('%10.4f' %covariances[k])
        print()

    else:
        printCov = 'Covariances'
        print('%-25s' %printCov, end=' ')
        for n in range(0, n_features):
            Xn = 'X %d' %(n + 1)
            print('%-8.4s' %Xn, end=' ')
        print()
        print('-----------------------------------------------------------------------------------------')

        if covariance_type == 'full':    
            for k in range(0, nClasses):
                for n1 in range(0, n_features):
                    ClassN = 'Class %d, X%d' %((k + 1), n1 + 1)
                    print('%-20s' %ClassN, end=' ')
                    for n2 in range(0, n_features):
                        print('%8.4f' %covariances[k,n1,n2], end=' ')
                    print()
                print()

        elif covariance_type == 'tied':
            for n1 in range(0,n_features):
                Xn = 'X %d' %(n1 + 1)
                print('%-20s' %Xn, end=' ')
                for n2 in range(0,n_features):
                    print('%8.4f' %covariances[n1,n2], end=' ')
                print()
            print()

        elif covariance_type == 'diag':
            for k in range(0,nClasses):
                ClassN = 'Class %d' %(k + 1)
                print('%-20s' %ClassN, end=' ')
                for n1 in range(0,n_features):
                    print('%8.4f' %covariances[k,n1], end=' ')
                print()
            print()

    if prediction_test == 'Yes':
        print()
        print('-----------------------------------------------------------------------------------------')
        print("Predicted Log-Likelihood:".ljust(45, ' '), str(round(llTestNormalized,2)).rjust(10,' '))
        print()

    


def processData(inds, indID, nClasses, 
        obsID, altID, choice, availAlts):
    """
    Function that takes the raw data and processes it to construct arrays and matrices that
    are subsequently used during estimation. 
    
    Parameters
    ----------
    inds : 1D numpy array of size nInds (total number of individuals in the dataset).
        Depicts total number of decision-makers in the dataset.    
    indID : 1D numpy array of size nRows.
        The jth element identifies the decision-maker corresponding to the jth row
        in the dataset.
    nClasses : Integer.
        Number of classes to be estimated by the model.
    expVarsClassMem ##Removed## : 2D numpy array of size (nExpVars x nRows).
        The (i, j)th element is the ith explanatory variable entering the class-membership 
        model for the decision-maker corresponding to the jth row in the dataset.
    availIndClasses ##Removed## : 2D numpy array of size (nClasses x nRows).
        Constraints on available latent classes. The (i,j)th element equals 1 if the ith 
        latent class is available to the decision-maker corresponding to the jth row in the 
        dataset, and 0 otherwise.
    obsID : 1D numpy array of size nRows.
        Identifies which rows in the dataset correspond to which observation.
    altID : 1D numpy array of size nRows.
        Identifies which rows in the dataset correspond to which alternative.
    choice : 1D numpy array of size nRows.
        Identifies which rows in the dataset correspond to alternatives that were 
        chosen during the corresponding observation.
    availAlts : List of size nClasses.
        Determines choice set constraints for each of the classes. The sth element is an 
        array containing identifiers for the alternatives that are available to decision-makers 
        belonging to the sth class.
        
    Returns
    -------
    nInds : Integer.
        Total number of individuals/decision-makers in the dataset.    
    altAv : List of size nClasses. 
        The sth element of which is a sparse matrix of size (nRows x nObs), where the (i, j)th 
        element equals 1 if the alternative corresponding to the ith column in expVarsMan is 
        available to the decision-maker corresponding to the jth observation, and 0 otherwise.
    altChosen : sparse matrix of size (nRows x nObs).
        The (i, j)th element of the returned matrix  is 1 if the alternative corresponding 
        to the ith row in the data file was chosen during observation j, and 0 otherwise.   
    obsAv : sprase matrix of size (nObs x nDms).
        The (i, j)th element of the returned matrix is 1 if observation i corresponds to 
        decision-maker j, and 0 otherwise.
    rowAv : sparse matrix of size (nRows x (nAlts * nDms)).
        The (i, ((n - 1) * nAlts) + j)th element of the returned matrix is 1 if the ith row 
        in the data file corresponds to the jth alternative and the nth decision-maker, 
        and 0 otherwise.  

    
    """ 
    # Class membership model
    nInds = inds.shape[0]

    # Class-specific model
    altAvTuple, altChosen, obsAv, rowAv = processClassSpecificPanel(inds, indID, obsID, altID, choice)
    nRows = altID.shape[0]
    nObs = np.unique(obsID).shape[0]

    altAv = []
    for k in range(0, nClasses):
        altAv.append(coo_matrix((imposeCSConstraints(altID, availAlts[k]), 
                (altAvTuple[0], altAvTuple[1])), shape = (nRows, nObs)))
    

    return (nInds, altAv, altChosen, obsAv, rowAv) 

    
def enumClassSpecificProbPanel(param, expVars, altAvMat, obsAv, rowAv, nDms, nAlts):
    """
    Function that calculates and enumerates the class specific choice probabilities 
    for each decision-maker in the sample and for each of the available alternatives
    in the choice set.
    
    Parameters
    ----------
    param : 1D numpy array of size nExpVars.
        Contains parameter values.
    expVars : 2D numpy array of size (nExpVars x (nRows)).
        Contains explanatory variables.
    altAvMat : sparse matrix of size (nRows x nObs).
        The (i, j)th element equals 1 if the alternative corresponding to the ith 
        column in expVars is available to the decision-maker corresponding to the 
        jth observation, and 0 otherwise.
    obsAv : sparse matrix of size (nObs x nInds).
        The (i, j)th element equals 1 if the ith observation in the dataset corresponds 
        to the jth decision-maker, and 0 otherwise.
    rowAv : sparse matrix of size (nRows x (nAlts * nDms)).
        The (i, ((n - 1) * nAlts) + j)th element of the returned matrix is 1 if the ith row 
        in the data file corresponds to the jth alternative and the nth decision-maker, 
        and 0 otherwise.  
    nDms : Integer.
        Total number of individuals/decision-makers in the dataset.
    nAlts : Integer.
        Total number of unique available alternatives to individuals in the sample.   
        
    Returns
    -------
    pAlt : 2D numpy array of size nInds x nAlts.
        The (i, j)th element of the returned 2D array is denotes the probability 
        of individual i choosing alternative j. 

    
    """ 

    v = np.dot(param[None, :], expVars)               # v is 1 x nRows
    ev = np.exp(v)                                    # ev is 1 x nRows
    ev[np.isinf(ev)] = 1e+20                          # As precaution when exp(v) is too large for machine
    ev[ev < 1e-200] = 1e-200                          # As precaution when exp(v) is too close to zero
    nev = ev * altAvMat                               # nev is 1 x nObs
    nnev = altAvMat * np.transpose(nev)               # nnev is nRows x 1
    p = np.divide(ev, np.transpose(nnev))             # p is 1 x nRows
    p[np.isinf(p)] = 1e-200                           # When none of the alternatives are available
    pAlt = p * rowAv                                  # pAlt is 1 x (nAlts * nDms)
    return pAlt.reshape((nDms, nAlts), order = 'C')

    
    
def calProb(nClasses, nInds, paramClassSpec, expVarsClassSpec, altAv, altChosen, obsAv, indWeights, means, means_dummy, mixing_coefficients, precisions_chol, covariance_type, X, X_Dummy):
    """
    Function that calculates the expectation of the latent variables in E-Step of the 
    EM Algorithm and the value of the log-likelihood function.
    
    Parameters
    ----------
    nClasses : Integer.
        Number of classes to be estimated by the model.
    nInds : Integer.
        Total number of individuals/decision-makers in the dataset.
    paramClassSpec : List of size nClasses.
        The jth element is a 1D numpy array containing the parameter estimates associated with 
        the explanatory variables entering the class-specific utilities for the jth latent class.
    expVarsClassSpec : List of size nClasses.
        Entails the utility specification for each of the latent classes.
        The sth element is a 2D numpy array of size (nExpVars x nRows) containing the explanatory 
        variables entering the class-specific utilities for the sth latent class.
        The (i, j)th element of the 2D array denotes the ith explanatory 
        variable entering the utility for the alternative corresponding to the jth row 
        in the data file.
    altAv : List of size nClasses. 
        The sth element of which is a sparse matrix of size (nRows x nObs), where the (i, j)th 
        element equals 1 if the alternative corresponding to the ith column in expVarsClassSpec is 
        available to the decision-maker corresponding to the jth observation, and 0 otherwise.
    altChosen : sparse matrix of size (nRows x nObs).
        The (i, j)th element of the returned matrix  is 1 if the alternative corresponding 
        to the ith row in the data file was chosen during observation j, and 0 otherwise.   
    obsAv : sprase matrix of size (nObs x nDms).
        The (i, j)th element of the returned matrix is 1 if observation i corresponds to 
        decision-maker j, and 0 otherwise. 
    indWeights : 1D numpy array of size nDms.
        Each element accounts for the associated weight for each individual in the data file
        to cater for the choice based sampling scheme.
    means, means_dummy, mixing_coefficients, precisions_chol, covariance_type, X, X_Dummy
    
    Returns
    -------
    pIndClass: the class membership probabilities
    p: the class-specific choice probabilities
    Gqnk: the expectation of qnk (class assignment: qnk = 1 if individual n is assigned to class k and 0 otherwise)
    ll : the value of log-likelihood.
    """    

    pIndClass_C, pIndClass_D = calClassMemProb(nClasses, means, means_dummy, mixing_coefficients, precisions_chol, covariance_type, X, X_Dummy)

    pIndClass = np.multiply(pIndClass_C, pIndClass_D)
    
    p = calClassSpecificProbPanel(paramClassSpec[0], expVarsClassSpec[0], altAv[0], altChosen, obsAv)
    for k in range(1, nClasses):
        p = np.vstack((p, calClassSpecificProbPanel(paramClassSpec[k], expVarsClassSpec[k], altAv[k], altChosen, obsAv)))
        # p (K x N)
    
    Gqnk = np.multiply(p, pIndClass.T)

    ll = np.sum(np.multiply(np.log(np.sum(Gqnk, axis = 0)), indWeights))

    Gqnk = np.divide(Gqnk, np.tile(np.sum(Gqnk, axis = 0), (nClasses, 1)))     # nClasses x nInds
    return pIndClass, p, Gqnk, ll
 
                                                                                                                                                                                                                                                                                                                                                                                     
def emAlgo(outputFilePath, outputFileName, outputFile, nClasses, covariance_type, X, X_Dummy, XTest, XTest_Dummy,
        prediction_test, GMM_Initialization, indID, obsID, altID, choice, indIDTest, obsIDTest, altIDTest, choiceTest, availAlts,
        expVarsClassSpec, expVarsClassSpecTest, namesExpVarsClassSpec, indWeights, indWeightsTest, paramClassSpec, reg_covar, tol, max_iter):
    """
    Function that implements the EM Algorithm to estimate the desired model specification. 
    
    Parameters
    ----------
    outputFilePath : String.
        File path to where all the output files should be stored.
    outputFileName : String.
        Name without extension that should be given to all output files.    
    outputFile : File.
        A file object to which the output on the display screen is concurrently written.
    nClasses : Integer.
        Number of classes to be estimated by the model.
    covariance_type, X, X_Dummy, XTest, XTest_Dummy, prediction_test, GMM_Initialization, 
    indID : 1D numpy array of size nRows.
        The jth element identifies the decision-maker corresponding to the jth row
        in the dataset.
    obsID : 1D numpy array of size nRows.
        Identifies which rows in the dataset correspond to which observation.
    altID : 1D numpy array of size nRows.
        Identifies which rows in the dataset correspond to which alternative.
    choice : 1D numpy array of size nRows.
        Identifies which rows in the dataset correspond to alternatives that were 
        chosen during the corresponding observation.
    indIDTest, obsIDTest, altIDTest, choiceTest: same as the above but for the test set.    
    availAlts : List of size nClasses.
        Determines choice set constraints for each of the classes. The sth element is an 
        array containing identifiers for the alternatives that are available to decision-makers 
        belonging to the sth class.
    expVarsClassSpec : List of size nClasses.
        Entails the utility specification for each of the latent classes.
        The sth element is a 2D numpy array of size (nExpVars x nRows) containing the explanatory 
        variables entering the class-specific utilities for the sth latent class.
        The (i, j)th element of the 2D array denotes the ith explanatory 
        variable entering the utility for the alternative corresponding to the jth row 
        in the data file.
    expVarsClassSpecTest: same as the above but for the test set.    
    namesExpVarsClassSpec : List of size nClasses.
        The jth element is a list containing the names of the explanatory variables
        entering the class-specific utilities for the jth latent class.
    indWeights : 1D numpy array of size nDms.
        Each element accounts for the associated weight for each individual in the data file
        to cater for the choice based sampling scheme.
    indWeightsTest: same as the above but for the test set.
    paramClassSpec : List of size nClasses.
        The jth element is a 1D numpy array containing the parameter estimates associated with 
        the explanatory variables entering the class-specific utilities for the jth latent class.
    reg_covar, tol, max_iter    
    Returns
    -------
    
    """ 
    
    startTime = datetime.now()
    print('Processing data')
    outputFile.write('Processing data\n')

    inds = np.unique(indID)
    n_samples, n_features = X.shape
    _, n_dummyfeatures = X_Dummy.shape

    (nInds, altAv, altChosen, obsAv, rowAv) \
            = processData(inds, indID, 
            nClasses, obsID, altID,
            choice, availAlts) 

    print('Initializing EM Algorithm...\n')
    outputFile.write('Initializing EM Algorithm...\n\n')

    # Initializing the parameters
    converged, iterCounter, llOld = False, 0, -np.inf

    resp, nk, mixing_coefficients, means, means_dummy, covariances, precisions_chol, log_det = initialize_parameters(X, X_Dummy, tol, reg_covar, max_iter, covariance_type, \
                                                                                                                     nClasses, GMM_Initialization)

    # calculating the null log-likelihod
    paramClassSpecNull = []    
    for k in range(0, nClasses):
        paramClassSpecNull.append(np.zeros(expVarsClassSpec[k].shape[0]))


    _, _, _, llNull = calProb(nClasses, nInds, paramClassSpecNull, expVarsClassSpec, altAv, altChosen, obsAv, indWeights,\
                                          means, means_dummy, mixing_coefficients, precisions_chol, covariance_type, X, X_Dummy)
    
    while not converged:
        
        # E-Step: Calculate the expectations of the latent variables, using the current 
        # values for the model parameters.
        pIndClass, _, Gqnk, llNew = calProb(nClasses, nInds, paramClassSpec, expVarsClassSpec, altAv, altChosen, obsAv, indWeights, means, means_dummy, mixing_coefficients, \
                                            precisions_chol, covariance_type, X, X_Dummy)

        currentTime = datetime.now().strftime('%a, %d %b %Y %H:%M:%S')
        print('<%s> Iteration %d: %.4f' %(currentTime, iterCounter, llNew))
        outputFile.write('<%s> Iteration %d: %.4f\n' %(currentTime, iterCounter, llNew))

        # M-Step: Use the weights derived in the E-Step to update the model parameters.

        #### M-Step For GMM
        """The effective number of points assigned to cluster k"""
        nk = Gqnk.T.sum(axis=0) + 10 * np.finfo(Gqnk.dtype).eps
        
        """mixing_coefficients of the gaussians"""
        mixing_coefficients = nk/sum(nk)

        """Means"""
        means = np.dot(Gqnk, X) / nk[:, np.newaxis]
        means_dummy = np.dot(Gqnk, X_Dummy) / nk[:, np.newaxis]

        """Covariance matrix"""
        covariances = estimate_gmm_covariances(covariance_type, Gqnk.T, X, nk, means, reg_covar, nClasses)

        """The Cholesky decomposition of the precision (Inverse of the covariance matrix)"""
        precisions_chol = compute_precisions_choleskey(covariance_type, covariances)

        """The log-det of the cholesky decomposition of matrices.
        log_det_precisions_chol : array-like, shape (nClasses,): the determinant of the precision matrix for each component.
        log_det_precisions_chol = 0.5*log_det_precision.
        matrix_col is precisions_chol"""
        log_det = compute_log_det_cholesky(precisions_chol, covariance_type, n_features) ### matrix_col is precisions_chol

        #### End of M-Step for GMM

        #### M-Step for the class-specific choice parameters
        for k in range(0, nClasses):
            cWeights = np.multiply(Gqnk[k, :], indWeights)
            paramClassSpec[k] = minimize(wtLogitPanel, paramClassSpec[k], args = (expVarsClassSpec[k], altAv[k], 
                    cWeights, altAv[k] * obsAv * cWeights[:, None], altChosen, 
                    obsAv, choice), method = 'BFGS', jac = True, tol = llTol, options = {'gtol': grTol})['x']

        #### End of M-Step for the class-specific choice parameters
            
        converged =  (abs(llNew - llOld) < emTol)
        llOld = llNew
        iterCounter += 1


    # Calculate standard errors for the class specific choice model                                     
    stdErrClassSpec = []
    for k in range(0, nClasses):
        stdErrClassSpec.append(calStdErrWtLogitPanel(paramClassSpec[k], expVarsClassSpec[k], altAv[k], 
                    Gqnk[k, :], altAv[k] * obsAv * Gqnk[k, :][:, None], 
                    altChosen, obsAv, choice))


    pIndClass, pChoice, Gqnk, llNew = calProb(nClasses, nInds, paramClassSpec, expVarsClassSpec, altAv, altChosen, obsAv, indWeights,\
                                              means, means_dummy, mixing_coefficients, precisions_chol, covariance_type, X, X_Dummy)

    llTestNormalized = 0
    pChoiceTest = 0

    (nInds, altAv, altChosen, obsAv, rowAv)\
    = processData(inds, indID, nClasses, obsID, altID, choice, availAlts)

    nAlts = np.unique(altID).shape[0]

    if prediction_test == 'Yes':
        #### Prediction Test
        indsTest = np.unique(indIDTest)
        n_samples_Test, n_features_Test = XTest.shape
        (nIndsTest, altAvTest, altChosenTest, obsAvTest, rowAvTest) = processData(indsTest, indIDTest, nClasses, obsIDTest, altIDTest, choiceTest, availAlts) 
        nAltsTest = np.unique(altIDTest).shape[0]
        #Normalized Model
        pIndClassTest, pChoiceTest, GqnkTest, llNewTest = calProb(nClasses, nIndsTest, paramClassSpec, expVarsClassSpecTest, altAvTest, altChosenTest, obsAvTest, indWeightsTest,\
                                                                  means, means_dummy, mixing_coefficients, precisions_chol, covariance_type, XTest, XTest_Dummy)
    
        pIndClassTestNormalized = np.divide(pIndClassTest.T, np.tile(np.sum(pIndClassTest.T, axis = 0), (nClasses, 1)))
        aTest=np.multiply(pChoiceTest, pIndClassTestNormalized)
        llTestNormalized = np.sum(np.log(np.sum(aTest, axis = 0)))


        #Sample Enumeration for Test Data
        pTest = enumClassSpecificProbPanel(paramClassSpec[0], expVarsClassSpecTest[0], altAvTest[0], obsAvTest, rowAvTest, nIndsTest, nAltsTest)
        for s in range(1, nClasses):
            pTest = np.hstack((pTest, enumClassSpecificProbPanel(paramClassSpec[s], expVarsClassSpecTest[s], altAvTest[s], obsAvTest, rowAvTest, nIndsTest, nAltsTest)))
        pTest = np.hstack((indsTest[:, None], pIndClassTestNormalized.T, pTest))
        pTest = np.hstack((pTest, pChoiceTest.T))
        ### this p will have: first, the class membership probabilities (pIndClassTestNormalized, e.g. P(k=1))
        ###                   Second, the panel (product of probabilities for each individual n) class specific probabilities for each alternative
        ###                   Thired, the panel choice probability per class

        # Choice probability per individual per observarion/scenario per individual
        pScenarioTest = calClassSpecificProbScenarios(paramClassSpec[0], expVarsClassSpecTest[0], altAvTest[0], altChosenTest, obsAvTest)
        for k in range(1, nClasses):
            pScenarioTest = np.vstack((pScenarioTest, calClassSpecificProbScenarios(paramClassSpec[k], expVarsClassSpecTest[k], altAvTest[k], altChosenTest, obsAvTest)))

        np.savetxt(outputFilePath + outputFileName + 'SampleEnumTest.csv', pTest, delimiter = ',')
        np.savetxt(outputFilePath + outputFileName + 'SampleEnumScenarioTest.csv', pScenarioTest, delimiter = ',')

    ####

    print('\nEnumerating choices for the sample')
    outputFile.write('\nEnumerating choices for the sample\n')


    
    # display model fit results and parameter estimation results            
    displayOutput(outputFile, startTime, llNew, llNull, llTestNormalized, prediction_test, nClasses,
            namesExpVarsClassSpec, paramClassSpec, stdErrClassSpec, obsID, X, X_Dummy, means, means_dummy, covariances, covariance_type, mixing_coefficients, pIndClass, pChoice, pChoiceTest) 

    # Write parameters to file and store them in an outputfile for the user
    with open(outputFilePath + outputFileName + 'Param.txt', 'wb') as f:                        
        for k in range(0, nClasses):
            np.savetxt(f, paramClassSpec[k][None, :], delimiter = ',')
        #np.savetxt(f, paramClassMem[None, :], delimiter = ',')

def lccm_fit(data,
             X,
             X_Dummy,
             dataTest,
             XTest,
             XTest_Dummy,
             prediction_test,
             GMM_Initialization,
             ind_id_col, 
             obs_id_col,
             alt_id_col,
             choice_col,
             n_classes,
             covariance_type,
             reg_covar,
             tol,
             max_iter,
             class_specific_specs,
             class_specific_labels, 
             indWeights = None,
             avail_alts = None,
             paramClassSpec = None,
             outputFilePath = '', 
             outputFileName = 'ModelResults'):
    """
    Takes a PyLogit-style dataframe and dict-based specifications, converts them into
    matrices, and invokes emAlgo().
    
    Parameters
    ----------
    data : pandas.DataFrame.
        Labeled data in long format (i.e., each alternative in a choice scenario is in a 
        separate row).
    X, X_Dummy, dataTest, XTest, XTest_Dummy, prediction_test, GMM_Initialization,
    ind_id_col : String.
        	Name of column identifying the decision maker for each row of data.
    obs_id_col : String.
        	Name of column identifying the observation (choice scenario).
    alt_id_col : String.
        	Name of column identifying the alternative represented.
    choice_col : String.
        	Name of column identifying whether the alternative represented by a row was 
         chosen during the corresponding observation. 
    n_classes : Integer.
        	Number of latent classes to be estimated by the model. 
    covariance_type, reg_covar, tol, max_iter,
    class_specific_spec : list of OrderedDicts, of length n_classes
        	Each OrderedDict represents the specification for one class-specific choice model.
         Specifications should have keys representing the column names to be used as 
         explanatory variables, and values that are lists of the applicable alternative
         id's. Specs will be passed to pylogit.choice_tools.create_design_matrix().
    class_specific_labels : list of OrderedDicts, of length n_classes
         Each OrderedDict entails the names of explanatory variables for one class-
         specific choice model. Labels should have keys representing the general name
         of the explnatory variable used, and values that are lists of the names of 
         the variable associated with the respective alternative as specified by the analyst.    	
    indWeights : 1D numpy array of size nDms.
        Each element accounts for the associated weight for each individual in the data file
        to cater for the choice based sampling scheme.
    avail_alts : list of length n_classes, optional
    	Which choice alternatives are available to members of each latent class? The sth
    	element is an array containing identifiers for the alternatives that are available
    	to decision-makers belonging to the sth latent class. If not specified, all
    	alternatives are available to members of all latent classes.
    paramClassSpec : List of size nClasses.
        The jth element is a 1D numpy array containing the parameter estimates associated with 
        the explanatory variables entering the class-specific utilities for the jth latent class.
    outputFilePath : str, optional
    	Relative file path for output. If not specified, defaults to 'output/'
    outputFileName : str, optional
    	Basename for output files. If not specified, defaults to 'ModelResults'
    	
    Returns
    -------
    None
    
    """
    outputFile = open(outputFilePath + outputFileName + 'Log.txt', 'w')
    
    # Generate columns representing individual, observation, and alternative id
    # ind_id_col = 'ID'
    # obs_id_col = 'custom_id'
    # alt_id_col = 'mode_id'
    indID = data[ind_id_col].values
    obsID = data[obs_id_col].values
    altID = data[alt_id_col].values
    
    # Generate the choice column and transpose it
    #choice_col = 'choice'
    choice = np.reshape(data[choice_col].values, (data.shape[0], 1))

    indIDTest = []
    obsIDTest = []
    altIDTest = []
    choiceTest = []

    if prediction_test == 'Yes':
        # Generate columns representing individual, observation, and alternative id for the test dataset
        indIDTest = dataTest[ind_id_col].values
        obsIDTest = dataTest[obs_id_col].values
        altIDTest = dataTest[alt_id_col].values
        # Generate the choice column and transpose it
        choiceTest = np.reshape(dataTest[choice_col].values, (dataTest.shape[0], 1))

    # NUMBER OF CLASSES: We could infer this from the number of choice specifications 
    # provided, but it's probably better to make it explicit because that gives us the 
    # option of taking a single choice specification and using it for all the classes (?)
    
    nClasses = n_classes
    
    
    # AVAILABLE ALTERNATIVES: Which choice alternatives are available to each latent
    # class of decision-makers? List of size nClasses, where each element is a list of
    # identifiers of the alternatives available to members of that class.
    # Default case is to make all alternative available to all decision-makers.
    
    if avail_alts is None:
    	availAlts = [np.unique(altID) for s in class_specific_specs]  
    else:
        availAlts = avail_alts
    
    # CLASS-SPECIFIC MODELS: Use PyLogit to generate design matrices of explanatory variables
    # for each of the class specific choice models, inluding an intercept as specified by the user.
    
    design_matrices = [pylogit.choice_tools.create_design_matrix(data, spec, alt_id_col)[0] 
    						for spec in class_specific_specs]

    expVarsClassSpec = [np.transpose(m) for m in design_matrices]

    expVarsClassSpecTest = []
    
    if prediction_test == 'Yes':
        design_matricesTest = [pylogit.choice_tools.create_design_matrix(dataTest, spec, alt_id_col)[0] 
    						for spec in class_specific_specs]
        expVarsClassSpecTest = [np.transpose(m) for m in design_matricesTest]
    
    # NOTE: class-specific choice specifications with explanatory variables that vary
    # by alternative should work automatically thanks to PyLogit, but the output labels 
    # WILL NOT work until we update the LCCM code to handle that. 
    
    # starting values for the parameters of the class specific models
    # making the starting value of the class specfic choice models random
    # in case the user does not specify those starting values.
    if paramClassSpec is None:
        paramClassSpec = []
        for s in range(0, nClasses):
            paramClassSpec.append(-np.random.rand(expVarsClassSpec[s].shape[0])/10)
    
    # weights to account for choice-based sampling
    # By default the weights will be assumed to be equal to one for all individuals unless the user
    # specifies the weights
    # indWeights is 1D numpy array of size nInds accounting for the weight for each individual in the sample
    # as given by the user
    indWeightsTest = []
    if indWeights is None:    
        indWeights = np.ones((np.unique(indID).shape[0]))
        if prediction_test == 'Yes':
            indWeightsTest = np.ones((np.unique(indIDTest).shape[0]))
    
    # defining the names of the explanatory variables for class specific model
    # getting the requried list elements that comprise string of names of
    # explanatory variables to be used in displaying parameter estimates in the output tables.
    namesExpVarsClassSpec = []
    for i in range(0, len(class_specific_labels)):
        name_iterator=[]
        #for key, value in class_specific_labels[i].iteritems() :
        for key, value in class_specific_labels[i].items() :
            if type(value) is list:
                name_iterator += value
            else:
                name_iterator.append(value)
        namesExpVarsClassSpec.append(name_iterator)

    # Invoke emAlgo()
    emAlgo(outputFilePath = outputFilePath, 
           outputFileName = outputFileName, 
           outputFile = outputFile, 
           nClasses = nClasses,
           covariance_type = covariance_type,
           X = X,
           X_Dummy = X_Dummy,
           XTest = XTest,
           XTest_Dummy = XTest_Dummy,
           prediction_test = prediction_test,
           GMM_Initialization = GMM_Initialization,
           indID = indID,
           obsID = obsID, 
           altID = altID, 
           choice = choice,
           indIDTest = indIDTest,
           obsIDTest = obsIDTest,
           altIDTest = altIDTest,
           choiceTest = choiceTest,
           availAlts = availAlts, 
           expVarsClassSpec = expVarsClassSpec,
           expVarsClassSpecTest = expVarsClassSpecTest,
           namesExpVarsClassSpec = namesExpVarsClassSpec, 
           indWeights = indWeights,
           indWeightsTest = indWeightsTest,
           paramClassSpec = paramClassSpec,
           reg_covar = reg_covar,
           tol = tol,
           max_iter = max_iter)
    
    outputFile.close()
    return
