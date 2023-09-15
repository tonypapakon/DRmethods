#Imports
import numpy as np
import cvxpy as cp
from tqdm import tqdm
from sklearn.exceptions import NotFittedError

'''
Implementation of the DSNPE method as described in the paper 'Discriminant sparse neighborhood preserving embedding for face recognition' 
by Jie Gui, Zhenan Sun, Wei Jia, Rongxiang Hu, Yingke Lei, Shuiwang Ji
'''

'''
Encapsulation of the DSNPE method inside a class. This class-based structure is similar to the design of scikit-learn’s 
transformers and models, which is a widely accepted standard for machine learning in Python.
'''

class DSNPE:
    #1. Initialization (init method): Introduced parameters for the number of
    # components (n_components), the epsilon threshold (epsilon), and the gamma value (gamma).
    def __init__(self, n_components, epsilon=0.01, gamma=1.0):
        self.n_components = n_components
        self.epsilon = epsilon
        self.gamma = gamma
        self.projection_matrix_ = None

    #2.	_compute_si method: This is a private helper method (indicated by the underscore prefix)
    # that computes the sparse coefficient for a given data point.
    def _compute_si(self, xi, Xi):
        n = Xi.shape[1]
        si = cp.Variable(n)
        obj = cp.norm(si, 1)
        constr = [cp.norm(xi - Xi @ si, 2) <= self.epsilon]
        prob = cp.Problem(cp.Minimize(obj), constr)
        prob.solve(solver='ECOS')
        return si.value

    #3.	fit method: This method trains the model using the
    # input data X and labels y to produce the projection matrix.
    def fit(self, X, y):
        n, p = X.shape
        S = np.zeros((n, n))
        for i in tqdm(range(n), desc="Calculating Projection Vector using DSNPE"):
            xi = X[i, :]
            Xi_k = X[y == y[i], :]
            S[i, y == y[i]] = self._compute_si(xi, Xi_k.T)

        Sa = np.eye(n) - S - S.T + S.T @ S
        X_sa = X.T @ Sa @ X

        X_sbt = np.zeros((p, p))
        classes = np.unique(y)
        for k in classes:
            X_sbt += np.cov(X[y == k].T)
        X_sw = np.cov(X.T)
        X_sb = X_sbt - X_sw

        A = (X @ (X_sa - self.gamma * X_sb)).T @ X

        eigenvalues, eigenvectors = np.linalg.eigh(A)
        top_indices = np.argsort(eigenvalues)[:self.n_components]
        self.projection_matrix_ = eigenvectors[:, top_indices]

        return self

    #4.	transform method: This method projects the input data X onto
    # the lower-dimensional subspace defined by the projection matrix.
    def transform(self, X):
        if self.projection_matrix_ is None:
            raise NotFittedError("This DSNPE instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        return X @ self.projection_matrix_



'''
def dsnpe_compute_si(xi, Xi, epsilon):
    n = Xi.shape[1]
    si = cp.Variable(n)
    obj = cp.norm(si, 1)
    constr = [cp.norm(xi - Xi @ si, 2) <= epsilon]
    prob = cp.Problem(cp.Minimize(obj), constr)
    prob.solve(solver='ECOS')
    return si.value

def do_dsnpe(X, y, d, epsilon=0.01, gamma=1.0):
    # PREPROCESSING
    n, p = X.shape

    S = np.zeros((n, n))
    for i in range(n):
        xi = X[i, :]
        Xi_k = X[y == y[i], :]
        S[i, y == y[i]] = dsnpe_compute_si(xi, Xi_k.T, epsilon)

    Sa = np.eye(n) - S - S.T + S.T @ S

    X_sa = X.T @ Sa @ X

    X_sbt = np.zeros((p, p))
    classes = np.unique(y)
    for k in tqdm(classes, desc="Calculating Projection Vector"):
        X_sbt += np.cov(X[y == k].T)  # Adjusted computation of X_sbt
    X_sw = np.cov(X.T)
    X_sb = X_sbt - X_sw

    A = (X @ (X_sa - gamma * X_sb)).T @ X  # Correct computation of A

    eigenvalues, eigenvectors = np.linalg.eigh(A)
    top_indices = np.argsort(eigenvalues)[:d]
    projection = eigenvectors[:, top_indices]

    # Return output
    Y = X @ projection
    result = {"Y": Y, "projection": projection}
    return result
'''