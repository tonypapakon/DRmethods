#Imports
import numpy as np
import cvxpy as cp
from tqdm import tqdm
from sklearn.exceptions import NotFittedError

'''
Implementation of the SPP method as described in the paper 'Sparsity preserving projections with applications to face recognition' 
by Lishan Qiao, Songcan Chen, Xiaoyang Tan
'''

'''
Encapsulation of the SPP method inside a class. This class-based structure is similar to the design of scikit-learn’s 
transformers and models, which is a widely accepted standard for machine learning in Python.

	1.	Initialization (init method):
	•	The n_components parameter indicates the number of components/dimensions you want in the reduced space.
	•	The epsilon is a threshold for the L1 minimization problem.
	•	The projection_matrix_ will store the projection matrix after the fit method is called.
	2.	fit method:
	•	The fitting process calculates the projection matrix based on the input data X.
	•	It does so by first constructing the weight matrix S through the L1 minimization for each data point.
	•	Then, it computes the beta matrix and subsequently extracts the top n_components eigenvectors to form the projection matrix.
	3.	transform method:
	•	This method projects the input data X onto the subspace defined by the projection matrix.
	•	If the fit method has not been called previously, it raises a NotFittedError, indicating the need to fit the model before transformation.
'''

class SPP:
    def __init__(self, n_components, epsilon=0.01):
        self.n_components = n_components
        self.epsilon = epsilon
        self.projection_matrix_ = None

    def fit(self, X, epsilon=0.05):
        m, n = X.shape
        S = np.zeros((n, n))  # Initialize the weight matrix

        for i in tqdm(range(n), desc="Calculating Projection Vector using SPP"):
            xi = X[:, i]  # Current data point
            si = cp.Variable(n)  # Sparse coefficient vector
            objective = cp.Minimize(cp.norm(si, 1))
            constraints = [cp.norm(xi - X @ si) <= epsilon, cp.sum(si) == 1]
            problem = cp.Problem(objective, constraints)
            problem.solve(solver='ECOS')  # Solve the L1 minimization problem
            S[:, i] = si.value

        # Compute beta matrix
        beta = S + S.T - S.T @ S

        # Compute eigenvectors of beta
        eigenvalues, eigenvectors = np.linalg.eigh(beta)

        # Select the top d eigenvectors
        sorted_indices = np.argsort(eigenvalues)[::-1]  # Sort eigenvalues in descending order
        selected_indices = sorted_indices[:self.n_components]  # Select the top d eigenvalues
        self.projection_matrix_ = np.real(eigenvectors[:, selected_indices])

        return self

    def transform(self, X):
        if self.projection_matrix_ is None:
            raise NotFittedError("This SPP instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        return X @ self.projection_matrix_



'''
def spp(X, d, epsilon):
    m, n = X.shape
    S = np.zeros((n, n))  # Initialize the weight matrix

    for i in tqdm(range(n), desc="Calculating Projection Vector using SPP"):
        xi = X[:, i]  # Current data point
        si = cp.Variable(n)  # Sparse coefficient vector
        objective = cp.Minimize(cp.norm(si, 1))
        constraints = [cp.norm(xi - X @ si) <= epsilon, cp.sum(si) == 1]
        problem = cp.Problem(objective, constraints)
        problem.solve(solver='ECOS')  # Solve the L1 minimization problem
        S[:, i] = si.value

    # Step 2: Eigenvector Extraction
    beta = S + S.T - S.T @ S
    eigenvalues, eigenvectors = np.linalg.eig(X @ beta @ X.T)

    # Step 3: Subspace Projection
    sorted_indices = np.argsort(eigenvalues)[::-1]  # Sort eigenvalues in descending order
    selected_indices = sorted_indices[:d]  # Select the top d eigenvalues
    selected_vectors = np.real(eigenvectors[:, selected_indices])

    # Return the projection matrix
    return selected_vectors
'''
