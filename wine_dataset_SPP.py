#Imports
import numpy as np
import cvxpy as cp
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt


#Apply SPP on wine dataset for testing

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

############ wine-dataset ############
print("\n#####################################################################\n")
print("Initializing wine dataset...\n")

# Load the wine dataset
wine = load_wine()

# Get the feature matrix
X = wine['data']

# Initialize the scaler
scaler = StandardScaler()

# Initialize the scaler
X = scaler.fit_transform(X)
def spp_plot(n_components, epsilon=0.05):
    # Apply SPP to data
    P = spp(X.T, d=n_components, epsilon=epsilon)
    Y = X @ P

    # Separate the projected points based on their class
    Y1 = Y[wine.target == 0]
    Y2 = Y[wine.target == 1]
    Y3 = Y[wine.target == 2]

    # Create a figure and a set of subplots
    fig, ax = plt.subplots()

    # Check the number of components and adjust the plot accordingly
    if n_components == 1:
        # Plot the projected points from each class along the x-axis
        ax.scatter(Y1, [0]*len(Y1), color='b', label='Class 1')
        ax.scatter(Y2, [0]*len(Y2), color='r', label='Class 2')
        ax.scatter(Y3, [0]*len(Y3), color='g', label='Class 3')
        ax.set_ylabel('')  # No label for the y-axis
    elif n_components == 2:
        # Plot the projected points from each class
        ax.scatter(Y1[:, 0], Y1[:, 1], color='b', label='Class 1')
        ax.scatter(Y2[:, 0], Y2[:, 1], color='r', label='Class 2')
        ax.scatter(Y3[:, 0], Y3[:, 1], color='g', label='Class 3')
        ax.set_ylabel('Component 2')

    # Set the title and labels
    ax.set_title('Projected Data')
    ax.set_xlabel('Component 1')

    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()

# Test the function
spp_plot(n_components=1, epsilon=0.05)
spp_plot(n_components=2, epsilon=0.05)