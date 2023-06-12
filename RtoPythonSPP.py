#Imports
from sklearn.preprocessing import StandardScaler
from scipy import linalg
import scipy.io as sio
import scipy
from time import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import normalize
import spams
import matplotlib.cm as cm
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import LabelEncoder
import os
import cv2
import numpy as np
from scipy.linalg import eigh
import cvxpy as cp
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

def aux_preprocess(data, method):
    if method == "scale":
        multiplier = 1 / np.std(data, axis=0)
        data = data @ np.diag(multiplier)
        info = {'type': 'scale', 'mean': np.zeros(data.shape[1]), 'multiplier': np.diag(multiplier)}
    elif method == "cscale":
        mean = np.mean(data, axis=0)
        data = data - mean
        multiplier = 1 / np.std(data, axis=0)
        data = data @ np.diag(multiplier)
        info = {'type': 'cscale', 'mean': mean, 'multiplier': np.diag(multiplier)}
    else:
        # add the other preprocess methods here
        raise ValueError('Invalid preprocessing method')
    return data, info


def is_singular(matrix, tol=1e-15):
    """Check if a matrix is singular by comparing its rank with its shape"""
    return np.linalg.matrix_rank(matrix) < min(matrix.shape) - tol


def is_psd(matrix, tol=1e-8):
    """Check if a matrix is positive semi-definite by attempting a Cholesky decomposition"""
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False


def aux_geigen(A, B, k, maximal=True):
    eps = 1e-10
    A = A + np.eye(A.shape[0])*eps
    #B = B + np.eye(B.shape[0])*eps
    # Check for positive semi-definiteness
    if not is_psd(A) or not is_psd(B):
        raise ValueError("Matrix not positive semi-definite")
    w, v = eigh(A, B, subset_by_index=(len(B) - k, len(B) - 1))
    return v


def spp_compute_si(xi, Xi, reltol):
    n = Xi.shape[1]
    si = cp.Variable(n)

    proj_xi = Xi @ np.linalg.lstsq(Xi, xi, rcond=None)[0]
    fraction_explained = np.linalg.norm(proj_xi) / np.linalg.norm(xi)
    rank_Xi = np.linalg.matrix_rank(Xi)
    print(rank_Xi)
    print(fraction_explained)

    constraints = [cp.norm(xi - Xi @ si, 2) <= reltol, cp.sum(si) == 1]
    problem = cp.Problem(cp.Minimize(cp.norm1(si)), constraints)
    problem.solve(solver=cp.ECOS, verbose=False)
    print("Status:", problem.status)
    return si.value

## Using SPAMS
# def spp_compute_si(xi, Xi, reltol):
#     n = Xi.shape[1]
#     proj_xi = Xi @ np.linalg.lstsq(Xi, xi, rcond=None)[0]
#     fraction_explained = np.linalg.norm(proj_xi) / np.linalg.norm(xi)
#     rank_Xi = np.linalg.matrix_rank(Xi)
#     print(rank_Xi)
#     print(fraction_explained)
#     # Setting up the parameters for the SPAMS solver
#     params = {"mode": 1, "lambda1": reltol, "pos": True}
#     # Making sure xi and Xi are in the correct format for the SPAMS solver
#     xi = np.asfortranarray(xi.reshape(-1, 1))
#     Xi = np.asfortranarray(Xi)
#     # Using SPAMS to solve the problem
#     si = spams.lasso(xi, D=Xi, return_reg_path=False, **params).toarray().flatten()
#     print(si)
#     print(np.sum(si))
#     # Normalizing the output to satisfy the sum to 1 constraint
#     si /= (np.sum(si) + 1e-10)
#     return si


def do_spp(X, ndim=2, preprocess='center', reltol=1e-4):
    # preprocess data
    X, trfinfo = aux_preprocess(X, preprocess)
    n = X.shape[0]

    # compute S
    S = np.zeros((n, n))
    for i in tqdm(range(n)):
        xi = X[i, :]
        Xi = X[np.arange(len(X))!=i, :].T
        print("Any NaN in X? ", np.isnan(X).any())  # Check X matrix
        print("Any NaN in xi? ", np.isnan(xi).any())  # Check xi vector

        S[i, np.arange(len(S))!=i] = spp_compute_si(xi, Xi, reltol)

    print("Any NaN in S? ", np.isnan(S).any())  # Check S matrix here

    # Sbeta and projection matrix
    Sbeta = S + S.T - (S.T @ S)
    print("Any NaN in Sbeta? ", np.isnan(Sbeta).any())  # Check Sbeta matrix here

    LHS = X.T @ Sbeta @ X
    RHS = X.T @ X

    print(np.isnan(X).any())
    print(np.isnan(Sbeta).any())

    if np.isnan(LHS).any() or np.isnan(RHS).any():
        print("NaN detected!")
    if np.isinf(LHS).any() or np.isinf(RHS).any():
        print("Infinity detected!")

    projection = aux_geigen(LHS, RHS, ndim, maximal=True)

    return {'Y': X @ projection, 'trfinfo': trfinfo, 'projection': projection}


# Test on LFW and iris datasets
def main():
    '''
    #load iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    '''

    #load lfw people dataset
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.3)
    X = lfw_people.data
    y = lfw_people.target

    print("\nAs loaded the X shape (samples,pixels) is:",X.shape)
    print("\nAs loaded the y shape is:",y.shape)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    #normalize data
    X_train = normalize(X_train, axis=1, norm='l2')
    X_test = normalize(X_test, axis=1, norm='l2')

    print("\nAfer normalization and splitting, the X_train shape (samples,pixels) is:",X_train.shape)
    print("\nAfer normalization and splitting, the X_test shape (samples,pixels) is:",X_test.shape)

    pca = PCA(n_components=None, whiten=True).fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    print("\nAfter PCA the X_train shape (samples,pixels) is:",X_train.shape)
    print("\nAfter PCA the X_test shape (samples,pixels) is:",X_test.shape)

    try:
        np.linalg.inv(X_train)
        print("\nThe matrix is not singular.")
    except np.linalg.LinAlgError:
        print("\nThe matrix is singular.")

    # check if full rank
    rank = np.linalg.matrix_rank(X_train)
    is_full_rank_1 = rank == min(X_train.shape)
    cond = np.linalg.cond(X_train)
    is_full_rank_2 = cond < 1 / np.finfo(X_train.dtype).eps
    print("\nIs the training data full-rank?", is_full_rank_1 and is_full_rank_2)

    # test different tolerance levels
    out1 = do_spp(X_train, ndim=2, reltol=0.1, preprocess='scale')
    out2 = do_spp(X_train, ndim=2, reltol=0.01, preprocess='scale')
    out3 = do_spp(X_train, ndim=2, reltol=0.001, preprocess='scale')

    # Apply the projection matrix to the test set
    X_test_transformed_1 = X_test @ out1['projection']
    X_test_transformed_2 = X_test @ out2['projection']
    X_test_transformed_3 = X_test @ out3['projection']

    # Train a Support Vector Classifier on the transformed training data
    svc1 = SVC().fit(out1['Y'], y_train)
    svc2 = SVC().fit(out2['Y'], y_train)
    svc3 = SVC().fit(out3['Y'], y_train)

    # Use the trained classifier to predict labels for the transformed test data
    y_pred_1 = svc1.predict(X_test_transformed_1)
    y_pred_2 = svc2.predict(X_test_transformed_2)
    y_pred_3 = svc3.predict(X_test_transformed_3)

    # print the accuracy of the classifier
    print(accuracy_score(y_test, y_pred_1))
    print(accuracy_score(y_test, y_pred_2))
    print(accuracy_score(y_test, y_pred_3))

    #print the confusion matrix
    print(confusion_matrix(y_test, y_pred_1))
    print(confusion_matrix(y_test, y_pred_2))
    print(confusion_matrix(y_test, y_pred_3))

    def plot_results(data, labels, title):
        # scatter plot
        plt.figure(figsize=(8,6))
        plt.scatter(data[:,0], data[:,1], c=labels)
        plt.title(title)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.grid(True)
        plt.show()

    # apply the function to the outputs
    #plot_results(out1['Y'], y_train, 'SPP with reltol=0.01')
    #plot_results(out1['Y'], y_test, 'SPP with reltol=0.01')


# Test on Yale Faces Dataset
    # def main():
    #     yale = sio.loadmat('Yale_32x32.mat')
    #     X = yale['fea']
    #     y = yale['gnd'].flatten()
    #
    #     # scale pixels to be [0,1]
    #     maxValue = np.amax(X)
    #     X = X / maxValue
    #     print(X.shape)
    #
    #     pca = PCA(n_components=None, whiten=True).fit(X)
    #     X_pca = pca.transform(X)
    #
    #     if is_singular(X_pca):
    #         raise ValueError("Input matrix is singular!")
    #     print("The matrix is not singular and has a shape of: ",X_pca.shape)
    #
    #     # test different tolerance levels
    #     out1 = do_spp(X_pca[:None,:], ndim=150, reltol=0.1, preprocess='scale')
    #
    #     import matplotlib.pyplot as plt
    #
    #     def plot_results(data, labels, title):
    #         # scatter plot
    #         plt.figure(figsize=(8,6))
    #         plt.scatter(data[:,0], data[:,1], c=labels)
    #         plt.title(title)
    #         plt.xlabel('Component 1')
    #         plt.ylabel('Component 2')
    #         plt.grid(True)
    #         plt.show()
    #
    #     # apply the function to the outputs
    #     plot_results(out1['Y'], y, 'SPP with reltol=0.01')


if __name__ == '__main__':
    main()