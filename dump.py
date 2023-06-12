#Imports
import scipy.linalg
import scipy.io as sio
import scipy
import numpy as np
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
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import LabelEncoder
import os
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris



#Sparsity Preserving Projections
def sparse_representation(X):
    # Initialize S matrix
    m, n = X.shape
    S = np.zeros((n, n))

    # Convert X to a matrix type that SPAMS can use
    X = np.asfortranarray(X, dtype=np.float64)
    print("Shape of X:" ,X.shape)

    # Set the parameters for SPAMS
    param = {'lambda1': 1e-8, 'numThreads': -1, 'mode': 2}

    # Loop through each sample to find s_i
    for i in tqdm(range(n)):
        # Define X_temp and x
        X_temp = np.delete(X, i, axis=1)
        print(f"Shape of X_temp_{i}: {X_temp.shape}")
        x = X[:, i].reshape(-1, 1)
        print(f"Shape of x_{i}: {x.shape}")

        # Use SPAMS to find the sparse representation
        s_i = spams.lasso(x, D=X_temp, return_reg_path=False, **param).toarray()
        print(f"Shape of s_{i}: {s_i.shape}")

        # Check for negative values and take their absolute value
        s_i[np.where(s_i < 0)] = np.abs(s_i[np.where(s_i < 0)])

        # Check for near-zero values and set them to zero
        s_i[np.where(np.abs(s_i) < 1e-10)] = 0

        # Normalize the new s_i matrix
        s_i_norm = np.linalg.norm(s_i, ord=1)
        if s_i_norm != 0:  # Check if the norm is not zero
            s_i = s_i / s_i_norm

        # Add a zero to the place that has been checked
        s_i = np.insert(s_i, i, 0)
        print(f"Shape of s_{i}: {s_i.shape}")

        # Store s_i in the S matrix
        S[:, i] = s_i.T

    return S


def snpe(X, n_components, S=None):
    if S is None:
        S = sparse_representation(X)
    # Compute S_beta
    S_beta = S + S.T - S.T @ S
    print("Shape of S_beta: ", S_beta.shape)
    print("Shape of X: ", X.shape)

    # Solve the generalized eigenvalue problem
    A = X @ S_beta @ X.T
    print("Shape of A: ", A.shape)
    B = X @ X.T
    print("Shape of B: ", B.shape)
    epsilon = 1e-6 # Small positive value
    np.fill_diagonal(B, np.diag(B) + epsilon) #add small positive value to the diagonal elements of B
    lamda, w = scipy.linalg.eigh(A,B)  # eigenvalues are assigned to lamda, eigenvectors to w

    # Sort the eigenvalues and eigenvectors in descending order
    idx = np.argsort(lamda)[::-1]
    lamda = lamda[idx]
    w = w[:, idx]

    w = normalize(w, axis=1)

    return w[:, :n_components]


def print_matrix_statistics(X):
    # Calculate the number of positive values
    num_positives = np.count_nonzero(X > 0)

    # Calculate the number of zero values
    num_zeros = np.count_nonzero(X == 0)

    # Calculate the number of negative values
    num_negatives = np.count_nonzero(X < 0)

    # Calculate the total number of values
    num_values = X.size

    # Calculate the range of positive values
    positive_values = X[X > 0]

    # Calculate the sparsity
    sparsity = num_zeros / num_values


    # Print the results
    print("Number of positive values: ", num_positives)
    print("Number of zero values: ", num_zeros)
    print("Number of negative values: ", num_negatives)
    print("Total number of values: ", num_values)
    print("Sparsity of the matrix: ", sparsity)
    if positive_values.size > 0:
        min_positive = np.min(positive_values)
        max_positive = np.max(positive_values)
        print("Range of positive values: [{}, {}]".format(min_positive, max_positive))
    else:
        print("No positive values found in the matrix")
    print()


def load_and_preprocess_YALE():
    data_dir = 'preprocessing'  # directory where the data is stored
    X = []  # list to store the image data
    y = []  # list to store the labels

    # loop over the directories inside the main data directory
    for dir in os.listdir(data_dir):
        dir_path = os.path.join(data_dir, dir)
        # skip if not a directory
        if not os.path.isdir(dir_path):
            continue
        # loop over the images inside each directory
        for filename in os.listdir(dir_path):
            # load the image as grayscale and resize it to 32x32 pixels
            img = cv2.imread(os.path.join(dir_path, filename), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (32, 32))
            # normalize the pixel values to be between 0 and 1
            img = img / 255.0
            # flatten the image to be a 1D array
            img = img.flatten()
            # add the image and label to the lists
            X.append(img)
            y.append(dir)

    # convert the lists to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # Normalize the data
    X_train = normalize(X_train, axis=1, norm='l2')
    X_test = normalize(X_test, axis=1, norm='l2')

    # Encode labels to integers
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    # Define target names and number of classes
    unique_labels = np.unique(y)
    target_names = [f"subject{str(label).zfill(2)}" for label in unique_labels]
    n_classes = len(target_names)

    return X_train, X_test, y_train, y_test, target_names, n_classes


def load_and_preprocess_YaleB():
    # Load Yale B dataset
    data = sio.loadmat('YaleB_32x32.mat')

    # Extract data (X) and labels (y)
    X = data['fea'].T
    y = data['gnd'].flatten()  #flatten the labels

    # Define target names and number of classes
    target_names = [str(x) for x in np.unique(y)]
    n_classes = len(target_names)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize the data
    X_train = normalize(X_train, axis=1, norm='l2')
    X_test = normalize(X_test, axis=1, norm='l2')

    return X_train, X_test, y_train, y_test, target_names, n_classes


def load_and_preprocess_LFW():
    # Load the LFW dataset
    data = fetch_lfw_people(min_faces_per_person=70, resize=0.4, color=False)

    X = data.data
    y = data.target
    target_names = data.target_names
    n_classes = len(target_names)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # Normalize the data
    X_train = normalize(X_train, axis=1, norm='l2')
    X_test = normalize(X_test, axis=1, norm='l2')

    # # Standardize the data
    # X_train = StandardScaler().fit_transform(X_train)
    # X_test = StandardScaler().fit_transform(X_test)

    return X_train, X_test, y_train, y_test, target_names, n_classes


def load_and_preprocess_yale():
    # Load the data
    yale = sio.loadmat('Yale_32x32.mat')
    X = yale['fea']
    y = yale['gnd'].flatten()

    # scale pixels to be [0,1]
    maxValue = np.amax(X)
    X = X / maxValue

    print("Imported X shape: ", X.shape)
    print("Imported Y shape: ", y.shape)

    # Define target names and number of classes
    target_names = [str(x) for x in np.unique(y)]
    n_classes = len(target_names)

    # Transpose X so that samples are along the first dimension, make each image a column
    X = X.T

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X.T, y, test_size=0.5, random_state=42)

    # Transpose X_train and X_test back to their original orientation
    # X_train = X_train.T
    # X_test = X_test.T

    return X_train, X_test, y_train, y_test, target_names, n_classes

def load_and_preprocess_iris():
    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names
    n_classes = len(target_names)

    # normalize the data
    X = normalize(X, axis=1, norm='l2')

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    return X_train, X_test, y_train, y_test, target_names, n_classes



X_train, X_test, y_train, y_test, target_names, n_classes = load_and_preprocess_yale()

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

print("\n-----------------------------------------------------------")

try:
    np.linalg.inv(X_train)
    print("The matrix is not singular.")
except np.linalg.LinAlgError:
    print("The matrix is singular.")

print("\n-----------------------------------------------------------")
print("-----                     Dataset                     -----")
print("-----------------------------------------------------------")
print("Number of features - initial dimension: ", X_train.shape[1])
print("Number of classes: ", n_classes)
print("Number of training samples: ", X_train.shape[0])
print("Number of testing samples: ", X_test.shape[0])

print("\n-----------------------------------------------------------")
print("----- Print statistics for the original training data -----")
print("-----------------------------------------------------------")
print_matrix_statistics(X_train)
print("----------------------------------------------------------")
print("----- Print statistics for the original testing data -----")
print("----------------------------------------------------------")
print_matrix_statistics(X_test)

# Define a PCA object
pca = PCA(n_components=None)

# Apply PCA to the training data
#X_train_pca = np.transpose(pca.fit_transform(X_train.T))
X_train_pca = pca.fit_transform(X_train)

# Apply PCA transformation to the testing data
#X_test_pca = np.transpose(pca.transform(X_test.T))
X_test_pca = pca.transform(X_test)

print("X_train_pca shape:", X_train_pca.shape)
print("X_test_pca shape:", X_test_pca.shape)

print(f"Number of components for PCA: {pca.n_components_}\n")

print("\n-----------------------------------------------------------")

try:
    np.linalg.inv(X_train_pca)
    print("The matrix is not singular.")
except np.linalg.LinAlgError:
    print("The matrix is singular.")

print("\n------------------------------------------------------------------")
print("----- Print statistics for the PCA-transformed training data -----")
print("------------------------------------------------------------------")
print_matrix_statistics(X_train_pca)
print("-----------------------------------------------------------------")
print("----- Print statistics for the PCA-transformed testing data -----")
print("-----------------------------------------------------------------")
print_matrix_statistics(X_test_pca)

# Plot the PCA-transformed training data
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train)

# Define number of components for SNPE
n_components_snpe = 70 # You should adjust this value as needed

print(f"Number of components for SNPE: {n_components_snpe}\n")

# Compute the sparse representation for the training data
S_train = sparse_representation(X_train_pca)
print("\nS_train shape:", S_train.shape)

print("\n-----------------------------------------------------------")
print("-----                S Matrix statistics              -----")
print("-----------------------------------------------------------")
print_matrix_statistics(S_train)

# Apply SNPE to the training data
X_train_snpe = snpe(X_train_pca, n_components=n_components_snpe, S=S_train)

# Apply the same SNPE transformation to the testing data
X_test_snpe = snpe(X_test_pca, n_components=n_components_snpe, S=S_train)  # use S_train, not a new S

print("\n-------------------------------------------------------------------")
print("----- Print statistics for the SNPE-transformed training data -----")
print("-------------------------------------------------------------------")
print_matrix_statistics(X_train_snpe)
print("------------------------------------------------------------------")
print("----- Print statistics for the SNPE-transformed testing data -----")
print("------------------------------------------------------------------")
print_matrix_statistics(X_test_snpe)


print("------------------------------------------------------------------")
print("-----                     1NN classifier                     -----")
print("------------------------------------------------------------------")
# Define a 1-Nearest Neighbors classifier
knn = KNeighborsClassifier(n_neighbors=1)

# Fit the classifier to the SNPE-transformed training data
knn.fit(X_train_snpe, y_train)

# Make predictions on the SNPE-transformed testing data
y_pred = knn.predict(X_test_snpe)

# calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("-> Accuracy: {:.2f}%".format(accuracy * 100))

# Print the classification report
#print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

# Plot the confusion matrix
print("\n-> Confusion matrix:\n",confusion_matrix(y_test, y_pred))

