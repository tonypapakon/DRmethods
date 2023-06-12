from sklearn.datasets import fetch_lfw_people
from time import time
import scipy
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from sklearn.preprocessing import normalize
import spams
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def sparse_representation(X):
    # Initialize S matrix
    m, n = X.shape
    S = np.zeros((n, n))

    # Create a Lasso model with the specified alpha
    #alpha = 1e-6
    #lasso_model = Lasso(alpha=alpha, max_iter=30000)

    # Create an ElasticNet model with the specified alpha and l1_ratio
    #elasticnet_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=30000)

    # Create a LinearRegression model
    #linear_model = LinearRegression()

    # Convert X to a matrix type that SPAMS can use
    X = np.asfortranarray(X)

    # Set the parameters for SPAMS
    param = {'lambda1': 0.1, 'numThreads': -1, 'mode': 2}
    #param = {'L': 10, 'eps': 0.1, 'numThreads': -1}


    # Loop through each sample to find s_i
    for i in tqdm(range(n)):
        # Define X_temp and x
        X_temp = np.delete(X, i, axis=1)
        #X_temp = X_temp.astype("double")
        #x = X[:, i]
        x = X[:, i].reshape(-1, 1)

        # Use SPAMS to find the sparse representation
        s_i = spams.lasso(x, D=X_temp, return_reg_path=False, **param).toarray()
        #s_i = spams.omp(x, D=X_temp, **param).toarray()

        # Fit the Lasso model
        #lasso_model.fit(X_temp, x)

        # Extract the solution s_i from the model
        #s_i = lasso_model.coef_

        # Fit the ElasticNet model
        #elasticnet_model.fit(X_temp, x)

        # Extract the solution s_i from the model
        #s_i = elasticnet_model.coef_

        # Fit the LinearRegression model
        #linear_model.fit(X_temp, x)

        # Extract the solution s_i from the model
        #s_i = linear_model.coef_

        # Check for negative values and take their absolute value
        s_i[np.where(s_i < 0)] = np.abs(s_i[np.where(s_i < 0)])

        # Check for near-zero values and set them to zero
        s_i[np.where(np.abs(s_i) < 1e-7)] = 0

        # Normalize the new s_i matrix
        s_i_norm = np.linalg.norm(s_i, ord=1)
        if s_i_norm != 0:  # Check if the norm is not zero
            s_i = s_i / s_i_norm

        # Add a zero to the place that has been checked
        s_i = np.insert(s_i, i, 0)

        # Store s_i in the S matrix
        S[:, i] = s_i.T
    return S


def snpe(X, n_components):
    S = sparse_representation(X)
    # Compute S_beta
    S_beta = S + S.T - S.T @ S

    # Solve the generalized eigenvalue problem
    A = X @ S_beta @ X.T
    B = X @ X.T
    epsilon = 1e-6  # Small positive value
    B += epsilon * np.eye(*B.shape) #add small positive value to the diagonal elements of B
    w, lamda = scipy.linalg.eigh(A, B)

    # Sort the eigenvalues and eigenvectors in descending order
    idx = np.argsort(w)[::-1]
    w = w[idx]
    lamda = lamda[:, idx]

    # Normalize the eigenvectors
    lamda = normalize(lamda, axis=1)

    return lamda[:, :n_components]

def print_matrix_statistics(X):
    # Calculate the number of positive values
    num_positives = np.count_nonzero(X > 0)

    # Calculate the number of zero values
    num_zeros = np.count_nonzero(X == 0)

    # Calculate the total number of values
    num_values = X.size

    # Calculate the range of positive values
    positive_values = X[X > 0]
    if positive_values.size > 0:
        min_positive = np.min(positive_values)
        max_positive = np.max(positive_values)
        print("Range of positive values: [{}, {}]".format(min_positive, max_positive))
    else:
        print("No positive values found in the matrix")

    # Print the results
    print("Number of positive values: ", num_positives)
    print("Number of zero values: ", num_zeros)
    print("Total number of values: ", num_values)
    print("\n")


###########################################
#                Dataset
###########################################
t0 = time()
# Load and preprocess the dataset
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4, color=False)
X = lfw_people.data
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
X_train = normalize(X_train, axis=1, norm='l2')
X_test = normalize(X_test, axis=1, norm='l2')

print("Initial dimensions: ", X_train.shape[1])

print("After normalization:")
print("X_train range: ", np.min(X_train), np.max(X_train))
print("X_test range: ", np.min(X_test), np.max(X_test))


sparsity_train = np.count_nonzero(X_train==0) / X_train.size
sparsity_test = np.count_nonzero(X_test==0) / X_test.size
print("Sparsity in training data: ", sparsity_train)
print("Sparsity in testing data: ", sparsity_test)


###########################################
#                  PCA
###########################################
print("\n###################################################")
print("                        PCA")
print("###################################################\n")
# Project the training data onto a PCA subspace
pca = PCA(n_components=0.98)  # Preserve 98% of the total variance
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# #all the values in these arrays will be non-negative
# X_train_pca = np.abs(X_train_pca)
# X_test_pca = np.abs(X_test_pca)

print("Dimensions after PCA: ", X_train_pca.shape[1])

print("After normalization:")
print("X_train range: ", np.min(X_train_pca), np.max(X_train_pca))
print("X_test range: ", np.min(X_test_pca), np.max(X_test_pca))

sparsity_train_pca = np.count_nonzero(X_train_pca==0) / X_train_pca.size
sparsity_test_pca = np.count_nonzero(X_test_pca==0) / X_test_pca.size
print("Sparsity in training data: ", sparsity_train_pca)
print("Sparsity in testing data: ", sparsity_test_pca)

print("Sparse representation using SPAMS Lasso:")
S = sparse_representation(X_train_pca)
print_matrix_statistics(S)

###########################################
#                  SNPE
###########################################
print("\n###################################################")
print("                        SNPE")
print("###################################################\n")

n_components = 150


# Apply the modified SNPE on the training data
print("Apply the SNPE on the training data")
X_train_snpe = snpe(X_train_pca, n_components)

print("Dimensions after modified SNPE: ", X_train_snpe.shape[1])

# Transform the testing data using the SNPE transformations learned from the training data
print("Transform the testing data learned from the training data")
X_test_snpe = snpe(X_test_pca, n_components)
print("Dimensions after SNPE test: ", X_test_snpe.shape[1])

S = sparse_representation(X_train_snpe)
print_matrix_statistics(S)


#print("done in % 0.3fs" % (time() - t0))


###########################################
# CROSS VALIDATION TO CHOOSE kNN PARAMETERS
###########################################
print("\n###################################################")
print("                 kNN classifier")
print("###################################################\n")
print("Fitting the classifier to the training set")
t0 = time()

# Create KNN classifier and perform GridSearchCV to find the best n_neighbors
params = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15]}

knn = KNeighborsClassifier()

grid = GridSearchCV(knn, params, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)

grid = grid.fit(X_train_snpe, y_train)

best_n_neighbors = grid.best_params_['n_neighbors']

print("done in % 0.3fs" % (time() - t0))
print("Best n_neighbors found: ", best_n_neighbors)
print("Best cross-validated accuracy:", grid.best_score_)

###########################################
#          Apply kNN classifier
###########################################
# Fit the model with the best n_neighbors found
knn_best = KNeighborsClassifier(n_neighbors=best_n_neighbors)
knn_best.fit(X_train_snpe, y_train)

print("Predicting people's names on the test set")
t0 = time()
# Predict and evaluate the model
y_predicted = knn_best.predict(X_test_snpe)
print("done in % 0.3fs" % (time() - t0))

# print classifiction results
print("Classification report:\n", classification_report(y_test, y_predicted, target_names=target_names, zero_division=0))

# calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_predicted)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# print confusion matrix
print("Confusion Matrix is:")
print(confusion_matrix(y_test, y_predicted, labels=range(n_classes)))


###########################################
#          Apply 1NN classifier
###########################################
print("\n###################################################")
print("                 1NN classifier")
print("###################################################\n")
print("Fitting the classifier to the training set")
# Fit the model with the best n_neighbors found
knn_best = KNeighborsClassifier(n_neighbors=1)
knn_best.fit(X_train_snpe, y_train)

print("Predicting people's names on the test set")
t0 = time()
# Predict and evaluate the model
y_predicted = knn_best.predict(X_test_snpe)
print("done in % 0.3fs" % (time() - t0))

# print classifiction results
print("Classification report:\n", classification_report(y_test, y_predicted, target_names=target_names, zero_division=0))

# calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_predicted)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# print confusion matrix
print("Confusion Matrix is:")
print(confusion_matrix(y_test, y_predicted, labels=range(n_classes)))


'''
###########################################
# CROSS VALIDATION TO CHOOSE SVM PARAMETERS
###########################################
print("\n###################################################")
print("                 SVM classifier")
print("###################################################\n")
print("Fitting the classifier to the training set")
t0 = time()

param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10, 15, 100]}

svm = SVC(kernel='rbf',probability=True, class_weight='balanced', random_state=42)

clf = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)

clf = clf.fit(X_train_snpe, y_train)

print("done in % 0.3fs" % (time() - t0))
print("Best parameters found: ", clf.best_params_)
print("Best cross-validated accuracy:", clf.best_score_)
#print("Best estimator found by grid search:", clf.best_estimator_)

best_classifier = clf.best_estimator_


###########################################
#          Apply SVM classifier
###########################################
# Fit the model with the best parameters found
best_classifier.fit(X_train_snpe, y_train)

print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test_snpe)
print("done in % 0.3fs" % (time() - t0))

# print classifiction results
print("Classification report:\n", classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# print confusion matrix
print("Confusion Matrix is:")
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
'''

# # Visualize the results
#
# colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'gray', 'orange']
# markers = ['o', '^', 's', 'p', '*', '+', 'x', 'D', '|', '_']
#
# # Plot the true class labels
# for i, target_name in enumerate(target_names):
#     plt.scatter(X_test_snpe[y_test == i, 0], X_test_snpe[y_test == i, 1], color=colors[i], marker=markers[i], label=target_name)
#
# # Plot the predicted class labels
# for i, target_name in enumerate(target_names):
#     plt.scatter(X_test_snpe[y_pred == i, 0], X_test_snpe[y_pred == i, 1], color=colors[i], marker=markers[i], facecolors='none')
#
# plt.xlabel('SNPE 1')
# plt.ylabel('SNPE 2')
# plt.legend()
# plt.show()

n_classes = len(target_names)
colors = cm.rainbow(np.linspace(0, 1, n_classes))
for i, target_name in enumerate(target_names):
    indices = np.where(y_train == i)[0]
    plt.scatter(X_train_snpe[indices, 0], X_train_snpe[indices, 1], color=colors[i], label=target_name)
plt.xlabel('SNPE 1')
plt.ylabel('SNPE 2')
plt.legend()
plt.show()
