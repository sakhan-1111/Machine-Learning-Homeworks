#************************************************************************************
# Shafiqul Alam Khan
# ML – HW#3 - Part 2
# Filename: HW3_Part_2.py
# Due: Oct. 6, 2023
#
# Objective:
# • Use a scikit-learn perceptron model to classify all targets.
# • Same requirements from Part 1
#*************************************************************************************

f = open("Output_HW3_Part2.txt", "w")
f.write('> Importing Packages...')
# Importing all required libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Perceptron
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, RocCurveDisplay
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
import seaborn as sns
import warnings
f.write('\n\t\t\t\t....DONE!\n')


####################################################
#
# Data Preprocessing
#
####################################################

# Read master dataset
f.write('\n> Loading the master DataFrame...')
master = pd.read_csv('master.csv')
f.write('\n\t\t\t\t...DONE!\n')

# Define X and y
f.write('\n> Defining X and y...')
X, y = master.iloc[:, 1:6], master.iloc[:, 6:]
f.write('\n\t\t\t\t...DONE!\n')

# Perform split
f.write('\n> Perform dataset split...')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
f.write('\n\t\t\t\t...DONE!\n')

# Normalize datasets 
X_train_norm = X_train.values
min_max_scaler = preprocessing.MinMaxScaler()
X_scaled = min_max_scaler.fit_transform(X_train_norm)
X_train_norm = pd.DataFrame(X_scaled)

X_test_norm = X_test.values
min_max_scaler = preprocessing.MinMaxScaler()
X_scaled = min_max_scaler.fit_transform(X_test_norm)
X_test_norm = pd.DataFrame(X_scaled)

# Ignore warning
warnings.filterwarnings('ignore')


####################################################
#
# Pipeline
#
####################################################

f.write('\n>> Making Piplines...')
# Make pipeline to perform PCA with standard scaler and perform Logistic Regression
pipe_ppn_pca = make_pipeline(StandardScaler(), PCA(n_components=2), Perceptron(random_state=42))

# Make pipeline to perform PCA with normalized data and perform Logistic Regression
pipe_ppn_pca_norm = make_pipeline(PCA(n_components=2), Perceptron(random_state=42))

# Make pipeline to perform LDA with standard scaler and perform Logistic Regression
pipe_ppn_lda = make_pipeline(StandardScaler(), LDA(solver='svd', n_components=2), Perceptron(random_state=42))

# Make pipeline to perform LDA with normalized data and perform Logistic Regression
pipe_ppn_lda_norm = make_pipeline(LDA(solver='svd', n_components=2), Perceptron(random_state=42))
f.write('\n\t\t\t\t...DONE!\n')


####################################################
#
# GridSearch & Cross-validation
#
####################################################

f.write('\n>> Setting parameters for grid search...')
# Setting parameters
param_grid = [{'perceptron__max_iter': [1000000, 100000, 10000, 1000, 100, 10],
               'perceptron__eta0': [1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001],
              'perceptron__validation_fraction': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]

f.write('\n\t\t\t\t...DONE!\n')


f.write('\n>> Performing GridSearchCV for grid search for pca (standard scaler)...\n')

# Set up grid search & cross validation for pca
gs_pca = GridSearchCV(estimator=pipe_ppn_pca, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=-1)

# Train grid search model
gs_pca = gs_pca.fit(X_train, y_train)

f.write('\n> Best score from GridSearchCV: ' + str(gs_pca.best_score_) + '\n')
f.write('\n> Best parameters from GridSearchCV: ' + str(gs_pca.best_params_) + '\n')
f.write('\n\t\t\t\t...DONE!\n')


f.write('\n>> Performing GridSearchCV for grid search for pca (normalized)...\n')

# Set up grid search & cross validation for pca
gs_pca_norm = GridSearchCV(estimator=pipe_ppn_pca_norm, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=-1)

# Train grid search model
gs_pca_norm = gs_pca_norm.fit(X_train_norm, y_train)

f.write('\n> Best score from GridSearchCV: ' + str(gs_pca_norm.best_score_) + '\n')
f.write('\n> Best parameters from GridSearchCV: ' + str(gs_pca_norm.best_params_) + '\n')
f.write('\n\t\t\t\t...DONE!\n')


f.write('\n>> Performing GridSearchCV for grid search for lda (standard scaler)...\n')

# Set up grid search & cross validation for pca
gs_lda = GridSearchCV(estimator=pipe_ppn_lda, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=-1)

# Train grid search model
gs_lda = gs_lda.fit(X_train, y_train)

f.write('\n> Best score from GridSearchCV: ' + str(gs_lda.best_score_) + '\n')
f.write('\n> Best parameters from GridSearchCV: ' + str(gs_lda.best_params_) + '\n')
f.write('\n\t\t\t\t...DONE!\n')


f.write('\n>> Performing GridSearchCV for grid search for lda (normalized)...\n')

# Set up grid search & cross validation for pca
gs_lda_norm = GridSearchCV(estimator=pipe_ppn_lda_norm, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=-1)

# Train grid search model
gs_lda_norm = gs_lda_norm.fit(X_train_norm, y_train)

f.write('\n> Best score from GridSearchCV: ' + str(gs_lda_norm.best_score_) + '\n')
f.write('\n> Best parameters from GridSearchCV: ' + str(gs_lda_norm.best_params_) + '\n')
f.write('\n\t\t\t\t...DONE!\n')


####################################################
#
# Selecting the best hyperparameters & Printing results
#
####################################################

f.write('\n>> Observing the outputs: \n')
f.write('\n>The best hyperparameters found to be "eta0 = 0.1", "max_iter = 1000000" and "validation_fraction = 0.1" for lda (standard scaler)\n')


# Printing accuracy with best hyperparameters found
f.write('\n>> Printing results with best hyperparameters found...\n')

# Perform LDA
lda = LDA(solver='svd', n_components=2)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

# Create an instance of Logistic Regression Classifier and fit the data.
ppn = Perceptron(eta0 = 0.1, max_iter = 1000000, validation_fraction = 0.1, random_state=42, n_jobs = -1)
ppn_fit = ppn.fit(X_train_lda, y_train)

# Testing the model data
y_pred = ppn.predict(X_test_lda)
y_pred = y_pred.reshape(4808,1)

# Printing results
f.write('\nMisclassified samples: %d' % (y_test != y_pred).sum() + '\n')
f.write('\nTraining Accuracy: %.4f' % ppn.score(X_train_lda, y_train) + '\n')
f.write('\nTest Accuracy: %.4f' % ppn.score(X_test_lda, y_test) + '\n')
f.write('\nPrecision: %.4f' % precision_score(y_true=y_test, y_pred=y_pred, pos_label='positive', average='macro') + '\n')
f.write('\nRecall: %.4f' % recall_score(y_true=y_test, y_pred=y_pred, pos_label='positive', average='macro') + '\n')
f.write('\nF1: %.4f' % f1_score(y_true=y_test, y_pred=y_pred, pos_label='positive', average='macro') + '\n')
f.write('\n\t\t\t\t...DONE!\n')


####################################################
#
# Plot the classification outcome
#
####################################################

f.write('\n>> Plotting the classification outcome of the best model... \n')

def plot_decision_regions(X,y,classifier,test_idx=None,resolution=0.02):
    markers = ('s','x','o','D')
    colors = ('red', 'blue', 'lightgreen','orange')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Plot all the samples
    X_test,y_test=X[test_idx,:],y[test_idx]
    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0],y=X[y==cl,1],alpha=0.8,c=cmap(idx),marker=markers[idx],label=cl)

    # Highlight test samples
    if test_idx:
        X_test,y_test =X[test_idx,:],y[test_idx]

    plt.scatter(X_test[:,0],X_test[:,1],facecolors='none', edgecolors='black', alpha=1.0, linewidths=1, marker='o', s=55, label='test set')


# Combine all training and test data to single object variables
X_combined_lda = np.vstack((X_train_lda, X_test_lda))
y_combined = np.hstack((y_train.to_numpy().flatten(), y_test.to_numpy().flatten()))
plot_decision_regions(X=X_combined_lda, y=y_combined, classifier=ppn, test_idx=range(y_train.size, y_train.size + y_test.size))

# Plot with labels and legend
plt.title('Decision Region using the LDA transformed/projected features')
plt.xlabel('LDA Feature 1 [Standard Scaler]')
plt.ylabel('LDA Feature 2 [Standard Scaler]')
plt.legend(loc='upper left')

# Save plot
plt.savefig('Output_HW_3_Part2_Decision_Region.png')
f.write('\n\t\t\t\t...DONE!\n')
plt.clf()


####################################################
#
# Plot the learning curves
#
####################################################

f.write('\n>> Plotting the learning curves of the best model... \n')

# The limit is 80-20 split of training data
train_sizes = [1, 100, 1000, 2000, 3000, 5000, 8000, 10000, 13000, 15386]

# Define train_sizes, train_scores & validation_scores
train_sizes, train_scores, validation_scores = learning_curve(estimator = Perceptron(eta0 = 0.1, max_iter = 1000000, validation_fraction = 0.1, random_state=42, n_jobs = -1), 
                                                              X = X, 
                                                              y = y, 
                                                              train_sizes = train_sizes,
                                                              cv = 10,
                                                              scoring = 'neg_mean_squared_error')

# Calculate means
train_scores_mean = -train_scores.mean(axis = 1)
validation_scores_mean = -validation_scores.mean(axis = 1)

# Plot the learning curve
plt.plot(train_sizes, train_scores_mean, label = 'Training error')
plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')
plt.ylabel('MSE', fontsize = 12)
plt.xlabel('Training set size', fontsize = 12)
plt.title('Learning curves for the Perceptron model', fontsize = 14, y = 1.03)
plt.legend()

# Save plot
plt.savefig('Output_HW_3_Part2_Learning_Curve.png')
f.write('\n\t\t\t\t...DONE!\n')
plt.clf()


####################################################
#
# Plot the confusion matrices
#
####################################################

f.write('\n>> Plotting the confusion matrices of the best model... \n')

# Create the confusion matrices
y_pred_train = ppn.predict(X_train_lda)
y_pred_train = y_pred_train.reshape(19232,1)

cfmat_train = confusion_matrix(y_true=y_train, y_pred=y_pred_train)
cfmat_test = confusion_matrix(y_true=y_test, y_pred=y_pred)

# Map the values of datasets
mapping = {'Lab':0, 'Corridor':1, 'Lobby':2, 'Hall':3}

# Plot & save training confusion matrix
sns.heatmap(cfmat_train,
            cmap='gnuplot',
            fmt='d',
            annot = True,
            cbar=True,
            xticklabels = mapping,
            yticklabels = mapping)
plt.ylabel('True label', fontsize = 12)
plt.xlabel('Predicted label', fontsize = 12)
plt.title('Training confusion matrix', fontsize = 14)
plt.savefig('Output_HW_3_Part2_Training_Confusion_Matrix.png')
plt.clf()

# Plot & save test confusion matrix
sns.heatmap(cfmat_test,
            cmap='gnuplot',
            fmt='d',
            annot = True,
            cbar=True,
            xticklabels = mapping,
            yticklabels = mapping)
plt.ylabel('True label', fontsize = 12)
plt.xlabel('Predicted label', fontsize = 12)
plt.title('Test confusion matrix', fontsize = 14)
plt.savefig('Output_HW_3_Part2_Test_Confusion_Matrix.png')
plt.clf()
f.write('\n\t\t\t\t...DONE!\n')


####################################################
#
# Plot the ROC AUC graph
#
####################################################

f.write('\n>> Plotting the ROC AUC graph of the best model... \n')

# Calculate fpr, tpr & thresholds
y_score = ppn._predict_proba_lr(X_test_lda)
fpr, tpr, thresholds = roc_curve(y_true = y_test, y_score = y_score[:, 1], pos_label = 4)

# Calculate ROC AUC
roc_auc = auc(fpr, tpr)

# Plot & save ROC AUC
display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
display.plot()
plt.title('ROC AUC graph', fontsize = 14)
plt.savefig('Output_HW_3_Part2_ROC_AUC.png')
plt.clf()
f.write('\n\t\t\t\t...DONE!\n')
f.write('\n******************PROGRAM is DONE *******************')
f.close()
