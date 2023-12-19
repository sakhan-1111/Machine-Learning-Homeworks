#************************************************************************************
# Shafiqul Alam Khan
# ML – HW#3 - Part 1
# Filename: HW3_Part_1.py
# Due: Oct. 6, 2023
#
# Objective:
# • Use a scikit-learn logical regression model to classify all targets
# • Use a pipeline, GridSearch, and cross-validation to setup, tune, train, and validate the model. Thigs to consider: Standardization vs. Normalization, PCA vs. LDA, etc.
# • Print out the best parameters, training & test accuracy values, include the precision, recall, and F1-score to a text file.
# • Plot the classification outcome of the best model, the training and validation accuracy learning curves of the best model, training and test confusion matrices of the best model,and the ROC AUC graph of the best model, save them as images for submission.
#*************************************************************************************

f = open("Output_HW3_Part1.txt", "w")
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
from sklearn.linear_model import LogisticRegression
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

# Load datasets
# Read Lab139_7.1 csv files
f.write('\n> Read Lab Data Files...')
lab1 = pd.read_csv('Measurements_Upload/Lab139_7.1/Loc_0612/Lab_139_1Ch1.csv', skiprows=2, delimiter = ';')
lab1 = lab1.iloc[:, 0:5]

lab2 = pd.read_csv('Measurements_Upload/Lab139_7.1/Loc_0612/Lab_139_2Ch1.csv', skiprows=2, delimiter = ';')
lab2 = lab2.iloc[:, 0:5]

lab3 = pd.read_csv('Measurements_Upload/Lab139_7.1/Loc_0612/Lab_139_3Ch1.csv', skiprows=2, delimiter = ';')
lab3 = lab3.iloc[:, 0:5]

lab4 = pd.read_csv('Measurements_Upload/Lab139_7.1/Loc_0612/Lab_139_4Ch1.csv', skiprows=2, delimiter = ';')
lab4 = lab4.iloc[:, 0:5]

lab5 = pd.read_csv('Measurements_Upload/Lab139_7.1/Loc_0612/Lab_139_5Ch1.csv', skiprows=2, delimiter = ';')
lab5 = lab5.iloc[:, 0:5]

lab6 = pd.read_csv('Measurements_Upload/Lab139_7.1/Loc_0612/Lab_139_6Ch1.csv', skiprows=2, delimiter = ';')
lab6 = lab6.iloc[:, 0:5]

lab7 = pd.read_csv('Measurements_Upload/Lab139_7.1/Loc_0612/Lab_139_7Ch1.csv', skiprows=2, delimiter = ';')
lab7 = lab7.iloc[:, 0:5]

lab8 = pd.read_csv('Measurements_Upload/Lab139_7.1/Loc_0612/Lab_139_8Ch1.csv', skiprows=2, delimiter = ';')
lab8 = lab8.iloc[:, 0:5]

lab9 = pd.read_csv('Measurements_Upload/Lab139_7.1/Loc_0612/Lab_139_9Ch1.csv', skiprows=2, delimiter = ';')
lab9 = lab9.iloc[:, 0:5]

lab10 = pd.read_csv('Measurements_Upload/Lab139_7.1/Loc_0612/Lab_139_10Ch1.csv', skiprows=2, delimiter = ';')
lab10 = lab10.iloc[:, 0:5]
f.write('\n\t\t\t\t...DONE!\n')

# Concatenating all lab dataframes
f.write('\n> Concatenating all the lab data files...')
lab_master = pd.concat([lab1, lab2, lab3, lab4, lab5, lab6, lab7, lab8, lab9, lab10], axis=0, ignore_index=True)
f.write('\n\t\t\t\t...DONE!\n')

# Adding encoded loaction to lab_master
f.write('\n> Adding encoded location to the lab master DataFrame...')
l_1 = []
for i in range(6010):
    l_1.append(1)

lab_master['location'] = l_1
f.write('\n\t\t\t\t...DONE!\n')


# Read Corridor_rm155_7.1 csv files
f.write('\n> Read Corridor Data Files...')
crd1 = pd.read_csv('Measurements_Upload/Corridor_rm155_7.1/Loc_0612/Lab_139_1Ch1.csv', skiprows=2, delimiter = ';')
crd1 = crd1.iloc[:, 0:5]

crd2 = pd.read_csv('Measurements_Upload/Corridor_rm155_7.1/Loc_0612/Lab_139_2Ch1.csv', skiprows=2, delimiter = ';')
crd2 = crd2.iloc[:, 0:5]

crd3 = pd.read_csv('Measurements_Upload/Corridor_rm155_7.1/Loc_0612/Lab_139_3Ch1.csv', skiprows=2, delimiter = ';')
crd3 = crd3.iloc[:, 0:5]

crd4 = pd.read_csv('Measurements_Upload/Corridor_rm155_7.1/Loc_0612/Lab_139_4Ch1.csv', skiprows=2, delimiter = ';')
crd4 = crd4.iloc[:, 0:5]

crd5 = pd.read_csv('Measurements_Upload/Corridor_rm155_7.1/Loc_0612/Lab_139_5Ch1.csv', skiprows=2, delimiter = ';')
crd5 = crd5.iloc[:, 0:5]

crd6 = pd.read_csv('Measurements_Upload/Corridor_rm155_7.1/Loc_0612/Lab_139_6Ch1.csv', skiprows=2, delimiter = ';')
crd6 = crd6.iloc[:, 0:5]

crd7 = pd.read_csv('Measurements_Upload/Corridor_rm155_7.1/Loc_0612/Lab_139_7Ch1.csv', skiprows=2, delimiter = ';')
crd7 = crd7.iloc[:, 0:5]

crd8 = pd.read_csv('Measurements_Upload/Corridor_rm155_7.1/Loc_0612/Lab_139_8Ch1.csv', skiprows=2, delimiter = ';')
crd8 = crd8.iloc[:, 0:5]

crd9 = pd.read_csv('Measurements_Upload/Corridor_rm155_7.1/Loc_0612/Lab_139_9Ch1.csv', skiprows=2, delimiter = ';')
crd9 = crd9.iloc[:, 0:5]

crd10 = pd.read_csv('Measurements_Upload/Corridor_rm155_7.1/Loc_0612/Lab_139_10Ch1.csv', skiprows=2, delimiter = ';')
crd10 = crd10.iloc[:, 0:5]
f.write('\n\t\t\t\t...DONE!\n')

# Concatenating all corridor dataframes
f.write('\n> Concatenating all the corridor data files...')
crd_master = pd.concat([crd1, crd2, crd3, crd4, crd5, crd6, crd7, crd8, crd9, crd10], axis=0, ignore_index=True)
f.write('\n\t\t\t\t...DONE!\n')

# Adding encoded loaction to crd_master
f.write('\n> Adding encoded location to the corridor master DataFrame...')
l_2 = []
for i in range(6010):
    l_2.append(2)

crd_master['location'] = l_2
f.write('\n\t\t\t\t...DONE!\n')


# Read Main_Lobby_7.1 csv files
f.write('\n> Read Lobby Data Files...')
lby1 = pd.read_csv('Measurements_Upload/Main_Lobby_7.1/Loc_0612/Lab_139_1Ch1.csv', skiprows=2, delimiter = ';')
lby1 = lby1.iloc[:, 0:5]

lby2 = pd.read_csv('Measurements_Upload/Main_Lobby_7.1/Loc_0612/Lab_139_2Ch1.csv', skiprows=2, delimiter = ';')
lby2 = lby2.iloc[:, 0:5]

lby3 = pd.read_csv('Measurements_Upload/Main_Lobby_7.1/Loc_0612/Lab_139_3Ch1.csv', skiprows=2, delimiter = ';')
lby3 = lby3.iloc[:, 0:5]

lby4 = pd.read_csv('Measurements_Upload/Main_Lobby_7.1/Loc_0612/Lab_139_4Ch1.csv', skiprows=2, delimiter = ';')
lby4 = lby4.iloc[:, 0:5]

lby5 = pd.read_csv('Measurements_Upload/Main_Lobby_7.1/Loc_0612/Lab_139_5Ch1.csv', skiprows=2, delimiter = ';')
lby5 = lby5.iloc[:, 0:5]

lby6 = pd.read_csv('Measurements_Upload/Main_Lobby_7.1/Loc_0612/Lab_139_6Ch1.csv', skiprows=2, delimiter = ';')
lby6 = lby6.iloc[:, 0:5]

lby7 = pd.read_csv('Measurements_Upload/Main_Lobby_7.1/Loc_0612/Lab_139_7Ch1.csv', skiprows=2, delimiter = ';')
lby7 = lby7.iloc[:, 0:5]

lby8 = pd.read_csv('Measurements_Upload/Main_Lobby_7.1/Loc_0612/Lab_139_8Ch1.csv', skiprows=2, delimiter = ';')
lby8 = lby8.iloc[:, 0:5]

lby9 = pd.read_csv('Measurements_Upload/Main_Lobby_7.1/Loc_0612/Lab_139_9Ch1.csv', skiprows=2, delimiter = ';')
lby9 = lby9.iloc[:, 0:5]

lby10 = pd.read_csv('Measurements_Upload/Main_Lobby_7.1/Loc_0612/Lab_139_10Ch1.csv', skiprows=2, delimiter = ';')
lby10 = lby10.iloc[:, 0:5]
f.write('\n\t\t\t\t...DONE!\n')

# Concatenating all lobby dataframes
f.write('\n> Concatenating all the lobby data files...')
lby_master = pd.concat([lby1, lby2, lby3, lby4, lby5, lby6, lby7, lby8, lby9, lby10], axis=0, ignore_index=True)
f.write('\n\t\t\t\t...DONE!\n')

# Adding encoded loaction to lby_master
f.write('\n> Adding encoded location to the lobby master DataFrame...')
l_3 = []
for i in range(6010):
    l_3.append(3)

lby_master['location'] = l_3
f.write('\n\t\t\t\t...DONE!\n')


# Read Sport_Hall_7.1 csv files
f.write('\n> Read Hall Data Files...')
hall1 = pd.read_csv('Measurements_Upload/Sport_Hall_7.1/Loc_0612/Lab_139_1Ch1.csv', skiprows=2, delimiter = ';')
hall1 = hall1.iloc[:, 0:5]

hall2 = pd.read_csv('Measurements_Upload/Sport_Hall_7.1/Loc_0612/Lab_139_2Ch1.csv', skiprows=2, delimiter = ';')
hall2 = hall2.iloc[:, 0:5]

hall3 = pd.read_csv('Measurements_Upload/Sport_Hall_7.1/Loc_0612/Lab_139_3Ch1.csv', skiprows=2, delimiter = ';')
hall3 = hall3.iloc[:, 0:5]

hall4 = pd.read_csv('Measurements_Upload/Sport_Hall_7.1/Loc_0612/Lab_139_4Ch1.csv', skiprows=2, delimiter = ';')
hall4 = hall4.iloc[:, 0:5]

hall5 = pd.read_csv('Measurements_Upload/Sport_Hall_7.1/Loc_0612/Lab_139_5Ch1.csv', skiprows=2, delimiter = ';')
hall5 = hall5.iloc[:, 0:5]

hall6 = pd.read_csv('Measurements_Upload/Sport_Hall_7.1/Loc_0612/Lab_139_6Ch1.csv', skiprows=2, delimiter = ';')
hall6 = hall6.iloc[:, 0:5]

hall7 = pd.read_csv('Measurements_Upload/Sport_Hall_7.1/Loc_0612/Lab_139_7Ch1.csv', skiprows=2, delimiter = ';')
hall7 = hall7.iloc[:, 0:5]

hall8 = pd.read_csv('Measurements_Upload/Sport_Hall_7.1/Loc_0612/Lab_139_8Ch1.csv', skiprows=2, delimiter = ';')
hall8 = hall8.iloc[:, 0:5]

hall9 = pd.read_csv('Measurements_Upload/Sport_Hall_7.1/Loc_0612/Lab_139_9Ch1.csv', skiprows=2, delimiter = ';')
hall9 = hall9.iloc[:, 0:5]

hall10 = pd.read_csv('Measurements_Upload/Sport_Hall_7.1/Loc_0612/Lab_139_10Ch1.csv', skiprows=2, delimiter = ';')
hall10 = hall10.iloc[:, 0:5]
f.write('\n\t\t\t\t...DONE!\n')

# Concatenating all hall dataframes
f.write('\n> Concatenating all the hall data files...')
hall_master = pd.concat([hall1, hall2, hall3, hall4, hall5, hall6, hall7, hall8, hall9, hall10], axis=0, ignore_index=True)
f.write('\n\t\t\t\t...DONE!\n')

# Adding encoded loaction to hall_master
f.write('\n> Adding encoded location to the hall master DataFrame...')
l_4 = []
for i in range(6010):
    l_4.append(4)

hall_master['location'] = l_4
f.write('\n\t\t\t\t...DONE!\n')

# Concatenating lab, corridor, lobby & hall dataframes to create & save master dataframe
f.write('\n> Creating and saving the master DataFrame...')
master = pd.concat([lab_master, crd_master, lby_master, hall_master], axis=0, ignore_index=True)
master.to_csv('master.csv')
f.write('\n\t\t\t\t...DONE!\n')

# Define X and y
f.write('\n> Defining X and y...')
X, y = master.iloc[:, :5], master.iloc[:, 5:]
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
pipe_lr_pca = make_pipeline(StandardScaler(), PCA(n_components=2), LogisticRegression(random_state=42))

# Make pipeline to perform PCA with normalized data and perform Logistic Regression
pipe_lr_pca_norm = make_pipeline(PCA(n_components=2), LogisticRegression(random_state=42))

# Make pipeline to perform LDA with standard scaler and perform Logistic Regression
pipe_lr_lda = make_pipeline(StandardScaler(), LDA(solver='svd', n_components=2), LogisticRegression(random_state=42))

# Make pipeline to perform LDA with normalized data and perform Logistic Regression
pipe_lr_lda_norm = make_pipeline(LDA(solver='svd', n_components=2), LogisticRegression(random_state=42))
f.write('\n\t\t\t\t...DONE!\n')


####################################################
#
# GridSearch & Cross-validation
#
####################################################

f.write('\n>> Setting parameters for grid search...')
# Setting parameters
param_grid = [{'logisticregression__C': [10000, 1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001],
               'logisticregression__solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
              'logisticregression__max_iter': [1, 10, 100, 1000, 10000]}]

f.write('\n\t\t\t\t...DONE!\n')


f.write('\n>> Performing GridSearchCV for grid search for pca (standard scaler)...\n')

# Set up grid search & cross validation for pca
gs_pca = GridSearchCV(estimator=pipe_lr_pca, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=-1)

# Train grid search model
gs_pca = gs_pca.fit(X_train, y_train)

f.write('\n> Best score from GridSearchCV: ' + str(gs_pca.best_score_) + '\n')
f.write('\n> Best parameters from GridSearchCV: ' + str(gs_pca.best_params_) + '\n')
f.write('\n\t\t\t\t...DONE!\n')


f.write('\n>> Performing GridSearchCV for grid search for pca (normalized)...\n')

# Set up grid search & cross validation for pca
gs_pca_norm = GridSearchCV(estimator=pipe_lr_pca_norm, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=-1)

# Train grid search model
gs_pca_norm = gs_pca_norm.fit(X_train_norm, y_train)

f.write('\n> Best score from GridSearchCV: ' + str(gs_pca_norm.best_score_) + '\n')
f.write('\n> Best parameters from GridSearchCV: ' + str(gs_pca_norm.best_params_) + '\n')
f.write('\n\t\t\t\t...DONE!\n')


f.write('\n>> Performing GridSearchCV for grid search for lda (standard scaler)...\n')

# Set up grid search & cross validation for pca
gs_lda = GridSearchCV(estimator=pipe_lr_lda, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=-1)

# Train grid search model
gs_lda = gs_lda.fit(X_train, y_train)

f.write('\n> Best score from GridSearchCV: ' + str(gs_lda.best_score_) + '\n')
f.write('\n> Best parameters from GridSearchCV: ' + str(gs_lda.best_params_) + '\n')
f.write('\n\t\t\t\t...DONE!\n')


f.write('\n>> Performing GridSearchCV for grid search for lda (normalized)...\n')

# Set up grid search & cross validation for pca
gs_lda_norm = GridSearchCV(estimator=pipe_lr_lda_norm, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=-1)

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
f.write('\n>The best hyperparameters found to be "C = 0.01", "max_iter = 10" and "solver = saga" for lda (normalized)\n')


# Printing accuracy with best hyperparameters found
f.write('\n>> Printing results with best hyperparameters found...\n')

# Perform LDA
lda = LDA(solver='svd', n_components=2)
X_train_lda = lda.fit_transform(X_train_norm, y_train)
X_test_lda = lda.transform(X_test_norm)

# Create an instance of Logistic Regression Classifier and fit the data.
lr = LogisticRegression(C=0.01, random_state=42, solver = 'saga', max_iter = 10, n_jobs = -1)
lr_fit = lr.fit(X_train_lda, y_train)

# Testing the model data
y_pred = lr.predict(X_test_lda)
y_pred = y_pred.reshape(4808,1)

# Printing results
f.write('\nMisclassified samples: %d' % (y_test != y_pred).sum() + '\n')
f.write('\nTraining Accuracy: %.4f' % lr.score(X_train_lda, y_train) + '\n')
f.write('\nTest Accuracy: %.4f' % lr.score(X_test_lda, y_test) + '\n')
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
plot_decision_regions(X=X_combined_lda, y=y_combined, classifier=lr, test_idx=range(y_train.size, y_train.size + y_test.size))

# Plot with labels and legend
plt.title('Decision Region using the LDA transformed/projected features')
plt.xlabel('LDA Feature 1 [Normalized]')
plt.ylabel('LDA Feature 2 [Normalized]')
plt.legend(loc='upper left')

# Save plot
plt.savefig('Output_HW_3_Part1_Decision_Region.png')
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
train_sizes, train_scores, validation_scores = learning_curve(estimator = LogisticRegression(C=0.01, random_state=42, solver = 'saga', max_iter = 10, n_jobs = -1), 
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
plt.title('Learning curves for the logistic regression model', fontsize = 14, y = 1.03)
plt.legend()

# Save plot
plt.savefig('Output_HW_3_Part1_Learning_Curve.png')
f.write('\n\t\t\t\t...DONE!\n')
plt.clf()


####################################################
#
# Plot the confusion matrices
#
####################################################

f.write('\n>> Plotting the confusion matrices of the best model... \n')

# Create the confusion matrices
y_pred_train = lr.predict(X_train_lda)
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
plt.savefig('Output_HW_3_Part1_Training_Confusion_Matrix.png')
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
plt.savefig('Output_HW_3_Part1_Test_Confusion_Matrix.png')
plt.clf()
f.write('\n\t\t\t\t...DONE!\n')


####################################################
#
# Plot the ROC AUC graph
#
####################################################

f.write('\n>> Plotting the ROC AUC graph of the best model... \n')

# Calculate fpr, tpr & thresholds
fpr, tpr, thresholds = roc_curve(y_true = y_test, y_score = lr_fit.predict_proba(X_test_lda)[:, 1], pos_label = 4)

# Calculate ROC AUC
roc_auc = auc(fpr, tpr)

# Plot & save ROC AUC
display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
display.plot()
plt.title('ROC AUC graph', fontsize = 14)
plt.savefig('Output_HW_3_Part1_ROC_AUC.png')
plt.clf()
f.write('\n\t\t\t\t...DONE!\n')
f.write('\n******************PROGRAM is DONE *******************')
f.close()
