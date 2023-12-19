#************************************************************************************
# Shafiqul Alam Khan
# ML – HW#3 - Part 4
# Filename: HW3_Part_4.py
# Due: Oct. 6, 2023
#
# Objective:
# • Use Ensemble Learning using the models from Parts 1, 2, and 3.
# • Implement the voting technique for classification
# • Print the accuracy, precision, recall and F1-score for the ensembled model
# • Plot the classification outcome and save it as an image for submission.
#*************************************************************************************

f = open("Output_HW3_Part4.txt", "w")
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
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
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
# Preparing the models
#
####################################################

f.write('\n>> Preparing the models...')

clf1 = LogisticRegression(C=0.01, random_state=42, solver = 'saga', max_iter = 10, n_jobs = -1)
clf2 = Perceptron(eta0 = 0.1, max_iter = 1000000, validation_fraction = 0.1, random_state=42, n_jobs = -1)
clf3 = DecisionTreeClassifier(criterion='gini', splitter = 'random', max_depth=25, random_state=42)

f.write('\n\t\t\t\t...DONE!\n')

####################################################
#
# Pipeline
#
####################################################

f.write('\n>> Making Piplines...')

pipe1 = Pipeline([['ld', LDA(solver='svd', n_components=2)], ['clf', clf1]]) # Feed normalized data
pipe2 = Pipeline([['ld', LDA(solver='svd', n_components=2)], ['clf', clf2]]) # ,, 
pipe3 = Pipeline([['pc', PCA(n_components=2)], ['clf', clf3]]) # ,,

clf_labels = ['Logistic regression', 'Perceptron', 'Decision Tree', 'Majority voting - ArgMax']

f.write('\n\t\t\t\t...DONE!\n')


####################################################
#
# Ensemble Learning implementing the voting technique 
#
####################################################

f.write('\n>> Performing the voting technique...\n')

# Voting technique
mv_clf = VotingClassifier(estimators=[('lr', pipe1), ('ppn', pipe2), ('dt', pipe3)], voting='hard')
all_clf = [pipe1, pipe2, pipe3, mv_clf]

# Ensemble Learning (voting technique) ---> Print accuracy 
for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf,
                             X=X_train_norm,
                             y=y_train,
                             cv=10,
                             scoring='accuracy')
    f.write("\n> Accuracy: %0.6f (+/- %0.2f) [%s]" %(scores.mean(), scores.std(), label))
f.write('\n')

# Ensemble Learning (voting technique) ---> Print precision 
for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf,
                             X=X_train_norm,
                             y=y_train,
                             cv=10,
                             scoring='precision_macro')
    f.write("\n> Precision: %0.6f (+/- %0.2f) [%s]" %(scores.mean(), scores.std(), label))
f.write('\n')

# Ensemble Learning (voting technique) ---> Print recall 
for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf,
                             X=X_train_norm,
                             y=y_train,
                             cv=10,
                             scoring='recall_macro')
    f.write("\n> Recall: %0.6f (+/- %0.2f) [%s]" %(scores.mean(), scores.std(), label))
f.write('\n')

# Ensemble Learning (voting technique) ---> Print f1-score
for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf,
                             X=X_train_norm,
                             y=y_train,
                             cv=10,
                             scoring='f1_macro')
    f.write("\n> F1-score: %0.6f (+/- %0.2f) [%s]" %(scores.mean(), scores.std(), label))
f.write('\n')

f.write('\n\t\t\t\t...DONE!\n')

####################################################
#
# Plot the classification outcome
#
####################################################

f.write('\n>> Plotting the confusion matrix... \n')

mv_clf.fit(X_train_norm, y_train)
y_pred = mv_clf.predict(X_test_norm)
y_pred_train = mv_clf.predict(X_train_norm)
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
plt.savefig('Output_HW_3_Part4_Training_Confusion_Matrix.png')
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
plt.savefig('Output_HW_3_Part4_Test_Confusion_Matrix.png')
plt.clf()

f.write('\n\t\t\t\t...DONE!\n')

f.write('\n******************PROGRAM is DONE *******************')
f.close()
