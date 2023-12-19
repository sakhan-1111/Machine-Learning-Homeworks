#************************************************************************************
# Shafiqul Alam Khan
# ML – HW#1 - Part 2
# Filename: HW1_Part_2.py
# Due: Sept. 6, 2023
#
# Objective:
# • Use a scikit-learn logistic regression model to classify all labels.
# • Use the “SCG” and “STR” features to train, predict, and plot the classification.
# • In your program, print out the training and test accuracy values to a text file.
# • Plot the classification image, save it as an image for submission.
# • Generate and log your excel sheet to record the parameter changes vs. test accuracy (min 10 test performed to find best accuracy)
#*************************************************************************************

# Importing all required libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings

# Load datasets
train = pd.read_excel(r'Data_User_Modeling_Dataset.xls', sheet_name='Training_Data')
test = pd.read_excel(r'Data_User_Modeling_Dataset.xls', sheet_name='Test_Data')

# Joining dataframes to get better ratio of test train split & test more ratios
data = pd.concat([train, test], ignore_index=True)

# Drop unnecessary samples
data = data[['SCG','STR', ' UNS']]

# Check for NaN / Null values
print(data.isna().sum())
print ('\n')

# Encodeing ' UNS' Column
b = []
encoder = preprocessing.LabelEncoder()
encoded = encoder.fit_transform(data[' UNS'])
b.append(encoded)
b_data = pd.DataFrame(b)
encoded_data = b_data.transpose()
encoded_data.columns = ['encoded_data']
data = data.join(encoded_data['encoded_data'])


# Performing train test split
X = data.drop([' UNS','encoded_data'], axis=1)
y = data['encoded_data']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=66, stratify=y)


# Scaling training and test data
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)

# Scaling test data
sc.fit(X_test)
X_test_std = sc.transform(X_test)

# Ingnore warnings
warnings.filterwarnings("ignore")

# Creating text file
f = open("Output_HW1_Part2.txt", "w")

# Looping through hyperparameters & printing outputs
S = ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
C = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001]

for s in S:
    # Create an instance of Logistic Regression Classifier and fit the data.
    lr = LogisticRegression(C=100, random_state=66, solver = s, max_iter = 100, n_jobs = -1)
    lr.fit(X_train_std, y_train)

    # Testing the model data
    y_pred = lr.predict(X_test_std)

    # Printing the results
    f.write('For C = 100, max_iter = 100 & Solver = ' + str(s))
    f.write('\n')
    f.write('Training Accuracy: %.2f' % lr.score(X_train_std, y_train))
    f.write('\n')
    f.write('Test Accuracy: %.2f' % lr.score(X_test_std, y_test))
    f.write('\n')
    f.write('\n')

for c in C:
    # Create an instance of Logistic Regression Classifier and fit the data.
    lr = LogisticRegression(C=c, random_state=66, solver = 'liblinear', max_iter = 100, n_jobs = -1)
    lr.fit(X_train_std, y_train)

    # Testing the model data
    y_pred = lr.predict(X_test_std)

    # Printing the results
    f.write('For C = ' + str(c) +', max_iter = 100 & Solver = liblinear')
    f.write('\n')
    f.write('Training Accuracy: %.2f' % lr.score(X_train_std, y_train))
    f.write('\n')
    f.write('Test Accuracy: %.2f' % lr.score(X_test_std, y_test))
    f.write('\n')
    f.write('\n')

f.write('Observing the outputs. The best hyperparameters found to be "C = 0.1" and "solver = liblinear" & max_iter = 100')
f.write('\n')
f.write('\n')

# Create an instance of Logistic Regression Classifier and fit the data.
lr = LogisticRegression(C=0.1, random_state=66, solver = 'liblinear', max_iter = 100, n_jobs = -1)
lr.fit(X_train_std, y_train)

# Testing the model data
y_pred = lr.predict(X_test_std)

# Print out the training and test accuracy values
f.write('Misclassified samples: %d' % (y_test != y_pred).sum())
f.write('\n')
f.write('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
f.write('\n')
f.write('Training Accuracy: %.2f' % lr.score(X_train_std, y_train))
f.write('\n')
f.write('Test Accuracy: %.2f' % lr.score(X_test_std, y_test))
f.write('\n')
f.write('\n')
f.close()

#Plot the classification outcome using this Method
def plot_decision_regions(X,y,classifier,test_idx=None,resolution=0.02):
    markers = ('s','x','o','v')
    colors = ('red', 'blue', 'lightgreen', 'darkorange')
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
    
    #Plot all the samples
    X_test,y_test=X[test_idx,:],y[test_idx]
    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0],y=X[y==cl,1],alpha=0.8,c=cmap(idx),marker=markers[idx],label=cl)
        plt.savefig('Output_HW1_Part2.png')
    #Highlight test samples
    if test_idx:
        X_test,y_test =X[test_idx,:],y[test_idx]
        plt.scatter(X_test[:,0],X_test[:,1],c='',alpha=1.0,linewidths=1,marker='o',s=55,label='test set')
        
# Combine all training and test data to single object variables
X_combined_std=np.vstack((X_train_std,X_test_std))
y_combined=np.hstack((y_train,y_test))
plot_decision_regions(X=X_combined_std, y=y_combined, classifier=lr)

#plot with labels and legend
plt.xlabel('Feature 2 [Standardized]')
plt.ylabel('Feature 1 [standardized]')
plt.savefig('Output_HW1_Part2.png')
plt.legend()
plt.show()
