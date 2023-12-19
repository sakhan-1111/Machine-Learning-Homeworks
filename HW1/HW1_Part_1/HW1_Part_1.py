#************************************************************************************
# Shafiqul Alam Khan
# ML – HW#1 - Part 1
# Filename: HW1_Part_1.py
# Due: Sept. 6, 2023
#
# Objective:
# • Use a scikit-learn perceptron model to classify all labels
# • Use the “STG” and “PEG” features to train, predict, and plot the classification.
# • In your program, print out the training and test accuracy values to a text file
# • Plot the classification outcome, save it as an image for submission.
# • Generate and log your excel sheet to record the parameter changes vs. test accuracy (min 10 test performed to find best accuracy)o Find the highest test accuracy by tuning the model parameters.
#*************************************************************************************


# Importing all required libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import warnings

# Load datasets
train = pd.read_excel(r'Data_User_Modeling_Dataset.xls', sheet_name='Training_Data')
test = pd.read_excel(r'Data_User_Modeling_Dataset.xls', sheet_name='Test_Data')

# Joining dataframes to get better ratio of test train split & test more ratios
data = pd.concat([train, test], ignore_index=True)

# Drop unnecessary samples
data = data[['STG','PEG', ' UNS']]

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=12, stratify=y)


# Scaling training and test data
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)

# Scaling test data
sc.fit(X_test)
X_test_std = sc.transform(X_test)

# Creating text file
f = open("Output_HW1_Part1.txt", "w")

# Looping through hyperparameters & printing outputs
it = [1000000, 100000, 10000, 1000, 100, 10]
et = [10, 1, 0.1, 0.01, 0.001, 0.0001]

for i in it:
    # Creating perceptron with hyperparameters
    ppn = Perceptron(max_iter=i, eta0=1, shuffle=True)
    
    # Training the model
    ppn.fit(X_train_std, y_train)
    
    # Testing the model data
    y_pred = ppn.predict(X_test_std)
    
    # Print results
    f.write('For max_iter = ' + str(i) + ' & eta0 = 1')
    f.write('\n')
    f.write('Train Accuracy: %.2f' % ppn.score(X_train_std, y_train))
    f.write('\n')
    f.write('Test Accuracy: %.2f' % ppn.score(X_test_std, y_test))
    f.write('\n')
    f.write('\n')

for e in et:
    # Creating perceptron with hyperparameters
    ppn = Perceptron(max_iter=10, eta0=e, shuffle=True)
    
    # Training the model
    ppn.fit(X_train_std, y_train)
    
    # Testing the model data
    y_pred = ppn.predict(X_test_std)
    
    # Print results
    f.write('For max_iter = 10 & eta0 = ' + str(e))
    f.write('\n')
    f.write('Train Accuracy: %.2f' % ppn.score(X_train_std, y_train))
    f.write('\n')
    f.write('Test Accuracy: %.2f' % ppn.score(X_test_std, y_test))
    f.write('\n')
    f.write('\n')

f.write('\n')
f.write('Observing the outputs. The best hyperparameters found to be "max_iter = 10" and "eta0 = 1"')
f.write('\n')
f.write('\n')

# Ingnore warnings
warnings.filterwarnings("ignore")

# Printing accuracy with best hyperparameters found
ppn = Perceptron(max_iter=10, eta0=1, shuffle=True)
ppn.fit(X_train_std, y_train)
y_pred = ppn.predict(X_test_std)
f.write('Train Accuracy: %.2f' % ppn.score(X_train_std, y_train))
f.write('\n')
f.write('Test Accuracy: %.2f' % ppn.score(X_test_std, y_test))
f.write('\n')
f.write('Best Misclassified samples: %d' % (y_test != y_pred).sum())
f.write('\n')
f.close()

# Plotting the result
mapping = {'High':0,'Low':1,'Middle':2,'VeryLow':3}
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=mapping, yticklabels=mapping, cmap = 'gnuplot')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('Output_Confusion_Matrix.png')
plt.show()
