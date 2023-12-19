#************************************************************************************
# Shafiqul Alam Khan
# ML – HW#2 - Part 1
# Filename: HW2_Part_1.py
# Due: Sept. 20, 2023
#
# Objective:
# • Use a scikit-learn SVM model to classify all targets.
# • Determine which features and parameters will provide the best outcomes using PCA or LDA
# • From the dataset:
#         -For each location, choose a single loc_number folder (use this one for all models).
#         -Within the folder, combine all the CSV files into a single file for that location.
#             -Label the data accordingly to the dataset information.
#         -You can either keep it separate or join all samples to a large master CSV file.
# • In your program, print out the training and test accuracy values and the best features values via the PCA or LDA to a text file from the best model experiment.
# • Generate the t-SNE and UMAP images.
# • Generate the training and testing Plot Decision image.
#*************************************************************************************

# Importing all required libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.manifold import TSNE
import umap
import umap.plot
import warnings

# Load datasets
# Read Lab139_7.1 csv files
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

# Concatenating all lab dataframes
lab_master = pd.concat([lab1, lab2, lab3, lab4, lab5, lab6, lab7, lab8, lab9, lab10], axis=0, ignore_index=True)

# Adding encoded loaction to lab_master
l_1 = []
for i in range(6010):
    l_1.append(1)

lab_master['location'] = l_1


# Read Corridor_rm155_7.1 csv files
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

# Concatenating all corridor dataframes
crd_master = pd.concat([crd1, crd2, crd3, crd4, crd5, crd6, crd7, crd8, crd9, crd10], axis=0, ignore_index=True)

# Adding encoded loaction to crd_master
l_2 = []
for i in range(6010):
    l_2.append(2)

crd_master['location'] = l_2


# Read Main_Lobby_7.1 csv files
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

# Concatenating all lobby dataframes
lby_master = pd.concat([lby1, lby2, lby3, lby4, lby5, lby6, lby7, lby8, lby9, lby10], axis=0, ignore_index=True)

# Adding encoded loaction to lby_master
l_3 = []
for i in range(6010):
    l_3.append(3)

lby_master['location'] = l_3


# Read Sport_Hall_7.1 csv files
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

# Concatenating all hall dataframes
hall_master = pd.concat([hall1, hall2, hall3, hall4, hall5, hall6, hall7, hall8, hall9, hall10], axis=0, ignore_index=True)

# Adding encoded loaction to hall_master
l_4 = []
for i in range(6010):
    l_4.append(4)

hall_master['location'] = l_4

# Concatenating lab, corridor, lobby & hall dataframes to create & save master dataframe
master = pd.concat([lab_master, crd_master, lby_master, hall_master], axis=0, ignore_index=True)
master.to_csv('master.csv')


# Define X and y
X, y = master.iloc[:, :6], master.iloc[:, 5:]

# Perform split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Scaling the datasets
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

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

# Perform PCA with Standard Scaler 
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

# Percentage of variance explained by each of the selected components.
explained_variance_pca = pca.explained_variance_ratio_

# Print PCA results
f = open("Output_HW2_Part1.txt", "w")
f.write('PCA with standard scaler > Percentage of variance explained by each of the selected components')
f.write('\n')
f.write(str(explained_variance_pca))
f.write('\n')
f.write('\n')


# Perform PCA with normalized data
pca = PCA(n_components=2)
X_train_pca_2 = pca.fit_transform(X_train_norm)
X_test_pca_2 = pca.transform(X_test_norm)

# Percentage of variance explained by each of the selected components.
explained_variance_pca = pca.explained_variance_ratio_

# Print PCA 2 results
f.write('PCA with normalized data > Percentage of variance explained by each of the selected components')
f.write('\n')
f.write(str(explained_variance_pca))
f.write('\n')
f.write('\n')


# Perform LDA
lda = LDA(solver='svd', n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)

# Percentage of variance explained by each of the selected components.
explained_variance_lda = lda.explained_variance_ratio_

# Print LDA results
f.write('LDA with standard scaler > Percentage of variance explained by each of the selected components')
f.write('\n')
f.write(str(explained_variance_lda))
f.write('\n')
f.write('\n')


# Perform SVM with PCA Data(standard scaler)
f.write('>> Perform SVM to test PCA or LDA which works better')
f.write('\n')
f.write('> Perform SVM with PCA Data (standard scaler):')
f.write('\n')

svm = SVC(kernel='rbf', random_state=42, gamma=0.1, C=10.0)
svm_fit = svm.fit(X_train_pca, y_train)
y_pred = svm.predict(X_test_pca)

f.write('Train Accuracy: %.2f' % svm.score(X_train_pca, y_train))
f.write('\n')
f.write('Test Accuracy: %.2f' % svm.score(X_test_pca, y_test))
f.write('\n')
f.write('\n')

# Perform SVM with PCA Data (Normalized)
f.write('> Perform SVM with PCA Data (normalized):')
f.write('\n')

svm = SVC(kernel='rbf', random_state=42, gamma=0.1, C=10.0)
svm_fit = svm.fit(X_train_pca_2, y_train)
y_pred = svm.predict(X_test_pca_2)

f.write('Train Accuracy: %.2f' % svm.score(X_train_pca, y_train))
f.write('\n')
f.write('Test Accuracy: %.2f' % svm.score(X_test_pca, y_test))
f.write('\n')
f.write('\n')


# Perform SVM with LDA Data
f.write('> Perform SVM with LDA Data:')
f.write('\n')

svm = SVC(kernel='rbf', random_state=42, gamma=0.1, C=10.0)
svm_fit = svm.fit(X_train_lda, y_train)
y_pred = svm.predict(X_test_lda)

f.write('Train Accuracy: %.2f' % svm.score(X_train_lda, y_train))
f.write('\n')
f.write('Test Accuracy: %.2f' % svm.score(X_test_lda, y_test))
f.write('\n')
f.write('\n')

# Perform SVM with LDA Data 2
f.write('> Perform SVM with LDA Data 2:')
f.write('\n')

svm = SVC(kernel='rbf', random_state=42, gamma=10, C=10.0)
svm_fit = svm.fit(X_train_lda, y_train)
y_pred = svm.predict(X_test_lda)

f.write('Train Accuracy: %.2f' % svm.score(X_train_lda, y_train))
f.write('\n')
f.write('Test Accuracy: %.2f' % svm.score(X_test_lda, y_test))
f.write('\n')
f.write('\n')

# Determine which features and parameters will provide the best outcomes using PCA orLDA
f.write('>> SVM with LDA provides both train accuracy & test accuray as 100%.')
f.write('\n')
f.write('Even with hyperparameter tweaking for accuracy for both remain 100%')
f.write('\n')
f.write('Which is too good to be true. Selecting PCA with standard scaler for analysis.')
f.write('\n')
f.write('\n')


# Looping through hyperparameters & printing outputs
k = ['linear', 'poly', 'rbf', 'sigmoid']
gm = [1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001]
c_val = [0.01, 0.1, 1, 10, 100, 1000, 10000]

for k_s in k:
    svm = SVC(kernel=k_s, random_state=42, gamma=0.1, C=10.0)
    svm_fit = svm.fit(X_train_pca, y_train)
    y_pred = svm.predict(X_test_pca)
    
    # Print results
    f.write('For kernel = ' + str(k_s) + ', gamma = 0.1 & C = 10')
    f.write('\n')
    f.write('Train Accuracy: %.2f' % svm.score(X_train_pca, y_train))
    f.write('\n')
    f.write('Test Accuracy: %.2f' % svm.score(X_test_pca, y_test))
    f.write('\n')
    f.write('\n')
    
for g in gm:
    svm = SVC(kernel='rbf', random_state=42, gamma=g, C=10.0)
    svm_fit = svm.fit(X_train_pca, y_train)
    y_pred = svm.predict(X_test_pca)
    
    # Print results
    f.write('For kernel = rbf, gamma = ' + str(g) + ' & C = 10')
    f.write('\n')
    f.write('Train Accuracy: %.2f' % svm.score(X_train_pca, y_train))
    f.write('\n')
    f.write('Test Accuracy: %.2f' % svm.score(X_test_pca, y_test))
    f.write('\n')
    f.write('\n')

for c_in in c_val:
    svm = SVC(kernel='rbf', random_state=42, gamma=1000, C=c_in)
    svm_fit = svm.fit(X_train_pca, y_train)
    y_pred = svm.predict(X_test_pca)
    
    # Print results
    f.write('For kernel = rbf, gamma = 1000 & C = ' + str(c_in))
    f.write('\n')
    f.write('Train Accuracy: %.2f' % svm.score(X_train_pca, y_train))
    f.write('\n')
    f.write('Test Accuracy: %.2f' % svm.score(X_test_pca, y_test))
    f.write('\n')
    f.write('\n')


# Selecting the best hyperparameters
f.write('>> Observing the outputs: ')
f.write('\n')
f.write('The best hyperparameters found to be "kernel = rbf", "gamma = 1000" and "C = 10".')
f.write('\n')
f.write('\n')


# Printing accuracy with best hyperparameters found
f.write('>> Printing accuracy with best hyperparameters found.')
f.write('\n')

svm = SVC(kernel='rbf', random_state=42, gamma=1000, C=10)
svm_fit = svm.fit(X_train_pca, y_train)
y_pred = svm.predict(X_test_pca)

f.write('Train Accuracy: %.2f' % svm.score(X_train_pca, y_train))
f.write('\n')
f.write('Test Accuracy: %.2f' % svm.score(X_test_pca, y_test))
f.write('\n')
f.close()


# Plot the best result
def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

fig, ax = plt.subplots()
# title for the plots
title = ('Decision surface of linear SVC')
# Set-up grid for plotting.
X_combined_std=np.vstack((X_train_pca, X_test_pca))
X0, X1 = X_combined_std[:, 0], X_combined_std[:, 1]
xx, yy = make_meshgrid(X0, X1)

plot_contours(ax, svm_fit, xx, yy, cmap='RdYlGn', alpha=0.8)
ax.scatter(X0, X1, c=y.values, cmap='RdYlGn', s=20, edgecolors='k')
ax.set_ylabel('PC2')
ax.set_xlabel('PC1')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title('Decision Region using the PCA transformed/projected features')
plt.savefig('Output_HW_2_Part1_Decision_Region.png')


# Generate & save t-SNE image
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
z = tsne.fit_transform(X)
sn = pd.DataFrame()
sn['y'] = y
sn['feature_1'] = z[:,0]
sn['feature_2'] = z[:,1]
sns.scatterplot(x='feature_1', y='feature_2', hue=sn.y.tolist(),
                palette=sns.color_palette('hls', 4),
                data=sn).set(title='2D t-SNE on Master Dataset')
plt.savefig('Output_HW_2_Part1_t-SNE.png')


# Generate & save UMAP image
mapper = umap.UMAP().fit(X)
umap.plot.points(mapper, labels=y.values.flatten())
plt.savefig('Output_HW_2_Part1_UMAP.png')
