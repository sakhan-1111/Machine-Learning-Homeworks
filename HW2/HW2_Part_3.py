#************************************************************************************
# Shafiqul Alam Khan
# ML – HW#2 - Part 3
# Filename: HW2_Part_3.py
# Due: Sept. 20, 2023
#
# Objective:
# • Use a scikit-learn decision tree model to classify all targets.
# • Same requirements from Part 1 – you should use the same generated dataset.
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.manifold import TSNE
import umap
import umap.plot
import warnings

# Load master dataset
master = pd.read_csv('master.csv')

# Define X and y
X, y = master.iloc[:, 1:6], master.iloc[:, 6:]

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
f = open("Output_HW2_Part3.txt", "w")
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


# Perform LDA with Standard Scaler
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

# Perform LDA with Normalized Data
lda = LDA(solver='svd', n_components=2)
X_train_lda_2 = lda.fit_transform(X_train_norm, y_train)
X_test_lda_2 = lda.transform(X_test_norm)

# Percentage of variance explained by each of the selected components.
explained_variance_lda = lda.explained_variance_ratio_

# Print LDA results
f.write('LDA with normalized data > Percentage of variance explained by each of the selected components')
f.write('\n')
f.write(str(explained_variance_lda))
f.write('\n')
f.write('\n')


# Perform Decision Tree with PCA Data(standard scaler)
f.write('>> Perform Decision Tree to test PCA or LDA which works better')
f.write('\n')
f.write('> Perform Decision Tree with PCA Data (standard scaler):')
f.write('\n')

tree = DecisionTreeClassifier(criterion='gini', splitter = 'best', max_depth=4, random_state=42)
tree.fit(X_train_pca, y_train)
y_pred = tree.predict(X_test_pca)

f.write('Train Accuracy: %.4f' % tree.score(X_train_pca, y_train))
f.write('\n')
f.write('Test Accuracy: %.4f' % tree.score(X_test_pca, y_test))
f.write('\n')
f.write('\n')

# Perform Decision Tree with PCA Data (Normalized)
f.write('> Perform Decision Tree with PCA Data (normalized):')
f.write('\n')

tree = DecisionTreeClassifier(criterion='gini', splitter = 'best', max_depth=4, random_state=42)
tree.fit(X_train_pca_2, y_train)
y_pred = tree.predict(X_test_pca_2)

f.write('Train Accuracy: %.4f' % tree.score(X_train_pca_2, y_train))
f.write('\n')
f.write('Test Accuracy: %.4f' % tree.score(X_test_pca_2, y_test))
f.write('\n')
f.write('\n')


# Perform Decision Tree with LDA Data (standard scaler)
f.write('> Perform Decision Tree with LDA Data (standard scaler):')
f.write('\n')

tree = DecisionTreeClassifier(criterion='gini', splitter = 'best', max_depth=4, random_state=42)
tree.fit(X_train_lda, y_train)
y_pred = tree.predict(X_test_lda)

f.write('Train Accuracy: %.4f' % tree.score(X_train_lda, y_train))
f.write('\n')
f.write('Test Accuracy: %.4f' % tree.score(X_test_lda, y_test))
f.write('\n')
f.write('\n')


# Perform Decision Tree with LDA Data (Normalized)
f.write('> Perform Decision Tree with LDA Data (Normalized):')
f.write('\n')

tree = DecisionTreeClassifier(criterion='gini', splitter = 'best', max_depth=4, random_state=42)
tree.fit(X_train_lda_2, y_train)
y_pred = tree.predict(X_test_lda_2)

f.write('Train Accuracy: %.4f' % tree.score(X_train_lda_2, y_train))
f.write('\n')
f.write('Test Accuracy: %.4f' % tree.score(X_test_lda_2, y_test))
f.write('\n')
f.write('\n')

# Determine which features and parameters will provide the best outcomes using PCA or LDA
f.write('>> Decision Tree with PCA provides lower train & test score than Decision Tree with LDA (standard scaler).')
f.write('\n')
f.write('Selecting LDA with standard scaler for analysis.')
f.write('\n')
f.write('\n')


# # Looping through hyperparameters & printing outputs
cri = ['gini', 'entropy', 'log_loss']
sp = ['best', 'random']
md = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]


for c in cri:
    tree = DecisionTreeClassifier(criterion=c, splitter = 'best', max_depth=4, random_state=42)
    tree.fit(X_train_lda, y_train)
    y_pred = tree.predict(X_test_lda)
    
    # Print results
    f.write('For criterion = "' + str(c) + '", splitter = "best" & max_depth = 4')
    f.write('\n')
    f.write('Train Accuracy: %.4f' % tree.score(X_train_lda, y_train))
    f.write('\n')
    f.write('Test Accuracy: %.4f' % tree.score(X_test_lda, y_test))
    f.write('\n')
    f.write('\n')
    
    
for s in sp:
    tree = DecisionTreeClassifier(criterion='entropy', splitter = s, max_depth=4, random_state=42)
    tree.fit(X_train_lda, y_train)
    y_pred = tree.predict(X_test_lda)
    
    # Print results
    f.write('For criterion = "entropy", splitter = "' + str(s) +  '" & max_depth = 4')
    f.write('\n')
    f.write('Train Accuracy: %.4f' % tree.score(X_train_lda, y_train))
    f.write('\n')
    f.write('Test Accuracy: %.4f' % tree.score(X_test_lda, y_test))
    f.write('\n')
    f.write('\n')
    
    
for m in md:
    tree = DecisionTreeClassifier(criterion='entropy', splitter = 'best', max_depth=m, random_state=42)
    tree.fit(X_train_lda, y_train)
    y_pred = tree.predict(X_test_lda)
    
    # Print results
    f.write('For criterion = "entropy", splitter = "best" & max_depth = ' + str(m) + '.')
    f.write('\n')
    f.write('Train Accuracy: %.4f' % tree.score(X_train_lda, y_train))
    f.write('\n')
    f.write('Test Accuracy: %.4f' % tree.score(X_test_lda, y_test))
    f.write('\n')
    f.write('\n')
    

# Selecting the best hyperparameters
f.write('>> Observing the outputs: ')
f.write('\n')
f.write('The best hyperparameters found to be criterion = "entropy", splitter = "best" & max_depth = 21.')
f.write('\n')
f.write('\n')


# Printing accuracy with best hyperparameters found
f.write('>> Printing accuracy with best hyperparameters found.')
f.write('\n')

tree = DecisionTreeClassifier(criterion='entropy', splitter = 'best', max_depth=21, random_state=42)
tree_fit = tree.fit(X_train_lda, y_train)
y_pred = tree.predict(X_test_lda)
    
# Print results
f.write('Train Accuracy: %.4f' % tree.score(X_train_lda, y_train))
f.write('\n')
f.write('Test Accuracy: %.4f' % tree.score(X_test_lda, y_test))
f.write('\n')
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
title = ('Decision surface of Decision Tree')

# Set-up grid for plotting.
X_combined_std=np.vstack((X_train_lda, X_test_lda))
X0, X1 = X_combined_std[:, 0], X_combined_std[:, 1]
xx, yy = make_meshgrid(X0, X1)

plot_contours(ax, tree_fit, xx, yy, cmap='RdYlGn', alpha=0.8)
ax.scatter(X0, X1, c=y.values, cmap='RdYlGn', s=20, edgecolors='k')
ax.set_ylabel('PC2')
ax.set_xlabel('PC1')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title('Decision Region of Decision Tree using the LDA transformed/projected features')
plt.savefig('Output_HW_2_Part3_Decision_Region.png')


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
plt.savefig('Output_HW_2_Part3_t-SNE.png')


# Generate & save UMAP image
mapper = umap.UMAP().fit(X)
umap.plot.points(mapper, labels=y.values.flatten())
plt.savefig('Output_HW_2_Part3_UMAP.png')
