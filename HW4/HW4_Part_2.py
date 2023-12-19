#************************************************************************************
# Shafiqul Alam Khan
# ML – HW#4 - Part 2
# Filename: HW4_Part_2.py
# Due: Oct. 18, 2023
#
# Objective:
# • Use k-means++ to observe clusters in the data using the LEAP cluster
# • Combine all the mini datasets into a single dataset
# • Determine all the same requirements from Part 1
#*************************************************************************************

s = open("Output_HW4_Part2.txt", "w")
s.write('>> Importing Packages...\n')
# Importing all required libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from yellowbrick.cluster import SilhouetteVisualizer
import matplotlib.pyplot as plt
import warnings
s.write('\n\t\t\t\t....DONE!\n')

####################################################
#
# Data Preprocessing
#
####################################################

# Load datasets
# Read data files for 2011 - 2015
s.write('\n>> Read Data Files...\n')
D_11 = pd.read_csv('gas+turbine+co+and+nox+emission+data+set/gt_2011.csv')
D_12 = pd.read_csv('gas+turbine+co+and+nox+emission+data+set/gt_2012.csv')
D_13 = pd.read_csv('gas+turbine+co+and+nox+emission+data+set/gt_2013.csv')
D_14 = pd.read_csv('gas+turbine+co+and+nox+emission+data+set/gt_2014.csv')
D_15 = pd.read_csv('gas+turbine+co+and+nox+emission+data+set/gt_2015.csv')
s.write('\n\t\t\t\t...DONE!\n')

# Concatenating all data files
s.write('\n>> Concatenating all Data files...\n')
master = pd.concat([D_11, D_12, D_13, D_14, D_15], axis=0, ignore_index=True)
s.write('\n\t\t\t\t...DONE!\n')

# Define X and y
s.write('\n>> Defining X and y...\n')

X = master.iloc[:, :9]
y_co = master.iloc[:, 9:10]
y_nox = master.iloc[:, 10:11]
s.write('\n\t\t\t\t...DONE!\n')

# Scaling the datasets
sc = StandardScaler()
X_std = sc.fit_transform(X)

# Perform PCA with Standard Scaler 
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

# Ignore warning
warnings.filterwarnings('ignore')

####################################################
#
# Elbow Method
#
####################################################

s.write('\n>> Creating Elbow Plot...\n')

# Elbow plot
distortions = []
for i in range(1, 21):
    km = KMeans(n_clusters=i,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=42)
    y_km = km.fit_predict(X_pca)
    distortions.append(km.inertia_)
    
plt.plot(range(1,21), distortions, marker='o')
plt.title('Elbow plot')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.savefig('Output_HW_4_Part2_Elbow_Plot.png')
plt.clf()

s.write('\n\t\t\t\t...DONE!\n')

####################################################
#
# Plot the clusters with its centers
#
####################################################

s.write('\n>> Creating Cluster Plots...\n')

# Plot clusters for 2011
km = KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300, random_state=42)
label = km.fit_predict(X_pca)
u_labels = np.unique(label)
centroids = km.cluster_centers_
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]
colors = ['green', 'orange', 'blue', 'black']
plt.scatter(centroids_x, centroids_y, marker = '*', s=150, linewidths = 5, zorder = 10, c=colors)
for i in u_labels:
    plt.scatter(X_pca[label == i , 0] , X_pca[label == i , 1] , label = i)
plt.legend()
plt.title('Cluster plot for 2011 data')
plt.savefig('Output_HW_4_Part2_Cluster_Plot.png')
plt.clf()

s.write('\n\t\t\t\t...DONE!\n')

####################################################
#
# Silhouettes
#
####################################################

s.write('\n>> Creating Silhouette Plots...\n')

# Silhouette plot
model = SilhouetteVisualizer(KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300, random_state=42))
model.fit(X_pca)
plt.title('Silhouette plot')
plt.savefig('Output_HW_4_Part2_Silhouette_Plot.png')
plt.clf()

s.write('\n\t\t\t\t...DONE!\n')

####################################################
#
# Distortion score
#
####################################################

s.write('\n>> Printing distortion score...\n')

s.write('\n\t> Distortion: %.2f' % km.inertia_ + '\n')

s.write('\n\t\t\t\t...DONE!\n')
         
s.write('\n******************PROGRAM is DONE *******************')
s.close()
