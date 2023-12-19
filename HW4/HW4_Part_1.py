#************************************************************************************
# Shafiqul Alam Khan
# ML – HW#4 - Part 1
# Filename: HW4_Part_1.py
# Due: Oct. 18, 2023
#
# Objective:
# • Use k-means++ to observe clusters in the data using the LEAP cluster
# • Determine the number of centroids by using the Elbow Method (provide the plot) for the 2011 dataset
# • Use the correct number of centroids and plot the clusters with its centers and silhouettes for each individual year
# • Determine the distortion score and save it to a text file for each individual year
#*************************************************************************************

s = open("Output_HW4_Part1.txt", "w")
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

# Define X and y
s.write('\n>> Defining X and y...\n')

# Define X & y for 2011 data
X_11 = D_11.iloc[:, :9]
y_co_11 = D_11.iloc[:, 9:10]
y_nox_11 = D_11.iloc[:, 10:11]

# Define X & y for 2012 data
X_12 = D_12.iloc[:, :9]
y_co_12 = D_12.iloc[:, 9:10]
y_nox_12 = D_12.iloc[:, 10:11]

# Define X & y for 2013 data
X_13 = D_13.iloc[:, :9]
y_co_13 = D_13.iloc[:, 9:10]
y_nox_13 = D_13.iloc[:, 10:11]

# Define X & y for 2014 data
X_14 = D_14.iloc[:, :9]
y_co_14 = D_14.iloc[:, 9:10]
y_nox_14 = D_14.iloc[:, 10:11]

# Define X & y for 2015 data
X_15 = D_15.iloc[:, :9]
y_co_15 = D_15.iloc[:, 9:10]
y_nox_15 = D_15.iloc[:, 10:11]


# Scaling the datasets
sc = StandardScaler()
X_11_std = sc.fit_transform(X_11)
X_12_std = sc.fit_transform(X_12)
X_13_std = sc.fit_transform(X_13)
X_14_std = sc.fit_transform(X_14)
X_15_std = sc.fit_transform(X_15)

# Perform PCA with Standard Scaler 
pca = PCA(n_components=2)
X_11_pca = pca.fit_transform(X_11_std)
X_12_pca = pca.fit_transform(X_12_std)
X_13_pca = pca.fit_transform(X_13_std)
X_14_pca = pca.fit_transform(X_14_std)
X_15_pca = pca.fit_transform(X_15_std)

# Ignore warning
warnings.filterwarnings('ignore')

s.write('\n\t\t\t\t...DONE!\n')

####################################################
#
# Elbow Method
#
####################################################

s.write('\n>> Creating Elbow Plots...\n')

# Elbow plot for 2011
distortions = []
for i in range(1, 21):
    km = KMeans(n_clusters=i,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=42)
    y_km = km.fit_predict(X_11_pca)
    distortions.append(km.inertia_)
    
plt.plot(range(1,21), distortions, marker='o')
plt.title('Elbow plot for 2011 data')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.savefig('Output_HW_4_Part1_Elbow_Plot_2011_Data.png')
plt.clf()

# Elbow plot for 2012
distortions = []
for i in range(1, 21):
    km = KMeans(n_clusters=i,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=42)
    y_km = km.fit_predict(X_12_pca)
    distortions.append(km.inertia_)
    
plt.plot(range(1,21), distortions, marker='o')
plt.title('Elbow plot for 2012 data')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.savefig('Output_HW_4_Part1_Elbow_Plot_2012_Data.png')
plt.clf()

# Elbow plot for 2013
distortions = []
for i in range(1, 21):
    km = KMeans(n_clusters=i,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=42)
    y_km = km.fit_predict(X_13_pca)
    distortions.append(km.inertia_)
    
plt.plot(range(1,21), distortions, marker='o')
plt.title('Elbow plot for 2013 data')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.savefig('Output_HW_4_Part1_Elbow_Plot_2013_Data.png')
plt.clf()

# Elbow plot for 2014
distortions = []
for i in range(1, 21):
    km = KMeans(n_clusters=i,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=42)
    y_km = km.fit_predict(X_14_pca)
    distortions.append(km.inertia_)
    
plt.plot(range(1,21), distortions, marker='o')
plt.title('Elbow plot for 2014 data')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.savefig('Output_HW_4_Part1_Elbow_Plot_2014_Data.png')
plt.clf()

# Elbow plot for 2015
distortions = []
for i in range(1, 21):
    km = KMeans(n_clusters=i,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=42)
    y_km = km.fit_predict(X_15_pca)
    distortions.append(km.inertia_)
    
plt.plot(range(1,21), distortions, marker='o')
plt.title('Elbow plot for 2015 data')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.savefig('Output_HW_4_Part1_Elbow_Plot_2015_Data.png')
plt.clf()

s.write('\n\t\t\t\t...DONE!\n')

####################################################
#
# Plot the clusters with its centers
#
####################################################

s.write('\n>> Creating Cluster Plots...\n')

# Plot clusters for 2011
km_11 = KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300, random_state=42)
label = km_11.fit_predict(X_11_pca)
u_labels = np.unique(label)
centroids = km_11.cluster_centers_
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]
colors = ['lightgreen', 'orange', 'blue', 'black']
plt.scatter(centroids_x, centroids_y, marker = '*', s=150, linewidths = 5, zorder = 10, c=colors)
for i in u_labels:
    plt.scatter(X_11_pca[label == i , 0] , X_11_pca[label == i , 1] , label = i)
plt.legend()
plt.title('Cluster plot for 2011 data')
plt.savefig('Output_HW_4_Part1_Cluster_Plot_2011_Data.png')
plt.clf()

# Plot clusters for 2012
km_12 = KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300, random_state=42)
label = km_12.fit_predict(X_12_pca)
u_labels = np.unique(label)
centroids = km_12.cluster_centers_
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]
colors = ['lightgreen', 'orange', 'blue', 'black']
plt.scatter(centroids_x, centroids_y, marker = '*', s=150, linewidths = 5, zorder = 10, c=colors)
for i in u_labels:
    plt.scatter(X_12_pca[label == i , 0] , X_12_pca[label == i , 1] , label = i)
plt.legend()
plt.title('Cluster plot for 2012 data')
plt.savefig('Output_HW_4_Part1_Cluster_Plot_2012_Data.png')
plt.clf()

# Plot clusters for 2013
km_13 = KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300, random_state=42)
label = km_13.fit_predict(X_13_pca)
u_labels = np.unique(label)
centroids = km_13.cluster_centers_
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]
colors = ['lightgreen', 'orange', 'blue', 'black']
plt.scatter(centroids_x, centroids_y, marker = '*', s=150, linewidths = 5, zorder = 10, c=colors)
for i in u_labels:
    plt.scatter(X_13_pca[label == i , 0] , X_13_pca[label == i , 1] , label = i)
plt.legend()
plt.title('Cluster plot for 2013 data')
plt.savefig('Output_HW_4_Part1_Cluster_Plot_2013_Data.png')
plt.clf()

# Plot clusters for 2014
km_14 = KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300, random_state=42)
label = km_14.fit_predict(X_14_pca)
u_labels = np.unique(label)
centroids = km_14.cluster_centers_
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]
colors = ['lightgreen', 'orange', 'blue', 'black', 'red']
plt.scatter(centroids_x, centroids_y, marker = '*', s=150, linewidths = 5, zorder = 10, c=colors)
for i in u_labels:
    plt.scatter(X_14_pca[label == i , 0] , X_14_pca[label == i , 1] , label = i)
plt.legend()
plt.title('Cluster plot for 2014 data')
plt.savefig('Output_HW_4_Part1_Cluster_Plot_2014_Data.png')
plt.clf()

# Plot clusters for 2015
km_15 = KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300, random_state=42)
label = km_15.fit_predict(X_15_pca)
u_labels = np.unique(label)
centroids = km_15.cluster_centers_
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]
colors = ['lightgreen', 'orange', 'blue', 'black']
plt.scatter(centroids_x, centroids_y, marker = '*', s=150, linewidths = 5, zorder = 10, c=colors)
for i in u_labels:
    plt.scatter(X_15_pca[label == i , 0] , X_15_pca[label == i , 1] , label = i)
plt.legend()
plt.title('Cluster plot for 2015 data')
plt.savefig('Output_HW_4_Part1_Cluster_Plot_2015_Data.png')
plt.clf()

s.write('\n\t\t\t\t...DONE!\n')

####################################################
#
# Silhouettes
#
####################################################

s.write('\n>> Creating Silhouette Plots...\n')

# Silhouette plot for 2011
model = SilhouetteVisualizer(KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300, random_state=42))
model.fit(X_11_pca)
plt.title('Silhouette plot for 2011 data')
plt.savefig('Output_HW_4_Part1_Silhouette_Plot_2011_Data.png')
plt.clf()

# Silhouette plot for 2012
model = SilhouetteVisualizer(KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300, random_state=42))
model.fit(X_12_pca)
plt.title('Silhouette plot for 2012 data')
plt.savefig('Output_HW_4_Part1_Silhouette_Plot_2012_Data.png')
plt.clf()

# Silhouette plot for 2013
model = SilhouetteVisualizer(KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300, random_state=42))
model.fit(X_13_pca)
plt.title('Silhouette plot for 2013 data')
plt.savefig('Output_HW_4_Part1_Silhouette_Plot_2013_Data.png')
plt.clf()

# Silhouette plot for 2014
model = SilhouetteVisualizer(KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300, random_state=42))
model.fit(X_14_pca)
plt.title('Silhouette plot for 2014 data')
plt.savefig('Output_HW_4_Part1_Silhouette_Plot_2014_Data.png')
plt.clf()

# Silhouette plot for 2015
model = SilhouetteVisualizer(KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300, random_state=42))
model.fit(X_15_pca)
plt.title('Silhouette plot for 2015 data')
plt.savefig('Output_HW_4_Part1_Silhouette_Plot_2015_Data.png')
plt.clf()

s.write('\n\t\t\t\t...DONE!\n')

####################################################
#
# Distortion score
#
####################################################

s.write('\n>> Printing distortion scores...\n')

s.write('\n\t> Distortion 2011: %.2f' % km_11.inertia_ + '\n')
s.write('\n\t> Distortion 2012: %.2f' % km_12.inertia_ + '\n')
s.write('\n\t> Distortion 2013: %.2f' % km_13.inertia_ + '\n')
s.write('\n\t> Distortion 2014: %.2f' % km_14.inertia_ + '\n')
s.write('\n\t> Distortion 2015: %.2f' % km_15.inertia_ + '\n')

s.write('\n\t\t\t\t...DONE!\n')
         
s.write('\n******************PROGRAM is DONE *******************')
s.close()
