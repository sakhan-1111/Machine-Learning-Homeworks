#************************************************************************************
# Shafiqul Alam Khan
# ML – HW#4 - Part 3
# Filename: HW4_Part_3.py
# Due: Oct. 18, 2023
#
# Objective:
# • Use the Agglomerative technique to determine the hierarchical tree using the LEAP cluster
# • Determine the dendrogram plot for Part 2
#*************************************************************************************

s = open("Output_HW4_Part3.txt", "w")
s.write('>> Importing Packages...\n')
# Importing all required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
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

# Ignore warning
warnings.filterwarnings('ignore')

####################################################
#
# Plot Dendrogram
#
####################################################

s.write('\n>> Plotting Dendrogram...\n')

# Define function to plot dendrogram
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
  
# Agglomerative technique
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
model = model.fit(X)

# Plot dendrogram
plt.title("Hierarchical Clustering Dendrogram")
plot_dendrogram(model, truncate_mode="level", p=3)
plt.xlabel("Number of points in node.")
plt.savefig('Output_HW_4_Part3_Dendrogram_Plot.png')
plt.clf()

s.write('\n\t\t\t\t...DONE!\n')

s.write('\n******************PROGRAM is DONE *******************')
s.close()