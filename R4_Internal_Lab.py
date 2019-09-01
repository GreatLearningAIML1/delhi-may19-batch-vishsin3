#!/usr/bin/env python
# coding: utf-8

# ### The data set has information about features of silhouette extracted from the images of different cars
# 
# Four "Corgie" model vehicles were used for the experiment: a double decker bus, Cheverolet van, Saab 9000 and an Opel Manta 400 cars. This particular combination of vehicles was chosen with the expectation that the bus, van and either one of the cars would be readily distinguishable, but it would be more difficult to distinguish between the cars.
# 
# 

# ### 1. Read the dataset using function .dropna() - to avoid dealing with NAs as of now

# In[272]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans 


# In[273]:


df = pd.read_csv('vehicle.csv')


# In[274]:


df=df.dropna()


# ### 2. Print/ Plot the dependent (categorical variable) - Class column

# Since the variable is categorical, you can use value_counts function

# In[275]:


df.describe().T


# In[276]:


df.info()


# In[277]:


x=df['class']
sns.countplot(x)


# ### Check for any missing values in the data 

# In[278]:


df.count()


# ### 3. Standardize the data 

# In[279]:


X= stats.zscore(df.loc[:,df.columns !='class'])


# In[280]:


from scipy.stats import zscore
df=df.loc[:,df.columns !='class']
df_X = df.apply(zscore)


# Since the dimensions of the data are not really known to us, it would be wise to standardize the data using z scores before we go for any clustering methods.
# You can use zscore function to do this

# ### K - Means Clustering

# ### 4. Plotting Elbow/ Scree Plot

# Use Matplotlib to plot the scree plot - Note: Scree plot plots distortion vs the no of clusters

# In[281]:


from sklearn.cluster import KMeans
from scipy.spatial import distance

# Let us check optimal number of clusters-
distortion = []

cluster_range = range( 1, 10)   
cluster_errors = []
cluster_sil_scores = []
for num_clusters in cluster_range:
  clusters = KMeans( num_clusters, n_init = 5)
  clusters.fit(X)
  labels = clusters.labels_                     # capture the cluster lables
  centroids = clusters.cluster_centers_         # capture the centroids
  cluster_errors.append( clusters.inertia_ )    # capture the intertia
  distortion.append(sum(np.min(distance.cdist(X, clusters.cluster_centers_, 'euclidean'), axis=1))/ X.shape[0])


# In[282]:


clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors} )
clusters_df[0:15]


# In[283]:


plt.figure(figsize=(12,6))
plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )


# ### Find out the optimal value of K

# In[266]:


# The elbow plot confirms our visual analysis that there are likely 2, 3 or 4 good clusters. But, with insight of data
# Let us start with 3 clusters


# ### Using optimal value of K - Cluster the data. 
# Note: Since the data has more than 2 dimension we cannot visualize the data. As an alternative, we can observe the centroids and note how they are distributed across different dimensions

# In[284]:


kmeans = KMeans(n_clusters=3)
kmeans = kmeans.fit(X)
labels = kmeans.predict(X)
centroids = kmeans.cluster_centers_

print("Centroid values")
print("sklearn")
print(centroids) # From sci-kit learn


# You can use kmeans.cluster_centers_ function to pull the centroid information from the instance

# In[ ]:





# ### 5. Store the centroids in a dataframe with column names from the original dataset given 

# In[285]:


centroid_df = pd.DataFrame(centroids, columns =list(df_X) )
centroid_df


# Hint: Use pd.Dataframe function 

# ### Use kmeans.labels_ function to print out the labels of the classes

# In[287]:


kmeans.labels_


# In[ ]:





# ## Hierarchical Clustering 

# ### 6. Variable creation

# For Hierarchical clustering, we will create datasets using multivariate normal distribution to visually observe how the clusters are formed at the end

# In[288]:


a = np.random.multivariate_normal([10, 0], [[3, 1], [1, 4]], size=[100,])
b = np.random.multivariate_normal([0, 20], [[3, 1], [1, 4]], size=[50,])
c = np.random.multivariate_normal([10, 20], [[3, 1], [1, 4]], size=[100,])


# In[ ]:





# In[289]:


new.shape


# In[290]:


df_a = pd.DataFrame(a)
df_b = pd.DataFrame(b)
df_c = pd.DataFrame(c)


# ### 7. Combine all three arrays a,b,c into a dataframe

# In[291]:


new = pd.DataFrame(np.concatenate((a,b,c)))


# In[292]:


new


# ### 8. Use scatter matrix to print all the 3 distributions

# In[ ]:


sns.pairplot(new)


# In[ ]:





# ### 9. Find out the linkage matrix

# In[293]:


from scipy.cluster.hierarchy import cophenet, dendrogram, linkage

Linkage_Matrix = linkage(new,'average')


# In[294]:


Linkage_Matrix


# Use ward as linkage metric and distance as Eucledian

# In[ ]:





# ### 10. Plot the dendrogram for the consolidated dataframe

# In[295]:


plt.figure(figsize=(10, 10))
plt.title('Agglomerative Hierarchical Clustering Dendogram')
plt.xlabel('sample index')
plt.ylabel('Distance')
dendrogram(Linkage_Matrix, leaf_rotation=90.,color_threshold = 30, leaf_font_size=8. )
plt.tight_layout()


# In[ ]:





# In[ ]:





# ### 11. Recreate the dendrogram for last 12 merged clusters 

# In[296]:


plt.figure(figsize=(10, 10))
plt.title('Agglomerative Hierarchical Clustering Dendogram')
plt.xlabel('sample index')
plt.ylabel('Distance')
dendrogram(Linkage_Matrix,truncate_mode='lastp',p=3, leaf_rotation=90.,color_threshold = 30, leaf_font_size=8. )
plt.tight_layout()


# Hint: Use truncate_mode='lastp' attribute in dendrogram function to arrive at dendrogram 

# ### 12. From the truncated dendrogram, find out the optimal distance between clusters which u want to use an input for clustering data

# In[297]:


from scipy.spatial.distance import pdist
c, coph_dists = cophenet(Linkage_Matrix, pdist(new))
c


# In[ ]:





# ### 13. Using this distance measure and fcluster function to cluster the data into 3 different groups

# In[305]:


from scipy.cluster.hierarchy import fcluster


# In[306]:


fcluster_plot = fcluster(Linkage_Matrix, 10, criterion ="distance" )
fcluster_plot


# ### Use matplotlib to visually observe the clusters in 2D space 

# In[312]:


plt.scatter(x =new[0], y=new[1])


# In[ ]:




