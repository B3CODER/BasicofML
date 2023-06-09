import matplotlib.pyplot as plt
import seaborn as sns ; sns.set()
import numpy as np
from sklearn.datasets._samples_generator import make_blobs

X,y_true = make_blobs(n_samples = 300 , centers= 4 , cluster_std= 0.50 , random_state=0)
plt.scatter(X[:, 0],X[:,1], s=50)
plt.show()

from sklearn.cluster import  KMeans
Kmeans =KMeans(n_clusters=4)
Kmeans.fit(X)

y_kmeans=Kmeans.predict(X)