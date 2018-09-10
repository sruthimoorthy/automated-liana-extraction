from sklearn.cluster import KMeans
import numpy as np

def cluster_leaf_vs_wood(train_data, save_file):
    """
        Simple clustering function that seprates leaf and wood into different clusters. If the plot has a lot of lianas it is advised to use classify_lianas function first and pass the non-liana segmented part to this function 

            Parameters
            ----------
            arr_with_features: arr
                Array of size (m x 18) where m is the number of points and out of 18, 3 fields correspond to x,y,z dimension of points and the other 15 come from the 3 eigen values corresponding to x, y and z dimension for 5 different radii of the pointcloud
            out_file_name: str
                Output file to save predictions
    """
    X_train = train_data[:, 3:]
    X_train[np.isnan(X_train)] = 0
    kmeans = KMeans(n_clusters=2).fit(X_train)
    labels = kmeans.labels_
    labelled_data = np.column_stack((train_data[:,0:3], labels))
    np.savetxt(save_file, labelled_data, fmt='%1.3f')
