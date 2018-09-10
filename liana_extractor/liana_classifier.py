import numpy as np
import pickle
import os

def predict_class(arr_with_features, out_file_name, model_file_name = 'liana_clf_model.sav'):
    """
        Function to predict if a point belongs to liana or other vegetation based on random forest classifier

            Parameters
            ----------
            arr_with_features: arr
                Array of size (m x 18) where m is the number of points and out of 18, 3 fields correspond to x,y,z dimension of points and the
                other 15 come from the 3 eigen values corresponding to x, y and z dimension for 5 different radii of the pointcloud
            out_file_name: str
                Output file to save predictions
            model_file_name: str
                Filename of the classifier to make the predictions
        """

    clf = pickle.load(open(os.path.abspath(model_file_name), 'rb'))

    X_xyz = arr_with_features[:,0:3]
    X_features = arr_with_features[:, 3:]
    X_features[np.isnan(X_features)] = 0

    y_pred = clf.predict(X_features)

    predicted_outcome = np.column_stack((X_xyz,X_features,y_pred))

    np.savetxt(out_file_name, predicted_outcome, fmt='%1.3f')
