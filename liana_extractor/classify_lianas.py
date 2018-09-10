import argparse
import generate_features
import numpy as np
import liana_classifier

def getCmdargs():
    """
    Get commandline arguments
    """
    p = argparse.ArgumentParser()
    p.add_argument("-nbr", "--nbrFile", help="Input file to generate neighbor kdtree")
    p.add_argument("-in", "--inDataFile", help="Input data file")
    p.add_argument("-out", "--outPredFile", help="Output file to save predictions")
    p.add_argument("-randpts", "--randompts", help="Number of random points to be chosen")
    cmdargs = p.parse_args()

    return cmdargs

def main():
    cmdargs = getCmdargs()
    nbr_data_file_name = cmdargs.nbrFile
    in_file_name = cmdargs.inDataFile
    out_file_name = cmdargs.outPredFile
    no_rand_pts = cmdargs.randompts

    nbr_arr = np.loadtxt(nbr_data_file_name)
    knn_arr = np.loadtxt(in_file_name)

    feature_arr = generate_features.get_eigen_values_with_radius(nbr_arr, knn_arr, no_rand_pts)
    liana_classifier.predict_class(feature_arr, out_file_name, model_file_name='liana_clf_model.sav')

if __name__== "__main__":
    main()

