import argparse
import generate_features
import numpy as np
import leaf_wood_classifier

def getCmdargs():
    """
    Get commandline arguments
    """
    p = argparse.ArgumentParser()
    p.add_argument("-nbr", "--nbrFile", help="Input file to generate neighbor kdtree")
    p.add_argument("-in", "--inDataFile", help="Input data file")
    p.add_argument("-ifclass", "--ifclass", help="If the last column of the input file contains class value from classify_liana step")
    p.add_argument("-out", "--outPredFile", help="Output file to save predictions")
    cmdargs = p.parse_args()

    return cmdargs

def main():
    cmdargs = getCmdargs()
    nbr_data_file_name = cmdargs.nbrFile
    in_file_name = cmdargs.inDataFile
    ifclass = int(cmdargs.ifclass)
    out_file_name = cmdargs.outPredFile

    nbr_arr = np.loadtxt(nbr_data_file_name)
    knn_arr = np.loadtxt(in_file_name)
    #print(knn_arr.shape)
    if(ifclass == 1):
        knn_arr = knn_arr[knn_arr[:,-1]  == 2,:-1]
    #print(knn_arr.shape)
    feature_arr = generate_features.get_eigen_values_with_radius(nbr_arr, knn_arr, 0)
    leaf_wood_classifier.cluster_leaf_vs_wood(feature_arr, out_file_name)

if __name__== "__main__":
    main()

