to extract liana from plot-level point cloud:
---------------------------------------------
python classify_lianas.py -nbr 'NBR_FILE_NAME' -in 'INPUT_FILE_NAME' -randpts 0 -out 'OUTPUT_FILE_NAME'

to extract liana from plot-level point cloud:
---------------------------------------------
python cluster_leaf_vs_wood.py -nbr 'NBR_FILE_NAME' -in 'INPUT_FILE_NAME' -ifclass 1 -out 'OUTPUT_FILE_NAME'

NBR_FILE_NAME: The path and name of the file containing the 3D point cloud of a bigger area to construct the neighborhood data
INPUT_FILE_NAME: The path and name of the file containing the 3D point cloud of a subset of points from points in NBR_FILE_NAME for which the eigen features are calculated
OUTPUT_FILE_NAME: The path and name of the file where the final results from liana classification or leaf vs wood classification is stored

The NBR_FILE_NAME and INPUT_FILE_NAME can be same or different. If the plot size is big (> 100 sq. metres), the program might run to memory issues. Thus it is advisable to split the plots
into smaller size (I sub-divided in such a way that each subplot had not more than 650,000 points) and still use the original plot for generating the neighbor array to eliminate the edge effects and use the smaller subplots to generate features at 
multiple spatial scales. 
In classify_lianas.py, "randpts > 0" when you don't want to split the big plot into smaller subplots but rather want to generate features for some randomly selected points from 
NBR_FILE_NAME.
In cluster_leaf_vs_wood.py, "ifclass = 1" if you want to separate leaf from woody points for the file where the points are already classified into lianas or non-lianas. For plots without liana classification, you can keep "ifclass = 0".