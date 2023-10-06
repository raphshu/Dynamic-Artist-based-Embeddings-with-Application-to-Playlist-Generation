import numpy as np

M, m = data.shape[0], len(central_points)

# Reshape central_points and clusters_cov_inverse to have a shape of (m, 1, D) and (m, D, D), respectively
central_points_reshaped = np.array(central_points).reshape(m, 1, -1)
clusters_cov_inverse_reshaped = np.array(clusters_cov_inverse).reshape(m, -1, data.shape[1])

# Reshape data to have a shape of (M, 1, D)
data_reshaped = data.values.reshape(M, 1, -1)

# Calculate the difference between each song and the centers of all clusters
difference = data_reshaped - central_points_reshaped

# Perform the matrix multiplication (difference * clusters_cov_inverse) for all clusters at once
product = np.matmul(difference, clusters_cov_inverse_reshaped)

# Perform the element-wise multiplication between difference and product
element_wise_product = difference * product

# Sum along the last axis (axis=-1) to get the squared distance for each song and cluster
distance_matrix = np.sum(element_wise_product, axis=-1)