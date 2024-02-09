import numpy as np

def linear_discriminant_analysis(w1, w2):
    # Construct the whole dataset
    x = np.concatenate((w1, w2), axis=0)

    # Compute the mean of the given two class and of the whole dataset
    u1 = np.mean(w1, axis=0)
    u2 = np.mean(w2, axis=0)
    u_total = np.mean(x, axis=0)

    # Compute the between-class variance
    n_u1 = np.expand_dims(u1 - u_total, axis=1)
    n_u2 = np.expand_dims(u2 - u_total, axis=1)
    sb = w1.shape[0] * np.dot(n_u1, n_u1.T) + w2.shape[0] * np.dot(n_u2, n_u2.T)

    # Compute the within-class variance
    sw = np.dot((w1 - u1).T, (w1 - u1)) + np.dot((w2 - u2).T, (w2 - u2))

    # Compute the Fisher's linear discriminant and solve the eigenvalue problem
    fisher = np.dot(np.linalg.inv(sw), sb)
    eigenvalue, eigenvector = np.linalg.eig(fisher)

    # Find the optimal projection vector and its corresponding eigenvalue
    max_idx = np.argmax(eigenvalue)
    return eigenvector[:, max_idx], eigenvalue[max_idx]


if __name__ == '__main__':
    # Construct the data points of each class
    w1 = np.array([[5, 3], [3, 5], [3, 4], [4, 5], [4, 7], [5, 6]])
    w2 = np.array([[9, 10], [7, 7], [8, 5], [8, 8], [7, 2], [10, 8]])

    # Execute LDA
    proj_vec, eigen_val = linear_discriminant_analysis(w1, w2)
    print(f'The projection vector is\n\t({proj_vec[0]:7.5f}, {proj_vec[1]:7.5f})')
    print(f'corresonding to the largest eigenvalue\n\t{eigen_val:7.5f}')