
# A set of helpers which are useful for debugging SIFT!
# Feel free to take a look around in case you are curious,
# but you shouldn't need to know exactly what goes on,
# and you certainly don't need to change anything

import numpy as np
import scipy.io as scio

import visualize


# Gives you the TA solution for the interest points you
# should find
def cheat_interest_points(eval_file, scale_factor):
    file_contents = scio.loadmat(eval_file)

    x1 = file_contents['x1']
    y1 = file_contents['y1']
    x2 = file_contents['x2']
    y2 = file_contents['y2']

    x1 = x1 * scale_factor
    y1 = y1 * scale_factor
    x2 = x2 * scale_factor
    y2 = y2 * scale_factor

    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    x2 = x2.reshape(-1)
    y2 = y2.reshape(-1)

    return x1, y1, x2, y2


def estimate_fundamental_matrix(Points_a, Points_b):
    # Get linear system of eqns
    # each row will be:
    # [u'u u'v u' v'u v'v v' u v 1]
    # found from rearranging the defn of the fundamental matrix
    # we assume the prime image is image B

    n = Points_b.shape[0]
    u_prime = np.copy(Points_b[:, 0])
    v_prime = np.copy(Points_b[:, 1])
    u = np.copy(Points_a[:, 0])
    v = np.copy(Points_a[:, 1])

    ############################
    # Normalize points
    # Calculate offset matrices combining images a and b
    c_u = np.mean(u)
    c_v = np.mean(v)
    c_u_prime = np.mean(u_prime)
    c_v_prime = np.mean(v_prime)

    offset_matrix = np.array([[1, 0, -c_u], [0, 1, -c_v], [0, 0, 1]])
    offset_matrix_prime = np.array([[1, 0, -c_u_prime], [0, 1, -c_v_prime], [0, 0, 1]])

    # Calculate scale matrices for images a and b
    s = 1 / np.std([[u - c_u], [v - c_v]])
    s_prime = 1 / np.std([[u_prime - c_u_prime], [v_prime - c_v_prime]])

    scale_matrix = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]])
    scale_matrix_prime = np.array([[s_prime, 0, 0], [0, s_prime, 0], [0, 0, 1]])

    T_a = scale_matrix @ offset_matrix
    T_b = scale_matrix_prime @ offset_matrix_prime

    # Normalize points from images a and b
    for i in range(0, n):
        norm = T_a @ np.transpose([u[i], v[i], 1])
        norm_prime = T_b @ np.transpose([u_prime[i], v_prime[i], 1])
        u[i] = norm[0]
        v[i] = norm[1]
        u_prime[i] = norm_prime[0]
        v_prime[i] = norm_prime[1]

    # Normalize points ends here
    ############################

    # create data matrix
    data_matrix = np.array([u_prime * u, u_prime * v, u_prime,
                            v_prime * u, v_prime * v, v_prime,