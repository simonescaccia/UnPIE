import numpy as np

def normalize_undigraph(A):

    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-0.5)

    DAD = np.dot(np.dot(Dn, A), Dn)

    return DAD