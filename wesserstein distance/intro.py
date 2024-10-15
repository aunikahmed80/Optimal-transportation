#Source : https://github.com/vincentherrmann/wasserstein-notebook/blob/master/Wasserstein_Kantorovich.ipynb

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import linprog
from matplotlib import cm



l = 10

P_r = np.array([12,7,4,1,19,14,9,6,3,2])
P_t = np.array([1,5,11,17,13,9,6,4,3,2])
P_r = P_r / np.sum(P_r)
P_t = P_t / np.sum(P_t)


def construct_A():
    A_r = np.zeros((l, l, l))
    A_t = np.zeros((l, l, l))

    for i in range(l):
        for j in range(l):
            A_r[i, i, j] = 1
            A_t[i, j, i] = 1

    A = np.concatenate((A_r.reshape((l, l ** 2)), A_t.reshape((l, l ** 2))), axis=0)
    print("A: \n", A, "\n")
    return  A

def construc_distance_mat():
    D = np.ndarray(shape=(l, l))

    for i in range(l):
        for j in range(l):
            D[i, j] = abs(range(l)[i] - range(l)[j])
    return D

def plot_distribution(dist, color):

    plt.bar(range(l), dist, 1, color=color, alpha=1)
    plt.axis('off')
    plt.ylim(0, 0.5)
    #plt.savefig("discrete_p_r.svg")
    print("P_r:")
    plt.show()

#plot_distribution(P_r,'blue')
#plot_distribution(P_t,'green')
A = construct_A()
D = construc_distance_mat()
b = np.concatenate((P_r, P_t), axis=0)
c = D.reshape((l**2))

opt_res = linprog(c, A_eq=A, b_eq=b, bounds=[0, None])
emd = opt_res.fun
gamma = opt_res.x.reshape((l, l))
print("EMD: ", emd, "\n")