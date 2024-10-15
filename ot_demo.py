import numpy as np
import matplotlib.pylab as pl
import ot
import ot.plot


# n = 50  # nb samples
#
# mu_s = np.array([0, 0])
# cov_s = np.array([[1, 0], [0, 1]])
#
# mu_t = np.array([4, 4])
# cov_t = np.array([[1, -.8], [-.8, 1]])
#
# xs = ot.datasets.make_2D_samples_gauss(n, mu_s, cov_s)
# xt = ot.datasets.make_2D_samples_gauss(n, mu_t, cov_t)
# print(xs)
#
# a, b = np.ones((n,)) / n, np.ones((n,)) / n  # uniform distribution on samples
#
# # loss matrix
# M = ot.dist(xs, xt)
# M /= M.max()
#
# pl.figure(1)
# pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
# pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
# pl.legend(loc=0)
# pl.title('Source and target distributions')
#
# pl.figure(2)
# pl.imshow(M, interpolation='nearest')
# pl.title('Cost matrix M')
#
#
# G0 = ot.emd(a, b, M)
#
# pl.figure(3)
# pl.imshow(G0, interpolation='nearest')
# pl.title('OT matrix G0')
#
# pl.figure(4)
# ot.plot.plot2D_samples_mat(xs, xt, G0, c=[.5, .5, 1])
# pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
# pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
# pl.legend(loc=0)
# pl.title('OT matrix with samples')
#
# pl.show()

import ot
a=[1, 1, 1,1]
b=[1, 1,2]
M=[[0., 1.], [1., 0.],[2, 1],[3, 2]]
print(M)
M = np.insert(M,2,0,axis=1)
M = np.transpose(M).copy()
#print(np.transpose(M))
r = ot.emd(b, a, M, )
print(r.shape[0])


# a=[1, 1,2]
# b=[1, 1,1,1]
# M=np.transpose(M)#[[0., 1.,2,3], [1., 0.,1,2],[0,0,0,0]]
# M = M.copy(order='C')
#
# r = ot.emd(a, b, M, )
# print(r)