
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import ot
import ot.plot
import matplotlib.pylab as pl

import pprint as pp
import matplotlib.pyplot as plt
mnist = input_data.read_data_sets('/tmp/MNIST_data', one_hot=False)

train_x = mnist.test.images
print ('mnist:' , np.shape(train_x))
s_idx, t_idx =9,12  #18,11,1,2

train_label = mnist.test.labels
print(train_label[:20])
print(train_label[s_idx])

def get_dig_coordinate(dig_img):

    dig_img = np.reshape(dig_img, [28,28])
    dig_pixel = np.where(dig_img>0)
    dig_pixel = np.squeeze(np.dstack((dig_pixel[0],dig_pixel[1]))) # coverting to list of (x,y) coordinate
    return dig_pixel

source_coor = get_dig_coordinate(train_x[s_idx])
target_coor = get_dig_coordinate(train_x[t_idx])#np.array([[x+20, y+10] for x,y in dig_pixel], dtype='float') #pixel shifted for better vizualization

n1, n2 = len(source_coor), len(target_coor)
def calc_mapping(source_coor, target_coor, dis_mat):

    sn, tn = len(source_coor), len(target_coor)
    s_t_swap =False
    M = dis_mat
    if tn> sn: # swap source target if target is larger
        temp = source_coor
        source_coor = target_coor
        target_coor = temp
        M = np.transpose(dis_mat).copy()
        s_t_swap = True

    s, t = {'all': source_coor}, {'all': target_coor,'unmapped':[],'mapped':target_coor}

    # always treat largest one as source then before return back do the transformation

    M = np.array(M, dtype='float')
    sn, tn = len(source_coor), len(target_coor)
    a, b = np.ones((sn,)), np.ones((tn,))

    # add a sink in target at 0,0 to dump all excess, add a column  0 in cost matrix with

    b = np.insert(b, len(b), sn-tn)
    M = np.insert(M,tn,0,axis=1)

    M /= M.max()
    G = ot.emd(a, b, M)
    mapped_s = []
    unmapped_s = []
    for c in range(G.shape[0]):
        if G[c,-1] < 1:  # mapped to sink
            mapped_s.append(source_coor[c])
        else:unmapped_s.append(source_coor[c])
    s['mapped'] = np.array(mapped_s, dtype='int32')
    s['unmapped'] = np.array(unmapped_s, dtype='int32')

    G = G[:, :-1]
    if s_t_swap:

        temp =s.copy()
        s = t
        t = temp
        G = np.transpose(G).copy() #copy is required to actually transpose in memory not just represetation


    return s,t,G

# a, b = np.ones((n1,)) / n1, np.ones((n2,)) / n2  # uniform distribution on samples
# print(np.squeeze(dig_pixel_target))
# 
# M = ot.dist(dig_pixel, dig_pixel_target)
# M = np.array(M, dtype='float')
# M /= M.max()


def draw_mapping(source_coor,target_coor,G):
    s_all = source_coor['all']
    t_all = target_coor['all']
    pl.figure(1)
    t_all = np.array([[x + 20, y + 20] for x, y in t_all], dtype='float') #pixel shifted for better vizualization
    ot.plot.plot2D_samples_mat(s_all, t_all, G, c=[0, 1, 0,.1])
    s_all = source_coor['mapped']
    t_all = target_coor['mapped']
    pl.plot(s_all[:, 0], s_all[:, 1], '+b', label='Source samples')
    if len(source_coor['unmapped'])> 0:
        pl.plot(source_coor['unmapped'][:, 0], source_coor['unmapped'][:, 1], '+c', label='Source samples')

    t_all = np.array([[x + 20, y + 20] for x, y in t_all], dtype='float') #pixel shifted for better vizualization
    pl.plot(t_all[:, 0], t_all[:, 1], 'xr', label='Target samples')
    if len(target_coor['unmapped'])> 0:
        t_all = np.array([[x + 20, y + 20] for x, y in target_coor['unmapped']], dtype='float')  # pixel shifted for better vizualization
        pl.plot(t_all[:, 0], t_all[:, 1], 'xc', label='Target samples')

    pl.legend(loc=0)
    pl.title('OT matrix with samples')
    pl.show()

M = ot.dist(source_coor, target_coor)
source_coor,target_coor,G = calc_mapping(source_coor, target_coor,M)
draw_mapping(source_coor,target_coor,G)