import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import ot
import ot.plot
import matplotlib.pylab as pl
import tensorflow as tf
import pprint as pp
import matplotlib.pyplot as plt
mnist = input_data.read_data_sets('/tmp/MNIST_data', one_hot=False)

train_x = mnist.test.images
print ('mnist:' , np.shape(train_x))
img_idx =12 #18,11,1,2

train_label = mnist.test.labels
print(train_label[:20])
print(train_label[img_idx])



def get_dig_coordinate(dig_img):

    dig_img = np.reshape(dig_img, [28,28])
    dig_pixel = np.where(dig_img>0)
    dig_pixel = np.squeeze(np.dstack((dig_pixel[0],dig_pixel[1]))) # coverting to list of (x,y) coordinate
    return dig_pixel

def get_eclidian_dis(s_pixel, t_pixel):

    M = ot.dist(s_pixel, t_pixel)
    M = np.array(M, dtype='float')
    M /= M.max()
    return  M

def get_data_indices_by_patches(img_idx):

    x = tf.placeholder(tf.float32, shape=[None, 784])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    patch_width = 3
    patchs = tf.extract_image_patches(images=x_image, ksizes=[1, patch_width, patch_width, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME')
    patch_dim = tf.shape(patchs,name='patch_dim')

    #patchs = tf.reshape(patchs, [patch_dim[0], patch_dim[1], patch_dim[2], patch_width, patch_width])
    patchs = tf.reshape(patchs, [-1, patch_width , patch_width])

    zero = tf.constant(.3, dtype=tf.float32)
    where = tf.greater(patchs, zero)
    indices = tf.where(where,name='indices')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        p, patch_dim = sess.run([indices, patch_dim],feed_dict = {x : train_x[img_idx:img_idx+1]})
        print('patchdim: ', patch_dim)
        pp.pprint(np.shape(p))
        return [p,patch_dim]

def  pca_map(z, total_patches):
    c = 1
    pca_dat = []
    for i in range(total_patches):
        temp = z[z[:,0] == i]
        M = np.array(temp[:, 1:3])
        #M = temp[:,1:3]
        #print M

        #if i == 44:
        if len(M) > 2:
            c += 1
            #print M.shape
            mu = M.mean(axis=0)
            data = M - mu
            U, S, V = np.linalg.svd(data.T)
            projected_data = np.dot(data, U)
            #pp.pprint(S)
            #print M
            #draw(M[:,0],M[:,1], U, S, mu, c)

            if np.abs(S[0] - S[1]) > 0.5 :
                pca_dat.append(np.vstack((U, S)))
            else:
                pca_dat.append([[0, 0], [0, 0], [0, 0]])
            #print data.shape

            #print i, '\t', len(M)
        else:
            pca_dat.append([[0,0],[0,0],[0,0]])
            #print U
    return pca_dat

def angles(img_idx):
    [indices, patch_dim] = get_data_indices_by_patches(img_idx)

    total_patches = patch_dim[0] * patch_dim[1] * patch_dim[2]
    pca_dat = np.array(pca_map(indices, total_patches))
    pca_pcs = pca_dat[:, 0, :]
    print(np.shape(pca_dat))

    pca_pcs = np.reshape(pca_pcs, [patch_dim[1], patch_dim[2], 2])
    pcs_x = pca_pcs[:, :, 0]
    pcs_y = pca_pcs[:, :, 1]

    pcs_y = -1 * np.flipud(pcs_y)
    pcs_x = np.flipud(pcs_x)
    # np.putmask(pcs_x, pcs_x*pcs_y >=0, abs(pcs_x))

    angles = np.arctan2(pcs_y, pcs_x) * 180 / np.pi
    angles[angles < 0] += 180

    return pcs_y, pcs_x, angles



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

    s, t = {'all': source_coor}, {'all': target_coor,'unmapped':np.empty( shape= [0,2],dtype='int32'),'mapped':np.array(target_coor, dtype='int32')}

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


def calc_cutoff(G, C):
    val = []
    for r in range(G.shape[0]):
        for c in range(G.shape[1]):
            if G[r][c]>0.0:
                val.append(C[r,c])
    std = np.sqrt(np.var(val))
    meu = np.mean(val)
    print("mean:{} std:{}".format(meu,std))
    cut_max = meu + 2*std
    return val, cut_max

def filter_costly_map(s_coor, t_coor, M, C, thr_upr):
    s, t = s_coor['all'], t_coor['all']
    mapped_s = []
    unmapped_s = s_coor['unmapped'] #unmapped due to 1 to 1 relation, less pixel in one
    mapped_t = []
    unmapped_t =  t_coor['unmapped']
    for i in range(s.shape[0]):
        for j in range(t.shape[0]):
            if M[i, j] > 0. and C[i, j] <thr_upr:
                mapped_s.append(s[i])
                mapped_t.append(t[j])
            elif M[i, j] > 0. and C[i, j] >= thr_upr:
                unmapped_s =np.vstack ((unmapped_s, s[i]) )
                unmapped_t =np.vstack ((unmapped_t, t[j]) )

    s_coor['mapped'] = np.array(mapped_s, dtype='int32')
    s_coor['unmapped'] = unmapped_s
    t_coor['mappped'] = np.array(mapped_t, dtype='int32')
    t_coor['unmapped'] = unmapped_t

    return s_coor, t_coor
def draw_overlapping(s, t,fig_id):

    pl.figure(fig_id)

    pl.plot(s[:, 0], s[:, 1], '+b', label='Source samples')
    pl.plot(t[:, 0], t[:, 1], 'xr', label='Target samples')
    pl.legend(loc=0)
    pl.title('OT matrix with samples')
    pl.show()

def draw_mapping(source_coor,target_coor,G, fig_id, thr):
    s_all = source_coor['all']
    t_all = target_coor['all']
    pl.figure(fig_id)
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

def draw_vector_field(X, Y, angles):

    plt.title('Arrows scale with plot width, not view')
    plt.axis('equal')


    Q = plt.quiver(X, Y, angles, scale=2, units='xy')
def draw_cost_density(cost):

    import seaborn as sns
    sns.distplot(cost, hist=True, rug=True,)
    plt.show()

s_idx, t_idx = 7,12#1, 12 (0,7)(1,2)(4,4)(5,1)(6,4)(7,9)
source_img, target_img = train_x[s_idx], train_x[t_idx]

s_pixel_coor = get_dig_coordinate(source_img)
t_pixel_coor = get_dig_coordinate(target_img)  # np.array([[x+20, y+10] for x,y in dig_pixel], dtype='float') #pixel shifted for better vizualization

s_x, s_y,s_angles = angles(s_idx)
t_x,t_y,t_angles = angles(t_idx)
print(np.shape(s_pixel_coor))
#print(s_pixel_coor[:,0])
s_angles = s_angles[s_pixel_coor[:,0],s_pixel_coor[:,1]]
t_angles = t_angles[t_pixel_coor[:,0],t_pixel_coor[:,1]]
D = np.subtract.outer(s_angles, t_angles)
D = np.abs(D)
D /= D.max()

n1, n2 = len(s_pixel_coor), len(t_pixel_coor)
a, b = np.ones((n1,)) / n1, np.ones((n2,)) / n2  # uniform distribution on samples
M = get_eclidian_dis(s_pixel_coor, t_pixel_coor)
#G = ot.emd2(a, b, D)
#G = ot.emd(a, b, M)


alpha = 20/21
cost = alpha*M+(1-alpha)*D
#cost = M+100*D
source_coor,target_coor,G = calc_mapping(s_pixel_coor, t_pixel_coor, cost)
c =np.multiply(cost,G)
print(c,np.max(c))
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
draw_vector_field(s_x, s_y, s_angles)
plt.subplot(1, 2, 2)
draw_vector_field(t_x, t_y, t_angles)

val, cutoff_dis = calc_cutoff(G,c)
print("cutoff max: ",cutoff_dis)
draw_overlapping(s_pixel_coor,t_pixel_coor,3)
source_coor, target_coor = filter_costly_map(source_coor, target_coor,G, cost, .6*cutoff_dis)
draw_mapping(source_coor,target_coor,c,4,4*cutoff_dis)
draw_cost_density(val)

# t_pixel_coor = np.array([[x + 20, y + 20] for x, y in t_pixel_coor], dtype='float')  # pixel shifted for better vizualization
# print(D)
# draw_ot(s_pixel_coor,t_pixel_coor,G)
#
#
#

