#matrixeggs.py
import numpy as np
import numpy.linalg as lin 
import matplotlib as mplt 
import matplotlib.pyplot as plt 
import matplotlib.patches as pat

def to_eggs_grayscale(image, dim1=3, dim2=3, egg_dim=2, use_rows=True):
    """
    For grayscale images, we can't compute covariance across channels
    so we use the rows or columns of the neighborhood as values.
    Otherwise, works identically to to_eggs()
    """
    #crop image to appropriate size
    length, width = image.shape[0], image.shape[1]
    if length%dim1!=0:
        image = image[:-(length%dim1)]
    if width%dim2!=0:
        image = image[:,:-(width%dim2)]
    length, width = image.shape[0], image.shape[1]    

    #calculate dimensions for matrix of eggs
    egglen = length//dim1
    eggwid = width//dim2
    #instantiate arrays to store eggs
    eggvecs = np.empty((egglen, eggwid, egg_dim, egg_dim), dtype=np.float64)
    eggvals = np.empty((egglen, eggwid, egg_dim), dtype=np.float64)

    #iterate across neighborhoods
    for i in range(egglen):
        for j in range(eggwid):
            nbhd = image[i*dim1:(i+1)*dim1, j*dim2:(j+1)*dim2]
            #handles rows/columns as vectors
            if use_rows: 
                cov = np.cov(nbhd)
            else:
                nbhd = nbhd.T
                cov = np.cov(nbhd)
            #Use pca to project down into lower dimesion
            #using eigh always results in O.N. matrix of 
            #eigenvectors for a symmetric matrix
            evals, evecs = lin.eigh(cov)
            #throw out least significant dimensions
            evecs = evecs[:,-egg_dim:]
            #project to lower dimension
            lower_dim = nbhd @ evecs

            #calculate evals/evecs for lower dimension
            evals, evecs = lin.eigh(np.cov(lower_dim.T))
            eggvecs[i,j] = evecs
            eggvals[i,j] = evals

    return eggvals, eggvecs


def to_eggs(image, dim1=15, dim2=15, egg_dim=2):
    '''
    For each dim1 x dim2 neighborhood in an image, use pca to project the values
    of the pixels down into egg_dim dimensions, then compute the 'eggs', which
    are the eigenvalues and eigenvectors of the covariance matrix of the 
    neighborhood in the lower dimension
    '''
    #crop image to appropriate size
    if image.shape[2] != 3:
        raise ValueError("Image must have three channels")
    length, width = image.shape[0], image.shape[1]
    if length%dim1!=0:
        image = image[:-(length%dim1)]
    if width%dim2!=0:
        image = image[:,:-(width%dim2)]
    length, width = image.shape[0], image.shape[1]    

    egglen = length//dim1
    eggwid = width//dim2
    eggvecs = np.empty((egglen, eggwid, egg_dim, egg_dim), dtype=np.float64)
    eggvals = np.empty((egglen, eggwid, egg_dim), dtype=np.float64)

    for i in range(egglen):
        for j in range(eggwid):
            #reshape a neighborhood for computing covariance across the pixels
            nbhd = image[i*dim1:(i+1)*dim1, j*dim2:(j+1)*dim2].reshape(dim1*dim2, 3)
            cov = np.cov(nbhd.T)
            #Use pca to project down into lower dimesion
            #using eigh always results in O.N. matrix of 
            #eigenvectors for a symmetric matrix
            evals, evecs = lin.eigh(cov)
            #throw out least significant dimensions
            evecs = evecs[:,-egg_dim:]
            #project to lower dimension
            lower_dim = nbhd @ evecs

            #calculate evals/evecs for lower dimension
            evals, evecs = lin.eigh(np.cov(lower_dim.T))
            eggvecs[i,j] = evecs
            eggvals[i,j] = evals

    return eggvals, eggvecs

def egg_dist(eggs1, eggs2):
    '''
    Computes neighborhood-wise distance between two egg matrices
    where the distance is the sum of the l2 norm between the eigenvalues
    and the riemannian distance between the eigenvectors (which form a 
    rotation matrix)

    As of now, only works if the target dimension is 2

    rot(x) = [cos x  sin x]
             [-sin x cos x]    
    '''
    (eggvals1, eggvecs1) = eggs1
    (eggvals2, eggvecs2) = eggs2
    dim1, dim2 = eggvals1.shape[0], eggvals1.shape[1]
    egg_dists = np.empty((dim1, dim2), dtype=np.float64)

    for i in range(dim1):
        for j in range(dim2):
            l2dist = lin.norm(eggvals1[i,j]-eggvals2[i,j])
            cos = (eggvecs1[i,j] @ eggvecs2[i,j].T)[0,0]
            theta = np.arccos(cos)
            #rescale theta to be between 0 and sqrt(2) so it has the same scale as l2dist
            SOndist = theta*(2**0.5)/np.pi
            #We can map a 2x2 matrices to angles 
            egg_dists[i,j] = l2dist + SOndist

    return egg_dists

def total_dist(egg_dists):
    return np.trace(egg_dists @ egg_dists.T)

def total_dist(eggs1, eggs2):
    return total_dist(egg_dist(eggs1, eggs2))

def plot_eggs(eggvals, eggvecs):
    biggest = np.amax(eggvals)*2
    xdim, ydim = eggvals.shape[0], eggvals.shape[1]
    ellipses = []
    for i in range(xdim):
        for j in range(ydim):
            width = eggvals[i,j,0]/biggest
            height = eggvals[i,j,1]/biggest
            angle = np.arccos(eggvecs[i,j,0,0])
            ellipses.append(pat.Ellipse((i+0.5,j+0.5), width, height, angle, fill=False))
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    for e in ellipses:
        ax.add_artist(e)
    ax.set_xlim(0, xdim+1)
    ax.set_ylim(0, ydim+1)        
    plt.savefig('test')


def plot_egg_dists(egg_dists):
    pass



from pylab import imshow, show, get_cmap
from numpy import random
from scipy import misc
print('getting raccoon from internet')
f = misc.face()
print(f.shape)
#misc.imsave('test4.jpg', f)
#plt.imshow(f)
#plt.close()
#plt.show()
print("converting to eggs")
eggvals, eggvecs = to_eggs(f,dim1=15,dim2=15)
print("plotting")
plot_eggs(eggvals, eggvecs)








