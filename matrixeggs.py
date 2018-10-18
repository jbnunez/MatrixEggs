#matrixeggs.py
import numpy as np
import numpy.linalg as lin 
import matplotlib as mplt 
import matplotlib.pyplot as plt 
import matplotlib.patches as pat
import scipy.ndimage as nd
from pylab import imshow, show, get_cmap
import cv2

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

def split_image(image):
    l, w = image.shape[0], image.shape[1]
    image_l = image[:, :(w//2), :]
    image_r = image[:, (w//2):, :]
    return(image_l, image_r)


def plot_egg_dists(egg_dists):
    pass


def dist_heat_map(image1, image2, dim1=15, dim2=15, egg_dim=2):
    egg1 = to_eggs(image1, dim1, dim2, egg_dim)
    egg2 = to_eggs(image2, dim1, dim2, egg_dim)
    distances = egg_dist(egg1, egg2)
    imshow(distances, cmap=get_cmap("Spectral"), interpolation='nearest')
    show()

    
def spot_the_difference(image_name ='spot.png', dim1=15, dim2=15, egg_dim=2):
    image = nd.imread(image_name)
    left, right = split_image(image)
    dist_heat_map(left, right, dim1, dim2, egg_dim)

def extract_frames(video_file='test_video.mp4'):
    '''
    Takes in the name of a video file and outputs
    a list of each frame
    '''
    images = []
    vidcap = cv2.VideoCapture(video_file)
    success,image = vidcap.read()
    success = True
    count = 0
    while success:
        images.append(image)
        success,image = vidcap.read()
        # print('Read a new frame: ' + str(success) + ", " + str(count))
        count += 1
    return images

def dist_heat_map_video(video_file='test_video.mp4', dim1=20, dim2=20, egg_dim=2):
    print("Beginning extraction...")
    images = extract_frames(video_file)
    print("Done with extraction.")
    eggs = []
    print("Converting to eggs...")
    for image in images:
        eggs.append(to_eggs(image, dim1, dim2, egg_dim))
    print("Getting distances...")
    distances = []
    for i in range(len(eggs)-1):
        distances.append(egg_dist(eggs[i], eggs[i+1]))
    return distances

def make_video(frames):
    pass










