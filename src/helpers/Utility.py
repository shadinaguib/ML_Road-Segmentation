"""
Utility functions for data augmentation
"""


import numpy as np
import matplotlib.image as mpimg
from imgaug import augmenters as iaa



# Helper functions

def load_image(infilename):
    '''
    Load image from file path.
    --------
    In : infilename : String with filepath
    Out : data : loaded image in array
    '''
    data = mpimg.imread(infilename)
    return data

def augment_image(img_path):
    '''
    Given a input path to the image, return array of augmented images. 
    -----------
    In : String with filepath to image
    Out: Array of images with original image, horizontal and vertical flipped image, and rotations of the image      
    '''
    rot_ang = [-60, -45, 15, 45, 60, 90, 180]
    img_ = []
    im = load_image(img_path)
    img_.append(im)
    img_.append(np.fliplr(im))
    img_.append(np.flipud(im))
    for ang in rot_ang :
        transf = iaa.Affine(rotate=ang, mode = 'reflect')
        img_.append(transf.augment_image(im))
    return img_
    
def mirror_exp(img, exp):
    '''
    Makes an expansion of the image by mirron with the 'exp' value
    -------
    In : Image and value to expand
    Out : Expanded Image
    '''
    
    w, h = len(img), len(img[0])
    
    assert (w >= exp) and (h >= exp)
    
    if (len(img.shape) < 3):
        right_exp = np.concatenate((img,np.fliplr(img[:, (w - exp):])), axis=1)
    else:
        right_exp = np.concatenate((img, np.fliplr(img[:, (w - exp):, :])), axis=1)

    if (len(img.shape) < 3):
        img_exp = np.concatenate((right_exp, np.flipud(right_exp[(h - exp):, :])), axis=0)
    else:
        img_exp = np.concatenate((right_exp, np.flipud(right_exp[(h - exp):, :, :])), axis=0)
    
    return img_exp