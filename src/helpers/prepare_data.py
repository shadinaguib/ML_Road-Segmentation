import numpy as np
import torch 
import os, sys
import matplotlib.image as mpimg
from helpers.Utility import mirror_exp, augment_image
#from Utility import augment_image

def img_crop(im, w, h, overlap):
    '''
    Extract patches of w*h with overlap from original image
    '''
    # Extract patches from a given image
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight-h+1, h-overlap):
        for j in range(0, imgwidth-w+1, w-overlap):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches


def extract_images(filename, indices, IMG_PATCH_SIZE, overlap, augmentation):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    """
    imgs = []
    for index in indices:
        imageid = f"satImage_{index:03}"
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            if augmentation:
                augmented_images = augment_image(image_filename)
                for img in augmented_images:
                    len_ratio = len(img)//IMG_PATCH_SIZE
                    a = (len_ratio+1)*IMG_PATCH_SIZE - len(img)# - len(img)%IMG_PATCH_SIZE
                    if len(img) % IMG_PATCH_SIZE:
                        img = mirror_exp(img, a)
                    imgs.append(img)
            else:
                img = mpimg.imread(image_filename)
                len_ratio = len(img)//IMG_PATCH_SIZE
                a = (len_ratio+1)*IMG_PATCH_SIZE - len(img)# - len(img)%IMG_PATCH_SIZE
                if len(img) % IMG_PATCH_SIZE:
                    img = mirror_exp(img, a)
                imgs.append(img)   

        else:
            print('File ' + image_filename + ' does not exist')

    num_images = len(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]
    N_PATCHES_PER_IMAGE = (IMG_WIDTH/IMG_PATCH_SIZE)*(IMG_HEIGHT/IMG_PATCH_SIZE)

    img_patches = [img_crop(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE, overlap) for i in range(num_images)]
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]

    print('Loaded images')
    return np.array(data)


def extract_test_images(filename, indices, IMG_PATCH_SIZE, overlap=0):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    """
    imgs = []
    for index in indices:
        imageid = f"test_{index}/test_{index}"
        #imageid = f'satImage_{index:03}'
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            img = mpimg.imread(image_filename)
            len_ratio = len(img)//IMG_PATCH_SIZE
            a = (len_ratio+1)*IMG_PATCH_SIZE - len(img)
            if len(img) % IMG_PATCH_SIZE:
                img = mirror_exp(img, a)
            imgs.append(img)
        else:
            print('File ' + image_filename + ' does not exist')

    num_images = len(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]
    N_PATCHES_PER_IMAGE = (IMG_WIDTH/IMG_PATCH_SIZE)*(IMG_HEIGHT/IMG_PATCH_SIZE)

    img_patches = [img_crop(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE, overlap) for i in range(num_images)]
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]

    #print('Loaded images')
    return np.array(data)

# Extract label images
def extract_labels(filename, indices, IMG_PATCH_SIZE, overlap, augmentation):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = []
    for index in indices:
        imageid = f"satImage_{index:03}"
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            #print('Loading ' + image_filename)
            #img = mpimg.imread(image_filename)
            if augmentation:
                augmented_images = augment_image(image_filename)
                for img in augmented_images:
                    len_ratio = len(img)//IMG_PATCH_SIZE
                    a = (len_ratio+1)*IMG_PATCH_SIZE - len(img)
                    if len(img) % IMG_PATCH_SIZE:
                        img = mirror_exp(img, a)
                    gt_imgs.append(img)
            else:
                img = mpimg.imread(image_filename)
                len_ratio = len(img)//IMG_PATCH_SIZE
                a = (len_ratio+1)*IMG_PATCH_SIZE - len(img)
                if len(img) % IMG_PATCH_SIZE:
                    img = mirror_exp(img, a)
                gt_imgs.append(img)
        else:
            print('File ' + image_filename + ' does not exist')
    #gt_imgs = np.asarray(gt_imgs)
    num_images = len(gt_imgs)
    gt_patches = [img_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE, overlap) for i in range(num_images)]
    labels = [gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))]
    #labels = data #np.asarray([value_to_class(np.mean(data[i])) for i in range(len(data))])
    print('Loaded labels')
    # Convert to dense 1-hot representation.
    return labels

def prepare_patches(images_path, gt_path, indices, IMG_PATCH_SIZE, overlap, augmentation):
    images = extract_images(images_path, indices, IMG_PATCH_SIZE, overlap, augmentation)
    gts = extract_labels(gt_path, indices, IMG_PATCH_SIZE, overlap, augmentation)

    return images, gts

def prepare_test_patches(images_path, indices, IMG_PATCH_SIZE):
    images = extract_test_images(images_path, indices, IMG_PATCH_SIZE)

    return images

def labels_to_patches(image, img_size, p_size, threshold):
    """
    Transform pixel-wise label image, to patch-wise label image.
    """
    patch_prediction_values = torch.zeros(1444)
    if torch.cuda.is_available():
        patch_prediction_values = patch_prediction_values.cuda()

    index=0
    for i in range(0, img_size, p_size):
        for j in range(0, img_size, p_size):    
            mean = torch.mean(image[j : j+p_size, i : i+p_size])
            if mean > threshold: 
                patch_prediction_values[index] = 1
            index += 1
    return patch_prediction_values