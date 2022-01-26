"""
Defines classes to handle datasets
"""

import torch
import numpy as np
from torch.utils.data import Dataset
import sys
import os
from helpers.prepare_data import prepare_patches, prepare_test_patches
import torch

LABELING_THRESHOLD = 128

class RoadDataset(Dataset):
    """
    Class to store the training road dataset and load them with he DataLoader
    """
    def __init__(self, images_path, gt_path, indices, patch_size, nb_images, overlap = 0, augmentation=True):
        # Store the images and groundtruth
        self.images, self.gd_truth = prepare_patches(images_path, gt_path, indices, patch_size, overlap, augmentation=augmentation)

    def __len__(self): # Function to use the DataLoader
        return len(self.images)

    def __getitem__(self, idx): # Function to use the DataLoader to get one batch of images
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        patched_image = np.transpose(self.images[idx], (2, 0, 1))
        patched_image = torch.from_numpy(patched_image)

        gt = torch.from_numpy(self.gd_truth[idx]*255).float().unsqueeze(0)
        # Round up to 1 or down to 0 
        gt[gt < 128] = 0
        gt[gt >= 128] = 1  

        return patched_image.float(), gt
        

class TestSet(Dataset):
    """
    Class to store the training road dataset and load them with he DataLoader
    """
    def __init__(self, images_path, indices, patch_size):
        self.images = prepare_test_patches(images_path, indices, patch_size)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        patched_image = np.transpose(self.images[idx], (2, 0, 1))
        patched_image = torch.from_numpy(patched_image)

        return patched_image.float()


'''class PostProcessing(Dataset):

    def __init__(self, model_outputs, ground_truths):
        self.images, self.gd_truth = model_outputs.cpu().detach().numpy(), ground_truths
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        patched_image = torch.from_numpy(self.images[idx]).unsqueeze(0)

        gt = torch.from_numpy(self.gd_truth[idx]*255).float().unsqueeze(0)
        gt[gt < 128] = 0
        gt[gt >= 128] = 1  

        return patched_image.float(), gt'''