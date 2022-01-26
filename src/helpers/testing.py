"""
Main training function.
"""
import os
import numpy as np
import matplotlib.image as mpimg
import re

foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch

# assign a label to a patch
def patch_to_label(patch):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0


import torch
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from patchify import unpatchify
#torch.manual_seed(42)
#torch.cuda.manual_seed(42)

def testing(model, test_loader):
    index = 0
    output_patches = []
    computed_images = []
    for batch in test_loader: # a batch corresponds to all patches from a single image
        output = model(batch)

        #output = np.rint(output.detach().numpy()).squeeze()
        #output_patches.append(output)
        #image = unpatchify(output.reshape(6, 6, 120, 120), (720, 720)).T
        computed_images.append(output[:608, :608])
        
    return computed_images #computed_images
        
        

