import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import os, sys
from models.model_UNet_beta import UNetBeta
from models.model_UNet_alpha import UNetAlpha

from helpers.prepare_data import labels_to_patches
import pandas as pd
from helpers.build_dataset import TestSet
from helpers.config import *
import datetime

# Load model from weights file
print('------ Loading model -----')
if torch.cuda.is_available():
    best_model = UNetBeta().cuda()
    best_model.load_state_dict(torch.load('models/Final_UNetBeta_patch80_depth256_batch64_epochs50_0.96308F1.pth'))
else:
    best_model = UNetBeta()
    best_model.load_state_dict(torch.load('models/Final_UNetBeta_patch80_depth256_batch64_epochs50_0.96308F1.pth', map_location='cpu'))
print('------ Done loading model -----')

# Build test set and test loader
print('------ Loading dataset -----')
test_indices = np.arange(1, TESTING_SIZE + 1)
test_set = TestSet(TEST_IMG_PATH, test_indices, 608)
test_loader = DataLoader(test_set, 1, shuffle=False)
print('------ Done loading dataset -----')

# Test model
index = 0
output_predictions = torch.zeros(1444*50)#.cuda()
print('------ Evaluating Model on test set -----')
if torch.cuda.is_available():
    output_predictions = output_predictions.cuda()
i = 0
for batch in test_loader: # a batch corresponds to all patches from a single image
    if torch.cuda.is_available():
        batch = batch.cuda()
    output = best_model(batch).squeeze()
    output_predictions[index:index+1444] = labels_to_patches(output, 608, p_size=16, threshold=0.2)
    index += 1444
# Create submission file
ids = []
for i in range(1, TESTING_SIZE+1):
    for j in range(0, 608, 16):
        for k in range(0, 608, 16):
            ids.append(f'{i:03}_{j}_{k}')

dic = {'id': ids, 
        'prediction': output_predictions.cpu().numpy()}
filename = datetime.datetime.now().strftime('%d-%m-%y-%H_%M_submission.csv')
print('------ All done! Submitting to csv file -----')
pd.DataFrame(data = dic).to_csv(filename, index=False)
