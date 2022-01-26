"""
Main training function.
"""
import torch
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from time import time
from datetime import datetime

#torch.manual_seed(42)
#torch.cuda.manual_seed(42)

def training2(epochs, model, criterion, optimizer, train_loader, val_loader, model_name):
    '''
    Train the model and validate it

    @param epochs : Number of epochs to train on 
    @param model : Type of model to train (UNet, ResNet, etc)
    @param criterions : Loss to train on (BCE, etc)
    @optimizer : Type of optimizer (SGD, Adam)
    '''
    print('-------------')
    print(f'Starting training process for {epochs} epochs : ')
    print('-------------')
    
    # Log gradients and model parameters
    #wandb.watch(model)

    best_score = 0
    best_model = model
    training_losses = []
    validation_losses = []

    for epoch in range(epochs):
        start = time()
        model.train()
        epoch_score = 0
        batch_id = 1
        
        for images, ground_truths in train_loader:
            #percentage = batch_id / 608 * 100
            #batch_id += 1
            #print("Progress : {:.2f} %% ".format(percentage))

            if torch.cuda.is_available():
                model, images, ground_truths = model.cuda(), images.cuda(), ground_truths.cuda()
            #print("here")
            output = model(images)
            loss = criterion(output, ground_truths)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            
            # Log metrics to visualize performance
            #wandb.log(﻿{﻿'Train Loss'﻿: train_loss/train_total, 'Train Accuracy'﻿: acc}﻿)

        training_losses.append(loss.item())
        # Validation score
        model.eval()
        scores = []
        accuracies = []
        val_losses = []
        for val_images, val_gd_truths in val_loader:
            val_output = model(val_images.cuda())
            val_losses.append(criterion(val_output, val_gd_truths.cuda()).item())
            a = np.rint(val_output.squeeze().ravel().data.cpu().numpy())
            b = val_gd_truths.squeeze().ravel().data.cpu().numpy()
            scores.append(f1_score(a, b, average='weighted'))
            accuracies.append(f1_score(a, b, average='micro'))
        print(f'')
        validation_losses.append(np.array(val_losses).mean())
        epoch_score = np.array(scores).mean()
        if epoch_score > best_score:
            best_score = epoch_score
            best_model = model
        t=time()-start
        print(f'Epoch {epoch} <{t:3.02}> : Training loss = {loss.item():.03} - Validation loss = {val_losses[-1]:.03} --- F1 score = {epoch_score:.04} - acc = {(np.array(accuracies).mean()):.04} ')
        # Save best model with date every 10 epochs
        if epoch>0 and (epoch+1)%10:
            model_name_ = f'{model_name}_epoch{epoch}.pth'
            torch.save(model.state_dict(), model_name_)

    return best_model, training_losses, validation_losses, best_score
