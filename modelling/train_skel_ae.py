curr_dir = '/home/vayzenbe/GitHub_Repos/docnet'

import sys
sys.path.insert(1, f'{curr_dir}')
import os, argparse
from collections import OrderedDict
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision import datasets
import torchvision.models as models
from LoadFrames import LoadFrames
from torch.utils.tensorboard import SummaryWriter
import cornet
import numpy as np
import pdb
writer = SummaryWriter(f'runs/skel_ae')
#Transformations for ImageNet
transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])])
inv_normalize = transforms.Normalize(
   mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],std=[1/0.229, 1/0.224, 1/0.225])
# specify loss function
criterion = nn.MSELoss()

epochs = 10
actNum = 1024

def save_model(model, epoch, optimizer, loss, file_path):

    print('Saving model ...', epoch)
    #torch.save(model.state_dict(), f'{weights_dir}/cornet_classify_{cond}_{epoch}.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        }, file_path)

def define_decoder():
    decoder = nn.Sequential(nn.Conv2d(3,1024,kernel_size=3, stride=2), nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1), nn.AdaptiveAvgPool2d(1),nn.ReLU(), nn.ConvTranspose2d(1024, 3, 224))
    #decoder = nn.Sequential(nn.Conv2d(3,1024,kernel_size=1, stride=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),nn.ReLU(), nn.ConvTranspose2d(1024, 3, 224))
    #maybe try this with max pool instead
    decoder = decoder.cuda()
    
    return decoder


'''
Train model
'''

train_dataset = LoadFrames(f'{curr_dir}/stim/blur/', transform=transform)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers = 4, pin_memory=True)

test_dataset = LoadFrames(f'{curr_dir}/stim/blur/', transform=transform)
valloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers = 4, pin_memory=True)

#Reset decoder for every object (i.e., make it like a fresh hab session)
#Create decoder
model = define_decoder()



lr = .01 #Starting learning rate
step_size = 20 #How often (epochs)the learning rate should decrease by a factor of 10
weight_decay = 1e-4
momentum = .9
n_epochs = 30
n_save = 5 #save model every X epochs
start_epoch = 0

optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad = True,weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size)

model.train()
valid_loss_min = np.Inf # track change in validation loss
nTrain = 1
nVal = 1
for epoch in range(0, epochs):
    print('Starting training')

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    
    
    ###################
    # train the model #
    ###################
    model.train()
    for data, label in trainloader:
        # move tensors to GPU if CUDA is available
        
        data = data.cuda()

            #print('moved to cuda')
        
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        #print(output.shape, data.shape)
        # calculate the batch loss
        loss = criterion(output, data)
        
        
        writer.add_scalar("Raw Train Loss", loss, nTrain) #write to tensorboard
        writer.flush()
        nTrain = nTrain + 1
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        #print(loss, nTrain)

        #clip to prevent exploding gradients
        #nn.utils.clip_grad_norm_(model.parameters,max_norm=2.0, norm_type=2)
        
        # perform a single optimization step (parameter update)
        optimizer.step()
        
        # update training loss
        train_loss += loss.item()*data.size(0)
        #print(train_loss)
    
    #scheduler.step()

    ######################    
    # validate the model #
    ######################
    model.eval()
    accuracy = 0
    with torch.no_grad():
        for data, label in valloader:
            # move tensors to GPU if CUDA is available
            
            data = data.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, data)
            writer.add_scalar("Raw Validation Loss", loss, nVal) #write to tensorboard
            writer.flush()
            nVal = nVal + 1
            #print('wrote to tensorboard')
            # update average validation loss 
            valid_loss += loss.item()*data.size(0)

    

    writer.add_scalar("Average Train Loss", train_loss, epoch) #write to tensorboard
    writer.add_scalar("Average Validation Loss", valid_loss, epoch) #write to tensorboard
    writer.flush()

    if valid_loss < valid_loss_min:
        valid_loss_min = valid_loss
        save_model(model, epoch, optimizer, loss, f'weights/docnet_skel_model.pt')

