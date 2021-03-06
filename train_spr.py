"""
Scripty to train an network using the VIN approach.
Note that this script takes in the terminal position as one of the inputs.
"""
import os
import time
import argparse
import numpy as np
np.random.seed(0)
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from IPython import embed
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from dataset.dataset_spr import SprInstructions
from vin_utility.utils import *
from models.model_spr import SprVin
from models.lookup_model import LookupModel
from models.text_model import TextModel
from models.attention_heatmap import AttentionHeatmap
from test_spr import test_manhattan

def train(net, trainloader, config, criterion, optimizer, use_GPU):
    print_header()
    for epoch in range(config.epochs): # Loop over dataset multiple times
        avg_error, avg_loss, num_batches = 0.0, 0.0, 0.0
        start_time = time.time()
        for i, data in enumerate(trainloader): # Loop over batches of data
            # Get input batch
            layouts, object_maps, inst, S1, S2, goals, labels = data

            if layouts.size()[0] != config.batch_size:
                continue # Drop those data, if not enough for a batch
            # Send Tensors to GPU if available
            if use_GPU:
                layouts = layouts.cuda()
                object_maps = object_maps.cuda()
                inst = inst.cuda()
                S1 = S1.cuda()
                S2 = S2.cuda()
                labels = labels.cuda()  
            # Wrap to autograd.Variable
            layouts, object_maps = Variable(layouts), Variable(object_maps)
            S1, S2, labels = Variable(S1), Variable(S2), Variable(labels)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs, predictions, v, r, heatmap = net(layouts, object_maps, inst, S1, S2, config)
            # Loss
            loss = criterion(outputs, labels)
            # Backward pass
            loss.backward()
            # Update params
            optimizer.step()
            # Calculate Loss and Error
            loss_batch, error_batch = get_stats(loss, predictions, labels)
            avg_loss += loss_batch
            avg_error += error_batch
            num_batches += 1  
        time_duration = time.time() - start_time
        # Print epoch logs
        print_stats(epoch, avg_loss, avg_error, num_batches, time_duration)
    print('\nFinished training. \n')


def test(net, testloader, config):  
    total, correct = 0.0, 0.0
    for i, data in enumerate(testloader):
        layouts, object_maps, inst, S1, S2, goals, labels = data
        if layouts.size()[0] != config.batch_size:
            continue # Drop those data, if not enough for a batch
        # Send Tensors to GPU if available
        if use_GPU:
            layouts = layouts.cuda()
            object_maps = object_maps.cuda()
            inst = inst.cuda()
            S1 = S1.cuda()
            S2 = S2.cuda()
            labels = labels.cuda()  
        # Wrap to autograd.Variable
        layouts, object_maps = Variable(layouts), Variable(object_maps)
        S1, S2 = Variable(S1), Variable(S2)
        inst = Variable(inst)
        # Forward pass
        outputs, predictions = net(layouts, object_maps, inst, S1, S2, config)
        # Select actions with max scores(logits)
        _, predicted = torch.max(outputs, dim=1, keepdim=True)
        # Unwrap autograd.Variable to Tensor
        predicted = predicted.data
        # Compute test accuracy
        # embed()
        correct += (predicted.view(-1) == labels).sum()
        total += labels.size()[0]
        
        # break
    print('Test Accuracy: {:.2f}%'.format(100*(correct/total)))


if __name__ == '__main__':
    # Automatic swith of GPU mode if available
    use_GPU = torch.cuda.is_available()
    # Parsing training parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', 
                        type=str, 
                        default='spr_data/', 
                        help='Path to data file')
    parser.add_argument('--imsize', 
                        type=int, 
                        default=10, 
                        help='Size of image')
    parser.add_argument('--lr', 
                        type=float, 
                        default=0.002, 
                        help='Learning rate, [0.01, 0.005, 0.002, 0.001]')
    parser.add_argument('--epochs', 
                        type=int, 
                        default=25, 
                        help='Number of epochs to train')
    parser.add_argument('--k', 
                        type=int, 
                        default=10, 
                        help='Number of Value Iterations')
    parser.add_argument('--l_i', 
                        type=int, 
                        default=2, 
                        help='Number of channels in input layer')
    parser.add_argument('--l_h', 
                        type=int, 
                        default=150, 
                        help='Number of channels in first hidden layer')
    parser.add_argument('--l_q', 
                        type=int, 
                        default=10, 
                        help='Number of channels in q layer (~actions) in VI-module')
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=128, 
                        help='Batch size')
    parser.add_argument('--save_dir', 
                        type=str, 
                        default='trained_models/', 
                        help='Directory to save models')
    parser.add_argument('--data_scale', 
                        type=int, 
                        default=25, 
                        help='Data point replication')
    parser.add_argument('--name_header', 
                        type=str, 
                        default="spr_vin", 
                        help='Data point replication')
    parser.add_argument('--map_type', 
                        type=str, 
                        default="local", 
                        help='Global or local maps')
    config = parser.parse_args()
    # Get path to save trained model

    if not os.path.isdir(config.save_dir):
        os.mkdir(config.save_dir)
    save_name = "{}_{}_{}epcs_{}its_{}datascale.pth".format(config.name_header, config.map_type, config.epochs,
                                                         config.k, config.data_scale)
    save_path = os.path.join(config.save_dir, save_name)
    print "Model will be saved as {}".format(save_path)

    #Load Spatial reasoning dataset
    map_type=config.map_type
    trainset = SprInstructions(config.datadir, mode=map_type,
                               annotation='human', train=True,
                               scale=config.data_scale)
    testset = SprInstructions(config.datadir, mode=map_type,
                              annotation='human', train=False, scale=1)
    
    # Create Dataloader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)

    #create the model
    objects = trainset.object_vocab_size
    text_vocab_size = trainset.text_vocab_size
    print "Vocab Sizes:\nObjects:%s\nText:%s\n" %(objects, text_vocab_size)
    obj_embed = 7
    object_model = LookupModel(objects, embed_dim=obj_embed)

    #text lstm model
    attn_kernel = 3
    attn_in = obj_embed
    attn_out = 1 # no. of heatmap channels
    lstm_out = (attn_kernel**2) * attn_in * attn_out
    instruction_model = TextModel(text_vocab_size, ninp=15,
                                nhid=30, nlayers=1,
                                out_dim=lstm_out)
    heatmap_model = AttentionHeatmap(instruction_model,
                                    attn_kernel,
                                    attn_in,
                                    attn_out)
    spr_model = SprVin(config, heatmap_model, object_model)


    
    

    # Use GPU if available
    if use_GPU:
         spr_model = spr_model.cuda() 
    # Loss
    criterion = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = optim.RMSprop(spr_model.parameters(), lr=config.lr, eps=1e-6)
    # Dataset transformer: torchvision.transforms
    transform = None
    
    # Train the model
    train(spr_model, trainloader, config, criterion, optimizer, use_GPU)
    # Test accuracy
    # test(spr_model, testloader, config)
    report_path = save_path.replace('.pth', '.txt')
    test_manhattan(spr_model, testloader, config, report_path)
    # Save the trained model
    torch.save(spr_model, save_path)
