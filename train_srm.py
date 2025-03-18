import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
import os
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt

from time import time
from models.main_model import main_file
from models.conditioning_network import conditioning_network

from utils.plot import error_bar,plot_std, train_test_error
from utils.plot_samples import save_samples
from utils.load_data import load_data

from args import args, device

import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch
import os
import numpy as np
from torchvision import transforms
from models.CombinedDataset import CombinedDataset
from models.s_t_network import convolution_network, fully_connected


def main():
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Example usage
    image_dir = 'srm_data/x/'
    timeseries_dir = 'srm_data/y/'

    combined_dataset = CombinedDataset(image_dir, timeseries_dir, transform=data_transforms)
    train_loader = DataLoader(combined_dataset, batch_size=2, shuffle=True)
    test_loader = train_loader # dummy
    sample_loader = train_loader # dummy

    # Iterate through the DataLoader
    for images, timeseries in train_loader:
        print(f"x batch shape: {images.shape}")
        print(f"y batch shape: {timeseries.shape}")

    # Scale and Shift networks
    network_s_t = convolution_network(args.hidden_layer_channel)     
    network_s_t2 = convolution_network(args.hidden_layer_channel2)
    network_s_t3 = fully_connected(args.hidden_layer3)

    #load networks
    INN_network = main_file(network_s_t, network_s_t2, network_s_t3).to(device)
    cond_network = conditioning_network().to(device)

    # Set up Adam optimizer
    combine_parameters = [parameters_net for parameters_net in INN_network.parameters() if parameters_net.requires_grad]
    for parameters_net in combine_parameters:
        parameters_net.data = 0.02 * torch.randn_like(parameters_net)
    combine_parameters += list(cond_network.parameters())
    optimizer = torch.optim.Adam(combine_parameters, lr=args.lr, weight_decay=args.weight_decay)

    def train(N_epochs, max_epochs):
        INN_network.train()
        cond_network.train()
        loss_mean = []
        for i, (x, y) in enumerate(train_loader):
            print(f"{i} of {len(train_loader)}, epoch: {N_epochs} of {max_epochs}")

            # DEBUG
            #tensor_2d_x = x[0].reshape((1250, 1250))
            #tensor_2d_y = y[0].reshape((1, 25))
            #vutils.save_image(tensor_2d_x, f"input_images/test_x_{i}_{N_epochs}.jpg")
            #utils.save_image(tensor_2d_y, f"input_images/test_y_{i}_{N_epochs}.jpg")

            x, y = x.to(device), y.to(device)

            # Unflattens the [1x4096] into [64x64]
            x = x.view(2,1,1024,1024)

            # for config_1  change this to y = y.view(16,2,64)
            # Ensures Y is [4 x 64]
            y = y.view(2,1,128,128)

            tic = time()

            # Gives results from running forward through Conditioning Network
            y1 = cond_network(y)
            c = y1[2]   
            c2 = y1[1]
            c3 = y1[0]
            c4 = y1[3]

            # Gives Z, result from running forward through INN network
            z,log_j = INN_network(x,c,c2,c3,c4,forward=True)

            # Find Loss and Backpropagate
            loss = torch.mean(z**2) / 2 - torch.mean(log_j) / ( 1 * 64 * 64)
            loss.backward()      
            loss_mean.append(loss.item())
            optimizer.step()
            optimizer.zero_grad()

        loss_mean1 = loss_mean
        return loss_mean1

    def test(epoch):
        INN_network.eval()
        cond_network.eval()
        loss_mean = []
        for i, (x, y) in enumerate(test_loader):            
            x, y = x.to(device), y.to(device)

            # Unflattens the [1x4096] into [64x64]
            x = x.view(2,1,1024,1024)

            # for config_1  change this to y = y.view(16,2,64)
            # Ensures Y is [4 x 64]
            y = y.view(2,1,128,128)

            # Run Y forward through conditioning network
            y1 = cond_network(y)
            c = y1[2]   
            c2 = y1[1]
            c3 = y1[0]
            c4 = y1[3]

            # Get Z and log of determinant from running forward through INN model
            z,log_j = INN_network(x,c,c2,c3,c4,forward=True)
            loss_val = torch.mean(z**2) / 2 - torch.mean(log_j) /( 1 * 64 * 64)
            loss_mean.append(loss_val.item())

        loss_mean1 = loss_mean
        return loss_mean1

    def sample2(epoch):
        INN_network.eval()
        cond_network.eval()
        loss_mean = []
        for batch_idx, (input, target) in enumerate(sample_loader):
            input, target = input.to(device), target.to(device)

            # Convert X to [64x64] from [1x4096]
            # Ensure Y is [4x64]
            input, target = input.view(2,1,1024,1024), target.view(2,1,128,128)# for config_1  change this to target = target.view(16,2,64)
            x = input.view(2,1,1024,1024)
            y = target.view(2,1,128,128) # for config_1  change this to y = target.view(16,2,64)

            labels_test = target    
            N_samples = 1000

            labels_test = labels_test[0,:,:]
            labels_test = labels_test.cpu().data.numpy()
            l = np.repeat(np.array(labels_test)[np.newaxis,:,:], N_samples, axis=0)
            l = torch.Tensor(l).to(device)

            # Sample randomly from Z
            z = torch.randn(N_samples,4096).to(device)
            with torch.no_grad():
                # Run labels forward through conditioning network
                y1 = cond_network(l)
                input = x.view(1,4096)
                c = y1[2]   
                c2 = y1[1]
                c3 = y1[0]
                c4 = y1[3]

                # This time, input Z to network, Forward = False
                val = INN_network(z,c,c2,c3,c4,forward=False)
            rev_x = val.cpu().data.numpy()

            if epoch % 10 == 0:
                input_test = input[0,:].cpu().data.numpy()
                input1 = input_test.reshape(1,1,64,64)
                samples1 = rev_x
                samples12 = samples1
                mean_samples1 = np.mean(samples1,axis=0)
                mean_samples1 = mean_samples1.reshape(1,1,64,64)
                samples1 = samples1[:2,:,:,:]
                x1 = np.concatenate((input1,mean_samples1,samples1),axis=0)
                save_dir = '.'
                save_samples(save_dir, x1, epoch, 2, 'sample', nrow=2, heatmap=True, cmap='jet')
                std_sample = np.std(samples12,axis=0)
                std_sample = std_sample.reshape(64,64)

                actual = input1
                pred = rev_x
                error_bar(actual,pred,epoch)
                io.savemat('./results/samples_%d.mat'%epoch, dict([('rev_x_%d'%epoch,np.array(rev_x))]))
                io.savemat('./results/input_%d.mat'%epoch, dict([('pos_test_%d'%epoch,np.array(input_test))]))
            if epoch == (args.epochs-1):
                std_sample = np.std(rev_x,axis=0)
                std_sample = std_sample.reshape(64,64)
                plot_std(std_sample,epoch)


    #==========================================================
    def mkdir(path):
        if not os.path.exists(path):
            os.makedirs(path)
    #==========================================================


    print('training start .............')
    mkdir('results')
    N_epochs = 102
    loss_train_all = []
    loss_test_all = []
    tic = time()


    for epoch in range(args.epochs):
        print('epoch number .......',epoch)
        loss_train = train(epoch, args.epochs)
        loss_train2 = np.mean(loss_train)
        loss_train_all.append(loss_train2)
        with torch.no_grad():
            #sample2(epoch)
            loss_test = test(epoch)
            loss_test = np.mean(loss_test)
            print(('mean NLL :',loss_test))
            loss_test_all.append(loss_test)

    epoch1 = 200
    torch.save(INN_network.state_dict(), f'INN_network_epoch{epoch1}.pt')
    torch.save(cond_network.state_dict(), f'cond_network_epoch{epoch1}.pt')

    loss_train_all = np.array(loss_train_all)
    loss_test_all = np.array(loss_test_all)

    print('saving the training error and testing error')
    io.savemat('test_loss.mat', dict([('testing_loss',np.array(loss_test_all))]))
    print('plotting the training error and testing error')

    train_test_error(loss_train_all,loss_test_all, epoch1)

    toc = time()
    print('total traning taken:', toc-tic)

if __name__ == "__main__":
    main()