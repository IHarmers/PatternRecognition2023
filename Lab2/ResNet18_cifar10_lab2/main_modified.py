'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pandas as pd
import numpy as np
from torchsummary import summary
from models.ourmodel import *

rootpath = "/content/gdrive/My Drive/Universiteit/PatternRecognition/Assignments/PR_Lab2"   # designated root path used in our runs of the code

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = ResNet18()
#net = OurModel()   # when using our model, de-comment this line and comment the previous line
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    global criterion
    
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_losses = []   # for storing training losses
    batch_accs = []   # for storing training accuracies
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        # print(outputs.shape)
        # plt.imshow(outputs[0,0,:,:].cpu().detach().numpy())
        # plt.show()
        
        loss = criterion(outputs, targets)
    
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        train_acc = 100.*correct/total
        batch_losses.append(train_loss/(batch_idx+1))      # appending the 'loss' as defined in the progress bar line below
        batch_accs.append(train_acc)
	
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), train_acc, correct, total)) 
                     
    return np.mean(batch_losses), np.mean(batch_accs)   # saving mean training loss and accuracy of each epoch


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    batch_losses = []   # for storing test losses
    batch_accs = []   # for storing test accuracies
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            test_acc = 100.*correct/total
            batch_losses.append(test_loss/(batch_idx+1))   # appending the 'loss' as defined in the progress bar line below
            batch_accs.append(test_acc)

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), test_acc, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        #torch.save(state, './checkpoint/ckpt.pth')
        torch.save(state, './checkpoint/ckpt.pth')

        best_acc = acc
        

    return np.mean(batch_losses), np.mean(batch_accs)   # saving mean test loss and accuracy of each epoch

def main():
    train_losses = []   # for storing mean training losses
    train_accs = []   # for storing mean training accuracies
    test_losses = []   # for storing mean test losses
    test_accs = []   # for storing mean test accuracies
    
    end_epoch = start_epoch+80
    for epoch in range(start_epoch, end_epoch):
        
        # Saving the mean training and test loss & accuracy metrics of each epoch.
        train_loss, train_acc = train(epoch)
        test_loss, test_acc = test(epoch)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        #scheduler.step()
    
    # Writing the results to a .csv file in Google Drive.
    # Note that your Google Drive must be mounted to the colab notebook in which this script will be running for this to work.
    df = pd.DataFrame(list(zip(*[list(range(start_epoch, end_epoch)), train_losses, train_accs, test_losses, test_accs])))
    df.columns = ['Epoch', 'Training Loss', 'Training Accuracy', 'Test Loss', 'Test Accuracy']
    df.to_csv(rootpath+'/Results/results_Q1.csv', encoding='utf-8', index=False)
        

def feature_maps(nr_images=3):
    """ Displaying the feature maps after each convolution block for the number of images specified by means of forward hooks, based on the discussion in [3]. """
    
    activation = {}   # for storing the feature/activation map after each block 
    
    # This function detaches the feature maps after each convolution block in the model via a so-called 'forward hook', as shown in the subsequent code lines, and stores it in a dictionary.
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    model = ResNet18()   # initializing ResNet18
    # Registering forward hooks on each block.
    model.conv1.register_forward_hook(get_activation('conv1'))
    model.layer1.register_forward_hook(get_activation('layer1'))
    model.layer2.register_forward_hook(get_activation('layer2'))
    model.layer3.register_forward_hook(get_activation('layer3'))
    model.layer4.register_forward_hook(get_activation('layer4'))
    # Names of the layers in the ResNet18 model.
    layers = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']  
    
    # This for loop iterates over the number of images and plots the feature maps after each block in a 5 by nr_images 'subplot matrix'.
    fig, axs = plt.subplots(nr_images, 5, figsize=[20, 20])
    for i in range(0, nr_images):
        data, _ = trainset[i]   # loading one training image
        data.unsqueeze_(0)
        output = model(data)   # passing image through network
        
        # This for loop plots the first feature map from each layer.      
        for k in range(0, len(layers)):
            act = activation[layers[k]].squeeze()
            img = axs[i, k].imshow(act[0])   # displaying only one feature map
            
            # Code lines from the matplotlib 'Simple Colorbar' tutorial to make the colorbar's height align with the subplot's height
            divider = make_axes_locatable(axs[i, k])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(img, cax=cax)
            
            axs[i, k].set_title(f"Image {i+1}: {layers[k]} Feature Map Shape: {np.shape(act[0])}", fontsize=8)   # setting title
    
    plt.savefig(rootpath+"/Images/Figure_Q2B2_image.png", bbox_inches="tight")   # saving image
    #plt.savefig(rootpath+"/Images/Figure_Q2B2_image.png")   # saving image
    
def model_summary(model):
    """Prints summary of parsed model."""
    summary(model.cuda(), (3, 32, 32))

if __name__ == "__main__":
    
    # In our runs, we have used these three functions to obtain our results.
    
    #main()
    feature_maps()
    #model_summary(OurModel())
