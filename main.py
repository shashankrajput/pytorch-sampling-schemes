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

from torch.utils.data import RandomSampler
import sys
import time
import random

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')

parser.add_argument('--sampling', default=None, type=str, help='sampling type, one of RR, SS, or SGD')
parser.add_argument('--model', default=None, type=str, help='model, one of VGG11, VGG13, VGG16, or VGG19')
parser.add_argument('--batchnorm', default=None, type=str, help='batchnorm, either true or false')
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

class RandomSamplerSS(RandomSampler):
    def __init__(self, train_dataset):
        super().__init__(train_dataset)
        self.epoch=1
        self.permutation=None
    def __iter__(self):
        if self.epoch==1:
            n = len(self.data_source)
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
            self.permutation = torch.randperm(n, generator=generator).tolist()
            yield from self.permutation
        else:
            # self.permutation.reverse()
            yield from self.permutation
        self.epoch=self.epoch+1

class RandomSamplerSS_classmix(RandomSampler):
    def __init__(self, train_dataset):
        super().__init__(train_dataset)
        self.epoch=1
        self.permutation=None
    def __iter__(self):
        if self.epoch==1:
            class_indices={}
            for (index, (_, label)) in enumerate(self.data_source):
                if label not in class_indices:
                    class_indices[label]=[]
                class_indices[label].append(index)
            
            for label in class_indices:
                random.shuffle(class_indices[label])
            
            self.permutation=[]
            num_classes=len(class_indices)
            for i in range(len(self.data_source)):
                self.permutation.append(class_indices[i%num_classes][int(i/num_classes)])
            yield from self.permutation
        else:
            # self.permutation.reverse()
            yield from self.permutation
        self.epoch=self.epoch+1


trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)

custom_sampler = None
sampling = args.sampling
if sampling=="SGD":
    custom_sampler = RandomSampler(data_source=trainset, replacement=True)
elif sampling=="SS":
    custom_sampler = RandomSamplerSS(trainset)
elif sampling=="SS_mix":
    custom_sampler = RandomSamplerSS_classmix(trainset)
elif sampling=="RR":
    custom_sampler = RandomSampler(data_source=trainset)
else:
    raise ValueError("--sampling argument should be one of SGD, SS, or RR.")

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, sampler=custom_sampler, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model

model = args.model
if model not in ['VGG11', 'VGG13', 'VGG16', 'VGG19']:
    raise ValueError("--model argument should be one of VGG11, VGG13, VGG16 or VGG19.")

batchnorm = args.batchnorm
if batchnorm not in ['true', 'false']:
    raise ValueError("--batchnorm should be either true or false.")

batchnorm = (batchnorm=='true')
btnm='batchnorm_false'
if batchnorm:
    btnm='batchnorm_true'

print('==> Building model..')
# print(model)
# print(batchnorm)
# print(btnm)
# exit()

net = VGG(model, batchnorm)
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
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
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


filename="./results/"+model+"_"+sampling+"_"+btnm

# Training
def train(epoch, start_time):
    print('\nEpoch: %d' % epoch)
    net.train()
    # train_loss = 0
    # correct = 0
    # total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # train_loss += loss.item()
        # _, predicted = outputs.max(1)
        # total += targets.size(0)
        # correct += predicted.eq(targets).sum().item()

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    

    train_loss = 0    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            train_loss += loss.item()
    print(train_loss)
    f = open(filename, "a")
    f.write(str(epoch)+" "+str(train_loss)+" "+str(time.time()-start_time))
    f.write("\n")
    f.close()



def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

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
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

open(filename, 'w').close()
start_time=time.time()
for epoch in range(start_epoch, start_epoch+200):
    train(epoch, start_time)
    # test(epoch)
    scheduler.step()