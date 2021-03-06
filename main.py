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
import pickle




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
            yield from self.permutation
        self.epoch=self.epoch+1


class RandomSamplerSS_flip(RandomSampler):
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
            self.permutation.reverse()
            yield from self.permutation
        self.epoch=self.epoch+1

class RandomSamplerSS_flip_mix(RandomSampler):
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
            self.permutation.reverse()
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
            yield from self.permutation
        self.epoch=self.epoch+1

class RandomSamplerSS_unmix(RandomSampler):
    def __init__(self, train_dataset, batch_size, num_priority_classes, priority_probability):
        super().__init__(train_dataset)
        self.epoch=1
        self.batch_size=batch_size
        self.num_priority_classes=num_priority_classes
        self.priority_probability=priority_probability
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
            class_list=list(range(num_classes))
            priority_class_list=[]
            for i in range(len(self.data_source)):
                if i%self.batch_size==0:
                    random.shuffle(class_list)
                    priority_class_list=class_list[:self.num_priority_classes]
            
                flag=False
                while (not flag):
                    chosen_class=-1
                    if (random.uniform(0,1) < self.priority_probability) and (len(priority_class_list)>0):
                        chosen_class=random.choice(priority_class_list)
                    else:
                        chosen_class=random.choice(class_list)

                    if len(class_indices[chosen_class])>0:
                        flag=True
                        self.permutation.append(class_indices[chosen_class].pop())
                    else:
                        class_list.remove(chosen_class)
                        if chosen_class in priority_class_list:
                            priority_class_list.remove(chosen_class)

            yield from self.permutation
        else:
            yield from self.permutation
        self.epoch=self.epoch+1


class RandomSamplerSS_flip_unmix(RandomSampler):
    def __init__(self, train_dataset, batch_size, num_priority_classes, priority_probability):
        super().__init__(train_dataset)
        self.epoch=1
        self.batch_size=batch_size
        self.num_priority_classes=num_priority_classes
        self.priority_probability=priority_probability
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
            class_list=list(range(num_classes))
            priority_class_list=[]
            for i in range(len(self.data_source)):
                if i%self.batch_size==0:
                    random.shuffle(class_list)
                    priority_class_list=class_list[:self.num_priority_classes]
            
                flag=False
                while (not flag):
                    chosen_class=-1
                    if (random.uniform(0,1) < self.priority_probability) and (len(priority_class_list)>0):
                        chosen_class=random.choice(priority_class_list)
                    else:
                        chosen_class=random.choice(class_list)

                    if len(class_indices[chosen_class])>0:
                        flag=True
                        self.permutation.append(class_indices[chosen_class].pop())
                    else:
                        class_list.remove(chosen_class)
                        if chosen_class in priority_class_list:
                            priority_class_list.remove(chosen_class)

            yield from self.permutation
        else:
            self.permutation.reverse()
            yield from self.permutation
        self.epoch=self.epoch+1


class RandomSamplerRR_classmix(RandomSampler):
    def __init__(self, train_dataset):
        super().__init__(train_dataset)
        self.epoch=1
        self.permutation=None
    def __iter__(self):
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
        self.epoch=self.epoch+1

class RandomSamplerSGD_classmix(RandomSampler):
    def __init__(self, train_dataset):
        super().__init__(train_dataset)
        self.epoch=1
        self.permutation=None
    def __iter__(self):
        class_indices={}
        for (index, (_, label)) in enumerate(self.data_source):
            if label not in class_indices:
                class_indices[label]=[]
            class_indices[label].append(index)
        
        for label in class_indices:
            random_indices = torch.randint(high=len(class_indices[label]), size=(len(class_indices[label]),), dtype=torch.int64, generator=self.generator).tolist()
            temp_list=[]
            for ind in random_indices:
                temp_list.append(class_indices[label][ind])
            class_indices[label]=temp_list
        
        self.permutation=[]
        num_classes=len(class_indices)
        for i in range(len(self.data_source)):
            self.permutation.append(class_indices[i%num_classes][int(i/num_classes)])
        yield from self.permutation
        self.epoch=self.epoch+1


# Training
def train(net, trainloader, device, criterion, optimizer):
    start_time=time.time()
    net.train()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    return time.time()-start_time


# Evaluation
def evaluate(net, trainloader_eval, testloader, device, criterion, optimizer):
    net.eval()

    train_loss = 0    
    test_loss = 0
    test_acc = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader_eval):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            train_loss += loss.item()
    
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            test_acc=100.0*correct/total
    

    evaluation = {}
    evaluation['Training Loss']=train_loss
    evaluation['Test Loss']=test_loss
    evaluation['Test Accuracy']=test_acc
    return evaluation






parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')

parser.add_argument('--sampling', default=None, type=str, help='sampling type, one of RR, SS, or SGD')
parser.add_argument('--model', default=None, type=str, help='model, one of VGG11, VGG13, VGG16, or VGG19')
parser.add_argument('--batchnorm', default=None, type=str, help='batchnorm, either true or false')
args = parser.parse_args()


model = args.model
if model not in ['VGG11', 'VGG13', 'VGG16', 'VGG19']:
    raise ValueError("--model argument should be one of VGG11, VGG13, VGG16, or VGG19.")

sampling = args.sampling
if sampling not in ['RR', 'SS', 'SS_flip', 'SS_flip_mix', 'SS_unmix', 'SS_flip_unmix', 'SGD', 'RR_mix', 'SS_mix', 'SGD_mix']:
    raise ValueError("--sampling argument should be one of RR, SS, SGD, RR_mix, SS_mix, or SGD_mix.")

batchnorm = args.batchnorm
if batchnorm not in ['true', 'false']:
    raise ValueError("--batchnorm should be either true or false.")

batchnorm = (batchnorm=='true')
btnm='batchnorm_false'
if batchnorm:
    btnm='batchnorm_true'




device = 'cuda' if torch.cuda.is_available() else 'cpu'


num_runs=3
num_epochs=300
num_priority_classes=10
priority_probability=0.00
batch_size=125
results={}
filename="./results_CIFAR100/"+model+"_"+sampling+"_"+btnm+"_"+str(num_priority_classes)+"_"+str(priority_probability)+"_lr"+str(args.lr)
print("Output will be written to "+filename)
with open(filename, 'w') as f:
    for run in range(num_runs):
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




        trainset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train)

        custom_sampler = None
        if sampling=="SGD":
            custom_sampler = RandomSampler(data_source=trainset, replacement=True)
        elif sampling=="SS":
            custom_sampler = RandomSamplerSS(trainset)
        elif sampling=="SS_flip":
            custom_sampler = RandomSamplerSS_flip(trainset)
        elif sampling=="SS_flip_mix":
            custom_sampler = RandomSamplerSS_flip_mix(trainset)
        elif sampling=="SS_mix":
            custom_sampler = RandomSamplerSS_classmix(trainset)
        elif sampling=="SS_unmix":
            custom_sampler = RandomSamplerSS_unmix(trainset, batch_size, num_priority_classes, priority_probability)
        elif sampling=="SS_flip_unmix":
            custom_sampler = RandomSamplerSS_flip_unmix(trainset, batch_size, num_priority_classes, priority_probability)
        elif sampling=="RR_mix":
            custom_sampler = RandomSamplerRR_classmix(trainset)
        elif sampling=="SGD_mix":
            custom_sampler = RandomSamplerSGD_classmix(trainset)
        elif sampling=="RR":
            custom_sampler = RandomSampler(data_source=trainset)
        else:
            raise ValueError("--sampling argument should be one of SGD, SS, or RR.")

        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, sampler=custom_sampler, num_workers=2)

        trainloader_eval = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=False, num_workers=2)

        testset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=2)




        print('==> Building model..')

        net = VGG(model, batchnorm)

        # model="ResNet101"
        # net = ResNet101()

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

        # model="SimpleDLA"
        # net = SimpleDLA()

        net = net.to(device)
        if device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=args.lr,
                            momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)


    
        results[run]={}
        for epoch in range(0, num_epochs):
                        
            train_time=train(net, trainloader, device, criterion, optimizer)
            evaluation=evaluate(net, trainloader_eval, testloader, device, criterion, optimizer)

            epoch_result={}
            epoch_result["Run"]=run
            epoch_result["Epoch"]=epoch
            epoch_result["Training Time"]=train_time
            lr_list=[]
            for param_group in optimizer.param_groups:
                lr_list.append(param_group['lr'])
            epoch_result["LR"]=lr_list
            epoch_result.update(evaluation)
        
            results[run][epoch]=epoch_result            
            print(epoch_result)

            f = open(filename, "a")
            f.write(str(epoch_result)+os.linesep)
            f.flush()

            scheduler.step()

with open(filename+".pkl",'wb') as f:
    pickle.dump(results,f)
