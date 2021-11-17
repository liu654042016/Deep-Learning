#https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

plt.ion()#interactive mode

#load data
#data augmentation and normalization for training
#just normalization for validation

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.486, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

data_dir = 'data/hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_works=4)
               for x in ['train', 'val']}
datasets_size = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def imshow(inp, title=None):
    """imshow for tensor"""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.arrsy([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)#pause a bit so that plots are update

#get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

#make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])

#training the model
def train_model(model, criterion, optimizer, schedule, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('_'*10)

        #each epoch has a tranining and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_correct = 0

            #iterate over data
            for inputs, label in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = label.to(device)

                #zero the parameter gradients
                optimizer.zero_grad()

                #forward
                #track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _,preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    #backward optimizer only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    #statistic
                running_loss +=  loss.item()*inputs.size(0)
                running_correct += torch.sum(preds==labels.data)
            if phase == 'train':
                schedule.step()
            epoch_loss = running_loss / datasets_size[phase]
            epoch_acc = running_correct.double() / datasets_size[phase]

            print('{} Loss, {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            #deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()
    time_elapsed = time.time()-since
    print('training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
    print('best val acc: {:4f}'.format(best_acc))

    #load best model weights
    model.load_state_dict(best_model_wts)
    return model

#visualing the model predition
def visualize_model(model, num_images=6):
    was_training  = model.training
    model.eval()
    image_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i ,(inputs, labels) in enumerate(dataloaders['val'])
            inputs = inputs.to(device)
            labels = inputs.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                image_so_far += 1
                ax = plt.subplot(num_images//2, 2, image_so_far)
                ax.axix('off')

                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if image_so_far == num_images:
                    model.train(model==was_training)
                    return
            model.train(model=was_training)

#finetuning the convnet

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
#here the size of each output sample is set to 2
#it can generalized to nn.Linear(num_ftrs, len(class_name))
model_ft.fc = nn.Linear(num_ftrs, 2)
model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

#observe that all parameters are being  optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

#decay lr by a factor of 0.1 evety 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

#train and evaluate
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)
visualize_model(model_ft)



#convnet as fixed features extractor
#we need freeze all the network except the final layer we need to set requries_grad == false to freeze the parameters
def test_fixed():
    model_conv = torchvision.models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False
    #paramter of newly constructed modules have required_grad = True by deafult
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)

    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.0001, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=25)

    visualize_model(model_conv)
