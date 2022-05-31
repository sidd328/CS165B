import torch
from torchvision.models import resnet18
from torch import nn
from torch.utils.data import DataLoader
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from sklearn import metrics
from sklearn import decomposition
from sklearn import manifold
from tqdm.notebook import trange, tqdm
import matplotlib.pyplot as plt
import numpy as np

import copy
import random
import time
import os

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

from torchvision import models
from torch.optim import lr_scheduler

from torch.utils.data import DataLoader 
from torch.utils.data import Dataset
from PIL import Image
import natsort

main_dir = "hw4_train"

test_transforms = transforms.Compose([
                           transforms.Grayscale(num_output_channels=1),
                           transforms.ToTensor()
                                     ])

train_data = ImageFolder(main_dir, transform=test_transforms)
loader = DataLoader(train_data, batch_size=1, num_workers=4)
mstd = next(iter(loader))
_mean = mstd[0].mean()
_std  =  mstd[0].std()



train_transforms = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=_mean, std=_std),
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomRotation(5),
                            transforms.RandomAdjustSharpness(2)
                                      ])

train_data = ImageFolder(main_dir, transform=train_transforms)

VALID_RATIO = 0.8

n_train_examples = int(len(train_data) * VALID_RATIO)
n_valid_examples = len(train_data) - n_train_examples

train_data, valid_data = torch.utils.data.random_split(train_data, [n_train_examples, n_valid_examples])

valid_data = copy.deepcopy(valid_data)
valid_data.dataset.transform = train_transforms

BATCH_SIZE = 64

train_iterator = torch.utils.data.DataLoader(train_data,
                                 shuffle=True,
                                 batch_size=BATCH_SIZE,)

valid_iterator = torch.utils.data.DataLoader(valid_data,
                                 batch_size=BATCH_SIZE,)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataloaders = {'train': train_iterator, 'val': valid_iterator}
dataset_sizes= {'train': len(train_data), 'val': len(valid_data)}

model = resnet18(NUM_CLASSES)
model.to(device)

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        #print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        #print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
               phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        #print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)

        return model



epochs = 10
model_ft = models.resnet18(pretrained=False)

# change input layer
# the default number of input channel in the resnet is 3, but our images are 1 channel. So we have to change 3 to 1.
# nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) <- default
model_ft.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) 

# change fc layer
# the number of classes in our dataset is 10. default is 1000.
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 10)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)

# decay LR by a factor of 0.1 every 5 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

# train model
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, dataset_sizes,
                       num_epochs=epochs)

PATH = F"ResNet.pth"
torch.save(model_ft.state_dict(), PATH)

def test(model, data_loader):
    model.eval()

    with open('prediction.txt', 'w') as f:
      with torch.no_grad():
          for data in test_loader:
              output = F.log_softmax(model(data), dim=1)
              _, pred = torch.max(output, dim=1)
              for i in pred.numpy():
                f.write(str(i))
                f.write('\n')
      f.close()


class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc)
        tensor_image = self.transform(image)
        return tensor_image

test_path = 'hw4_test' 

my_dataset = CustomDataSet(test_path, transform=test_transforms)
test_loader = data.DataLoader(my_dataset, batch_size=50, shuffle=False, drop_last = False)

PATH = "ResNet.pth"
model = model_ft
model.load_state_dict(torch.load (PATH))

test(model, test_loader)
