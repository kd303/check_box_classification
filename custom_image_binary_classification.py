from pickletools import optimize
from tkinter import Y
from datasets import Dataset
import torch
import torchvision
import torchvision.transforms as transforms


import torch.nn as nn
import torch.nn.functional as F  # change for new PyTorch versions, old may not have nn.functional
import torch.optim as optim

import torch.utils.data as data


class CheckBoxClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 5) # [3 in channels, 6 out channels, kernel size]
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5) # nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        # print('shape of x ----------------------------- ', x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        # print('shape of x after conv1 ----------------------------- ', x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        # print('shape of x after conv2 ----------------------------- ', x.shape)
        x = torch.flatten(x, 1) ## Do not want batch to be fallted, but Conv2d - has 16 * 6 * 5 features, 
        # print('shape of x after flattening ----------------------------- ', x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



def run(TRAIN_DATA_PATH = "D:/code/py/train", VALIDATION_DATA_PATH = "D:/code/py/test",
            BATCH_SIZE = 8):
    
    transform_img = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])
    
    model = CheckBoxClassification()
    # copy to device
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)

    
    train_datasets = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=transform_img)
    validation_datasets = torchvision.datasets.ImageFolder(root=VALIDATION_DATA_PATH, transform=transform_img)

    train_data_loader = data.DataLoader(train_datasets, batch_size=BATCH_SIZE, shuffle = True, num_workers = 2 )
    validation_data_loader = data.DataLoader(validation_datasets, batch_size=BATCH_SIZE, shuffle = True, num_workers = 2 )

    training(model, train_data_loader, validation_data_loader, optimizer, 2, criterion=criterion, n_epochs=10)
    

def training (model, train_loader, validation_loader, optimizer, print_at_iter,criterion, n_epochs = 3):
    running_loss = 0.0
    clses = ['checked', 'unchecked']
    _clses = iter(clses)
    for epoch in range(n_epochs):
        print(f'Epoch ---------------------------------: {epoch} -------------------------------')
        train_one_epoch(model, train_loader, optimizer, print_at_iter, running_loss,criterion, epoch)
        validate_model(validation_loader, model, _clses)
    print('Training compelete !')

def train_one_epoch(model, train_loader, optimizer, print_at_iter, running_loss, criterion, epoch):
    running_loss = 0.0
    model.train()
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
            
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        #  print('#################### Loss ##################', loss.item())
        if i%print_at_iter == (print_at_iter - 1):
            # print('#################### running Loss ##################', running_loss)
            print(f'[ epoch {epoch + 1 }, mini batch : {i + 1:3d},  loss: {running_loss/print_at_iter} ')
            running_loss = 0


def validate_model(validate_loader, model, classes):
    _classes = iter(classes)
    # correct_pred = {classname: 0 for classname in _classes}
    correct_pred = {'checked': 0, 'unchecked': 0}
    total_pred = {'checked': 0, 'unchecked': 0}
    class_dict = {0: 'checked', 1: 'unchecked'}
    with torch.no_grad():
        total = 0.0
        correct = 0.0
        for data in validate_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            # # For class wise accuracy
            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    # print(label.item())
                    correct_pred[class_dict[label.item()]] += 1
                else:
                    correct_pred[class_dict[label.item()]] += 1
                total_pred[class_dict[label.item()]] += 1
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # print(correct)
        print('Total ' , total_pred)
        print('Correct ' ,correct_pred)
        print(f'Valiation Accuracy of the network on total {total} test images {correct} : {100 * correct//total} %')

        # for classname, correct_count in correct_pred.items():
        #     accuracy = 100* float(correct_count)/total_pred[classname]
        #     print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} % ')


def save_model(name_of_model , model):
    PATH = './' + name_of_model
    torch.save(PATH, model.state_dict())
