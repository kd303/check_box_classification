import torch
import torchvision
import torchvision.transforms as transforms


import torch.nn as nn
import torch.nn.functional as F  # change for old PyTorch versions, old may not have nn.functional
import torch.optim as optim

import torch.utils.data as data

from sklearn.metrics import confusion_matrix, classification_report

class CheckBoxClassification(nn.Module):
    def __init__(self):
        super().__init__()
        # O = (W - K + 2P)/s + 1
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 5) # stride = 1, padding = 0, default, W=28, K = 5; input = [batch_size, 3, 28, 28] 
        # Ouput of conv1 = (28 - 5 + 0)/1 + 1 => 24,  Output of pool 1 = W / K -> 24 /2 = 12
        self.pool = nn.MaxPool2d(2,2) # stride = None, padding = 0, stride = kernel_size  input = [(28-5 + 2 * 0)/1 = [batch_size, 6, 24, 24]]
        # Output of pool 1 = W / K -> 24 /2 = 12, [batch_size, 6, 12, 12]
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5) # input = [batch_size, 6, (W-K)/S] -> [batch_size, 6, 12, 12 ]]
        # Output of conv 2 = (W - K + 2P)/s + 1 -> (12 - 5 + 0)/1 + 1 = 8, [batch_size, 16, 12, 12], 
        # Output Maxpool 2 - W / K -> 8 / 2 = 4, [batch_size, 16, 4, 4]
        self.fc1 = nn.Linear(16 * 4 * 4, 120) # input = [batch_size, 16, 4, 4] -> [batch_size, 16, 4, 4]
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1) # changed this to 1
    
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
        x = self.fc3(x)
        return x

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



def run(TRAIN_DATA_PATH = "D:/code/py/train", VALIDATION_DATA_PATH = "D:/code/py/test",
            BATCH_SIZE = 8, LR_RATE=0.001, EPOCH = 10):
    
    ## transforming to (28,28) becuase of CNN is hardcoded
    transform_img = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    model = CheckBoxClassification()
    # copy to device
    model.to(device)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr = LR_RATE, momentum=0.9)

    
    train_datasets = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=transform_img)
    validation_datasets = torchvision.datasets.ImageFolder(root=VALIDATION_DATA_PATH, transform=transform_img)
    
    print('train_datasets : ', train_datasets.classes)
    print('validation_datasets : ', validation_datasets.classes)

    train_data_loader = data.DataLoader(train_datasets, batch_size=BATCH_SIZE, shuffle = True, num_workers = 2 )
    validation_data_loader = data.DataLoader(validation_datasets, batch_size=BATCH_SIZE, shuffle = True, num_workers = 2 )

    training(model, train_data_loader, validation_data_loader, optimizer, 8, criterion=criterion, n_epochs=EPOCH)
    
def binary_acc(y_pred, y_test):
    # converting logits
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_sum = (y_pred_tag == y_test).sum().float()
    avg_acc = correct_sum/y_test.shape[0]
    # print('From binary acc functon ', avg_acc, torch.round(avg_acc * 100))
    return torch.round(avg_acc * 100)

def training (model, train_loader, validation_loader, optimizer, print_at_iter,criterion, n_epochs = 3):
    running_loss = 0.0
    for epoch in range(n_epochs):
        print(f' ---------------------------------: Epoch : {epoch} : -------------------------------')
        train_one_epoch(model, train_loader, optimizer, print_at_iter, running_loss,criterion, epoch)
    
    validate_model(validation_loader, model)
    print('Training compelete !')

def train_one_epoch(model, train_loader, optimizer, print_at_iter, running_loss, criterion, epoch):
    
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    
    for i, (inputs, labels) in enumerate(train_loader):
         
        inputs, labels = inputs.to(device), labels.to(device)
            
        optimizer.zero_grad()
        outputs = model(inputs)
        # print('output : ', outputs.dtype, 'labels ', labels.dtype)
        loss = criterion(outputs, labels.float().unsqueeze(1))
        acc = binary_acc(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
        
        total_acc += acc.item()
        total_loss += loss.item()

        if i%print_at_iter == (print_at_iter - 1):
            # print('#################### running Loss ##################', running_loss)
            print(f'[ epoch {epoch+1}, mini batch : {i+1:3d},  loss: {total_loss/i}, mini batch acc : {total_acc/i:.3f}')
        
           
    print(f'[ epoch end {epoch+1},  loss: {total_loss/len(train_loader)}, training acc: {total_acc/len(train_loader):.2f}')

def validate_model(validate_loader, model):
    y_pred_list = []
    labels_np = []
    outputs_logits = []
    model.eval()
    with torch.no_grad():
        for (images, labels) in validate_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            pred = torch.round(torch.sigmoid(outputs))
            # pred = torch.round(outputs)
            # outputs_logits.extend(torch.sigmoid(outputs).cpu().numpy())
            y_pred_list.extend(pred.cpu().numpy())
            labels_np.extend(labels.cpu().numpy())
            
    # print(outputs_logits)
    print('y_pred_list ', len(y_pred_list), 'labels_np ', len(labels_np))
    # print(y_pred_list)
    print(confusion_matrix(labels_np, y_pred_list))
    print(classification_report(labels_np, y_pred_list))
  

def save_model(name_of_model , model):
    PATH = './' + name_of_model
    torch.save(PATH, model.state_dict())
