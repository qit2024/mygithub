import torch as t
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import torchvision.models as models
from tqdm import tqdm
import torch

# 查看当前路径
work_dir = os.getcwd()
print(work_dir)

train_path = work_dir + '/dogs-vs-cats-0/train/'
test_path = work_dir + '/dogs-vs-cats-0/test1/'
img_name = os.listdir(train_path)
test_img = os.listdir(test_path)
classes = set()
for img in img_name:
    cls = img.split('.')[0]
    classes.add(cls)
classes = sorted(classes)
n_class = len(classes)
class_to_num = dict(zip(classes, range(2)))
print(class_to_num)
num_to_class = {v: k for k, v in class_to_num.items()}
print(num_to_class)
labels = []
img_paths = []
for img in img_name:
    imgp = train_path + img
    img_paths.append(imgp)
    cls = img.split('.')[0]
    labels.append(class_to_num[cls])
test_img_name = []
for i in test_img:
    imgp = test_path + i
    test_img_name.append(imgp)

def Data_Transform(mode='train'):
    if mode == 'train':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
        ])
    return transform

import random
random.seed(2023)
random.shuffle(img_paths)
random.seed(2023)
random.shuffle(labels)
cut = int(len(img_paths) * 0.15)

class CatDogDataset(Dataset):
    def __init__(self, mode='train'):
        self.mode = mode
        train_img = img_paths[cut:]
        train_label = labels[cut:]
        valid_img = img_paths[:cut]
        valid_label = labels[:cut]

        if self.mode == 'train':
            self.img = train_img
            self.label = train_label
        elif self.mode == 'valid':
            self.img = valid_img
            self.label = valid_label
        elif self.mode == 'test':
            self.img = test_img_name

        self.data_len = len(self.img)
        print(f'finish reading the {mode} set of dataset ({self.data_len} sample found)')

    def __getitem__(self, index):
        img = Image.open(self.img[index])
        transforms = Data_Transform(self.mode)
        img = transforms(img)
        if self.mode == 'test':
            return img
        else:
            label = self.label[index]
            return img, label

    def __len__(self):
        return len(self.img)

train_dataset = CatDogDataset(mode='train')
val_dataset = CatDogDataset(mode='valid')
test_dataset = CatDogDataset(mode='test')
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=False, num_workers=0)
val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, num_workers=0)
test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False, num_workers=0)

model_ft = models.resnet50(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)
model_ft.to('cpu')
lr = 1e-4
weight_decay = 1e-3
num_epochs = 2
device = 'cpu'
criterion = nn.CrossEntropyLoss()
optimizer = t.optim.Adam(model_ft.parameters(), lr=lr, weight_decay=weight_decay)
from torch.cuda.amp import GradScaler
scaler = GradScaler()

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = []
    train_acc = []
    for batch in tqdm(train_loader):
        imgs, labels = batch
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = model(imgs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        avg_acc = (output.argmax(dim=1) == labels).float().mean()
        train_acc.append(avg_acc)
        train_loss.append(loss)
    train_loss = sum(train_loss) / len(train_loss)
    train_accs = sum(train_acc) / len(train_acc)
    print(f"[ Train / {epoch + 1}/{num_epochs} ] loss = {train_loss}, acc = {train_accs}")

def valid(model, device, val_loader, epoch, base_acc):
    with torch.no_grad():
        model.eval()
        valid_loss = []
        valid_accs = []
        for batch in tqdm(val_loader):
            imgs, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            avg_acc = (outputs.argmax(dim=1) == labels).float().mean()
            valid_accs.append(avg_acc)
            valid_loss.append(loss)
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_accs = sum(valid_accs) / len(valid_accs)
        print(f"[ Valid | {epoch + 1}/{num_epochs} ] loss = {valid_loss}, acc = {valid_accs}")
        best_acc = 0.0
        if valid_accs > best_acc:
            t.save(model.state_dict(), '66model_' + str(epoch) + '_' + str(valid_accs) + '.pth')
            print('66model_' + str(epoch) + '_' + str(valid_accs) + '.pth')
            best_acc = valid_accs
            print('当前最好模型精度：{}%'.format(best_acc * 100))
for epoch in range(num_epochs):
    train(model_ft, device, train_loader, optimizer, epoch)
    valid(model_ft, device, val_loader, epoch, best_acc)
predict
model_ft = model_ft.to(device)
model_ft.load_state_dict(torch.load('66model_0_tensor(0.9883).pth'))
model_ft.eval()
predictions = []
for batch in tqdm(test_loader):
    imgs = batch
    with torch.no_grad():
        output = model_ft(imgs.to(device))
    predictions.extend(output.argmax(dim=-1).cpu().numpy().tolist())
preds = []
for i in predictions:
    preds.append(num_to_class[i])
print(preds)
