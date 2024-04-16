# Add PyTorch to project
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# Import necessary libraries for data loading and preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Using GPU if available
DEVICE = torch.device("cuda")  if torch.cuda.is_available() else torch.device("cpu")
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else DEVICE


train_config = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
validation_config = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

train_dataset = datasets.ImageFolder('./data/cats_and_dogs_filtered/train/', train_config)
validation_dataset = datasets.ImageFolder('./data/cats_and_dogs_filtered/validation/', validation_config)

BATCH_SIZE = 32

train_dataset_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_dataset_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ㅂ배치 추출하여 내역 확인 
images, labels = next(iter(train_dataset_loader))

labels_map = { v:k for k, v in train_dataset.class_to_idx.items()}
figure = plt.figure(figsize = (6, 7))
cols, rows = 1,4

for i in range(1, cols * rows + 1) :
    sample_idx = torch.randint(len(images), size=(1,)).item()
    img, label = images[sample_idx], labels[sample_idx].item()

    figure.add_subplot(rows, cols, i)

    plt.title(labels_map[label])
    plt.axis("off")

    plt.imshow(torch.permute(img, (1, 2, 0)))
plt.show()

pretrainsed_model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)

class TransferLearningModel(nn.Module):
    def __init__(self, pretraind_model, feature_extractor):
        super().__init__()

        if(feature_extractor):
            for param in pretraind_model.parameters():
                param.requires_grad = False
            
        pretraind_model.heads = nn.Sequential(
            nn.Linear(pretraind_model.heads[0].in_features, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 2)
        )
        self.model = pretraind_model
    
    def forward(self, data):
        logits = self.model(data)
        return logits


feature_extractor = False 
model = TransferLearningModel(pretrainsed_model, feature_extractor).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-6)
loss_function = nn.CrossEntropyLoss()

def model_train(dataloader, model, loss_function, optimizer):
    model.train()

    train_loss_sum = 0
    train_correct = 0
    train_total = 0

    total_train_batch = len(dataloader)

    for images, labels in dataloader:
        x_train = images.to(DEVICE)
        y_train = labels.to(DEVICE)

        outputs = model(x_train)
        loss = loss_function(outputs, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item()

        train_total += y_train.size(0)
        train_correct += ((torch.argmax(outputs, dim=1) == y_train).type(torch.float32)).sum().item()

    train_avg_loss = train_loss_sum / total_train_batch
    train_avg_accuracy = 100 * train_correct / train_total 

    return (train_avg_loss, train_avg_accuracy)

def model_evaluate(dataloader, model, loss_function, optimizer):
    model.eval()

    with torch.no_grad():
        val_loss_sum = 0
        val_correct = 0
        val_total = 0

        total_val_batch = len(dataloader)

        for images, labels in dataloader:
            x_val = images.to(DEVICE)
            y_val = labels.to(DEVICE)

            outputs = model(x_val)
            loss = loss_function(outputs, y_val)

            val_loss_sum += loss.item()

            val_total += y_val.size(0)
            val_correct += ((torch.argmax(outputs, dim=1) == y_val).type(torch.float32)).sum().item()
        
        val_avg_loss = val_loss_sum / total_val_batch
        val_avg_accuracy = 100 * val_correct / val_total

    return (val_avg_loss, val_avg_accuracy)


train_loss_list = []
train_accuracy_list = []

val_loss_list = []
val_accuracy_list = []

from datetime import datetime 

start_time = datetime.now()

EPOCHS = 10

for epoch in range(EPOCHS):
    train_avg_loss, train_avg_accuracy = model_train(train_dataset_loader, model, loss_function, optimizer)
    train_loss_list.append(train_avg_loss)
    train_accuracy_list.append(train_avg_accuracy)

    val_avg_loss, val_avg_accuracy = model_evaluate(validation_dataset_loader, model, loss_function, optimizer)

    val_loss_list.append(val_avg_loss)
    val_accuracy_list.append(val_avg_accuracy)


    print('epoch:', '%02d' % (epoch + 1), 'train loss=', '{:.4f}'.format(train_avg_loss), 
          ' train accuracy=', '{:.4f}%'.format(train_avg_accuracy),
          ' validation loss=', '{:.4f}'.format(val_avg_loss),
          ' validation accuracy=', '{:.4f}%'.format(val_avg_accuracy))

end_time = datetime.now()

print('elapsed time => ', end_time - start_time)

# 정확도 및 오차 테스트 

import matplotlib.pyplot as plt

plt.title('Loss Trend')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()

plt.plot(train_loss_list, label='train loss')
plt.plot(val_loss_list, label='validation loss')
plt.legend()
plt.show()


plt.title('Accuracy Trend')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.grid()

plt.plot(train_accuracy_list, label='train accuracy')
plt.plot(val_accuracy_list, label='validation accuracy')
plt.legend()
plt.show()


