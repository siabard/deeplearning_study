import torch 
from torch import nn 
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else DEVICE

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())

train_dataset_size = int(len(train_dataset) * 0.85)
validation_dataset_size = int(len(train_dataset) * 0.15)

train_dataset, validation_dataset = random_split(train_dataset, [train_dataset_size, validation_dataset_size])

BATCH_SIZE = 12
train_dataset_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_dataset_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataset_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

class DeepLearningModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(7 * 7 * 64, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout25 = nn.Dropout(p = 0.25)
        self.dropout50 = nn.Dropout(p = 0.5)
    
    def forward(self, data):
        data = self.conv1(data)
        data = torch.relu(data)
        data = self.pooling(data)
        data = self.dropout25(data)

        data = self.conv2(data)
        data = torch.relu(data)
        data = self.pooling(data)
        data = self.dropout25(data)
        
        data = data.view(-1, 7 * 7 * 64)
        data = self.fc1(data)
        data = torch.relu(data)
        data = self.dropout50(data)

        logits = self.fc2(data)
        return logits

model = DeepLearningModel().to(DEVICE)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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


def model_test(dataloader, model):
    model.eval()

    with torch.no_grad():
        test_loss_sum = 0
        test_correct = 0
        test_total = 0

        total_test_batch = len(dataloader)

        for images, labels in dataloader:
            x_test = images.to(DEVICE)
            y_test = labels.to(DEVICE)

            outputs = model(x_test)
            loss = loss_function(outputs, y_test)

            test_loss_sum += loss.item()
            test_total += y_test.size(0)
            test_correct += ((torch.argmax(outputs, dim=1) == y_test).type(torch.float32)).sum().item()
        
        test_avg_loss = test_loss_sum / total_test_batch
        test_avg_accuracy = 100 * test_correct / test_total

    return (test_avg_loss, test_avg_accuracy)

from datetime import datetime 

train_loss_list = []
train_accuracy_list = []

val_loss_list = []
val_accuracy_list = []

start_time = datetime.now()

EPOCHS = 20

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
print(model_test(test_dataset_loader, model))

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


