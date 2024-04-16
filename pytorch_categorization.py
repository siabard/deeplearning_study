import numpy as np

loaded_data = np.loadtxt('./data/diabetes.csv', delimiter=',')

x_train_np = loaded_data[:, 0:-1]
y_train_np = loaded_data[:, [-1]]

print('x_train_np.shape', x_train_np.shape)
print('y_train_np.shape', y_train_np.shape)

import torch
from torch import nn 

x_train = torch.Tensor(x_train_np)
y_train = torch.Tensor(y_train_np)

class LogisticModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.logistic_stack = nn.Sequential(
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    
    def forward(self, data):
        prediction = self.logistic_stack(data)
        return prediction

model = LogisticModel()

loss_function = nn.BCELoss() # 로지스틱문제에는 Binary Cross Entropy 함수로 손실함수를 사용
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

train_loss_list = []
train_accuracy_list = []

nums_epoch = 5000

for epoch in range(nums_epoch):
    outputs = model(x_train)
    loss = loss_function(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    train_loss_list.append(loss.item())
    predicted = outputs > 0.5
    accuracy = (predicted.float() == y_train).sum().item() / len(y_train)
    train_accuracy_list.append(accuracy)
    
    if epoch % 500 == 0:
        print('Epoch {}: Loss - {}, Accuracy - {}'.format(epoch, loss.item(), accuracy))

import matplotlib.pyplot as plt
plt.title('Loss Trend')
plt.plot(train_loss_list, label = 'train loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.legend(loc='best')
plt.show()

plt.title('Accuracy Trend')
plt.plot(train_accuracy_list, label = 'train accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.legend(loc='best')
plt.show()
