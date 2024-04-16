import torch
import numpy as np

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from torch import nn 

x_train = torch.Tensor([1, 2, 3, 4, 5, 6]).view(6, 1)
y_train = torch.Tensor([3, 4, 5, 6, 7, 8]).view(6, 1)

class MyNeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1, 1),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

model = MyNeuralNetwork()

loss_function = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
nums_epoch = 2000

for epoch in range(nums_epoch + 1):
    prediction = model(x_train)
    loss = loss_function(prediction, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('epoch = ', epoch , ' current_loss = ', loss.item())

x_test = torch.Tensor([-3.1, 3.0, 1.2, -2.5])