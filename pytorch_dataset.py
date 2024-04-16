import torch

x_train = torch.Tensor([1, 2, 3, 4, 5, 6]).view(6, 1)
y_train = torch.Tensor([2, 4, 6, 8, 10, 12]).view(6, 1)

from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, x_train, y_train) -> None:
        super().__init__()
        self.x_train = x_train
        self.y_train = y_train
    
    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]

    def __len__(self) -> int:
        return self.x_train.shape[0]

dataset = CustomDataset(x_train, y_train)
train_loader = DataLoader(dataset, batch_size=3, shuffle=True)

from torch import nn 
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1, 1),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

