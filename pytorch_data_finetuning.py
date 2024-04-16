import torch

from torch import nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from torch.optim import Adam 

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else DEVICE

train_config = transforms.Compose([transforms.Resize((224, 224)), 
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor()])

validation_config = transforms.Compose([transforms.Resize((224, 224)), 
                                     transforms.ToTensor()])

test_config = transforms.Compose([transforms.Resize((224, 224)), 
                                   transforms.ToTensor()])

train_dataset = datasets.ImageFolder(root='./data/pytorch_dog_image/train', transform=train_config)
validation_dataset = datasets.ImageFolder(root='./data/pytorch_dog_image/validation', transform=validation_config)
test_dataset = datasets.ImageFolder(root='./data/pytorch_dog_image/test', transform=test_config)

train_dataset_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_dataset_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
test_dataset_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

pretrained_model = models.vit_b_16(weights = models.ViT_B_16_Weights.DEFAULT)

class TransferLearningModel(nn.Module):
    def __init__(self, pretrained_model, feature_extractor):
        super().__init__()

        # Feature Extractor 인지를 확인해서
        # Feature Extractor라면 내부 weight 값을 수정하지 않도록 한다.
        if feature_extractor:
            for param in pretrained_model.parameters():
                param.requires_grad = False

        # Classfier 부분(heads)를 수정한다.
        # 마지막 레이어의 값은 최종 출력 노드의 갯수와 같아야한다.
        pretrained_model.heads = torch.nn.Sequential(
            nn.Linear(pretrained_model.heads[0].in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 4)
        )
        
        self.model = pretrained_model
    
    def forward(self, data):
        logits = self.model(data)
        return logits
    
feature_extractor = False # True면 feature extractor로 사용한다.
model = TransferLearningModel(pretrained_model, feature_extractor).to(DEVICE)
optimizer = Adam(model.parameters(), lr = 1e-6) # 파인튜닝에서는 학습률을 매우 작게 설정해야함.
loss_function = nn.CrossEntropyLoss()

def model_train(dataloader, model, loss_function, optimizer):
    model.train() # 모델을 학습모드로 전환 

    train_loss_sum = train_correct = train_total = 0

    total_train_batch = len(dataloader)

    # images 에는 입력 이미지가 들어오고 lagels에는 라벨이 들어온다.
    for images, labels in dataloader:
        x_train = images.to(DEVICE)
        y_train = labels.to(DEVICE)

        outputs = model(x_train)
        loss = loss_function(outputs, y_train)

        optimizer.zero_grad() # 기존 gradients를 0으로 만든다.
        loss.backward() # 역전파를 수행한다.
        optimizer.step() # 파라미터를 갱신한다.

        train_loss_sum += loss.item()
        train_total += y_train.size(0)
        train_correct += ((torch.argmax(outputs, 1) == y_train).type(torch.float32)).sum().item()
    
    train_avg_loss = train_loss_sum / total_train_batch
    train_accuracy = 100* train_correct / train_total

    return (train_avg_loss, train_accuracy)

def model_evaluate(dataloader, model, loss_function):
    model.eval() # 신경망을 추론 모드로 전환

    # 추론 모드에서는 모델 파라미터를 업데이트하지않으므로 미분을 하지 않음
    with torch.no_grad():
        val_loss_sum = val_correct = val_total = 0

        total_val_batch = len(dataloader)

        for images, labels in dataloader:
            x_val = images.to(DEVICE)
            y_val = labels.to(DEVICE)

            outputs = model(x_val) # 추론을 수행한다.
            loss = loss_function(outputs, y_val) # 손실을 계산한다.
            val_loss_sum += loss.item()
            val_total += y_val.size(0)
            val_correct += ((torch.argmax(outputs, 1) == y_val).type(torch.float32)).sum().item()
    
    val_avg_loss = val_loss_sum / total_val_batch
    val_accuracy = 100* val_correct / val_total

    return (val_avg_loss, val_accuracy)

train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

from datetime import datetime

start_time = datetime.now()
EPOCHS = 40 

for epoch in range(EPOCHS):
    #== Model Train ==
    train_avg_loss, train_accuracy = model_train(train_dataset_loader, model, loss_function, optimizer)
    train_loss_list.append(train_avg_loss)
    train_acc_list.append(train_accuracy)

    #== Model 평가 ==
    val_avg_loss, val_accuracy = model_evaluate(validation_dataset_loader, model, loss_function)
    val_loss_list.append(val_avg_loss)
    val_acc_list.append(val_accuracy)


    print('epoch:', '%02d' % (epoch + 1), 'train loss=', '{:.4f}'.format(train_avg_loss), 
          ' train accuracy=', '{:.4f}%'.format(train_accuracy),
          ' validation loss=', '{:.4f}'.format(val_avg_loss),
          ' validation accuracy=', '{:.4f}%'.format(val_accuracy))
    


# 모델의 학습 및 평가 결과를 그래프로 표시 
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

plt.plot(train_acc_list, label='train accuracy')
plt.plot(val_acc_list, label='validation accuracy')
plt.legend()
plt.show()



def model_test(dataloader, model, loss_function):
    model.eval() # 평가모드로 전환 

    with torch.no_grad():
        test_loss = 0
        test_correct = 0
        test_total = 0

        total_test_batch = len(dataloader)

        for images, labels in dataloader:
            x_test = images.to(DEVICE)
            y_test = labels.to(DEVICE)

            outputs = model(x_test)
            loss = loss_function(outputs, y_test)
            test_loss += loss.item()
            test_total += y_test.size(0)
            test_correct += ((torch.argmax(outputs, 1) == y_test)).sum().item()
        
        test_avg_loss = test_loss / total_test_batch
        test_avg_accuracy = test_correct / test_total

    return (test_avg_loss, test_avg_accuracy)

#+ 모델 테스트

test_image_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
test_images, test_labels = next(iter(test_image_loader))

x_test = test_images.to(DEVICE)
y_test = test_labels.to(DEVICE)

outputs = model(x_test)

_, preds = torch.max(outputs, 1) # 예측 값 추출 

#+ TEST 결과 출력
import matplotlib.pyplot as plt

labels_map = { class_index:class_name for class_name, class_index in train_dataset.class_to_idx.items()}

plt.figure(figsize=(10, 10))
for pos in range(16):
    plt.subplot(4, 4, pos+1)
    sample_idx = torch.randint(len(x_test), size=(1,)).item()
    img, label, pred = test_images[sample_idx], test_labels[sample_idx].item(), preds[sample_idx].item()
    #plt.imshow(img.view(28, 28).numpy())
    plt.imshow(torch.permute(img, (1, 2, 0)))
    plt.title('{} \n({})'.format(labels_map[label], labels_map[pred]))
    plt.axis('off')
plt.show()