import torch 
from torch import nn
from torchvision import models # 다양한 사전 학습모델을 포함한 모듈 

pretrained_model = models.vgg16(weights=models.VGG16_Weights.DEFAULT) # 사전 학습모델과 가중치 

# print(pretrained_model)

class MyTransferLearningModel(nn.Module):
    def __init__(self, pretrained_model, feature_extractor):
        super().__init__()

        # feature_extractor가 True이면 기존 Feature Extractor를 씀
        # false 인 경우는 Fine Tuning을 진행
        if(feature_extractor):
            for param in pretrained_model.parameters():
                param.requires_grad = False
        
        # 학습 데이터에 맞게 새로운 분류기를 만들어 준 후에, 기존 사전학습모델 Classfier 부분을
        # 새로운 classfier로 바꾸어야함
        pretrained_model.classifier = nn.Sequential(
            nn.Linear(pretrained_model.classifier[0].in_features, 128),
            nn.Linear(128, 2)
        )

        self.model = pretrained_model

    def forward(self, data):
        logits = self.model(data)
        return logits

feature_extractor = True 
model = MyTransferLearningModel(pretrained_model, feature_extractor)
# Fine Tuning을 하려면 전이학습 학습율은 매우 작게 정의해야함
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
loss_fn = nn.CrossEntropyLoss()