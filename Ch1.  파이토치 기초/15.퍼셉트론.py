### 퍼셉트론(Perceptron)

# 입력값을 토대로 특정연산을 진행했을 때 임계값(Threshold)보다 크면 전달, 작으면 전달하지 않음(크면1, 작으면0)
# 계단함수를 적용해 전달


### 단층 퍼셉트론(Single Layer Perceptron)

# 하나의 계층을 갖는 모델
# 논리게이트 중 XOR연산을 구현하지 못한다 => 다층 퍼셉트론

### 다층 퍼셉트론(Multi-Layer Perceptron, MLP)

# 단층 퍼셉트론을 여러 개 쌓아 은닉층을 생성
# 은닉층을 2개 이상 연결 == 심층 신경망(Deep Neural Network, DNN)
# 은닉층이 늘어날수록 더 복잡한 구조의 문제를 해결할 수 있다. 하지만 더 많은 학습데이터와 연산량 필요

# 다층 퍼셉트론의 학습 구조
# 1. 입력층부터 출력층까지 순전파 진행
# 2. 출력값으로 오차 계산
# 3. 오차를 퍼셉트론의 역방향으로 보내면서 입력된 노드의 기여도 측정(손실 함수를 편미분,연쇄 법칙을 통해 기울기 계산)
# 4. 입력층에 도달할 때 까지 노드의 기여도 측정
# 5. 모든 가중치에 최적화 알고리즘 수행


### 퍼셉트론 모델 실습 


### 단층 퍼셉트론 구조

# import torch
# import pandas as pd
# from torch import nn
# from torch import optim
# from torch.utils.data import Dataset,DataLoader

# class CustomDataset(Dataset):
#     def __init__(self,file_path):
#         df = pd.read_csv(file_path)
#         self.x1 = df.iloc[:,0].values
#         self.x2 = df.iloc[:,1].values
#         self.y = df.iloc[:,2].values
#         self.length = len(df)

#     def __getitem__(self,index):
#         x = torch.FloatTensor([self.x1[index],self.x2[index]])
#         y = torch.FloatTensor([self.y[index]])
#         return x,y
#     def __len__(self):
#         return self.length
    
# class CustomModel(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.layer = nn.Sequential(
#             nn.Linear(2,1),
#             nn.Sigmoid()
#         )

#     def forward(self,x):
#         x = self.layer(x)
#         return x
    
# train_dataset = CustomDataset("C:\\Users\\rlaxo\\Desktop\\datasets\\perceptron.csv")
# train_dataloader = DataLoader(train_dataset,batch_size=64,shuffle=True,drop_last=True)

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = CustomModel().to(device)
# criterion = nn.BCELoss().to(device)
# optimizer = optim.SGD(model.parameters(), lr = 0.01)

# for epoch in range(10000):
#     cost = 0.0

#     for x,y in train_dataloader:
#         x = x.to(device)
#         y = y.to(device)

#         output = model(x)
#         loss = criterion(output,y)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         cost += loss
#     cost = cost/len(train_dataloader)

#     if (epoch + 1) % 1000 == 0:
#         print(f'Epoch : {epoch+1:4d}, Cost : {cost:.3f}')

# with torch.no_grad():
#     model.eval()
#     inputs = torch.FloatTensor([
#         [0,0],
#         [0,1],
#         [1,0],
#         [1,1]
#     ]).to(device)

#     output = model(inputs)

#     print('===============')
#     print(output)
#     print(output >= 0.5) 



# Epoch : 1000, Cost : 0.692
# Epoch : 2000, Cost : 0.692
# Epoch : 3000, Cost : 0.692
# Epoch : 4000, Cost : 0.692
# Epoch : 5000, Cost : 0.692
# Epoch : 6000, Cost : 0.692
# Epoch : 7000, Cost : 0.692
# Epoch : 8000, Cost : 0.692
# Epoch : 9000, Cost : 0.692
# Epoch : 10000, Cost : 0.692
# ===============
# tensor([[0.4668],
#         [0.4987],
#         [0.5038],
#         [0.5356]], device='cuda:0')
# tensor([[False],
#         [False],
#         [ True],
#         [ True]], device='cuda:0')
    
### 하나의 계층으로는 XOR문제를 해결하지 못한다!


### 다층 퍼셉트론 구조
# 모델부분만 변경
import torch
import pandas as pd
from torch import nn
from torch import optim
from torch.utils.data import Dataset,DataLoader

class CustomDataset(Dataset):
    def __init__(self,file_path):
        df = pd.read_csv(file_path)
        self.x1 = df.iloc[:,0].values
        self.x2 = df.iloc[:,1].values
        self.y = df.iloc[:,2].values
        self.length = len(df)

    def __getitem__(self,index):
        x = torch.FloatTensor([self.x1[index],self.x2[index]])
        y = torch.FloatTensor([self.y[index]])
        return x,y
    def __len__(self):
        return self.length
    
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(2,2),
            nn.Sigmoid()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(2,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = self.layer1(x) # layer1에다가 데이터 입력, 결과를 x에 저장
        x = self.layer2(x) # layer2에다가 layer1의 출력값 입력, 결과를 x에 저장
        return x

    
    
train_dataset = CustomDataset("C:\\Users\\rlaxo\\Desktop\\datasets\\perceptron.csv")
train_dataloader = DataLoader(train_dataset,batch_size=64,shuffle=True,drop_last=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CustomModel().to(device)
criterion = nn.BCELoss().to(device)
optimizer = optim.SGD(model.parameters(), lr = 0.01)

for epoch in range(10000):
    cost = 0.0

    for x,y in train_dataloader:
        x = x.to(device)
        y = y.to(device)

        output = model(x)
        loss = criterion(output,y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cost += loss
    cost = cost/len(train_dataloader)

    if (epoch + 1) % 1000 == 0:
        print(f'Epoch : {epoch+1:4d}, Cost : {cost:.3f}')

with torch.no_grad():
    model.eval()
    inputs = torch.FloatTensor([
        [0,0],
        [0,1],
        [1,0],
        [1,1]
    ]).to(device)

    output = model(inputs)

    print('===============')
    print(output)
    print(output >= 0.5) 
    
# Epoch : 1000, Cost : 0.692
# Epoch : 2000, Cost : 0.684
# Epoch : 3000, Cost : 0.557
# Epoch : 4000, Cost : 0.186
# Epoch : 5000, Cost : 0.082
# Epoch : 6000, Cost : 0.050
# Epoch : 7000, Cost : 0.036
# Epoch : 8000, Cost : 0.028
# Epoch : 9000, Cost : 0.023
# Epoch : 10000, Cost : 0.019
# ===============
# tensor([[0.0164],
#         [0.9780],
#         [0.9784],
#         [0.0151]], device='cuda:0')
# tensor([[False],
#         [ True],
#         [ True],
#         [False]], device='cuda:0')