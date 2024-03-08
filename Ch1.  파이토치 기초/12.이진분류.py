### 비선형회귀를 이용한 이진분류

### 사용자 정의 데이터 세트
# x1,x2,x3 에는 데이터 y에는 True or False인 데이터
# class CustomDataset(Dataset):
#     def __init__(self,file_path):
#         df = pd.read_csv(file_path)
#         self.x1 = df.iloc[:,0].values
#         self.x2 = df.iloc[:,1].values
#         self.x3 = df.iloc[:,2].values
#         self.y = df.iloc[;,3].values
#         self.length = len(df)

#     def __getitem__(self,index):
#         x = torch.FloatTensor([self.x1[index],self.x2[index],self.x3[index]])
#         y = torch.FloatTensor([self.y[index]])
#         return x,y
    
#     def __len__(self):
#         return self.length
    
# forward 함수를 직접 실행 하지 않아도, 객체의 파라미터에 매개변수만 전달하면 forward는 자동으로 실행 된다.(nn.Module의 특징)
# nn.Module은 모든 Network의 Base Class이다.
# class CustomModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layer = nn.Sequential( #nn.Sequential : 여러 계층을 하나로 묶음, 묶어진 계층은 수차적으로 실행(가독성을 높임)
#             nn.Linear(3,1),
#             nn.Sigmoid()
#         )
#     def forward(self,x):
#         x = self.layer(x)
#         return x
    
### 이진 교차 엔트로피
# criterion = nn.BCELoss().to(device) #nn.BCELoss() 클래스로 criterion객체 생성, GPU를 쓰기 위해 .to(device)





##### 전체 코드

import torch
import pandas as pd
from torch import nn    
from torch import optim
from torch.utils.data import Dataset,DataLoader,random_split

class CustomDataset(Dataset):
    def __init__(self,file_path):
        df = pd.read_csv(file_path)
        self.x1 = df.iloc[:,0].values
        self.x2 = df.iloc[:,1].values
        self.x3 = df.iloc[:,2].values
        self.y = df.iloc[:,3].values
        self.length = len(df)
    
    def __getitem__(self,index):
        x = torch.FloatTensor([self.x1[index],self.x2[index],self.x3[index]])
        y = torch.FloatTensor([int(self.y[index])])
        return x,y
    
    def __len__(self):
        return self.length
    
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(3,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = self.layer(x)
        return x
    
dataset = CustomDataset('C:\\Users\\rlaxo\\Desktop\\datasets\\binary.csv')
dataset_size = len(dataset)
train_size = int(dataset_size * 0.8)
validation_size = int(dataset_size * 0.1)
test_size = dataset_size - train_size - validation_size

train_dataset,validation_dataset,test_dataset = random_split(dataset,[train_size,validation_size,test_size],torch.manual_seed(4)) #동일한 난수 발생

train_dataloader = DataLoader(train_dataset,batch_size=64,shuffle=True,drop_last=True)
validation_dataloader = DataLoader(validation_dataset,batch_size=4,shuffle=True,drop_last=True)
test_dataloader = DataLoader(test_dataset,batch_size=4,shuffle=True,drop_last=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CustomModel().to(device)
criterion = nn.BCELoss().to(device)
optimizer = optim.SGD(model.parameters(),lr =0.001)

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
    cost = cost / len(train_dataloader)

    if (epoch + 1)%1000 == 0:
        print(f'Epoch : {epoch+1:4d}, Model : {list(model.parameters())}, Cost : {cost:.3f}')


with torch.no_grad(): 
    model.eval()
    for x,y in validation_dataloader:
        x = x.to(device)
        y = y.to(device)

        outputs = model(x)
        
        print(outputs)
        print(outputs >= torch.FloatTensor([0.5]).to(device))
        print('===============================')


# Epoch : 1000, Model : [Parameter containing:
# tensor([[0.0013, 0.0014, 0.0023]], device='cuda:0', requires_grad=True), Parameter containing:
# tensor([-1.0278], device='cuda:0', requires_grad=True)], Cost : 0.733
# Epoch : 2000, Model : [Parameter containing:
# tensor([[0.0069, 0.0051, 0.0075]], device='cuda:0', requires_grad=True), Parameter containing:
# tensor([-2.0449], device='cuda:0', requires_grad=True)], Cost : 0.638
# Epoch : 3000, Model : [Parameter containing:
# tensor([[0.0172, 0.0186, 0.0239]], device='cuda:0', requires_grad=True), Parameter containing:
# tensor([-2.8759], device='cuda:0', requires_grad=True)], Cost : 0.664
# Epoch : 4000, Model : [Parameter containing:
# tensor([[0.0325, 0.0306, 0.0325]], device='cuda:0', requires_grad=True), Parameter containing:
# tensor([-3.5635], device='cuda:0', requires_grad=True)], Cost : 0.455
# Epoch : 5000, Model : [Parameter containing:
# tensor([[0.0203, 0.0205, 0.0220]], device='cuda:0', requires_grad=True), Parameter containing:
# tensor([-4.1518], device='cuda:0', requires_grad=True)], Cost : 0.436
# Epoch : 6000, Model : [Parameter containing:
# tensor([[0.0290, 0.0316, 0.0295]], device='cuda:0', requires_grad=True), Parameter containing:
# tensor([-4.6603], device='cuda:0', requires_grad=True)], Cost : 0.430
# Epoch : 7000, Model : [Parameter containing:
# tensor([[0.0243, 0.0232, 0.0278]], device='cuda:0', requires_grad=True), Parameter containing:
# tensor([-5.1090], device='cuda:0', requires_grad=True)], Cost : 0.411
# Epoch : 8000, Model : [Parameter containing:
# tensor([[0.0358, 0.0346, 0.0341]], device='cuda:0', requires_grad=True), Parameter containing:
# tensor([-5.5132], device='cuda:0', requires_grad=True)], Cost : 0.370
# Epoch : 9000, Model : [Parameter containing:
# tensor([[0.0298, 0.0338, 0.0319]], device='cuda:0', requires_grad=True), Parameter containing:
# tensor([-5.8811], device='cuda:0', requires_grad=True)], Cost : 0.328
# Epoch : 10000, Model : [Parameter containing:
# tensor([[0.0384, 0.0411, 0.0413]], device='cuda:0', requires_grad=True), Parameter containing:
# tensor([-6.2173], device='cuda:0', requires_grad=True)], Cost : 0.339
# tensor([[0.8786],
#         [0.9464],
#         [0.9668],
#         [0.9885]], device='cuda:0')
# tensor([[True],
#         [True],
#         [True],
#         [True]], device='cuda:0')
# ===============================
# tensor([[0.5168],
#         [0.8726],
#         [0.0829],
#         [0.0136]], device='cuda:0')
# tensor([[ True],
#         [ True],
#         [False],
#         [False]], device='cuda:0')
# ===============================
# tensor([[0.4455],
#         [0.7365],
#         [0.0272],
#         [0.6681]], device='cuda:0')
# tensor([[False],
#         [ True],
#         [False],
#         [ True]], device='cuda:0')
# ===============================
# tensor([[0.7427],
#         [0.9252],
#         [0.4112],
#         [0.6437]], device='cuda:0')
# tensor([[ True],
#         [ True],
#         [False],
#         [ True]], device='cuda:0')
# ===============================
# tensor([[0.9720],
#         [0.9201],
#         [0.9714],
#         [0.7376]], device='cuda:0')
# tensor([[True],
#         [True],
#         [True],
#         [True]], device='cuda:0')
# ===============================
# tensor([[0.9943],
#         [0.9748],
#         [0.6840],
#         [0.9552]], device='cuda:0')
# tensor([[True],
#         [True],
#         [True],
#         [True]], device='cuda:0')
# ===============================
# tensor([[0.0367],
#         [0.9323],
#         [0.4227],
#         [0.9869]], device='cuda:0')
# tensor([[False],
#         [ True],
#         [False],
#         [ True]], device='cuda:0')
# ===============================
# tensor([[0.9178],
#         [0.9536],
#         [0.9181],
#         [0.0169]], device='cuda:0')
# tensor([[ True],
#         [ True],
#         [ True],
#         [False]], device='cuda:0')
# ===============================
# tensor([[0.3354],
#         [0.8513],
#         [0.4314],
#         [0.9441]], device='cuda:0')
# tensor([[False],
#         [ True],
#         [False],
#         [ True]], device='cuda:0')
# ===============================
# tensor([[0.2099],
#         [0.9349],
#         [0.6196],
#         [0.7493]], device='cuda:0')
# tensor([[False],
#         [ True],
#         [ True],
#         [ True]], device='cuda:0')
# ===============================
# tensor([[0.9290],
#         [0.9281],
#         [0.9827],
#         [0.4263]], device='cuda:0')
# tensor([[ True],
#         [ True],
#         [ True],
#         [False]], device='cuda:0')
# ===============================
# tensor([[0.8383],
#         [0.0549],
#         [0.9693],
#         [0.9448]], device='cuda:0')
# tensor([[ True],
#         [False],
#         [ True],
#         [ True]], device='cuda:0')
# ===============================
# tensor([[0.8938],
#         [0.1654],
#         [0.3584],
#         [0.8749]], device='cuda:0')
# tensor([[ True],
#         [False],
#         [False],
#         [ True]], device='cuda:0')
# ===============================
# tensor([[0.2595],
#         [0.5029],
#         [0.0160],
#         [0.1465]], device='cuda:0')
# tensor([[False],
#         [ True],
#         [False],
#         [False]], device='cuda:0')
# ===============================
# tensor([[0.9862],
#         [0.9457],
#         [0.9531],
#         [0.7552]], device='cuda:0')
# tensor([[True],
#         [True],
#         [True],
#         [True]], device='cuda:0')
# ===============================
# tensor([[0.8788],
#         [0.0200],
#         [0.4472],
#         [0.0306]], device='cuda:0')
# tensor([[ True],
#         [False],
#         [False],
#         [False]], device='cuda:0')
# ===============================
# tensor([[0.1237],
#         [0.9883],
#         [0.4719],
#         [0.8058]], device='cuda:0')
# tensor([[False],
#         [ True],
#         [False],
#         [ True]], device='cuda:0')
# ===============================
# tensor([[0.8043],
#         [0.0864],
#         [0.1394],
#         [0.0668]], device='cuda:0')
# tensor([[ True],
#         [False],
#         [False],
#         [False]], device='cuda:0')
# ===============================
# tensor([[0.9555],
#         [0.4336],
#         [0.8460],
#         [0.1737]], device='cuda:0')
# tensor([[ True],
#         [False],
#         [ True],
#         [False]], device='cuda:0')
# ===============================
# tensor([[0.1620],
#         [0.9579],
#         [0.3943],
#         [0.2597]], device='cuda:0')
# tensor([[False],
#         [ True],
#         [False],
#         [False]], device='cuda:0')
# ===============================
# tensor([[0.8664],
#         [0.4596],
#         [0.7571],
#         [0.0208]], device='cuda:0')
# tensor([[ True],
#         [False],
#         [ True],
#         [False]], device='cuda:0')
# ===============================
# tensor([[0.0188],
#         [0.9318],
#         [0.9139],
#         [0.8395]], device='cuda:0')
# tensor([[False],
#         [ True],
#         [ True],
#         [ True]], device='cuda:0')
# ===============================
# tensor([[0.7320],
#         [0.1854],
#         [0.8805],
#         [0.6979]], device='cuda:0')
# tensor([[ True],
#         [False],
#         [ True],
#         [ True]], device='cuda:0')
# ===============================
# tensor([[0.9803],
#         [0.8506],
#         [0.0204],
#         [0.9849]], device='cuda:0')
# tensor([[ True],
#         [ True],
#         [False],
#         [ True]], device='cuda:0')
# ===============================
# tensor([[0.5283],
#         [0.9516],
#         [0.5586],
#         [0.3482]], device='cuda:0')
# tensor([[ True],
#         [ True],
#         [ True],
#         [False]], device='cuda:0')
# ===============================




    
