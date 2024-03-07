# 비선형 회귀 
# y = w1x^2 + w2x + b

# non_linear_csv를 읽기위해 panda 라이브러리 추가

#################################################################
# {추가 공부 내용}

# csv파일
# : Comma Seperated Value로 쉼표 ','로 분리된 텍스트 파일
# 이름, 성별, 키
# 데이콘, 남자, 180
# 홍길동, 남자, 175
# 아이유, 여자, 163

# pd.read_csv('파일경로/파일이름.csv') : 해당 경로에 있는 csv파일 불러오기 

#################################################################


# .iloc[]
# : 데이터프레임의 행이나 컬럼에 인덱스 값으로 접근

import torch
import pandas as pd
from torch import nn
from torch import optim
from torch.utils.data import Dataset,DataLoader

### 사용자 정의 데이터세트
# file_path 정의
# pdf.read.svc(file_path) : file_path에 있는 
class CustomDataset(Dataset):
    def __init__(self,file_path):
        df = pd.read_csv(file_path)
        self.x = df.iloc[:,0].values
        self.y = df.iloc[:,1].values
        self.length = len(df)

    # y = w1x^2 + w2x + b을 위해 
    # x의 값은 [x^2,x] y의 값은 [y]를 반환
    def __getitem__(self,index):
        x = torch.FloatTensor([self.x[index]**2,self.x[index]])
        y = torch.FloatTensor([self.y[index]])
        return x,y
    # 현재 데이터의 길이 제공
    def __len__(self):
        return self.length
    
### 사용자 정의 모델
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2,1)

    def forward(self,x):
        x = self.layer(x)
        return x
    
### 사용자 정의 데이터세트와 데이터로더
train_dataset = CustomDataset("C:\\Users\\rlaxo\Desktop\\datasets\\non_linear.csv")
train_dataloader = DataLoader(train_dataset,batch_size=128,shuffle=True,drop_last=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CustomModel().to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.SGD(model.parameters(),lr=0.0001)

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


### 모델평가 
# 현재는 임의의 값을 입력하여 결과를 확인
# 임의의 값으로 모델을 확인하거나 평가할 때는 torch.no_grad 클래스 활용
# no_grad 클래스는 기울기 계산을 비활성화하는 클래스(자동미분X, 메모리사용량 줄임)
with torch.no_grad():
    model.eval() #모델 평가모드로 전환
    inputs = torch.FloatTensor(
        [
            [1**2,1],
            [5**2,5],
            [11**2,11]
        ]
    ).to(device)
    outputs = model(inputs)
    print(outputs)

### 모델저장
torch.save(
    model,
    "C:/Users/rlaxo/Desktop/파이토치 트랜스포머 교재 모델저장/model.pt"
)

torch.save(
    model.state_dict(),
    "C:/Users/rlaxo/Desktop/파이토치 트랜스포머 교재 모델저장/model_state_dict.pt"

)

    

# Epoch : 1000, Model : [Parameter containing:
# tensor([[ 3.1140, -1.7010]], device='cuda:0', requires_grad=True), Parameter containing:
# tensor([-0.3456], device='cuda:0', requires_grad=True)], Cost : 0.388
# Epoch : 2000, Model : [Parameter containing:
# tensor([[ 3.1127, -1.7029]], device='cuda:0', requires_grad=True), Parameter containing:
# tensor([-0.2741], device='cuda:0', requires_grad=True)], Cost : 0.356
# Epoch : 3000, Model : [Parameter containing:
# tensor([[ 3.1116, -1.7023]], device='cuda:0', requires_grad=True), Parameter containing:
# tensor([-0.2087], device='cuda:0', requires_grad=True)], Cost : 0.316
# Epoch : 4000, Model : [Parameter containing:
# tensor([[ 3.1107, -1.7026]], device='cuda:0', requires_grad=True), Parameter containing:
# tensor([-0.1491], device='cuda:0', requires_grad=True)], Cost : 0.272
# Epoch : 5000, Model : [Parameter containing:
# tensor([[ 3.1102, -1.7025]], device='cuda:0', requires_grad=True), Parameter containing:
# tensor([-0.0944], device='cuda:0', requires_grad=True)], Cost : 0.211
# Epoch : 6000, Model : [Parameter containing:
# tensor([[ 3.1090, -1.7028]], device='cuda:0', requires_grad=True), Parameter containing:
# tensor([-0.0443], device='cuda:0', requires_grad=True)], Cost : 0.221
# Epoch : 7000, Model : [Parameter containing:
# tensor([[ 3.1082, -1.7032]], device='cuda:0', requires_grad=True), Parameter containing:
# tensor([0.0015], device='cuda:0', requires_grad=True)], Cost : 0.173
# Epoch : 8000, Model : [Parameter containing:
# tensor([[ 3.1074, -1.7030]], device='cuda:0', requires_grad=True), Parameter containing:
# tensor([0.0434], device='cuda:0', requires_grad=True)], Cost : 0.174
# Epoch : 9000, Model : [Parameter containing:
# tensor([[ 3.1071, -1.7029]], device='cuda:0', requires_grad=True), Parameter containing:
# tensor([0.0817], device='cuda:0', requires_grad=True)], Cost : 0.136
# Epoch : 10000, Model : [Parameter containing:
# tensor([[ 3.1065, -1.7029]], device='cuda:0', requires_grad=True), Parameter containing:
# tensor([0.1167], device='cuda:0', requires_grad=True)], Cost : 0.134
    
