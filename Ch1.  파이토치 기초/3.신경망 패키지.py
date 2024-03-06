# 신경망 패키지를 활용한 모델 구현
# torch.nn : 신경망 패키지를 포함(네트워크 정의,자동미분,계층 정의 etc)
# : 신경망 학습 과정을 빠르고 간편하게 구현 가능

### 선형 변환 클래스를 활용한 모델 구현

# from torch import nn
# import torch
### 선형 변환 클래스
# layer = torch.nn.Linear(
#     in_features,
#     out_features,
#     bias = True,
#     device = None,
#     dtype = None
# )

# y = Wx + b 형태의 선형변환을 입력데이터에 적용
# in_features와 outfeatures를 설정해 인스턴스를 생성
# 입력된 in_features와 동일한 텐서만 입력 받을 수 있다.
# 순방향 연산을 진행해 출력 데이터 차원 크기의 차원으로 반환
# bias,device,dtype 설정가능


### 모델선언
# Before
# weight = torch.zeros(1,requires_grad=True)
# bias = torch.zeros(1,requires_grad=True)

# Now
# 선형 변환된 형태로 
# model = nn.Linear(1,1,bias=True)
# criterion = torch.nn.MSELoss()
# learning_rate = 0.001


### 평군 제곱 오차 클래스
# criterion = torch.nn.MSELoss()


### 순방향 연산
# for epoch in range(10000):
#     output = model(x)
#     cost = criterion(output,y)




### 신경망 패키지 적용

import torch
from torch import nn
from torch import optim

x = torch.FloatTensor(
    [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],
     [11],[12],[13],[14],[15],[16],[17],[18],[19],[20],
     [21],[22],[23],[24],[25],[26],[27],[28],[29],[30]]
)
y = torch.FloatTensor(
    [[0.94],[1.98],[2.88],[3.92],[3.96],[4.55],[5.64],[6.3],[7.44],[9.1],
     [8.46],[9.5],[10.67],[11.16],[14],[11.83],[14.4],[14.25],[16.2],[16.32],
     [17.46],[19.8],[18],[21.34],[22],[22.5],[24.57],[26.04],[21.6],[28.8]]
)

model = nn.Linear(1,1)  # 입력 데이터에 맞게 파라미터 생성 ex) 입력데이터 1개 => 파라미터 1개
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr=0.001)

for epoch in range(10000):
    output = model(x)
    cost = criterion(output,y)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if (epoch + 1)%1000 == 0:
        print(f'Epoch : {epoch+1:4d}, Model : {list(model.parameters())}, Cost : {cost:.3f}')


# Epoch : 1000, Model : [Parameter containing:
# tensor([[0.8527]], requires_grad=True), Parameter containing:
# tensor([0.0967], requires_grad=True)], Cost : 1.440
# Epoch : 2000, Model : [Parameter containing:
# tensor([[0.8626]], requires_grad=True), Parameter containing:
# tensor([-0.1045], requires_grad=True)], Cost : 1.398
# Epoch : 3000, Model : [Parameter containing:
# tensor([[0.8687]], requires_grad=True), Parameter containing:
# tensor([-0.2297], requires_grad=True)], Cost : 1.382
# Epoch : 4000, Model : [Parameter containing:
# tensor([[0.8726]], requires_grad=True), Parameter containing:
# tensor([-0.3076], requires_grad=True)], Cost : 1.376
# Epoch : 5000, Model : [Parameter containing:
# tensor([[0.8750]], requires_grad=True), Parameter containing:
# tensor([-0.3561], requires_grad=True)], Cost : 1.374
# Epoch : 6000, Model : [Parameter containing:
# tensor([[0.8765]], requires_grad=True), Parameter containing:
# tensor([-0.3862], requires_grad=True)], Cost : 1.373
# Epoch : 7000, Model : [Parameter containing:
# tensor([[0.8774]], requires_grad=True), Parameter containing:
# tensor([-0.4050], requires_grad=True)], Cost : 1.372
# Epoch : 8000, Model : [Parameter containing:
# tensor([[0.8780]], requires_grad=True), Parameter containing:
# tensor([-0.4167], requires_grad=True)], Cost : 1.372
# Epoch : 9000, Model : [Parameter containing:
# tensor([[0.8783]], requires_grad=True), Parameter containing:
# tensor([-0.4240], requires_grad=True)], Cost : 1.372
# Epoch : 10000, Model : [Parameter containing:
# tensor([[0.8785]], requires_grad=True), Parameter containing:
# tensor([-0.4285], requires_grad=True)], Cost : 1.372

# weight 와 bias 변수를 사용하지 않아도 된다!
# 출력시 model.parameters()로 출력