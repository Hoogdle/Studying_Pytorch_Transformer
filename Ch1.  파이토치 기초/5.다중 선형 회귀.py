

# 다중 선형 회귀
# y1 = w1x1 + w2x2 + b1
# y2 = w3x1 + w4x2 + b2

# torch.utils.data : 데이터세트와 데이터로더 포함

import torch
from torch import nn    
from torch import optim
from torch.utils.data import TensorDataset,DataLoader

train_x = torch.FloatTensor([
    [1,2],[2,3],[3,4],[4,5],[5,6],[6,7]
])
train_y = torch.FloatTensor([
    [0.1,1.5],[1,2.8],[1.9,4.1],[2.8,5.4],[3.7,6.7],[4.6,8]
])

# 위 데이터를 데이터세트, 데이터로더에 적용
train_dataset = TensorDataset(train_x,train_y) # dataset은 초기화 값을 *args 형태로 입력 받음 -> 여러 개의 데이터 입력받을 수 있음
train_dataloader = DataLoader(train_dataset,batch_size=2,shuffle =True,drop_last=True) #배치 크기2(2 개의 데이터 샘플과 정답을 가져옴)
# 데이터 순서를 무작위로 변경, drop_last == 배치 크기에 맞지 않는 배치 제거 ex) data 5개 배치2 => 1개 남음, 학습에 포함되지 않음


# 모델, 오차함수, 최적화 함수 선언
model = nn.Linear(2,2,bias=True)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.001)


# 데이터로더 적용
for epoch in range(20000):
    cost = 0.0

    for batch in train_dataloader:
        x,y= batch
        output = model(x)
        
        loss = criterion(output,y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cost += loss
    
    cost = cost / len(train_dataloader)

    if (epoch + 1)%1000 == 0:
        print(f'Epoch : {epoch+1:4d}, Model : {list(model.parameters())}, Cost : {cost:.3f}')

# Epoch : 1000, Model : [Parameter containing:
# tensor([[0.3730, 0.4373],
#         [0.9158, 0.4012]], requires_grad=True), Parameter containing:
# tensor([-0.8589, -0.2730], requires_grad=True)], Cost : 0.014
# Epoch : 2000, Model : [Parameter containing:
# tensor([[0.4641, 0.3900],
#         [0.8985, 0.4102]], requires_grad=True), Parameter containing:
# tensor([-0.9974, -0.2467], requires_grad=True)], Cost : 0.004

# ~~~

# Epoch : 20000, Model : [Parameter containing:
# tensor([[0.5589, 0.3411],
#         [0.8806, 0.4194]], requires_grad=True), Parameter containing:
# tensor([-1.1410, -0.2194], requires_grad=True)], Cost : 0.000