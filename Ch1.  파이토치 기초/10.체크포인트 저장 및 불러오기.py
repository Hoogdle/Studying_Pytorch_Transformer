### 체크포인트 저장 / 불러오기

### 체크포인트
# : 학습 과정의 특정 지점마다 저장하는 것

### 비선형 회귀에서 체크포인트 적용 (저장)

import torch
import pandas as pd
from torch import nn    
from torch import optim
from torch.utils.data import Dataset,DataLoader

# 중략

checkpoint = 1
for epoch in range(10000):

# 중략
# 다양한 정보를 저장하기 위해 딕셔너리 형식으로 값을 할당
    cost = cost / len(train_dataloader)

    if (epoch + 1) % 1000 == 0:
        torch.save(
            {
                "model" : "CustomModel",
                "epoch" : epoch,  #반복 횟수 저장(필수)
                "model_state_dict" : model.state_dict(), # 모델 상태 저장(필수)
                "optimizer_state_dict" : optimizer.state_dict(), # 최적화 상태 저장(필수)
                "cost" : cost,
                "description" : f"CustomModel 체크포인트-{checkpoint}",
            },
            f"경로/checkpoint-{checkpoint}.pt"
        )
        checkpoint += 1


### 비선형 회귀에서 체크포인트 적용 (불러오기)

import torch 
import pandas as pd
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

# 중략

checkpoint = torch.load("경로/checkpoint-6.pt")
model.load_state_dict(checkpoint["model_state_dict"]) # 체크포인트의 모델 데이터를 모델에 적용
optimizer.load_state_dict(checkpoint["optimizer_state_dict"]) # 체크포인트의 파라미터 데이터를 파라미터에게 적용
checkpoint_epoch = checkpoint["epoch"]
checkpoint_description = checkpoint["description"]
print(checkpoint_description)

for epoch in range(checkpoint_epoch +1,10000):
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
        
        if (epoch + 1)%1000 == 0:
            print(f'Epoch : {epoch+1:4d}, Model : {list(model.parameters())}, Cost : {cost:.3f}')
