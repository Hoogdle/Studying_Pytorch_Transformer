### 데이터세트 분리

# 모델 평가에서 매번 임의의 값을 넣어주기는 번거롭다.
# 따라서 전체 데이트세트를 2가지 또는 3가지로 나눠서 다룬다.

# 2가지로 나누는 경우
# 훈련용 데이터 // 테스트 데이터

# 3가지로 나누는 경우
# 훈련용 데이터 // 검증용 데이터 // 테스트 데이터

# 훈련용 데이터 : 모델을 학습하는데 사용되는 데이터세트
# 검증용 데이터 : 학습이 완료된 모델을 검증하기 위해 사용되는 데이터세트, 주로 구조가 다른(계층,에폭,학습률(하이퍼파라미터)) 모델의 성능 비교를 위해 사용되는 데이터세트
# 테스트 데이터 : 검증용 데이터를 통해 결정된 성능이 가장 우수한 모델을 최종 테스트하기 위한 목적으로 사용되는 데이터세트, 새로운 데이터 세트로 모델을 평가한다.

# 훈련용 데이터(모델 학습) => 검증용 데이터(최적의 하이퍼파라미터) => 테스트 데이터(최종 모델의 성능 테스트)

# 보통 6:2:2 또는 8:1:1로 데이터세트를 분리해 사용한다.

# 데이터세트를 분리하기 위해 torch.utils.data 에서 random_split(무작위 분리)함수를 포함
import torch
import pandas as pd
from torch import nn    
from torch import optim
from torch.utils.data import Dataset,DataLoader,random_split

# 중략

dataset = CustomDataset("경로")
dataset_size = len(dataset)
train_size = int(dataset*0.8)
validation_size = int(dataset*0.1)
test_size = dataset_size - train_size - validation_size

train_dataset, validation_dataset, test_dataset = random_split(dataset,[train_size,validation_size,test_size])
print(f'Training  Data Size : {len(train_dataset)}')
print(f'Validation Data Size : {len(validation_dataset)}')
print(f'Testing Data Size : {len(test_dataset)}')

train_dataloader = DataLoader(train_dataset,batch_size=16,shuffle=True,drop_last=True)
validation_dataloader = DataLoader(validation_dataset,batch_size=4,shuffle=True,drop_last=True)
test_dataloader = DataLoader(test_dataset,batch_size=4,shuffle=True,drop_last=True)

# 중략

with torch.no_grad():
    model.eval()
    
    for x,y in validation_dataloader:
        x = x.to(device)
        y = y.to(device)

        outputs = model(x)
        print(f'X : {x}')
        print(f'X : {y}')
        print(f'Outputs : {outputs}')
        print('--------------------')

##########################################################
print(2)
### 무작위 분리 함수(random_split)

subset = torch.utils.data.random_split(
    dataset,
    lengths,
    generator
)

# 무작위 분리 함수는 lengths만큼의 크기의 데이터세트의 서브셋을 생성한다
# ex) lengts == [300,100,50] ==> 서브셋은 총 3개 순서대로 300개 100개 50개의 데이터를 가지는 데이터 세트
# 분리 길의의 총합은 전체 데이터 갯수와 같아야 한다!
# generator는 서브셋에 포함될 무작위 데이터들의 난수 생성 시드를 의미, 시드값에 따라 포함되는 데이터의 배치가 달라진다.

# ex) (연습)
dataset = CustomDataset("경로/data.csv")
dataset_size = len(dataset)
train_size = int(dataset_size * 0.8)
validation_size = int(dataset_size* 0.1)
test_size = dataset_size - train_size - validation_size

train_dataset, validation_dataset, test_dataset = random_split(dataset,[train_size,validation_size,test_size])
print(f'Training Data Size : {len(train_dataset)}')
print(f'Validation Data Size : {len(validation_dataset)}')
print(f'Testing Data Size : {len(test_dataset)}')

train_dataloader = DataLoader(train_dataset,batch_size=16,shuffle=True,drop_last=True)
validation_dataloader = DataLoader(validation_dataset,batch_size=4,shuffle=True,drop_last=True)
test_dataloader = DataLoader(train_dataset,batch_size=4,shuffle=True,drop_last=True)
