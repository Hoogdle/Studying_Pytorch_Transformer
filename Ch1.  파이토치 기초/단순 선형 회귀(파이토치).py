# 파이토치 기능으로 단순 선형 회귀 모델 구현
# 머신 러닝에서 비용함수는 최적화 알고리즘의 비용함수이다.

import torch
from torch import optim # torch.optim, 최적화 함수가 포함돼 있는 모듈 #딥러닝 모델의 가중치를 최적화


# 넘파이와는 동일한 구조의 데이터 사용, 단 ndarray형식이 아닌 FloatTensor형식
## 데이터 세트
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

## 하이퍼파라미터 초기화
# 0의 값을 갖는 텐서를 생성하며 텐서의 크기는 1로 설정 #(1,)==(1,1) 
# requires_grad : 모든 텐성 대한 연산을 추적, 역전파 메소드를 호출하여 기울기를 계싼하고 저장(자동미분 기능 사용여부)
bias = torch.zeros(1,requires_grad=True)
weight = torch.zeros(1,requires_grad=True)  
learning_rate = 0.001

## 넘파이에서는 학습 반복 구문으로 최적화 코드 구현
## 파이토치는 다르다!

## 이번 예제에서 사용하는 최적화 함수는 "확률적 경사 하강법(optim.SGD)"

## 확률적 경사 하강법 : 모든 데이터에 대한 연산 X, 일부 데이터만 계산하여 빠르게 최적화된 값을 찾음, 미니 배치 형태로 전체 데이터를 N 등분하여 학습 진행.

optimizer = optim.SGD([weight,bias],lr=learning_rate) #weight와 bias를 최적화 대상으로 설정

for epoch in range(10000):
    hypothesis = x * weight + bias
    cost = torch.mean((hypothesis - y)**2)

    optimizer.zero_grad() #optimizer 변수에 포함시킨 매개변수의 기울기 0으로 초기화(텐서의 기울기는 누적해서 더해짐)
    cost.backward() #역전파 수행, optimizer 변수에 포함시킨 매개변수들의 기울기가 새롭게 계산(가중치와 편향에 대한 기울기 계산)
    optimizer.step() #역전파의 결과를 최적화 함수(cost fucntion)에 적용 (weight와 bias의 변화)
    if (epoch + 1) % 1000 == 0:
        print(f'Epoc : {epoch+1:4d}, Weight : {weight.item():.3f}, Bias : {bias.item():.3f}, Cost : {cost:.3f}')


# Epoc : 1000, Weight : 0.864, Bias : -0.138, Cost : 1.393
# Epoc : 2000, Weight : 0.870, Bias : -0.251, Cost : 1.380
# Epoc : 3000, Weight : 0.873, Bias : -0.321, Cost : 1.375
# Epoc : 4000, Weight : 0.875, Bias : -0.364, Cost : 1.373
# Epoc : 5000, Weight : 0.877, Bias : -0.391, Cost : 1.373
# Epoc : 6000, Weight : 0.878, Bias : -0.408, Cost : 1.372
# Epoc : 7000, Weight : 0.878, Bias : -0.419, Cost : 1.372
# Epoc : 8000, Weight : 0.878, Bias : -0.425, Cost : 1.372
# Epoc : 9000, Weight : 0.879, Bias : -0.429, Cost : 1.372
# Epoc : 10000, Weight : 0.879, Bias : -0.432, Cost : 1.372
        
# 넘파이보다 낮은 학습률 0.001을 설정했지만 더 빠른 속도로 최적의 가중치와 편향을 찾아냄.

