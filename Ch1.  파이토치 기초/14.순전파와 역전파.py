### 순전파(Forward Propagation)
# : 가중치가 설정된 네트워크에 입력값 x을 넣어 선형결합(예측값)을 출력
# : 딥러닝에서는 각 계층마다 예측값을 활성화 함수에 넣은 후 출력된 값과 실제값을 비교하여 오차를 계산

### 역전파(Back Propagation)
# : 네트워크의 가중치와 편향을 예측값과 실제값의 사이의 오류를 최소화 하도록 업데이트 
# : 순전파 과정에서 나온 오차를 활용해 각 계층의 가중치와 편향을 최적화
# : 각각의 가중치와 편향을 최적화 하기 위해 연쇄법칙을 활용


### 모델 구조와 초깃값 

from torch import nn
import torch
from torch import optim
class CustomModel(nn.Model):
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

        self.layer1[0].weight.data = torch.nn.Parameter(
            torch.Tensor([[0.4352,0.3545],  # 첫 번째 노드로 가는 가중치
                          [0.1951,0.4835]]) # 두 번째 노드로 가는 가중치
        )

        self.layer1[0].bias.data = torch.nn.Parameter(
            torch.Tensor([-0.1419,0.0439])
        )

        self.layer2[0].weight.data = torch.nn.Parameter(
            torch.Tensor([[-0.1725,0.1129]])
        )

        self.layer2[0].bias.data = torch.nn.Parameter(
            torch.Tensor([-0.3043])
        )
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CustomModel.to(device)
criterion = nn.BCELoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=1)


### 순전파 계산
# 1. 할당된 가중치에다가 각 데이터를 곱하여 선형 결합을 만든다
# 2. 선형 결합의 값을 활성화 함수의 입력으로 넣어 값을 산출한다.
# 3. 다음 층에서 할당된 가중치를 2의 결과와 선형결합을 한다.
# 4. 3에서의 선형결합 값을 활성화 함수의 입력으로 넣어 값을 산출한다.
# 즉 output = model(x) 가 "순전파 과정"


### 오차 계산
# loss = criterion(output,y)

### 역전파 계산
# : 계층의 역순으로 가중치와 편향을 갱신

# 즉, 순전파를 통해 오차를 계싼, 역전파를 통해 가중치 갱신 위 두 과정을 반복하며 학습을 진행해 나간다.
# Layer를 지나가는 과정들은 연속적으로 합성함수를 씌우는 과정과 같다! => 연쇄법칙 가능! 
# 순전파 과정속에서 각 노드의 선형결합의 합, 활성화 함수를 거친 다음의 값 모두를 알고 있기 때문에 역전파 과정의 연쇄법칙은 항상 찾을 수 있다
# ex) Loss를 활성화 함수를 거친 값(a)로 미분하면 해당식은 a의 관한 식으로 변형된다. 우리는 a를 알기 때문에 a를 정의역으로 했을 때의 Loss의 기울기 또한 알 수 있다.
#     연쇄적으로 이 과정을 적용하면 loss의 가중치에 대한 기울기를 알 수 있고 이를 learning rate와 곱한 값을 현재 가중치에 빼서 가중치를 업데이트한다.

# 모든 가중치와 편향을 갱신하면 학습이 1회 진행된 것이다. 갱신된 가중치와 편향으로 다음 학습을 진행하게 되고 오차는 계속 줄어들게 된다.

