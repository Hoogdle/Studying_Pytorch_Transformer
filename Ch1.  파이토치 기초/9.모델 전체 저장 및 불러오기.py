### 모델 전체 저장 및 불러오기

# 모델 학습은 오래 시간 소요 => 학습 결과를 저장하고 불러와 활용할 수 있어야 함.
# 직렬화(Serialize), 역직렬화(Deserialize)를 통해 객체를 저장하고 불러올 수 있다.
# 모델을 저장하려면 피클(Pickle)을 활용해 객체 구조를 바이너리 프로토콜(Binary Protocols)로 직렬화; 모델에 사용된 텐서나 매개변수 저장
# 모델을 불러오려면 저장된 객체 파일을 역직렬화해  현재 프로세스의 메모리에 업로드; 계산된 텐서나 매개변수를 불러올 수 있음

# 주로 모델 학습이 모두 완료된 이후 모델을 저장하거나, 특정 에폭이 끝날 때마다 저장
# 모델 파일의 확장자는 주로 .pt 또는 .pth로 저장

### 모델 전체 저장/불러오기
# 학습에 사용된 모델 클래스의 구조와 학습 상태 등을 모두 저장(계층구조,매개변수) 
# 모델 파일만으로도 동일한 구조 구현 가능
# *** 모델의 코드가 저장되는 것은 아니다! 로드한 모델을 실행하려면 해당 모델의 클래스가 선언돼 있어야 한다! ***

# torch.save(
#     model, #모델 인스턴스
#     path #경로
# )


### 모델 불러오기 함수
# model = torch.load(
#     path, #모델이 저장되어 있는 경로
#     map_location #모델을 불러올 때 적요하려는 장치
# )

### 모델 불러오기 예시
import torch
from torch import nn    

# 모델을 불러오는 경우에도 동일한 형태의 클래스가 선언돼 있어야 한다. 선언되지 않으면 AttributeError 오류 발생!
# 모델 클래스가 어딘가에 선언돼 있기만 하면 된다!
# 모델 클래스 선언문
######################################
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2,1)
    
    def forward(self,x):
        x = self.layer(x)
        return x
######################################
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load("모델저장경로",map_location=device)
print(model)


with torch.no_grad():
    model.eval()
    inputs = torch.FloatTensor(
        [
            [1**2,1],
            [5**2,5],
            [11*2,11]
        ]
    ).to(device)
    outputs = model(inputs)
    print(outputs)


### 모델 전체 파일은 가지고 있지만 모델 구조를 모르는 경우
# 모델 구조를 출력해 확인할 수 있다.
# 이 후 모델 클래스에 동일한 형태로 모델 매개 변수를 구현한다.
# *** 단순한 모델의 경우는 모델을 복원할 수 있다. 하지만 복잡한 모델의 경우 사실상 불가능***
# 모델의 구조(클래스)는 모델데이터와 같이 갖고 있어야 한다!

import torch
from torch import nn

class CustomMode(nn.Module):
    pass    

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load("경로",map_location=device)
print(model)  


# CustoModel(
#     (layer) : Linear(in_features=2,out_features=1,bias=True)
# )