### 가중치 초기화(Weight Initialization)

# 모델의 초기 가중치 값을 설정하는 것.
# 적절한 가중치 초기화는 기울기 폭주 또는 소실 문제를 완화하며 모델의 수렴 속도를 향상시킬 수 있다.

### 상수 초기화
# 가중치를 특정 상수값으로 초기화
# ex) 0 또는 0.1 
# 상수의 값으로는 특정값(Costant),단위행렬(Unit Matirx),디랙 델타 함수(Dirac Delta Fucntion)이 있따.
# W = a(a는 상수)
# 일반적으로 사용되지 않는다, 모든 가중치 초기값을 같은 값으로 초기화하면 '대칭파괴(Breaking Symmetry)'문제 발생
# ex) 가중치가 모두 0이라면 loss 값도 0, 역전파 과정에서 가중치 업데이트가 진행되지 않는다.

### 무작위 초기화
# 가중치 값을 무작위 값 또는 특정 분포 형태로 초기화
# 무작위(Random),균등분포(Uniform Distribution),정규분포(Normal Distribution),잘린 정규 분포(Truncated Normal Distribution),희소 정규 분포 초기화 등이 있다.
# 노드의 가중치와 편향을 무작위로 할당해 대칭 파괴 문제 방지
# 계층이 깊어질수록 활성화 값이 양 끝단에 치우쳐짐, 기울기 소실 현상 발생

### 제이비어 & 글로럿 초기화
# 균등분포나 정규분포를 사용해 가중치를 초기화
# 각 노드의 출력 분산이 입력 분산과 동일하도록
# 평균이 0인 정규분포와 현재 계층의 입력 및 출력 노드 수를 기반으로 계산되는 표준 편차로 가중치 초기화 수행
# 시그모이드나 하이퍼볼릭 탄젠트를 활성화 함수로 사용하는 네트워크에 효과적

### 카이밍 & 허 초기화
# 균등 분포나 정규 분포를 사용해 가중치를 초기화 하는 방법
# 각 노드의 출력 분산이 입력 분산과 동일하도록 가중치를 초기화 
# 단, 현재 계층의 입력 뉴런수를 기반으로 가중치 초기화 수행
# ReLU를 활성화 함수로 사용하는 네트워크에 효과적

### 직교 초기화(Orthogonal Initialization)
# 특잇값 분해(Singular Value Decomposition,SVD)를 활용해 자기 자신을 제외한 나머지 모든 열, 행 벡터들과 직교이면서 동시에 단위벡터인 행렬을 만드는 방법
# 직교 행렬의 고윳값의 절댓값은 1이기 때문에 행렬 곱을 여러 번 수행하더라도 기울기 폭주나 기울기 소실이 발생하지 않는다. 

### 가중치 초기화 실습
# 가중치 초기화는 모델의 클래스를 구축하고 모델의 매개변수의 초기값을 설정할 때 주로 사용된다.
# from torch import nn    

# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layer = nn.Sequential(
#             nn.Linear(1,2),
#             nn.Sigmoid()
#         )
#         self.fc = nn.Linear(2,1)
#         self._init_weights() # 가중치 초기화 진행
#     def _init_weights(self):
#         nn.init.xavier_uniform_(self.layer[0].weight) #nn.init.xavier_uniform_ 으로 가중치를 제이비어로 초기화 # 선형변환에서의 가중치를 초기화 하기 위해 index=0 첨가
#         self.layer[0].bias.data.fill_(0.01) #fill_ 메서드로 편향 초기화

#         nn.init.xavier_uniform_(self.fc.weight) ##nn.init.xavier_unifrom_ 으로 가중치를 제이비어로 초기화 # Sequential이 아니므로 index 필요 없음
#         self.fc.bias.data.fill_(0.01)

# model = Net()

### 가중치 초기화 모듈화 버전

from torch import nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(1,2),
            nn.Sigmoid()
        )

        self.fc = nn.Linear(2,1)
        self.apply(self._init_weights) #가중치 초기화 메서드를 범용적으로 변경 #텐서의 각 요소에 임의의 함수 적용하여 새로운 텐서 반환

    def _init_weights(self,module): #moduel == 초기화 메서드에서 선언한 모델의 매개변수 #다른 이름으로 사용해도 상관 없음  #모든 계층에서의 파라미터를 각각 초기화
        if isinstance(module,nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias,0.01)
        print(f"Apply : {module}")

model = Net()


# Apply : Linear(in_features=1, out_features=2, bias=True)
# Apply : Sigmoid()
# Apply : Sequential(
#   (0): Linear(in_features=1, out_features=2, bias=True)
#   (1): Sigmoid()
# )
# Apply : Linear(in_features=2, out_features=1, bias=True)
# Apply : Net(
#   (layer): Sequential(
#     (0): Linear(in_features=1, out_features=2, bias=True)
#     (1): Sigmoid()
#   )
#   (fc): Linear(in_features=2, out_features=1, bias=True)
# )