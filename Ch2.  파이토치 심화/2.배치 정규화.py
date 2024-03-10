### 배치 정규화(Batch Normalization)
# : 내부 공변량 변화(Internal Covariate Shift)를 줄여 과대적합을 방지하는 기술(내부 공변량 변화 : 계층마다 입력 분포가 변경되는 현상)

# 일반적으로 학습에서는 배치단위로 나눠서 학습 진행, 배치단위로 나눠서 학습하게 되면 상위 계층의 매개변수가 갱신될 때마다 현재 계층에 전달 되는 데이터의 분포도 달라지게 된다.
# 배치 단위의 데이터로 인해 계속 변환되는 입력 분포를 학습해야 하기 때문에 인공 신경망의 성능과 안정성이 낮아져 학습 속도가 저하된다.
# 내부 공변량 변화가 발생하면 n은닉층에서 n+1은닉층으로 값이 전달될 때 입력값이 균일하지 않아 가중치가 제대로 갱신되지 않을 수 있다.

# 배치 정규화는 미니 배치의 입력을 정규화하는 방식으로 동작.

### Normalization 이란?
# https://yngie-c.github.io/deep%20learning/2021/02/08/batch_normalization/
# 서로 다른 데이터가 가지는 범위는 서로 다르며 최적화 경로를 찾을 때 편차가 큰 파라미터를 갱신하는 방향으로 학습이 진행되기 때문에 비효율적인 최적화 경로를 찾게 됨.
# 정규화를 통해 각 특성마다의 범위를 동일하게 만든다.

### Batch Normalization 이란?
# 입력층에서의 Normalization을 해도 은닉층에서 각 파라미터를 거쳐서 나온 값의 범위는 각기 다르기에 다시 문제에 빠진다.
# ICS(내부 공변량 변화) : 은닉층마다 입력되는 값의 분포가 달라지는 현상
# FC(Fully Connected) => BN(Batch Normalization) => Activation Fuction 의 과정으로 진행

### 배치 정규화 수행

import torch
from torch import  nn

x = torch.FloatTensor(
    [
        [-0.6577,-0.5797,0.6360],
        [0.7392,0.2145,1.523],
        [0.2432,0.5662,0.3222]
    ]
)

# tensor([[-1.3246, -1.3492, -0.3757],
#         [ 1.0912,  0.3077,  1.3686],
#         [ 0.2334,  1.0415, -0.9928]], grad_fn=<NativeBatchNormBackward0>)

print(x.size())
print(nn.BatchNorm1d(3)(x))


### 배치 정규화 클래스

# m = torch.nn.BatchNorm1d(
#     num_features,
#     eps = 1e-05
# )

# 1차원 배치 정규화 클래스는 특징 갯수(num_features)를 입력받아 배치 정규화 수행 (특징 갯수 == 텐서의 특징 개수 == 입력 데이터의 채널 수)
# eps는 분모가 0이 되는 현상을 방지하는 작은 상수

