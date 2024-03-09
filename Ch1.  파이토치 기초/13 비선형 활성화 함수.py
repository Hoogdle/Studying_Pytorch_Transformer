### 비선형 활성화 함수
# 네트워크에 비선형성을 적용하기 위해 인공 신경망에서 사용
# 실제 세계의 입출력 관계는 대부분 비선형적인 관계 ex) 사람의 나이와 키, 선형적인 함수로는 이 관계를 표현할 수 없다.

### 계단 함수(Step-Function) (이진 활성화 함수 Binary Actiavtion Function)
# : 퍼셉트론에서 최초로 사용한 활성화 함수
# : 계단 함수의 입력값의 합이 임계값을 넘으면 1 넘지 않으면 0을 출력
# : 입계값에서 불연속점을 가져 미분이 불가능해 딥러닝에서는 사용되지 않는다.(역전파 과정에서 데이터가 극단적으로 변경되기도함)
# Step(x) = 1 (if x>=0) , 0(x<=0)

### 임계값 함수(Threshold Function)
# : 임계값(Threshold)보다 크면 입력값(x)를 그대로 전달 작으면 특정 값으로 변경
# : 선형함수와 계단함수의 조합 
# Threshold(x) = x (if x>threshold), value(if x<=threshold)

### 시그모이드 함수(Sigmoid Fucntion)
# : 모든 입력값을 0과 1사이의 값으로 mapping
# : 이진 분류 신경망의 출력 계층에서 활성화 함수로 사용
# : 단순 형태의 미분식을 가지며, 입력값에 따라 출력값이 급격히 변하지 않는다.
# : 매우 큰 값이 입력돼도 최대 1의 값을 가지므로 기울기 소실이 발생
# 인공 신경망은 기울기를 이용해 최적의 값을 찾아가는데 시그모이드 함수 특성상 기울기가 매우 작음, 계층이 많아지면 값이 0으로 수렴
# 따라서 은닉층에서는 사용하지 않고 주로 출력층에서 사용한다. 


### 기울기 소실
# : 다층의 파라미터를 조절하기 위해 역전파 과정을 수행
# : 역전파 과정은 미분을 출력층에서 부터 시작하여 차례로 곱하면서 진행 됨
# : 각 층마다 활성화 함수로 sigmoid와 같은 함수를 사용하면 기울기의 값이 항상 작기 때문에 연속으로 곱하다보면 기울기의 손실이 온다.

### 하이퍼볼릭 탄젠트 함수(Hyperbolic Tangent Function)
# : 시그모이드와 형태가 유사 but 출력값의 중심이 0 (시그모이드는 0.5)
# : 출력값의 범위는 -1 ~ 1 사이 , 시그모이드에서는 발생하지 않는 음수 값 반환 가능


### ReLU 함수(Rectified Linear Unit Function)
# : 0보다 작거나 같으면 0을 반환 0보단 크면 선형 결합의 결과를 반환
# : 기울기소실이 발생하지 않는다.
# : 수식이 간단해 연산 속도가 빠르다
# : 입력값이 음수이면 항상 0을 반환하므로 가중치나 편향이 갱신되지 않을 수도 있다.
# : 선형 결합의 합이 음수이면 해당 노드는 더 이상 값을 갱신하지 않아 '죽은 뉴런(Dead Neuron,Dying ReLU)'가 될 수 있다.'
# ReLU(x) = x (if x>0), 0(if x<=0)


### LeakyReLU 함수 (Leaky Rectified Linear Unit Function)
# : ReLU 함수에서의 죽은 뉴런 현상을 방지하기 위해 고안됌.
# : 음수인 경우 작은 값이라도 출력시켜 기울기를 갱신
# LeakyReLU(x) = x (if x>0), negative_s_lope * x (if x<=0)

### PReLU 함수 (Parametric Rectified Linear Unit Function)
# : LeakyReLU와는 동일하지만 음수 기울기값을 고정값이 아닌, 학습을 통해 갱신되는 값으로 간주함
# : 음수 기울기는 지속적으로 값이 변경됌.
# PReLU(x) = x (if x>0). a*x (if x<=0)

### ELU 함수(Exponential Linear Unit Function)
# : 지수함수를 활용한 부드러운 곡선의 형태
# : 입력값이 0일 때에도 출력값이 급격하게 변하지 않음 => 경사 하강법의 수렴속도가 비교적 빠르다.
# : 복잡한 연산을 진행하므로 학습 속도는 더 느림
# ELU(x) = x (if x>0), negative_slope*(exp(x)-1) (if x<=0)

### 소프트맥스 함수(Softmax Function)
# : 차원 벡터에서 특정 출력값이 k번째 클래스에 속할 확률을 계산
# : 클래에 속할 확률을 계산하는 활성화 함수이므로 은닉층에서 사용되지 않고 출력층에서 사용
# : 네트워크 출력을 가능한 클래스에 대한 확률 분포로 매핑

