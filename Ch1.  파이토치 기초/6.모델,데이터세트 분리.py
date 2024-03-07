# 모델 구현은 신경망 패키지의 모듈(Module)패키지 활용
# 새로운 모델 클래스를 생성하려면 모듈 클래스를 상속받아 임의의 서브 클래스를 생성한다.


### 모듈 클래스
# __init__ 과 forward를 재정의하여 활용
# __init__(초기화 메서드) : 신경망에 사용될 계층을 초기화
# forward(순방향 메서드) : 모델이 어떤 구조를 갖게 될지를 정의
# 모듈 객체를 호출하는 순간 순방향 메서드가 실행됌.

# class Modle(nn.Module):
#     def __init__(self):
#         super().__init__() #부모에서의 init 메소드를 자식에서도 수행 #부모 클래스의 속성을 사용할 수 있음
#         # self.conv1과 self.conv2는 모델의 매개변수
#         self.conv1 = nn.Conv2d(1,20,5) 
#         self.conv2 = nn.Conv2d(20,20,5)
#     # 초기화 메서드에서 선언한 모델 매개변수를 활용해 신경망 구조를 설계
#     # super함수로 부모 클래스를 초기화 했으므로 역방향 연산은 정의하지 않아도 괜찮다.
#     def forward(self,x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self,conv2(x))
#         return x