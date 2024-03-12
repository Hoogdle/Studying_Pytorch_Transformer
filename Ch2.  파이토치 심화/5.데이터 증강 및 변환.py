### 데이터 증강(Data Augmentation)
# 데이터가 가지는 고유한 특징을 유지한 채 변형하거나 노이즈를 추가해 데이터 세트의 크기를 인위적으로 늘리는 방법
# 학습 데이터 수집이 어려운 경우 기존 학습 데이터를 재가공해 원래 데이터와 유사하지만 새로운 데이터를 생성할 수 있다.
# 너무 많은 변형이나 너무 많은 노이즈를 추가 한다면 기존 데이터의 특징이 파괴될 수 있다.
# 특정 알고리즘을 통해 생성되므로 데이터 수집보다 더 많은 비용이 들 수도 있다.


### 텍스트 데이터
# 텍스트 데이터 증강은 자연어 처리 모델을 구성할 때 데이터세트의 크기를 늘리기 위해 사용
# 해당 교재에서는 '자연어 처리 데이터 증강(NLPAUG)' 라이브러리를 활용해 텍스트 데이터를 증강.
# 해당 라이브러리는 음성 데이터 증강 또한 지원.


### 삽입 및 삭제
# 삽입 :  의미 없는 문자나 단어, 문장 의미에 영향을 끼치지 않는 수식어 등을 추가하는 방법, 임의의 단어나 문자를 기존 텍스트에 덧붙여 사용.
# 삭제 : 임의의 단어나 문자를 삭제해 데이터의 특징을 유지하는 방법
# 두 프로세스 모두 "문장의 의미는 유지한 채 시퀀스를 변경"

### 단어 삽입

# import nlpaug.augmenter.word as naw

# texts = [
#     "Those who can imagine anything, can create the impossible.",
#     "We can only see a short distance ahead, but we can see plenty there that needs to be done.",
#     "If a machine is expected to be infallible, it cannot also be intelligent"
# ]

# aug = naw.ContextualWordEmbsAug(model_path="bert-base-uncased",action="insert") # ContextualWordEmbsAug 클래스를 활용하여 Bert 모델을 통해 현재 문장 상황에 맞는 단어를 찾아 삽입하여 반환, insert는 단어 추가와 단어 대체 기능을 제공
# augmented_texts = aug.augment(texts) # ContextualWordEmbsAug 클래스의 augment 클래스를 활용하여 기존 데이터를 증강 무조건 리스트 구조로 반환

# for text, augmented in zip(texts,augmented_texts):
#     print(f'src : {text}')
#     print(f'dst : {augmented}')
#     print('----------------------')

# src : Those who can imagine anything, can create the impossible.
# dst : now those who do can imagine seemingly anything, can create the next impossible.
# ----------------------
# src : We can only see a short distance ahead, but we can see plenty there that needs to be done.
# dst : we can only also see mostly a short safe distance up ahead, but we can see plenty there too that still needs much to be done.
# ----------------------
# src : If a machine is expected to be infallible, it cannot also be intelligent
# dst : if thus a machine is expected to indeed be infallible, all it usually cannot also be entirely intelligent
# ----------------------
    



### 문자 삭제

# import nlpaug.augmenter.char as nac


# texts = [
#     "Those who can imagine anything, can create the impossible.",
#     "We can only see a short distance ahead, but we can see plenty there that needs to be done.",
#     "If a machine is expected to be infallible, it cannot also be intelligent"
# ]

# aug = nac.RandomCharAug(action="delete") #RandomCharAug 클래스로 무작위로 문자 삭제,insert,substitue,swap,delete 기능 제공
# augmented_texts = aug.augment(texts)

# for text,augmented in zip(texts,augmented_texts):
#     print(f'src : {text}')
#     print(f'dst : {augmented}')
#     print('==================')

# src : Those who can imagine anything, can create the impossible.
# dst : hos who can magn anything, can cree the ossible.
# ==================
# src : We can only see a short distance ahead, but we can see plenty there that needs to be done.
# dst : We can only see a shr distance hed, but we can see enty hre ht nes to be oe.
# ==================
# src : If a machine is expected to be infallible, it cannot also be intelligent
# dst : If a machine is xpeed to be infalbl, it cnnt lo be inlignt
    

### 교체 및 대체
# 교체는 단어나 문자의 위치를 교환하는 방법
# ex) '문제점을 찾지 말고 해결책을 찾으라' => '해결책을 찾으라 문제점을 찾지 말고'
# 잘못된 교체로 이상한 문자 생성가능, 데이터 특성에 따라 주의해 사용

# 대체는 단어나 문자를 임의의 단어나 문자로 바꾸거나 동의어로 변경하는 방법
# ex) '사과' => '바나나', '해' => '태양'

### 단어 교체 

# import nlpaug.augmenter.word as naw

# texts = [
#     "Those who can imagine anything, can create the impossible.",
#     "We can only see a short distance ahead, but we can see plenty there that needs to be done.",
#     "If a machine is expected to be infallible, it cannot also be intelligent"
# ]

# aug = naw.RandomWordAug(action='swap')
# augmented_texts = aug.augment(texts)

# for text,augmented in zip(texts,augmented_texts):
#         print(f'src : {text}')
#         print(f'dst : {augmented}')
#         print('==================')

# src : Those who can imagine anything, can create the impossible.
# dst : Can those who imagine anything can, the create impossible.
# ==================
# src : We can only see a short distance ahead, but we can see plenty there that needs to be done.
# dst : We only can see short a distance ahead, but see we can plenty that there needs done to be.
# ==================
# src : If a machine is expected to be infallible, it cannot also be intelligent
# dst : If a is machine expected to be infallible, it cannot also be intelligent
# ==================

# RandomWordAug 클래스를 활요해 무작위로 단어를 교체, 삽입,대체,교체,삭제 기능 제공
# 문맥을 파악하지 않고 교체하므로 사용시 주의
        

### 단어 대체(1)
# import nlpaug.augmenter.word as naw

# texts = [
#     "Those who can imagine anything, can create the impossible.",
#     "We can only see a short distance ahead, but we can see plenty there that needs to be done.",
#     "If a machine is expected to be infallible, it cannot also be intelligent"
# ]

# aug = naw.SynonymAug(aug_src="wordnet")
# augmented_texts = aug.augment(texts)

# for text,augmented in zip(texts,augmented_texts):
#         print(f'src : {text}')
#         print(f'dst : {augmented}')
#         print('==================')
# src : Those who can imagine anything, can create the impossible.
# dst : Those who fanny envisage anything, fire produce the impossible.
# ==================
# src : We can only see a short distance ahead, but we can see plenty there that needs to be done.
# dst : We can only see a short length ahead, only we can see enough there that need to make up done.
# ==================
# src : If a machine is expected to be infallible, it cannot also be intelligent
# dst : If a machine is expected to embody infallible, it cannot too personify well informed
# ==================
        
# SynonymAug 클래스는 워드넷(WordNet) 데이터베이스나 의역 데이터베이스(PPDB)를 화룡해 데이터를 증강
# 해당 기능은 문맥을 파악하여 동의어로 변경하는 것이 아닌 데이터베이스 내 유의어나 동의어로 변경함.(문맥과 전혀 다른 문장이 생성될 수도 있다)


### 단어 대체(2)
# import nlpaug.augmenter.word as naw

# texts = [
#     "Those who can imagine anything, can create the impossible.",
#     "We can only see a short distance ahead, but we can see plenty there that needs to be done.",
#     "If a machine is expected to be infallible, it cannot also be intelligent"
# ]
# reversed_tokens = [
#     ["can","can't","cannot","could"]
# ]

# reversed_aug = naw.ReservedAug(reserved_tokens=reversed_tokens)
# augmented_texts = reversed_aug.augment(texts)

# for text,augmented in zip (texts,augmented_texts):
#         print(f'src : {text}')
#         print(f'dst : {augmented}')
#         print('==================')

# src : Those who can imagine anything, can create the impossible.
# dst : Those who could imagine anything, could create the impossible.
# ==================
# src : We can only see a short distance ahead, but we can see plenty there that needs to be done.
# dst : We could only see a short distance ahead, but we could see plenty there that needs to be done.
# ==================
# src : If a machine is expected to be infallible, it cannot also be intelligent
# dst : If a machine is expected to be infallible, it could also be intelligent
# ==================

# ReserveAug 클래스는 입력 데이터에 포함되 단어를 특정한 단어로 대체하는 기능을 제공
# 가능한 모든 조합을 생성하거나 특정 글자나 문자를 reserved_tokens에서 선언한 데이터로 변경


### 역번역(Back-Translation)
# 입력 텍스트를 특정 언어로 번역한 다음 다시 본래의 언어로 번역하는 방법 ex) 영어를 한국어로 번역한 다음 다시 영어로 번역하는 과정
# 본래의 언어로 번역하는 과정에서 원래 텍스트와 유사한 텍스트가 생성됌.(패러프레이징-Paraphrasing 효과)
# 역번역은 모델 성능에 크게 영향을 받음, 기계 번역의 품질을 평가하는데 사용하기도 함.

### 역번역
# import nlpaug.augmenter.word as naw

# texts = [
#     "Those who can imagine anything, can create the impossible.",
#     "We can only see a short distance ahead, but we can see plenty there that needs to be done.",
#     "If a machine is expected to be infallible, it cannot also be intelligent"
# ]

# back_translation = naw.BackTranslationAug(
#         from_model_name="facebook/wmt19-en-de", # 영어에서 독일어로 번역
#         to_model_name="facebook/wmt19-de-en" # 독일어에서 영어로 역번역
# )

# augmented_texts = back_translation.augment(texts)

# for text,augmented in zip (texts,augmented_texts):
#         print(f'src : {text}')
#         print(f'dst : {augmented}')
#         print('==================')

# 원문과 크게 의미가 달리지지 않음, 모델의 성능에 영향을 많이 받는다. 
# 두 개의 모델을 활용해 데이터를 증강하므로 데이터 증강법 중 가장 많은 리소스를 소모한다.
        


### 이미지 데이터
# 이미지 데이터 증강 방법으로는 '회전','대칭','이동','크기 조정' 등이 있다.
# 이 교재에서는 토치비전(torchvision) 라이브러리와 이미지 증강(imgaug) 라이브러리를 활용해 이미지 데이터를 증강

# 이미지 데이터 증강 방법은 토치비전 라이브러리의 변환(transforms) 모듈을 통해 수행할 수 있다.


### 통합 클래스 및 변환 적용 방식

# from PIL import Image
# from torchvision import transforms

# transform = transforms.Compose( # 통합 클래스 #여러 모델의 매개변수를 묶어주는 Sequential과 동일한 역할
#     [
#         transforms.Resize(size=(512,512)), # 512 X 512 크기로 변환
#         transforms.ToTensor() # 텐서 타입으로 변환(PIL.Image => Tensor)
#         # [0~255] 범위의 픽셀값을 [0.0~1.0] 사이의 값으로 최대 최소 정규화를 수행, [높이,너비,채널] => [채널,높이,너비] 형태로 변환
#     ]
# )

# image = Image.open('C:\\Users\\rlaxo\\Desktop\\datasets\\images\\cat.jpg')
# transformed_image = transform(image)

# print(transformed_image.shape) #torch.Size([3, 512, 512])



### 회전 및 대칭
# 학습 이미지를 회전하거나 대칭한 이미지 데이터를 추가해 학습을 진행한다면 변형된 이미지가 들어오더라도 강건한 모델을 구축할 수 있다.

# from PIL import Image
# from torchvision import transforms

# transform = transforms.Compose( 
#     [
#         transforms.RandomRotation(degrees = 30, expand = False, center = None), # expand = True : 회전시 생기는 여백이 생성되지 않는다. # center를 입력하지 않으면 왼쪽 상단을 기준으로 회전
#         transforms.RandomHorizontalFlip(p=0.5), #50% 확률로 대칭 수행 수평선 기준
#         transforms.RandomVerticalFlip(p=0.5) #50% 확률로 대칭 수행 수직선 기준
#     ]
# )

# image = Image.open('C:\\Users\\rlaxo\\Desktop\\datasets\\images\\cat.jpg')
# transformed_image = transform(image)

# transformed_image.show()

# 이미지를 -30~30(degree) 사이로 회전시키면서 수평 대칭과 수직 대칭을 50% 확률로 적용.


### 자르기 및 패딩
# 객체 인식과 같은 모델 구성시 학습 데이터의 크기가 일정하지 않거나, 주요한 객체가 일부 영역에만 작게 존재할 수 있다.
# 이런 경우 객체가 존재하는 위치로 이미지를 잘라 불필요한 특징을 감소시키거나 패딩을 주어 이미지 크기를 동일한 크기로 맞출 수 있따.

# from PIL import Image
# from torchvision import transforms

# transform = transforms.Compose( 
#     [
#         transforms.RandomCrop(size=(512,512)), # 512 X 512 사이즈로 이미지 자르기
#         transforms.Pad(padding=50, fill=(127,127,255),padding_mode="constant") # padding, 이미지를 확장, padding = 50 : 50만큼 확장, 패딩은 모든 방향으로 적용, 612 X 612 크기로 변환
#         # 패딩 : 이미지에 경계를 덧대는 방식, 본래의 이미지 크기 유지 가능
#         # padding_mode = "constant" fill(127,127,255) , RGB(127,127,255)로 테두리가 생성됨.
#     ]
# )

# image = Image.open('C:\\Users\\rlaxo\\Desktop\\datasets\\images\\cat.jpg')
# transformed_image = transform(image)

# transformed_image.show()


### 크기 조정
# 학습을 원활하게 진행하기 위해서는 이미지의 크기가 모두 일정해야 한다.
# from PIL import Image
# from torchvision import transforms

# transform = transforms.Compose( 
#     [
#         transforms.Resize(size=(512,512))
#     ]
# )

# image = Image.open('C:\\Users\\rlaxo\\Desktop\\datasets\\images\\cat.jpg')
# transformed_image = transform(image)

# transformed_image.show()


### 변형
# 기하학적 변환을 통해 이미지를 변경
# 기하학적 변환 : 인위적인 확대,축소,위치 변경, 회전, 왜곡으로 이미지의 형태를 변환, 아핀 변환과 원근 변환으로 나뉨
# 아핀 변환 : 2 X 3 행렬을 사용하여 행렬 곱셈에 벡터 합을 활용해 표현할 수 있는 변환
# 원근 변환 : 3 X 3 행렬을 사용, 호모그래피로 모델링할 수 있는 변환

# 아핀 변환
# 각도(degree),이동(translate),척도(scale),전단(shear)을 입력해 이미지를 변형.
# from PIL import Image
# from torchvision import transforms

# transform = transforms.Compose( 
#     [
#         transforms.RandomAffine(
#             degrees=15, translate=(0.2,0.2),
#             scale=(0.8,1.2),shear=15
#         )
#     ]
# )

# image = Image.open('C:\\Users\\rlaxo\\Desktop\\datasets\\images\\cat.jpg')
# transformed_image = transform(image)

# transformed_image.show()


# 색상 변환
# 이미지 데이터의 특징은 픽셀값의 분포나 패턴에 크게 좌우된다.
# 이미지를 분석할 때 특정 색상에 편향되지 않도록 픽셀값을 변환하거나 정규화하면 모델을 더 일반화해 분석 성능을 향상시킬 수 있따.

# 색상 변환 및 정규화
# from PIL import Image
# from torchvision import transforms

# transform = transforms.Compose( 
#     [
#         transforms.ColorJitter( # 색상 변환 클래스, 밝기(brightness),대비(contrast),채도(saturation),색상(hue)를 변환
#             brightness=0.3,contrast=0.3,
#             saturation=0.3,hue=0.3
#         ),
#         transforms.ToTensor(), # 정규화 클래스는 PIL.Image 형식이 아닌 Tensor 형식으로 입력을 받는다.
#         transforms.Normalize( # 픽셀의 평균과 표준펴차를 활용해 정규화, 데이터를 정규화해 모델 성능을 높임.
#             mean=[0.485,0.456,0.406],
#             std=[0.229,0.224,0.225]
#         ),
#         transforms.ToPILImage()

#     ]
# )

# image = Image.open('C:\\Users\\rlaxo\\Desktop\\datasets\\images\\cat.jpg')
# transformed_image = transform(image)

# transformed_image.show()




### 노이즈
# 이미지 처리 모델은 주로 합성곱 연산을 통해 학습을 진행.
# 노이즈 추가는 특정 픽셀값에 편향되지 않도록 임의의 노이즈를 추가해 모델의 일반화 능력을 높이는데 사용된다.
# 학습 데이터에 직접 포함하지 않더라도 테스트 데이터에 노이즈를 추가해 일반화 능력이나 강건성을 평가하는데 사용된다.
# import numpy as np
# from PIL import Image
# from torchvision import transforms
# from imgaug import augmenters as iaa # 이미지 증강 클래스 사용

# class IaaTransforms:
#     def __init__(self): # 이미지 증강 방법 설정
#         self.seq = iaa.Sequential([
#             iaa.SaltAndPepper(p=(0.03,0.07)), #점잡음 적용
#             iaa.Rain(speed=(0.3,0.7)) #빗방울 레이어 적용
#         ])
# #__call__, 클래스의 인스턴스를 함수처럼 호출 가능하게 만들어줌.
# # class Test:
# #     def __call__(self, x):
# #         return x**2
 
# # T = Test()
# # print(T(5))  # 출력: 25
#     def __call__(self,images): 
#         images = np.array(images) # 이미지 증강 라이브러리의 증강 클래스는 넘파이의 ndarray 클래스를 입력과 출력으로 사용
#         augmented = self.seq.augment_image(images)
#         return Image.fromarray(augmented) # ndarray를 다시 PIL Image 형식으로 변환
    
# transform = transforms.Compose([
#     IaaTransforms()
# ])

# image = Image.open('C:\\Users\\rlaxo\\Desktop\\datasets\\images\\cat.jpg')
# transformed_image = transform(image)

# transformed_image.show()


### 컷아웃 및 무작위 지우기 
# 컷아웃 : 이미지에서 임의의 사각형 영역을 삭제하고 0의 픽셀값으로 채우는 방법
# 무작위 지우기 : 이미지에서 임의의 사각혀 영역을 삭제하고 무작위 픽셀값으로 채우는 방법
# 컷아웃은 동영상에서 폐색 영역에 대해 모델을 더 강건하게 만들어주고 무작위 지우기는 일부 영역이 누락되었을 때 더 강건한 모델을 만들 수 있게함.
# 즉, 이미지의 객체 일부가 누락되더라도 모델을 견고하게 만드는 증강 방법

# 무작위 지우기
# from PIL import Image
# from torchvision import transforms

# transform = transforms.Compose( 
#     [
#         transforms.ToTensor(),
#         transforms.RandomErasing(p=1.0,value=0), #컷아웃 #지우는 확률 100퍼
#         transforms.RandomErasing(p=1.0,value="random"), #무작위 지우기 #지우는 확률 100퍼
#         transforms.ToPILImage()
#     ]
# )

# image = Image.open('C:\\Users\\rlaxo\\Desktop\\datasets\\images\\cat.jpg')
# transformed_image = transform(image)

# transformed_image.show()


### 혼합 및 컷믹스
# 혼합 : 두 개 이상의 이미지를 혼합(Blending)해 새로운 이미지를 생성하는 방법, 픽셀 값을 선형으로 결합해 새 이미지 생성
# 생성된 이미지는 두 개의 이미지가 겹쳐 흐릿한 형상을 지님
# 혼합된 데이터로 학습시 레이블링이 다르게 태깅돼 있어어도 더 낮은 오류를 보이며, 다중 레이블 문제(하나의 객체가 두 개 이상의 클래스에 포함되는 것)에도 견고한 모델을 만들 수 있다.

# 컷믹스 : 이미지 패치 영역에 다른 이미지를 덮어씌우는 방법.
# 패치 위에 새로운 패치를 덮어씌워 비교적 자연스러운 이미지를 구성한다.
# 특정 영역을 기억해 인식하는 문제를 완화하며, 이미지 전체를 보고 판단할 수 있게 일반화 할 수 있다.
 
# 혼합은 이미지 크기만 맞다면 쉽게 가능하지만, 컷믹스는 패치 영역의 크기와 비율을 고려해 덮어씌워야 한다.

### 혼합
# import numpy as np
# from PIL import Image
# from torchvision import transforms

# class Mixup:
#     def __init__(self,target,scale,alpha=0.5,beta=0.5): # target : 혼합하려는 이미지, scale : 이미지 크기 조절, alpha,beta : 혼합 비율 설정
#         self.target = target
#         self.scale = scale
#         self.alpha = alpha
#         self.beta = beta
#     def __call__(self,image):
#         image = np.array(image)
#         target = self.target.resize(self.scale)
#         target = np.array(target)
#         mix_image = image * self.alpha + target * self.beta
#         return Image.fromarray(mix_image.astype(np.uint8))
    
# transform = transforms.Compose(
#     [
#         transforms.Resize((512,512)),
#         Mixup(
#             target = Image.open('C:\\Users\\rlaxo\\Desktop\\datasets\\images\\dog.jpg'),
#             scale = (512,512),
#             alpha = 0.5,
#             beta = 0.5
#         )
#     ]
# )

# image = Image.open('C:\\Users\\rlaxo\\Desktop\\datasets\\images\\cat.jpg')
# transformed_image = transform(image)

# transformed_image.show()


# 텍스트 및 이미지 증강 방법은 모든 데이터에 적용하는 것이 아닌, 일부 데이터에만 적용해 증강한다.
# 데이터 증강은 모델 학습에 있어서 보편적으로 사용되는 방법, 부족한 데이터를 확보하고 모델의 일반화 성능을 최대로 끌어올릴 수 있다.