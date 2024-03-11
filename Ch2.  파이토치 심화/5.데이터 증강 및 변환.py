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

# transforms = transforms.Compose( # 통합 클래스
#     [
#         transforms.Resize(size=(512,512)), # 512 X 512 크기로 변환
#         transforms.ToTensor() # 텐서 타입으로 변환(PIL.Image => Tensor)
#         # [0~255] 범위의 픽셀값을 [0.0~1.0] 사이의 값으로 최대 최소 정규화를 수행, [높이,너비,채널] => [채널,높이,너비] 형태로 변환
#     ]
# )

# image = Image.open('C:\\Users\\rlaxo\\Desktop\\datasets\\images\\cat.jpg')
# transformed_image = transforms(image)

# print(transformed_image.shape) #torch.Size([3, 512, 512])



### 회전 및 대칭
# 학습 이미지를 회전하거나 대칭한 이미지 데이터를 추가해 학습을 진행한다면 변형된 이미지가 들어오더라도 강건한 모델을 구축할 수 있다.

# from PIL import Image
# from torchvision import transforms

# transforms = transforms.Compose( 
#     [
#         transforms.RandomRotation(degrees = 30, expand = False, center = None), # expand = True : 회전시 생기는 여백이 생성되지 않는다. # center를 입력하지 않으면 왼쪽 상단을 기준으로 회전
#         transforms.RandomHorizontalFlip(p=0.5), #50% 확률로 대칭 수행 수평선 기준
#         transforms.RandomVerticalFlip(p=0.5) #50% 확률로 대칭 수행 수직선 기준
#     ]
# )

# image = Image.open('C:\\Users\\rlaxo\\Desktop\\datasets\\images\\cat.jpg')
# transformed_image = transforms(image)

# transformed_image.show()

# 이미지를 -30~30(degree) 사이로 회전시키면서 수평 대칭과 수직 대칭을 50% 확률로 적용.

