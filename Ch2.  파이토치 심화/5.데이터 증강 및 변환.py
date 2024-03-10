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

import nlpaug.augmenter.word as naw

texts = [
    "Those who can imagine anything, can create the impossible.",
    "We can only see a short distance ahead, but we can see plenty there that needs to be done.",
    "If a machine is expected to be infallible, it cannot also be intelligent"
]

aug = naw.ContextualWordEmbsAug(model_path="bert-base-uncased",action="insert") # ContextualWordEmbsAug 클래스를 활용하여 Bert 모델을 통해 현재 문장 상황에 맞는 단어를 찾아 삽입하여 반환, insert는 단어 추가와 단어 대체 기능을 제공
augmented_texts = aug.augment(texts) # ContextualWordEmbsAug 클래스의 augment 클래스를 활용하여 기존 데이터를 증강 무조건 리스트 구조로 반환

for text, augmented in zip(texts,augmented_texts):
    print(f'src : {text}')
    print(f'dst : {augmented}')
    print('----------------------')

# src : Those who can imagine anything, can create the impossible.
# dst : now those who do can imagine seemingly anything, can create the next impossible.
# ----------------------
# src : We can only see a short distance ahead, but we can see plenty there that needs to be done.
# dst : we can only also see mostly a short safe distance up ahead, but we can see plenty there too that still needs much to be done.
# ----------------------
# src : If a machine is expected to be infallible, it cannot also be intelligent
# dst : if thus a machine is expected to indeed be infallible, all it usually cannot also be entirely intelligent
# ----------------------
    

