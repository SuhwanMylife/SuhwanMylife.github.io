---
layout: post
title: "Graph Travel 2화 DeepWalk 활용한 Node Classification"
tags:
  - Graph
  - Tutorial
---

<br>

첫 GNN 알고리즘 (Node Classification)

사용할 Dataset: Cora 데이터셋

- 논문 인용 네트워크를 나타내는 데이터셋

**인접 행렬(Adjecency Matrix)의 중요성**

인접 행렬: Graph의 기본적인 Structure Information 제공

큰 그래프의 복잡한 구조 탐색하고 이해하기 위해서 Adjecency Matrix의 도움 필요

- 정확히는 Random Walk라는 알고리즘이 Adjecency Matrix의 Graph Structure information을 활용하도록 해야 함

# **Random Walk**

Random Walk: Graph의 특정 Node에서 시작해 임의의 Direction으로 이동하는 과정

노드간의 연결성(Connectivity)뿐만 아니라 Node 간의 전체적인 연결 패턴 파악이 가능

- 노드 방문 과정에서 랜덤하게 선택된 노드들 사이에서도 함께 자주 나타나는 Node들은 서로 ‘가까운 관계’라 판단 가능
- Network homophily hypothesis 네트워크 동질성 가정
    - 가까운 거리에 있는 Node들은 서로 비슷한 Attribute(특성, Feature)를 갖는다

**Random Walk 작동 방식**

1. Start Node(시작 노드) 선택
2. 현재 노드에 연결된 이웃 노드 중 하나를 random하게 선택해 그 방향으로 이동
3. 일정 횟수 혹은 조건을 만족할 때까지 2번 과정 반복

Random Walk 구현 코드

```python
def random_walk(G, start_node, walk_length):
    walk = [start_node] # 시작 노드를 포함하는 리스트 walk를 생성
    # walk 리스트는 무작위 경로의 노드를 저장하는 데 사용

    for i in range(walk_length - 1):
        # 시작 노드가 이미 walk안에 있으니, -1.
        neighbors = list(G.neighbors(walk[-1]))
        if len(neighbors) == 0:
            break
        next_node = random.choice(neighbors)
        walk.append(next_node)
    return walk
```

random walk의 세부 동작 방식이나 사용 목적에 따라 Startnode의 선택이나 walk의 수 등을 다르게 설정

- 일부 노드만 중점적으로 분석하고자 하는 경우 그 노드들에서만 random walk 수행
- or 일부 노드를 샘플링해 random walk 수행

random walk 개념 차용하여 local information 사용하는 것은 전체 그래프를 분석하는 것보다 효율적인 방법을 제공

- 전체 그래프 처리 시 computational cost(계산 시간)가 높아 비효율적
- 때문에 random walk를 활용하여 적은 정보들을 사용해 global attribute를 추정

```python
print(random_walk(G,0,20))
```

![Untitled](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/b0fe15b5-2683-4f81-95bf-2ff6b01cf0c8)

특정 노드가 반복해서 출력되는 현상

- 빈번하게 출력되는 만큼 특정 노드가 그래프 내에서 중요한 역할을 하는 것일 수 있음!

이제 좀 더 어려운 개념으로…

Random Walk는 Node Embedding 생성에 사용되는 **DeepWalk**, Node2Vec같은 알고리즘의 기반임!

- Node 사이의 경로를 sampling하는 데에 Random walk 사용
- 두 알고리즘은 Node similarity를 capture한 Vector Representation(Embedding)을 학습

여기서 **임베딩(embedding)**이란?

고차원 공간에서 단어나 이미지와 같은 데이터 조각을 수학적으로 표현한 벡터뭉치

# **Deep Walk**

Deep Walk: Graph Embedding을 생성하는 알고리즘 중 하나

- Graph Network의 노드 간 Similarity Score(유사도 점수)를 측정하는 방식으로 작동
- 서로 가까운 노드들의 유사도 점수는 높게, 멀리 떨어진 노드들의 유사도 점수는 낮게 설정하여 그래프 네트워크의 Structure Feature(구조적 특성)을 학습하고, **각 Node를 효과적으로 represent하는 low dimentional vector를 생성**하는 알고리즘

Deep Walk 이해를 위해 Random Walk와 비교해봅시다.

### **Random vs Deep**

Random walk: 그래프 상의 노드들을 무작위로 순회하는 가장 베이직한 방법론

DeepWalk: Random walk를 사용하여 Node를 Vector Space에 투영하는 방법론

- DeepWalk = Random walk + Word2Vec?

**Random Walk의 역할**: Random walk의 실행결과로 Node Sequence(일종의 문장)를 생성. Sequence에 포함되는 각 Node는 ‘단어’에 해당

**Word2Vec의 역할**: 일반적으로는 단어들의 벡터 표현(임베딩)을 학습하는 데 사용되는 ML 모델

- 비슷한 문맥에서 등장하는 단어들이 벡터 공간에서 가까이 위치하도록 단어들의 벡터를 조정하는 방식으로 작동
- 최종적으로 모델 임베딩은 단어 사이의 의미적 관계를 반영

⇒ 생성된 sequence(문장)를 활용하여 Word2Vec(특히 **Skip-gram** 아키텍처) 모델이 각 노드의 **Vector Representation**(벡터 표현, Embedding)을 학습

여기서 **skip-gram** 아키텍처, **Vector Representation**이 뭔데?

**skip-gram** 아키텍처: 중심 단어를 기반으로 주변 단어를 예측하는 방식으로 작동

- 주변 단어의 문맥을 이해하는 데 효과적

**Vector Representation**: 단어나 노드를 고차원 벡터로 표현하는 것 자체를 의미

- 여기서는 Node의 attribute를 캡처하며, 이는 원래의 데이터 공간에서는 알아챌 수 없는 패턴을 발견하는 데 도움이 될 수 있음

⇒ Deep Walk의 목표: **Feature Representations of Nodes(=embedding, Vector Representation)**를 만들어내는 것

지금부터의 과정

1. Unbiased Random Walk 구현
2. Word2Vec을 통해 Node Embedding 학습
3. 임베딩을 t-SNE를 통해 2차원으로 축소하여 시각화
4. Random Forest 모델로 분류 작업 수행

random walk

```python
def unbiased_random_walk(G, node, walk_length):
    walk = [node]
    for _ in range(walk_length - 1):
        neighbors = list(G.neighbors(walk[-1]))
        if len(neighbors) == 0:
            break
        walk.append(random.choice(neighbors))
    return walk

# Random Walk 생성
walks = [unbiased_random_walk(G, node, 10) for node in range(data.num_nodes) for _ in range(10)]
```

위 코드에서 Biased random walk 함수 정의하고 walk 추출, 이를 10번 반복

```python
# String 형태로 변환 (Word2Vec 입력을 위해)
walks = [[str(node) for node in walk] for walk in walks]

# Word2Vec 학습
model = Word2Vec(walks, vector_size=100, window=5, min_count=0, hs=1, sg=1, workers=4, epochs=10)

# 노드 임베딩 추출
embeddings = np.array([model.wv.get_vector(str(i)) for i in range(data.num_nodes)])
```

Word2Vec 학습을 위해 walks 내의 모든 노드를 문자열로 변환

Word2Vec 모델 학습 결과로 Node가 걸어간 경로들로 Embedding 추출

```python
# t-SNE를 통한 시각화
embeddings_2d = TSNE(n_components=2).fit_transform(embeddings)
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=data.y)
plt.show()
```

![Untitled 1](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/4f6c798f-bbf9-46a3-b86a-b125ae42cb3f)

t-sne: 복잡한 고차원을 2~3차원으로 축소

- 노드 간의 관계 이해에 도움

임베딩이 잘 학습되었는지, 원하는 attribute를 잘 capture하고 있는지 확인

한 가지 궁금증, 임베딩 시 train_set과 test_set 노드를 합쳐서 random walk 수행 후 임베딩하는 것이 모델링 과정에서 전혀 상관없는 것인가?

- 이후 모델 훈련 과정까지 쭉 보니 train_set과 test_set 노드를 합쳐 임베딩하고 있는 것 같은데, 내가 정확하게 본 것인지, 이렇게 해도 문제가 없다면 왜 문제가 없는지가 궁금함. 아마도 튜토리얼이니까 그냥 합쳐서 한 것일지도?

# Node Classification

각 Node마다 주어진 label을 활용하여 Node classification 진행

Node Classification: 그래프 내의 각 노드를 특정 카테고리나 클래스로 분류하는 작업

- 예) 한 비행기에 탑승한 승객들의 취미, 직업 국적 등 예측, 유저의 행동 패턴 예측, 추천 시스템에서 유저에게 가장 relevance가 높은 아이템 추천
- 여기서 GNN의 포인트: 그래프를 구성하는 Node의 Attribute(=Feature, 특성)을 학습
    - 학습 과정에서 target node의 정보뿐 아니라 연결되어 있는 neighborhood nodes의 정보도 함께 고려

다음의 과정을 통해 이번 여행 마무리

1. RandomForestClassifier에 Embedding 학습
2. Node classification 진행
3. 분류 보고서 확인

```python
# 레이블이 있는 노드만 선택
labels = data.y.numpy()
idx_train = data.train_mask.numpy()
idx_test = data.test_mask.numpy()

X_train, y_train = embeddings[idx_train], labels[idx_train]
X_test, y_test = embeddings[idx_test], labels[idx_test]

# 랜덤 포레스트 분류기 학습
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# 예측 및 성능 평가
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))
```

![Untitled 2](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/6f5c03bb-c243-40f6-ae1e-adbc8bb8c243)

**결과 분석**

class 3: support가 가장 높은 반면, recall이 상대적으로 낮음

class 2: precision과 recall이 모두 높아 잘 분류되고 있는 걸로 보임

전체적으로는 accuracy가 0.68로, test set의 약 68%를 정확하게 분류했지만 class별 f1-score 수치 격차가 큼

- 모델별로 데이터의 특정 패턴을 더 잘 학습한 경우 or 데이터 불균형이 있는 경우
    
    ⇒ 모델 개선 시 class별 성능 차이를 줄이는 데에 집중하는 게 좋겠음
    

1번 튜토리얼에서 봤던 하늘색 점 뭉치를 이번에는 label 정보를 포함해서 다시 출력해보자.

```python
# 하나, 둘, 셋, 치즈! 📸

plt.figure(figsize= (12,12), dpi= 300)
plt.axis('off')
nx.draw_networkx(G,
               pos = nx.spring_layout(G, seed = 0),
               node_color = labels,
               node_size = 300,  # 노드 크기 조정
               cmap = 'coolwarm',
               font_size = 14,
               font_color = 'white',
               edge_color = 'grey',  # 엣지 색상 설정
               width = 1,  # 엣지 두께 설정
               with_labels = False)  # 노드 라벨 표시 여부. True로 바꾸면 노드 이름이 출력됩니다!
```

![Untitled 3](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/d5e91fbb-c612-4bce-a425-cd92bae4cbbd)

더 이상 처음에 보았던 하늘색 점 뭉치가 아니다!

