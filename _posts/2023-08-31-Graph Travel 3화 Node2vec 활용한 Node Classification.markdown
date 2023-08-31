---
layout: post
title: "Graph Travel 3화 Node2vec 활용한 Node Classification"
tags:
  - Graph
  - Tutorial
---

<br>

1. Embedding 퀄리티를 높일 수 있는 방법은 뭘까요?
2. Deepwalk와 Node2vec의 차이점은 무엇일까요?

```python
# 필요한 모든 라이브러리 임포트
import torch
import torch_geometric
from torch_geometric.datasets import Planetoid
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import optuna
from sklearn.metrics import accuracy_score

# CORA 데이터셋 로드
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# torch_geometric의 데이터를 NetworkX 그래프로 변환
edge_index = data.edge_index
edges = edge_index.t().numpy()
G = nx.from_edgelist(edges, create_using=nx.Graph())
```

# Node2vec이란?

DeepWalk와 마찬가지로 Random Walk와 Word2Vec 알고리즘을 동일하게 사용한다.

**그럼 DeepWalk랑 뭐가 다른데?**

Random Walk시의 Bias 유뮤

**Walk에 Bias 부여하는 것이 미치는 영향**

Graph의 본질은 Relation이기에, 우리는 어떤 노드까지가 실제로 연관성이 있다고 판단할지를 정의해야만 한다.

- 풀고자 하는 Task에 적합한 Sampling Strategy 수립할 필요 있음
    - 멀리까지 갈지, 혹은 주변의 다양한 노드들을 탐색할지 결정
    - DFS, BFS
    - BFS: 이전에 보았던 것과 비슷한 주변 노드들을 둘러보는 전략. 가장 가까운 노드를 먼저 방문하고 그 다음으로 가까운 노드를 방문하는 식으로 샘플링
    - DFS: 이전에 본 적 없는 Node들을 둘러보는 전략. 한 뱡향으로 최대한 깊게 탐색하고, 더 이상 방문할 노드가 없으면 이전 노드로 돌아와 다른 방향을 탐색하는 식으로 샘플링

**Search Bias Probability**

DFS, BFS를 적절히 조정하여 최적의 여행 계획을 세울 수 있음: 파라미터 p, q

- p: return parameter
    
    값이 높을수록 본 적 없는 새로운 Node를 찾는다. 이 값이 크면 이전에 방문한 노드로 돌아가는 것을 방지.
    
- q: in-out parameter
    
    타겟 노드의 이웃이 아닌 노드로 이동하는 확률을 제어. 이 값이 크면 타겟 노드의 주변을 탐색하며 너무 멀리 가지 않게 되고, 반대로 작아지면 간접적으로 연결되어 있는 노드를 방문하며 먼 범위까지 탐색.
    

```python
# torch_geometric의 데이터를 NetworkX 그래프로 변환
edge_index = data.edge_index
edges = edge_index.t().numpy()
graph_nx = nx.from_edgelist(edges, create_using=nx.Graph())

# 중심 노드를 선택 (여기서는 0을 선택)
center_nodes = [0]

# 모든 이웃 노드를 저장할 리스트 초기화
all_neighbors = []

# 중심 노드들의 모든 이웃 노드를 all_neighbors에 추가
for selected_node in center_nodes:
    neighbors = list(graph_nx.neighbors(selected_node))
    all_neighbors += neighbors

# 중심 노드들도 all_neighbors에 추가
all_neighbors += center_nodes

# all_neighbors에 포함된 모든 노드들만으로 구성된 subgraph를 추출
subgraph_nx1 = graph_nx.subgraph(all_neighbors)

# 그래프 그리기
pos = nx.spring_layout(subgraph_nx1, seed=42)  # Spring layout 사용
nx.draw(subgraph_nx1, pos, with_labels=True, node_color="skyblue")

plt.show()
```

![Untitled](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/3ad66d6c-4cdf-4ff9-976c-3ceeef94a421)

방금 배운 개념을 위 그래프에 적용해보자.

먼저, 우리가 walk를 시작한 타겟 노드는 node 1862, 현재 node 0에 도착해 있다고 가정. 이 다음 walk를 결정하는 bias를 어떻게 이해해볼 수 있을까?

1. node 0이 고를 수 있는 선택지는 3가지: node 1862, node 2582, node 633
2. 이 때, 파라미터 p가 높다면 node 0 → node 1862로 되돌아갈 확률이 낮아짐
    - p값이 높아질수록 walk는 DFS처럼 작동
    - q가 높다면 node 0 → node 633으로 나아갈 확률이 높아짐 - BFS처럼 작동
3. node 1682와 node 2582 사이에 edge가 있음 ⇒ node 0 → node 2852로 나아갈 unnormalized probability는 1
    - random walk가 node 0로부터 node 2852로 이동할 확률은 1. 이를 통해 이전 노드를 떠나 현재 노드 주변의 이웃을 탐색하도록 장려가 가능해짐

이렇게 Node Sampling 전략 수립 끝! 이제 여행의 방향을 주체적으로 정해봅시다.

# Social Network Concept

잠시 다른 이야기로, 두 가지 컨셉을 짚고 넘어갑시다: Structural Equivalence, Homophily

- 두 개념은 유저의 선호도를 추론하고, 관심을 가질 만한 항목을 예측하는 데 중요한 역할
    - 우리가 이후에 구현할 추천 시스템의 효율성과 정확성을 위해 미리 알아봅시다!

**Structural Equivalence**

두 개 이상의 노드가 유사한 연결 패턴을 가지고 있는지 판단할 때 사용

- 노드 A와 B의 neighbor nodes 체크했을 때 겹치는 노드가 많다면 Structural Equivalent하다고 말할 수 있음

**Homophily**

비슷한 특성을 가진 노드들은 서로 연결될 가능성이 높다는 개념

- 노드 C와 D의 Attribute가 비슷하다면, 두 노드가 neighborhood일 가능성(Edge로 연결되어 있을 가느성)이 높다고 할 수 있음

그렇다면, 두 개념은 상호 배타적?

- 그렇지 않습니다. 두 개념이 서로 연관될 수는 있지만, 반드시 서로에게 직접적인 영향을 미치는 것은 아니다.
    - 예) 서로 동일한 선호나 취향을 가진다고 해서 비슷한 사회적 위치나 인간관계를 가지고 있다는 법은 없다.

이제 새로운 전략을 가지고 더 복잡한 그래프를 만나봅시다.

```python
# torch_geometric의 데이터를 NetworkX 그래프로 변환
edge_index = data.edge_index
edges = edge_index.t().numpy()
graph_nx = nx.from_edgelist(edges, create_using=nx.Graph())

# 중심 노드를 선택 (여기서는 1666을 선택)
center_nodes = [1666]

# 모든 이웃 노드를 저장할 리스트 초기화
all_neighbors = []

# 중심 노드들의 모든 이웃 노드를 all_neighbors에 추가
for selected_node in center_nodes:
    neighbors = list(graph_nx.neighbors(selected_node))
    all_neighbors += neighbors

    # 이웃의 이웃(1차 이웃)도 all_neighbors에 추가
    for neighbor in neighbors:
        neighbors_of_neighbor = list(graph_nx.neighbors(neighbor))
        all_neighbors += neighbors_of_neighbor

# 중심 노드들도 all_neighbors에 추가
all_neighbors += center_nodes

# 중복을 제거하기 위해 all_neighbors를 set로 변환 후 다시 list로 변환
all_neighbors = list(set(all_neighbors))

# all_neighbors에 포함된 모든 노드들만으로 구성된 subgraph를 추출
subgraph_nx2 = graph_nx.subgraph(all_neighbors)

# 그래프 그리기
pos = nx.spring_layout(subgraph_nx2, seed=42)  # Spring layout 사용
nx.draw(subgraph_nx2, pos, with_labels=True, node_color="skyblue")

plt.show()
```

![Untitled 1](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/37e4c5c6-9cbb-4d88-97d7-15c1e29d89c9)

어디까지 이웃 노드로 고려하는 게 좋을지 제어하는 것은 Graph 학습에서 굉장히 중요한 문제이다. walk 자체가 sequence, 즉 embedding에 관여하기 때문이다.

파라미터 p와 q를 적절히 조정함으로써 random walk가 node 1454까지 닿게 할 수도 있고, node 48의 주변 노드들만 돌아다닐 수도 있다!

# Node2Vec 구현

Biased Random Walk부터 구현해봅시다. 나머지는 어디서 많이 본 코드이다.

```python
# biased random walk 코드 

import networkx as nx
import numpy as np
import random

def biased_random_walk(G, start_node, walk_length, p=1, q=1):
    walk = [start_node]

    while len(walk) < walk_length:
        cur_node = walk[-1]
        cur_neighbors = list(G.neighbors(cur_node))

        if len(cur_neighbors) > 0:
            if len(walk) == 1:
                walk.append(random.choice(cur_neighbors))
            else:
                prev_node = walk[-2]

                # 노드 이동 확률 설정
                probability = []
                for neighbor in cur_neighbors:
                    if neighbor == prev_node:
                        # Return parameter
                        probability.append(1/p)
                    elif G.has_edge(neighbor, prev_node):
                        # Stay parameter
                        probability.append(1)
                    else:
                        # In-out parameter
                        probability.append(1/q)

                probability = np.array(probability)
                probability = probability / probability.sum()  # normalize

                next_node = np.random.choice(cur_neighbors, p=probability)
                walk.append(next_node)
        else:
            break

    return walk 

def generate_walks(G, num_walks, walk_length, p=1, q=1):
    walks = []
    nodes = list(G.nodes())
    for _ in range(num_walks):
        random.shuffle(nodes)  # to ensure randomness
        for node in nodes:
            walk_from_node = biased_random_walk(G, node, walk_length, p, q)
            walks.append(walk_from_node)
    return walks
```

![Untitled 2](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/633ab4fc-228b-45e3-b66e-8a8ab45b44af)

이제 위 함수와 Word2Vec 알고리즘을 결합해 봅시다. 이제 DeepWalk에서 한 층 더 발전된 Node2Vec 모델이 될 겁니다!

```python
# Random Walk 생성
walks = generate_walks(G, num_walks=10, walk_length=20, p=9, q=1)

# String 형태로 변환 (Word2Vec 입력을 위해)
walks = [[str(node) for node in walk] for walk in walks]

# Word2Vec 학습
model = Word2Vec(walks, vector_size=128, window=5, min_count=0,  hs=1, sg=1, workers=4, epochs=10)

# 노드 임베딩 추출
embeddings = np.array([model.wv.get_vector(str(i)) for i in range(data.num_nodes)])
# 이제 각 노드는 128차원의 vector 를 가지게 됩니다.
node_id = '2'  # 노드 한 개를 살펴볼까요?
vector = model.wv[node_id]

vector
```

![Untitled 3](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/c4ba65dd-86f8-4aa0-bf59-64ae631642bb)

위 벡터는 그래프의 복잡한 구조와 노드 간의 관계를 저차원의 연속적인 벡터 공간에 표현한 것

- 위 숫자들은 노드 ‘2’가 그래프 내에서 어떤 역할을 하는지, 어떤 다른 노드들과 Structural Equivalence(유사한 패턴)을 가지는지, 비슷한 특성(Homophily)을 가지고 있는지 등의 정보를 나타내고 있음
- 이러한 벡터는 노드 간의 similarity 계산, 그래프 기반의 다양한 prediction에 사용될 수 있다.
    - similarity를 잘 계산한다면 추천 시스템을 만들 수 있다.
- 임베딩은 ML 모델의 input으로 사용되기도 한다.
    - 노드 분류, 링크 예측, 클러스터링 등의 task를 풀기 위한 앙상블이 가능
    
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

![Untitled 4](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/bee73a5d-91cc-4125-95ef-92574dfcbfb2)

p와 q를 임의로 숫자를 조정해 가면서 베이스라인 모델의 성능을 테스트했는데, CORA graph에서는 파라미터 p가 높을 때 성능이 좋았다. 이는 본 적 없는 노드들은 더 많이, DFS처럼 탐색할 때 성능이 좋아진다는 뜻이다.

- CORA dataset은 homophilic한 컨셉을 가진 graph라는 것과 마찬가지
    - 유사한 특성을 가진 노드들이 서로 연결되는 경향이 있는 것이다. 이를 random walk의 bias로 반영했을 때 성능이 더 좋아진다는 것인데, 잘 생각해보면 CORA dataset이 논문 인용관계를 담고 있었기 때문에 어찌보면 당연한 결과이다.

DeepWalk 알고리즘의 성능은 이미 꽤나 좋은 것으로 검증되어 있다. 때문에 Baysian optimizing(베이지안 최적화)로 파라미터 튜닝을 해봅시다.

```python
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 10, 1000)
    max_depth = trial.suggest_int('max_depth', 1, 100)
    
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

rf_best = RandomForestClassifier(n_estimators=study.best_params['n_estimators'], 
                                  max_depth=study.best_params['max_depth'], 
                                  random_state=42)
rf_best.fit(X_train, y_train)

y_pred = rf_best.predict(X_test)
print(classification_report(y_test, y_pred))
```

![Untitled 5](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/30044947-4728-4a44-aefd-168e68677ef3)

accuracy가 0.68로 상승했다. 파라미터 p와 q를 더 적절히 튜닝하면 훨씬 나은 성능을 내겠지?

