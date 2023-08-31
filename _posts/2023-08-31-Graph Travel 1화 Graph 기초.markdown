---
layout: post
title: "Graph Travel 1화 Graph 기초"
tags:
  - Graph
  - Tutorial
---

<br>

CORA 데이터셋 활용해 튜토리얼 진행

- Node: 논문, Edge: 논문 간의 인용 관계
- 노드를 Vertex라 부르기도 함

```python
# 필요한 모든 라이브러리 임포트
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

# CORA 데이터셋 로드
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

data
```

![Untitled](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/2a902101-1845-4351-84b2-b7e74c25365f)

# Graph & Degree

데이터를 그래프로 시각화

PyTorch Geometric의 Data 객체를 networkx 그래프로 변환하여 그래프 시각화

- networkx의 from_edgelist 메서드를 사용하여 edge list를 받아 networkx 그래프 객체 생성

```python
# torch_geometric의 데이터를 NetworkX 그래프로 변환
edge_index = data.edge_index
edges = edge_index.t().numpy()
graph_nx = nx.from_edgelist(edges, create_using=nx.Graph())

# 그래프 그리기
plt.figure(figsize=(10, 10))
pos = nx.spring_layout(graph_nx, seed=42)  # Spring layout 사용
nx.draw_networkx(graph_nx, pos, with_labels=False, node_size=50, node_color="skyblue")
plt.title("Cora Graph Visualization")
plt.show()
```

![Untitled 1](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/e87a6b93-9e19-4dc3-94b0-a3f1cbdf7247)

이 때 궁금했던 게 ‘edge에 대한 정보만 있지, node에 대한 정보가 없는데 왜 이런 식으로 노드들의 위치가 정해져서 그려지는 거지?’였는데, 공식 문서를 보고 답을 알 수 있었다.

> Position nodes using Fruchterman-Reingold force-directed algorithm.
> 
> 
> The algorithm simulates a force-directed representation of the network treating edges as springs holding nodes close, while treating nodes as repelling objects, sometimes called an anti-gravity force. Simulation continues until the positions are close to an equilibrium.
> 

프로흐터만-라인골드의 force-directed 알고리즘을 사용하여 노드를 배치

- 에지로 연결된 node는 서로 exert attractive forces(인력력, 끌어당기는 힘)를 발휘하는 반면, 모든 노드는 서로 반발력을 가진다는 아이디어 기반의 알고리즘

그래서 대부분의 edge가 달려 있는 노드들은 뭉쳐 있고, 그렇지 않은 소수의 노드들은 밖으로 멀리 떨어져 있음을 알 수 있다.

그래프 중심에 많은 노드들이 존재한다는 것은 이 노드들이 서로 많은 연결(edge link)을 가질 가능성이 높다는 의미 **(Hub Node)**
반면에, 그래프의 바깥 원에서 발견되는 노드들은 일반적으로 적은 수의 연결 **(Peripheral Node)**

# Node Degree

그래프의 Degree 분포 확인

Degree: 노드의 차수

- 해당 노드에 연결되어 있는 edge의 개수

```python
# node degree 계산
degrees = [degree for node, degree in nx.degree(graph_nx)]
degree_count = Counter(degrees)

# Degree 분포 출력
plt.figure(figsize=(10, 6))
plt.bar(degree_count.keys(), degree_count.values())
plt.title("Degree Distribution of Cora Graph")
plt.xlabel("Degree")
plt.ylabel("Count")
plt.show()
```

![Untitled 2](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/d3044544-9707-43ad-b1ed-a824f71818a7)

# Subgraph

그래프 분석 과정에서 전체 그래프를 여러 subgraph로 나누어 분석하는 경우가 많다. 또는 특정 특성을 가진 edge만을 분석하고자 할 때도 subgraph를 사용한다.

```python
# 특정 노드를 선택
selected_node = 0

# 선택한 노드와 직접적으로 연결된 노드들만으로 구성된 subgraph를 추출
neighbors = list(graph_nx.neighbors(selected_node))
neighbors.append(selected_node)
subgraph_nx = graph_nx.subgraph(neighbors)

# 그래프 그리기
plt.figure(figsize=(6, 6))
pos = nx.spring_layout(subgraph_nx, seed=42)  # Spring layout 사용
nx.draw_networkx(subgraph_nx, pos, with_labels=True, node_color="skyblue")
plt.title("Subgraph Centered at Node {}".format(selected_node))
plt.show()

# node degree 출력
print(f"node degree({selected_node}) = {subgraph_nx.degree[selected_node]}")
```

![Untitled 3](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/5021612f-30bc-4a9a-b44c-5cbfea506af1)

# Directrional Graph

방향성을 가진 edge

```python
# 특정 노드를 선택
selected_node = 0

# 선택한 노드와 직접적으로 연결된 노드들만으로 구성된 서브그래프를 추출
neighbors = list(graph_nx.neighbors(selected_node))
neighbors.append(selected_node)
subgraph_nx = graph_nx.subgraph(neighbors)

# 서브그래프를 방향성 그래프로 변환
subgraph_nx_directed = nx.DiGraph()

# 선택된 노드에서 그 이웃 노드들로 향하는 방향성 추가
for neighbor in neighbors:
    if neighbor != selected_node:
        subgraph_nx_directed.add_edge(selected_node, neighbor)

# 그래프 그리기
plt.figure(figsize=(6, 6))
pos = nx.spring_layout(subgraph_nx_directed, seed=42)  # Spring layout 사용
nx.draw_networkx(subgraph_nx_directed, pos, with_labels=True, node_color="skyblue", arrows=True)
plt.title("Directed Subgraph Centered at Node {}".format(selected_node))
plt.show()

# node degree 출력
print(f"node out-degree({selected_node}) = {subgraph_nx_directed.out_degree[selected_node]}")
print(f"node in-degree({selected_node}) = {subgraph_nx_directed.in_degree[selected_node]}")
```

![Untitled 4](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/494263c7-ad97-4584-9341-c45ecb66b0d3)

subgraph에 방향성 부여 시, 원본 그래프에 있는 방향성 정보를 활용해 방향성 그래프(DiGraph)로 생성하거나 변환하는 과정이 필요하다.

# Undirected Graph

edge에 방향성이 없는 그래프

# Weight

edge에 direction뿐만 아니라 weight도 추가될 수 있다.

- 예) 별점으로 상품의 선호도 반영, 채팅 횟수로 사람 간의 친밀도 측정 등

unweighted graph: 가중치가 없는 그래프

weighted graph: 가중치가 있는 그래프

이 튜토리얼에서 다루는 dataset은 unweighted graph. 하지만, 튜토리얼이니 가중치를 무작위로 생성하여 edge에 부여한 후 시각화해봅시다.

```python
# Cora 데이터셋 로드
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# torch_geometric의 데이터를 NetworkX 그래프로 변환
edge_index = data.edge_index
edges = edge_index.t().numpy()
graph_nx = nx.from_edgelist(edges, create_using=nx.Graph())

# 특정 노드를 선택
selected_node = 0

# 선택한 노드와 직접적으로 연결된 노드들만으로 구성된 subgraph를 추출
neighbors = list(graph_nx.neighbors(selected_node))
neighbors.append(selected_node)
subgraph_nx = graph_nx.subgraph(neighbors)

# edge의 수만큼 무작위 가중치를 생성하고, 이를 각 edge의 가중치로 설정
weights = np.random.rand(subgraph_nx.number_of_edges())
for i, edge in enumerate(subgraph_nx.edges):
    subgraph_nx.edges[edge]['weight'] = weights[i]

# 그래프 그리기
pos = nx.spring_layout(subgraph_nx, seed=42)  # Spring layout 사용
nx.draw(subgraph_nx, pos, with_labels=True, node_color="skyblue")

# edge의 가중치를 그래프에 표시
edge_labels = nx.get_edge_attributes(subgraph_nx, 'weight')
nx.draw_networkx_edge_labels(subgraph_nx, pos, edge_labels=edge_labels)

plt.title("Subgraph Centered at Node {} with Random Edge Weights".format(selected_node))
plt.show()

# node degree 출력
print(f"node degree({selected_node}) = {subgraph_nx.degree[selected_node]}")
```

![Untitled 5](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/1cb6332a-a808-4e41-8613-0f9d070f9291)

# Connected Graphs

Connected Graph: 그래프의 모든 노드가 edge를 통해 서로 연결되어 있는 그래프

이번엔 노드 3개를 중심으로 하는 subgraph를 생성하고 시각화해봅시다.

```python
# 중심 노드를 선택 (여기서는 0, 1, 2를 선택)
center_nodes = [0, 1, 2]

for i, selected_node in enumerate(center_nodes):
    # 선택한 노드와 직접적으로 연결된 노드들만으로 구성된 subgraph를 추출
    neighbors = list(graph_nx.neighbors(selected_node))
    neighbors.append(selected_node)
    subgraph_nx = graph_nx.subgraph(neighbors)

    # edge의 수만큼 무작위 가중치를 생성하고, 이를 각 edge의 가중치로 설정
    weights = np.random.rand(subgraph_nx.number_of_edges())
    for i, edge in enumerate(subgraph_nx.edges):
        subgraph_nx.edges[edge]['weight'] = weights[i]

    # 그래프 그리기
    plt.figure(i)
    pos = nx.spring_layout(subgraph_nx, seed=42)  # Spring layout 사용
    nx.draw(subgraph_nx, pos, with_labels=True, node_color="skyblue")

    # edge의 가중치를 그래프에 표시
    edge_labels = nx.get_edge_attributes(subgraph_nx, 'weight')
    nx.draw_networkx_edge_labels(subgraph_nx, pos, edge_labels=edge_labels)

    plt.title("Subgraph Centered at Node {} with Random Edge Weights".format(selected_node))
    plt.show()

    # node degree 출력
    print(f"node degree({selected_node}) = {subgraph_nx.degree[selected_node]}")

    # 서브그래프의 연결성 확인
    print(f"Is subgraph centered at node {selected_node} connected? {nx.is_connected(subgraph_nx)}")
```

![Untitled 6](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/59ef5fc8-79f5-4cbd-8061-281c24d3c63e)

![Untitled 7](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/03967083-9552-4b5b-b8ce-2d70f19a8ba0)

![Untitled 8](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/0ea5e14e-b1f0-4b07-b6c3-5b71fc1cb04c)

여기서 주의깊게 볼 부분이 ‘Is subgraph centered at node connected?’인데, 지금은 3개의 subgraph에 대해 모두 True가 나왔지만, 3개를 모아놓으면 어떻게 될까요?

```python
# 중심 노드를 선택 (여기서는 1, 2를 선택)
center_nodes = [0, 1, 2]

# 모든 이웃 노드를 저장할 리스트 초기화
all_neighbors = []

# 중심 노드들의 모든 이웃 노드를 all_neighbors에 추가
for selected_node in center_nodes:
    neighbors = list(graph_nx.neighbors(selected_node))
    all_neighbors += neighbors

# 중심 노드들도 all_neighbors에 추가
all_neighbors += center_nodes

# all_neighbors에 포함된 모든 노드들만으로 구성된 subgraph를 추출
subgraph_nx = graph_nx.subgraph(all_neighbors)

# 그래프 그리기
pos = nx.spring_layout(subgraph_nx, seed=42)  # Spring layout 사용
nx.draw(subgraph_nx, pos, with_labels=True, node_color="skyblue")

# node degree 출력
for selected_node in center_nodes:
    print(f"node degree({selected_node}) = {subgraph_nx.degree[selected_node]}")

# 그래프의 연결성 확인
print(f"Is subgraph connected? {nx.is_connected(subgraph_nx)}")

plt.show()
```

![Untitled 9](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/e1fdb4dd-b926-4c0a-bfa3-66a2f06e3273)

이제는 False가 나옵니다. Node 0가 동떨어져 있는 모습을 볼 수 있습니다.

# Centrality

앞서 노드의 중심성에 대해 이야기했는데, 그럼 ‘어떤 노드가 가장 중심에 있을까요? 어떤 노드가 가장 중요할까요?’를 답하기 위해 centrality를 측정해야 한다.

centrality에는 여러 종류가 있고, 각각 다른 측면의 node importance를 나타낸다.

1. Degree Centrality
    
    노드의 중심성을 결정하는 가장 간단한 방법. 노드가 가지고 있는 직접적인 이웃 노드의 수에 비례.
    
2. Closeness Centrality
    
    모든 다른 노드들과의 ‘가까움’을 측정. 특정 노드로부터 그래프 내의 모든 다른 노드까지의 최단 경로의 평균 길이를 계산.
    
3. Betweeness Centrality
    
    노드가 다른 노드들 사이에서 얼마나 중요한 ‘다리’ 역할을 하는지 측정. 그래프 내의 최단 경로들에서 얼마나 자주 등장하는지에 따라 결정.
    
4. Eigenvector Centrality
    
    노드의 중요성이 그 이웃의 중요성에 의해 결정되는 경우를 측정. 노드의 중심성을 그 노드와 직접적으로 연결된 이웃 노드들의 중요성에 따라 결정.
    
    - 중요한 이웃을 많이 가진 노드가 중요한 노드
    - Google의 PageRank 알고리즘에서 사용된 원리와 유사

```python
# Betweenness Centrality
betweenness_centrality = nx.betweenness_centrality(subgraph_nx)
print(f"Betweenness Centrality: {betweenness_centrality[0]}") # 노드 0에 대해서만 계산

# Eigenvector Centrality
# eigenvector_centrality = nx.eigenvector_centrality(subgraph_nx)
# print(f"Eigenvector Centrality: {eigenvector_centrality[0]}") # 노드 0에 대해서만 계산

# Closeness Centrality
# 우리의 subgraph에서는 에러를 일으킵니다. 작동하지 않아요!
closeness_centrality = nx.closeness_centrality(subgraph_nx)
print(f"Closeness Centrality: {closeness_centrality[0]}") # 노드 0에 대해서만 계산
```

![Untitled 10](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/9fc83470-35ca-412c-b442-0b85c4cde534)

여기서 주의할 점

- 튜토리얼에서는 단 하나의 노드에 대해 centrality를 계산했으나, 실제로 centrality 계산 시 그래프의 전체 연결 구조를 고려해야 한다.
- 특히, subgraph의 centrality를 전체 그래프의 다른 부분과 비교하려는 경우, centrality는 반드시 전체 그래프에 대해 계산되어야 한다.
- 방향성을 고려한 centrality 계산 시에는 각 함수의 인자로 subgraph_nx.to_directed() 등을 넘겨줘야 한다.
- Eigenvector centrality는 그래프 연결 구조에 따라 계산이 불가능.
    - 그래프가 두 개 이상의 연결 요소로 나뉘어져 있거나, 그래프가 bipartite(이분 그래프) 구조인 경우에 그렇다. 이 튜토리얼에서 사용하는 subgraph에도 에러가 발생한다.

# Density & Adjacency Matrix

데이터의 **‘복잡성’**을 수치화해보자.

Density(그래프의 밀도)를 통해서 수치화가 가능하다.

- 그래프의 복잡성 판단, 데이터의 특성 이해, 성능 향상에 중요한 역할을 한다.

Density: [연결되어 있는 엣지/그래프에 존재하는 모든 엣지]의 비율

- 0~1 사이
- 1: 모든 노드가 서로 연결되어 있는 완전 그래프
- 0: edge가 하나도 없는 그래프

Adjacency Matrix: 그래프의 노드들이 서로 어떻게 연결되어 있는지를 나타내는 행렬

- 행렬의 크기: (노드 수) x (노드 수)
- 행과 열: 그래프의 노드
- 행렬 원소 값: 두 노드 간의 edge 여부 or 가중치

노드 0, 1, 2 subgraph의 Density를 계산하고 Adjacency Matrix를 만들어봅시다.

```python
# 그래프의 Density 계산
density = nx.density(subgraph_nx)
print(f"Density: {density}")

# Adjacency Matrix 만들기
adj_matrix = nx.adjacency_matrix(subgraph_nx)

# Adjacency Matrix 출력하기
print("Adjacency Matrix:")
print(adj_matrix.todense())  # .todense()를 사용하여 sparse matrix를 dense matrix로 변환
```

![Untitled 11](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/4d582196-94e0-48f0-9104-9b14030cdeb1)

**코드 결과 해석**

Density: 0.167 (가능한 모든 edge 중 약 16.7%만이 실제로 연결되어 있음을 의미)

