---
layout: post
title: "Competition with spatiotemporal data (Tabular Playground Series - Mar 2022)"
---

<br>

# 필사한 노트북

**TPS Mar22 - Top 6% Solution - EDA / FE / Blending 블로그 링크**
**TPS-Mar-22, FE, model selection 블로그 링크**

<br>

# 대회 개요

Kaggle에서 2021년부터 매달 여는 Tabular Playground Series 중 22.03에 열린 대회입니다.

<br>

### **Goal**

목적: 미국 대도시의 12시간 교통 흐름 예측하기

<br>

### Metric

test set의 각 기간에 대한 predicted congestion & actual congestion values 사이의 mean absolute error(평균 절대 오차)로 평가

<br>

### Data

**train.csv** April ~ September of 1991 사이의 65개 도로의 measurements of traffic congestion으로 구성

**test.csv** 1991-06-30에 location & direction of travel로 식별된 도로에 대한 시간별 예측 수행

**x** 도로의 동서 좌표

**y** 도로의 남북 좌표

**direction** 도로의 진행 방향

**congestion** 도로의 정체 수준 (0~100)

아래 figure: 도로별 location & direction 시각화

![Untitled](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/3f23d263-8efe-4e99-9728-e12ca6374fbf)

