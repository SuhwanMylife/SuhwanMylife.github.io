---
layout: post
title: "Grubb’s test란?"
use_math: true
---

<br>

참고자료

[Grubb's test 이상치 탐지](https://exmemory.tistory.com/19)

[Grubbs's test](https://en.wikipedia.org/wiki/Grubbs's_test)

<br>

가정: 모집단이 정규분포임을 가정

⇒ 정규성 가정에 기반을 두고 있으며, 테스트 이전에 정규분포를 따르는지 반드시 확인해야 한다.

<br>

Grubb’s test는 한 번에 하나의 이상치만 탐지

⇒ 특이치가 감지되지 않을 때까지 계속해서 반복해야 한다.

<br>

반복하는 과정에서
**1. 샘플의 갯수가 줄어들고**, 그에 따라 
**2. 탐지 확률이 바뀜에 따라 이상치가 아닌 경우에도 이상치**라고 하기 때문에 
**표본의 크기가 6 이하일 경우에 종료**하는 것이 좋다.

<br>

Grubb’s test의 귀무가설: dataset에 이상치가 없다.

Grubb’s test의 대립가설: dataset에 이상치가 하나는 존재한다.

- 귀무가설: 모집단의 특성에 대해 옳다고 제안하는 잠정적인 주장
- 대립가설: 귀무가설이 거짓이라면 대안적으로 참이 되는 가설

<br>

Grubb’s test의 **Test Statistic(검정 통계량)**

$
G = {\max\limits_{i=1,...,N} |Y_i-\bar{Y}|\over s}
$

- $\bar{Y}$: 표본평균, $s$: 표준편차
- 표본 평균에서 샘플 중 가장 큰 절대 편차를 가져오는 것

<br>

Grubb’s test의 **Critical Value(임계치)**

- 임계값(Critical Value): 귀무가설 하 검정 통계량의 분포에서 귀무 가설을 기각해야 하는 값의 집합을 정의하는 점

![캡처](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/b4fe5c43-47cc-4640-83a7-437dc11071ed)

- 양측 검정
- 자유도: N-2, 유의 수준: α/(2*N*)
    - 자유도가 뭐임?
    - 유의 수준: 통계적인 가설검정에서 사용되는 기준값. 일반적으로 유의 수준을 α로 표시하고, 95%의 신뢰도를 기준으로 한다면 0.05값이 유의수준 값이 된다.

