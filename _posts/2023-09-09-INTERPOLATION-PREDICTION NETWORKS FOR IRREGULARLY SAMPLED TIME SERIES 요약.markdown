---
layout: post
title: "INTERPOLATION-PREDICTION NETWORKS FOR IRREGULARLY SAMPLED TIME SERIES 요약"
tags:
  - Time Series
  - paper review
use_math: true
---

<br>

논문에서 제공하는 코드를 돌리기 전, 논문에서 이야기하는 대략적인 메시지를 파악하고자 논문의 내용을 따로 정리해봤습니다. 정리한 후 코드 실행이 가능한 개발환경을 세팅하여 코드를 정상적으로 실행한 후 논문의 내용과 코드를 번갈아 따라가며 이해하고자 하였습니다. 하지만 설계된 모델 학습 시작 과정에서 outofmemory 현상이 발생하여 코드를 돌리는 데 실패한 뒤로 이 논문을 이해하려는 과제를 접었습니다. 실패한 과제이지만 많은 시간을 투자했던 과제이기 때문에 추가해봤습니다. 코드는 아래의 링크에서 가져왔습니다.

[https://github.com/mlds-lab/interp-net](https://github.com/mlds-lab/interp-net)

논문 링크: [https://arxiv.org/pdf/1909.07782v1.pdf](https://arxiv.org/pdf/1909.07782v1.pdf)

<br>

# 목표

sparse & irregularly sampled된 multivariate time series의 지도 학습 문제 해결을 위한 새로운 딥러닝 아키텍처 제시

- 예측 네트워크 적용에 이어 semi-parametric interpolation(보간) network 사용을 기반
- interpolation network

<br>

# 1. Introduction

**irregularly sampled time series**: a sequence of samples with irregular intervals between their observation times (관측 시간 사이의 간격이 불규칙한 일련의 샘플)

- particular interest in the supervised learning: 별도의 보간(separate interpolation) 또는 귀속(imputation, 결측값 대체) 단계 없이 다변량 희소 및 불규칙하게 샘플링된 시계열 입력으로 사용하여 end-to-end 학습을 직접 수행하는 방법
    - end-to-end learning: 입력에서 출력까지 파이프라인 네트워크 없이 신경망으로 한 번에 처리
        - 별다른 전처리과정 없이 데이터 그대로를 모델의 입력으로 사용하여 목적에 맞는 출력값을 뽑아낸다.
        - 파이프라인 네트워크: 전체 네트워크를 이루는 부분적인 네트워크
        - 관련 설명: [https://velog.io/@jeewoo1025/What-is-end-to-end-deep-learning](https://velog.io/@jeewoo1025/What-is-end-to-end-deep-learning)

<br>

보간-예측(Interpolation-Prediction) 네트워크 제시

- 보간-예측 네트워크: 다변량 희소(multivariate sparse, 다차원 데이터) 및 불규칙하게 샘플링된 데이터를 사용하여 지도 학습을 위한 새로운 모델 아키텍처
- **interpolation network**로 구성된 몇 가지 semi-parametric interpolation layer 사용 후 모든 표준 딥러닝 모델에 활용 가능한 예측 네트워크를 적용
    - semi-parametric interpolation: 반모수 보간, 통계 추론 기법 중 하나인 듯…?
        - 모수, 비모수, 반모수
            
            모수: 모든 확률분포는 한 개 이상의 모수를 가지고 있으며 이는 확률분포 모양을 결정한다.
            
            - 예) 정규분포: 2개의 모수(평균, 분산)
            
            모수적 모델: 알려진 확률분포를 기반으로 해당 모수를  추정하는 과정이 포함되어 있는 모델을 통칭
            
            - 모수적 모델의 예: 선형회귀모델
                - 선형회귀모델은 모델 구축 시 알려진 확률분포(정규분포)를 가정하기 때문에 모수적 방법론에 속한다.
            
            비모수적 모델의 예: K-최근접이웃
            
            - 가장 가까운 거리에 있는 K개의 관측치를 결정한 후 이들의 특성을 이용해 관심 관측치를 예측하는 과정을 거치는 알고리즘
            - 이 과정에서 확률분포의 개념은 전혀 사용되지 않기 때문에 비모수적 모델
            
            반모수적 모델의 예: 인공신경망
            
            - 연결선의 가중치(w)인 ‘모수’가 존재하나, 이 모수는 확률분포와는 무관하게 얻어지는 것이기 때문
            
            **⇒ 모수적 모델과 비모수적 모델을 구분하기에 적절한 기준은
            '데이터의 양이나 분포에 의존하지 않고 일정 개수의 모수로 모델이 표현되는가'에 있다.**
            
본 연구에서는 GRU 네트워크를 **예측 네트워크**로 사용

<br>

이 아키텍처는…

- 입력 시계열의 explicit multi-timescale representation 계산 가능, 과도(단시간 이벤트)에 대한 정보를 더 광범위한 추세에서 분리

각 입력 시계열에 대해 세 가지 출력 시계열 생성

- 입력의 광범위한 추세를 모델링하는 부드러운 보간
- 짧은 시간 척도 모델링 과도(transients, 일시적)
- 로컬 관찰 주파수를 모델링하는 강도 함수

⇒ **읽으면서 위 3개가 뭔지 알아야 함!**

<br>

분류 및 회귀 작업 모두에 대해 제안된 아키텍처 평가

또한 이 아키텍처가 생성할 수 있는 정보 채널의 완전한 절제 테스트를 수행하여 분류 및 회귀 성능에 미치는 영향을 평가한다.

<br>

# 3. Model Framework

제시 순서: 표기법 - 모델 아키텍처 - 학습 기준

<br>

## 3.1 Notation

$D$: Dataset ($s_n, y_n$)

$y_n$: 분류, 실제값(in the case of regression)

$s_n$: sparse & 불규칙하게 샘플링된 다변량 시계열

time series $d$: 서로 다른 시간에 관측치를 가질 수 있음 & 서로 다른 총 관측치 $L_{dn}$을 가질 수 있음

$s_{dn} = (\bf{t}_{dn}, \bf{x}_{dn})$: 데이터 사례 n에 대한  시계열 d

- $\bf{t}_{dn} = [t1_{dn}, ..., t_{L_{dn}dn}]$: 관측치가 정의된 시점 list
- $\bf{x}_{dn} = [x1_{dn}, ..., x_{L_{dn}dn}]$: 관측된 값의 해당 목록

<br>

## 3.2 Model Architecture

The overall model architecture: **interpolation network & prediction network**

- **interpolation network**: multivariate, sparse and irregularly sampled input time series를 기준 시점 집합(a set of reverence time points) $r = [r_1, ..., r_T]$에 대해 보간
    - 모든 시계열이 공통 시간 간격 내에 정의된다 가정 (???)
    ~~*기껏 서로 다른 관측치 가질 수 있다고 해놓고…?
    뭐 일단 Model Architecture를 설명하는 도입부니까 새로 개념을 정리한다고 생각해야할 듯?*~~
        - 예) MIMIC-III Dataset에 대한 승인 후 처음 24 또는 48시간 (???)
        - T 기준 시점 $r_t$는 그 간격 내에 균등하게 이격되도록 선택된다.
    - 본 연구에서는 각 계층이 다른 유형의 보간을 수행하는 2계층 보간 네트워크(two-layer interpolation network)를 제안
- **prediction network**: 보간 네트워크의 출력을 입력으로 사용하여 대상 변수에 대한 예측 $\hat{y_n}$ 생성
    - any standard supervised neural network architecture (fully-connected feedforward(?), convolutional, recurrent, etc)
        - feedforward: 입력 층(input layer)으로 데이터가 입력되고, 1개 이상으로 구성되는 은닉 층(hidden layer)을 거쳐서 마지막에 있는 출력 층(output layer)으로 출력 값을 내보내는 과정을 말함
    - 서로 다른 prediction networks의 사용과 관련하여 완전히 모듈화되어 있다. (???)
- interpolation network 훈련을 위해 prediction network로부터의 지도 학습 신호 외에 비지도 학습 신호를 제공하는 auto-encoding component를 포함한다.
- Figure 1: 모델의 architecture

![Untitled](https://github.com/SuhwanMylife/CJDaehan_competition/assets/70688382/381f5f9a-40c0-4a53-8ca8-bdc9a832d388)

- Low-pass filter: 특정한 차단 주파수 이상 주파수의 신호를 감쇠시켜 차단 주파수 이하의 주파수 신호만 통과시키는 필터
- High-pass filter: 특정 차단 주파수보다 높은 주파수의 신호를 통과시키고 차단 주파수보다 낮은 주파수의 신호를 감쇠시키는 전자 필터
- Cross-channel Interpolation: 교차 채널 보간???
- Transient Component: 일시적 구성???

<br>

### 3.2.1 Interpolation Network

**The goal of the interpolation newtork:** 
$T$ 기준 시점 $r = [r_1, ..., r_T]$에서 정의된 input multivariate time series의 **각 $D$차원의 보간물 모음을 제공**하는 것

- 각 $D$ input series에 대해 총 $C = 3$개의 출력을 사용
- 세 가지 출력(아래에서 자세히 설명): smooth trends(부드러운 추세), transients(과도현상), and observation intensity information(관찰 강도 정보) 캡처
- $f\theta(r, s_n)$: interpolation network의 output $\hat s_n$을 계산하는 함수
- output $\hat{s_n}$: 모든 입력 $s_n$에 대해 치수 $(DC) \times T$를 갖는 고정 크기 배열이다.

The **first layer** in the interpolation network: 각 $D$ 시계열에 대해 **세 개의 semi-parametric univarate transformations(반모수 일변량 변환)**을 별도로 수행

- 각 transformations: 연속 시간 관측을 수용하기 위해 **RBF(방사형 기저 함수) 네트워크**를 기반으로 한다.
    - RBF(Radial Basis Function): [참고자료](https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=9409290274&logNo=221553102800)
- transformations: 저역 통과(low-pass)(or smooth) interpolation $\sigma_d$, 고역 통과(high-pass)(or non-smooth) $\gamma_d$ 및 강도 함수(intensity function) $\lambda_d$
    - 이러한 변환은 식 1, 2, 3, 4와 같이 각 data case와 각 input series $d$에 대한 기준 시점 $r_k$에서 계산된다.
    - smooth interpolation $\sigma_d$는 매개변수 $\alpha_d$가 있는 제곱 지수 커널(exponential kernel)를 사용하는 반면, non-smooth interpolation $\gamma_d$는 매개변수 καd가 > 1인 제곱 지수 커널을 사용한다.

The second interpolation layer: 모든 시계열에 걸쳐 학습 가능한 상관관계 $\rho_{dd'}$를 고려하여 각 기준 시점의 모든 $D$ 시계열에 걸쳐 정보를 merge

- 이로 인해, each input dimension $d$에 대해 cross-dimension interpolation $\chi_d$ 발생
- Equation 5) 첫 번째 층의 high-pass(or non-smooth) interpolation $\gamma_d$와 smooth cross-dimension interpolation $\chi_d$ 사이의 difference로 각 input dimension $d$에 대한 transient component(과도 성분) $\tau_d$를 추가로 정의

In the experiments presented in the next section,
$d$차원당 총 3개의 interpolation network outputs를 prediction network에 대한 input으로 사용

- smooth, cross-channel interpolants $\chi_d$를 사용하여 smooth trends를 포착
- transient components $\tau_d$를 사용하여 transients(과도 성분)를 포착
- intensity functions $\lambda_d$를 사용하여 관측치가 제 때에 발생하는 위치에 대한 정보를 포착

<br>

### 3.2.2 Prediction Network

Following the application of the interpolation network, 모든 $D$ dimensions가 규칙적인 간격의 기준 시점 $r_1, ..., r_T$에 정의된 $C$ 출력으로 다시 표현되었다. (위 과정에 의해 생긴 출력 $C$)

- 우리의 실험에서, 위에서 설명한 대로 $C = 3$을 사용함
- Again, the complete set of interpolation network outputs(전체 보간 네트워크 출력 집합)를 size $(DC) \times T$의 행렬로 나타낼 수 있는 $\hat{\bf{s}_n} = f\theta(\bf{r}, s_n)$라고 한다. ~~(전체 아키텍처의 출력을 말하는 듯.)~~

<br>

prediction network의 input: $\hat{\bf{s}_n}$
data case $n$에 대한 target value $y_n$의 prediction $\hat{y_n} = g_\phi(\hat{\bf{s}_n}) = g_\phi(f_\theta(\bf{r, s_n}))$을 출력해야 한다.

- 위 모델의 components(구성요소)에 대한 여러 가지 선택 사항 존재
    - 예) matrix $\hat{\bf{s_n}}$은 single long vector로 변환되어 **standard multi-layer feedforward network**(?)에 입력으로 제공될 수 있다.
        - temporal convolutional model or recurrent model like a GRU or LSTM과 같은 반복 모델을 행렬 $\hat{\bf{s_n}}$의 time slices에 적용할 수 있다.
        - 본 연구에서는 GRU 네트워크를 prediction network로 활용하는 실험 수행할 거임

<br>

### 3.2.3 Learning

model parameter 학습을 위해
supervised & unsupervised component로 구성된 composite objective function(복합 목적함수) 사용

- 훈련 데이터의 양을 고려했을 때, supervised components만으로는 impolation network parameters를 학습하기에 충분하지 않음
- unsupervised components: autoencoder와 같은 loss function에 해당
- However, semi-parametric RBF interpolation layers(반모수 RBF 보간 레이어)는 RBF kernel parameters를 매우 큰 값으로 설정하여 input points를 정확하게 맞추는 ability를 가지고 있다. (??? ~~이게 무슨 문제가 있다는건가?~~)

<br>

위 solution을 피하고 interpolation layers가 input data를 적절히 interpolate하는 방법을 learn하도록 하려면, 학습 중에 일부 관찰된 data point $x_{jdn}$에 대해서만 reconstruction loss(재구성 손실)를 계산해야 한다.

- 이는 high-capacity autoencoders에서 잘 알려진 문제이며, 과거 연구에서는 useful structure를 학습하지 않고 input data를 trivially memorizing을 피하기 위해 유사한 전략 사용했음

<br>

loss의 autoencoder component를 구현하기 위해 각 data point $(t_{jdn}, x_{jdn})$에 대한 masking 변수 $m_{jdn}$ 세트를 도입한다.

- If $m_{jdn}$, interpolation network에 대한 input으로 data point $(t_{jdn}, x_{jdn})$을 제거하고, autoencoder loss를 assessing(평가)할 때 이 time point의 predicted value(예측값)를 포함한다.
- masking된 $\bf{s}_n$값의 subset of values(하위 집합)을 나타내기 위해 short notation $\bf{m}_n \odot \bf{s}_n$  사용
- masking되지 않은 $\bf{s}_n$값의 하위 집합을 나타내기 위해 short notation  $(1-\bf{m}_n) \odot \bf{s}_n$  사용
- 시점 $t_{jdn}$에서 masked된 입력에 대해 예측하는 $\hat{x}_{jdn}$값은 해당 시점에서의 smooth cross-channel interpolant의 값으로, $\hat{x}_{jdn} = h^\chi_\theta(t_{jdn}, (1-\bf{m}_n)\odot \bf{s}_n)$를 기준으로 계산된다.

<br>

이제는 proposed framework에 대한 learning objective(학습 목표)를 정의할 수 있다.

- $l_P$를 prediction network의 loss로 한다.
    - classification - cross-entropy loss / regression - squared error
- $l_I$: interpolation network autoencoder loss (standard squared error 사용)
- $l_2$: interpolation & prediction networks parameters 모두에 대한 regularizers(정규화기)
- $\delta_I, \delta_P, \delta_R$: objective function(목적함수)의 성분들 간의 균형을 제어하는 hyper-parameters

![Untitled 1](https://github.com/SuhwanMylife/CJDaehan_competition/assets/70688382/a8e875c0-c66d-47cb-af5e-a8f4e9084e8c)

