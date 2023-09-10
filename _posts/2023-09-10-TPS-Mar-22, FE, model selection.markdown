---
layout: post
title: "TPS-Mar-22, FE, model selection"
---

<br>

이전에 같은 competition을 진행한 다른 notebook을 필사한 후 정리한 게 있으니, 여기서는 포인트만 집어서 정리해보자.

<br>

원본 데이터

![Untitled](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/b96f4cca-db89-4431-92ea-045b5217efcd)

<br>

# Outliers

## Labor day

Labor day에 해당하는 데이터를 outliers 취급하여 df_train에서 제외시킨다.

```python
labor_day = pd.to_datetime('1991-09-02').dayofyear
df_train = df_train[df_train['time'].dt.dayofyear != labor_day]
```

![Untitled_1](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/c09f99bd-78bc-4530-b1e0-92061fab6079)

- Labor day: first Monday in September

<br>

## Mondays only

```python
df_train = df_train[df_train.time.dt.weekday == 0]
```

![Untitled_2](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/6e00159b-3c7a-4294-a27f-e86ea63bfa5e)

미래 교통량을 예측하는 시점이 월요일이기 때문에 월요일 데이터만 남긴 모습

<br>

# Feature Engineering

## **Date and time features**

Date & time features 전처리 과정이다. 이전에 필사했던 Notebook과는 약간 차이가 있어 보인다.

```python
def add_datetime_features(df):
    df['month']   = df['time'].dt.month
    df['day']     = df['time'].dt.day
    df['weekday'] = df['time'].dt.weekday
    df['weekend'] = (df['time'].dt.weekday >= 5)
    df['hour']    = df['time'].dt.hour
    df['minute']  = df['time'].dt.minute
    df['afternoon'] = df['hour'] >= 12
    
    # number of 20' period in a day
    df['moment']  = df['time'].dt.hour * 3 + df['time'].dt.minute // 20

add_datetime_features(df_train)
add_datetime_features(df_test)

df_train
```

![Untitled_3](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/e8a22a8e-5480-4393-94b5-42867342452b)

<br>

## **Morning congestion averages**

일별 도로별 평균 congestion을 산출하여 df_train에 merge한다.

```python
df_mornings = df_train[(df_train.hour >= 6) & (df_train.hour < 12)]
morning_avgs = pd.DataFrame(df_mornings.groupby(['month', 'day', 'road']).congestion.median().astype(int)).reset_index()
morning_avgs = morning_avgs.rename(columns={'congestion':'morning_avg'})
morning_avgs
```

![Untitled_4](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/6f7a6953-dcad-41e4-a968-b4830ddf8abe)

```python
df_train = df_train.merge(morning_avgs, on=['month', 'day', 'road'], how='left')
df_test = df_test.merge(morning_avgs, on=['month', 'day', 'road'], how='left')
df_train
```

![Untitled_5](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/818bfe39-ebee-46ce-81f9-486a1a626f5a)

<br>

## **Congestion Min, Max, Median**

요일별로 분마다 congestion의 Min, Max, Median 산출하여 merge

```python
mins = pd.DataFrame(df_train.groupby(['road', 'weekday', 'hour', 'minute']).congestion.min().astype(int)).reset_index()
mins = mins.rename(columns={'congestion':'min'})
df_train = df_train.merge(mins, on=['road', 'weekday', 'hour', 'minute'], how='left')
df_test = df_test.merge(mins, on=['road', 'weekday', 'hour', 'minute'], how='left')

maxs = pd.DataFrame(df_train.groupby(['road', 'weekday', 'hour', 'minute']).congestion.max().astype(int)).reset_index()
maxs = maxs.rename(columns={'congestion':'max'})
df_train = df_train.merge(maxs, on=['road', 'weekday', 'hour', 'minute'], how='left')
df_test = df_test.merge(maxs, on=['road', 'weekday', 'hour', 'minute'], how='left')

medians = pd.DataFrame(df_train.groupby(['road', 'weekday', 'hour', 'minute']).congestion.median().astype(int)).reset_index()
medians = medians.rename(columns={'congestion':'median'})
df_train = df_train.merge(medians, on=['road', 'weekday', 'hour', 'minute'], how='left')
df_test = df_test.merge(medians, on=['road', 'weekday', 'hour', 'minute'], how='left')
df_train
```

![Untitled_6](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/1476db66-d719-4bcf-a920-14b9cf9c1742)

