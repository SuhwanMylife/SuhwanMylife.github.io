---
layout: post
title: "Kaggle 노트북 필사 (TPS Mar22 - Top 6% Solution - EDA / FE / Blending)"
tags:
  - Kaggle
  - Time Series
  - Tablar Playground Series
---

<br>

[TPS Mar22 - Top 6% Solution - EDA / FE / Blending](https://www.kaggle.com/code/javigallego/tps-mar22-top-6-solution-eda-fe-blending)

이 competition에 업로드된 code 중 3번째로 필사한 notebook인데, EDA 하는 방식도 색다르고 modeling하는 방식도 새롭게 알게 돼서 가장 먼저 정리해보려 한다. 대략적으로 이 notebook의 특징을 언급하고 내용 정리해보도록 하겠다.

# This Notebook has

- 메모리 절약 **(Reducing Memory Usage)**
- time column을 여러 columns로 나누는 feature engineering **(Datetime Features)**
- pandas_profiling 사용한 데이터에 대한 전체적인 EDA
- 수학적인 요소를 첨가한 feature engineering **(Cyclical Features)**
- 상호의존정보 **(Mutual Information, MI)**
- Hyperparameter Tuning - **Optuna**
- CatBoost, XGBoost
- 앙상블의 일종인 **Blending**

<br>

# 목차

1. **pre-work**
    1. **Import Modules**
    2. **Reducing Memory Usage**
    3. **Datetime Features**
        - 시간을 여러 columns로 나누는 작업
2. **EDA (Exploratory Data Analysis)**
3. **Feature Engineering**
    1. **Dataset Score**
        - baseline model 모델링
    2. **Cyclical Features**
        - direction을 sin & cos 구성 요소로 분해
    3. **Geography and Direction**
        - 각 도로 간의 상관관계 측정
    4. **Mutual Information (상호의존정보)**
        - mutual information(MI): 한 knowledge of one quantity가 다른 knowledge에 대한 불확실성을 줄이는 정도 측정
4. **Modeling**
    1. **Catboost**
    2. **xgboost**
    3. **Blending and Submitting**

<br>

# 1. pre-work

## 1.1 Import Modules

```python
# %%capture: 실행되는 명령에 대한 정보의 결과를 저장
# %%capture
# !pip install -U lightautoml

import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_profiling as pp # ProfileReport(): EDA tool
import seaborn as sns
from IPython.display import display # 이미지 커널에 출력할 때
from pandas.api.types import CategoricalDtype # 데이터에 순위 지정(categorical)

from category_encoders import MEstimateEncoder # ??
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression # 두 random variable들이 얼마나 mutual dependence한지 measure
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# Algorithms
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

## Optuna: 하이퍼파라미터 튜닝 framework
# Optuna - Bayesian Optimization
import optuna
from optuna.samplers import TPESampler

# Plotly
import plotly.express as px # 데이터 값을 선으로 표현하는 그래프?
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.offline as offline
import plotly.graph_objs as go

warnings.filterwarnings('ignore')

# feature engineering 후 종종 사용할 method
def plot_feature_importance(importance, names, model_type): # feature별 중요하게 작용하는 정도 측정 
    
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)
    
    # Create a DataFrame using a Dictionary
    data = {'feature_names':feature_names, 'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)
    
    # Sort the DataFrame in order decreasing feature impotance
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)
    
    # Define size of bar plot
    plt.figure(figsize=(20, 10))
    # Plot Seaborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    # Add chart labels
    plt.title(model_type + ' Feature Importance')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Names')
```

<br>

## 1.2 Reducing Memory Usage

각 컬럼별로 최대값과 최소값을 확인하여 이에 맞는 최소한의 datatype으로 축소시키거나 유지하는 코드이다. 이 데이터 기준으로는 23%의 메모리 절약 효과가 있다고 한다.

```python
# 각 컬럼별 최대값, 최소값을 토대로 데이터의 타입 축소 혹은 유지
def reduce_mem_usage(df, verbose=True):
    numerics = ['int8','int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2 # df.memory_usage(): 특정 DataFrame의 메모리 사용량 check
    print(f"previous datatype: \n{df.dtypes}\n")

    for col in df.columns:
        col_type = df[col].dtypes

        if col_type in numerics: # 숫자로 이루어진 경우만 다룸
            c_min = df[col].min()
            c_max = df[col].max()
            print(f"c_min: {c_min}")
            print(f"c_max: {c_max}")

            if str(col_type)[:3] == 'int':
                # np.iinfo(): int의 표현가능한 수의 한계 반환
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                # np.finfo(): float의 표현가능한 수의 한계 반환
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    print()
    end_mem = df.memory_usage().sum() / 1024**2
    print(f"modified datatype: \n{df.dtypes}\n")

    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
 
    return df

df_data = reduce_mem_usage(df_data)
```

```markdown
previous datatype: 
row_id          int64
time           object
x               int64
y               int64
direction      object
congestion    float64
dtype: object

c_min: 0
c_max: 851174
c_min: 0
c_max: 2
c_min: 0
c_max: 3
c_min: 0.0
c_max: 100.0

modified datatype: 
row_id          int32
time           object
x                int8
y                int8
direction      object
congestion    float32
dtype: object

Mem. usage decreased to 59.85 Mb **(23.0% reduction)**
```

<br>

## 1.3 Datetime Features

날짜를 여러 열로 나누어보자

```python
df_data.time = pd.to_datetime(df_data.time)
df_data['year'] = df_data.time.dt.year
df_data['month'] = df_data.time.dt.month
df_data['week'] = df_data.time.dt.isocalendar().week # isocalendar(): 해당 날짜가 1년의 몇 번째 주차인지 반환
df_data['hour'] = df_data.time.dt.hour
df_data['minute'] = df_data.time.dt.minute
df_data['day_of_week'] = df_data.time.dt.day_name() # 요일명
df_data['day_of_year'] = df_data.time.dt.dayofyear
df_data['is_weekend'] = (df_data.time.dt.dayofweek >= 5).astype("int")
df_data = df_data.set_index('time')
df_data
```

![Untitled](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/3c755f90-b731-453d-932b-ab26f981a5ec)

<br><br>

# 2. Exploratory Data Analysis

## 2.1 pandas profiling

아래의 간단한 명령어를 사용하여 우리가 사용할 데이터와 관련된 많은 정보를 탐색할 수 있다.

자세한 내용)

[01-05 판다스 프로파일링(Pandas-Profiling)](https://wikidocs.net/47193)

```python
def load_data():
    data_dir = Path("../input/tabular-playground-series-mar-2022")
    df_train = pd.read_csv(data_dir / "train.csv")
    df_test = pd.read_csv(data_dir / "test.csv")
    # Merge the splits so we can process them together
    df = pd.concat([df_train, df_test])
    return df

df_data = load_data()
pp.ProfileReport(df_data)
```

![Untitled 1](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/7fb0b870-9321-4829-b2cd-351a19dfcdbe)

<br>

## 2.2 General Congestion Analysis

Congestion을 예측해야 하기 때문에 Congestion 관련 EDA를 쭉 진행했다.

```python
from itertools import cycle
palette = cycle(px.colors.sequential.Viridis)
df_graph = df_data[df_data['congestion'].isnull()==False]

# Defining all our palette colours
primary_blue = "#496595"
primary_blue2 = "#85a1c1"
primary_blue3 = "#3f4d63"
primary_grey = "#c6ccd8"
primary_black = "#202022"
primary_bgcolor = "#f4f0ea"

# "coffee" pallette turqoise-gold.
f1 = "#a2885e"
f2 = "#e9cf87"
f3 = "#f1efd9"
f4 = "#8eb3aa"
f5 = "#235f83"
f6 = "#b4cde3"

# plot 큰 뼈대 만들기
# chart
fig = make_subplots(rows=3, cols=2, specs=[[{"type":"bar"}, {"type":"scatter"}], [{"colspan":2}, None], [{'type':'histogram'}, {'type':'bar'}]],
				column_widths=[0.4, 0.6], vertical_spacing=0.1, horizontal_spacing=0.1,
				subplot_titles=("Mean Congestion per Day of Week", "Hourly Congestion Trend", "Daily Congestion Trend", "Congestion Distribution", "Congestion Value Counts"))

# Upper Left chart
df_day = df_graph.groupby(['day_of_week']).agg({"congestion":"mean"}).reset_index().sort_values(by='congestion', ascending=False)
values = list(range(7))
fig.add_trace(go.Bar(x=df_day['day_of_week'], y=df_day['congestion'], marker=dict(color=values, colorscale="Viridis"),
                              name="Day of Week"), row=1, col=1)

# 각 축 기준으로 그래프 설정 수정
fig.update_xaxes(showgrid=False, linecolor='gray', linewidth=2, zeroline=False, row=1, col=1)
fig.update_yaxes(showgrid=False, linecolor='gray', linewidth=2, zeroline=False, row=1, col=1)

## Upper Right chart
# 요일별 평균 congestion
df_hour = df_graph.groupby(['hour']).agg({"congestion":"mean"}).reset_index('hour')
fig.add_trace(go.Scatter(x=df_hour['hour'], y=df_hour['congestion'], mode='lines+markers',
                        marker=dict(color=primary_blue3), name='Hourly Congestion'), row=1, col=2)

# Rectangle to highlight range
fig.add_vrect(x0=12.5, x1=18.5,
             fillcolor=px.colors.sequential.Viridis[4],
             layer="below", 
             opacity=0.25,
             line_width=0,
             row=1, col=2
)

fig.add_annotation(dict(
        x=7.9, y=df_hour.loc[8, 'congestion']+0.45,
        text="There is a <b>peak at <br>8am</b> coinciding with<br>going to work.",
        ax='-20', ay='-60',
        showarrow=True, arrowhead=7, arrowwidth=0.7
), row=1, col=2)

fig.add_annotation(dict(
        x=15.50, y=49,
        text="Midday hours are <br><b>the rush hours</b>.",
        showarrow=False
), row=1, col=2)

fig.add_annotation(dict(
        x=18.5, y=df_hour.loc[18, 'congestion'],
        text="From 6pm <br>on <b>congestion<br> ratio falls</b>.",
        ax="50", ay="-40",
        showarrow=True,
        arrowhead=7, arrowwidth=0.7
), row=1,  col=2)

fig.update_xaxes(showgrid=False, linecolor='gray', linewidth=2, zeroline=False, row=1, col=2)
fig.update_yaxes(showgrid=False, linecolor='gray', linewidth=2, row=1, col=2)

## Medium Chart
df_week = df_graph.groupby(['day_of_year']).agg({"congestion":"mean"}).reset_index()
fig.add_trace(go.Scatter(x=df_week['day_of_year'], y=df_week['congestion'], mode='lines',
                        marker=dict(color=px.colors.sequential.Viridis[5]),
                        name='Daily Congestion'), row=2, col=1)

# seasonal_decompose(): 시계열 성분 분해 시 사용, 추세(Trend), 계절성(seasonal), 잔차(residual) 파악
# Trend: 관측치들의 평균 측정
# seasonal: 빈도가 항상 일정하며 알려져 있는 경우
# residual: 관측값과 대응되는 적합값(fitted value)과 관측값의 차이
from statsmodels.tsa.seasonal import seasonal_decompose
decomp = seasonal_decompose(df_week['congestion'], period=61, model='additive', extrapolate_trend='freq')

fig.add_trace(go.Scatter(x=df_week['day_of_year'], y=decomp.trend, # trend 출력
                         mode='lines', marker=dict(color=primary_blue3),
                         name='Congestion Trend'), row=2, col=1)

fig.update_xaxes(showgrid=False, linecolor='gray', linewidth=2, row=2, col=1)
fig.update_yaxes(gridcolor='gray', gridwidth=0.15, linecolor='gray', linewidth=2, row=2, col=1)

## Left Bottom Chart
# add_trace(): Figure에 새로운 Trace 추가
fig.add_trace(go.Histogram(x=df_graph.congestion, name='Congestion Distribution', marker=dict(color=px.colors.sequential.Viridis[3])), row=3, col=1)

fig.update_xaxes(showgrid = False, showline = True, linecolor = 'gray', linewidth = 2, row = 3, col = 1)
fig.update_yaxes(showgrid = False, gridcolor = 'gray', gridwidth = 0.5, showline = True, linecolor = 'gray', linewidth = 2, row = 3, col = 1)

# Right Bottom Chart
con_bar = df_graph.copy()
con_bar['congestion_group'] = pd.cut(con_bar.congestion, bins=[0,20,40,60,80,100], labels=['0-20', '20-40', '40-60', '60-80', '80-100'])
con_bar = con_bar.groupby('congestion_group').agg({'congestion':'count'}).reset_index().sort_values(by='congestion')

values = list(range(5))
fig.add_trace(go.Bar(x=con_bar['congestion'], y=con_bar['congestion_group'], marker = dict(color=values, colorscale="Viridis_r"), name = 'Congestion Values', orientation = 'h'),
                      row=3, col=2)

fig.update_xaxes(showgrid = False, linecolor='gray', linewidth = 2, zeroline = False, row=3, col=2)
fig.update_yaxes(showgrid = False, linecolor='gray',linewidth=2, zeroline = False, row=3, col=2)

fig.add_annotation(dict(
        x=con_bar.loc[4,'congestion']+0.15, y=0,
        text="Highest congestion <br>ratios are <b> more unusual</b>.",
        ax="110",
        ay="-20",
        showarrow = True,
        arrowhead = 7,
        arrowwidth = 0.7
), row=3, col=2)

# update_layout(): 그래프 생성 후 layout 정보 업데이트
# General Styling
fig.update_layout(height=1100, bargap=0.2,
                  margin=dict(b=50,r=30,l=100),
                  title = "<span style='font-size:36px; font-family:Times New Roman'>Congestion Analysis</span>",                  
                  plot_bgcolor='rgb(242,242,242)',
                  paper_bgcolor = 'rgb(242,242,242)',
                  font=dict(family="Times New Roman", size= 14),
                  hoverlabel=dict(font_color="floralwhite"),
                  showlegend=False)
```

![Untitled 2](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/ec21d748-4ccc-4042-91fc-9687d80f6dda)

![Untitled 3](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/acc75606-3de5-4863-bf51-bf0b2c14c33c)

![Untitled 4](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/d988d3a5-6e29-40a7-9c20-e4375e15c764)

<br>

**Interpret**

- 평일 congestion: similar congestion rate, 주말 congestion이 가장 적고 일요일이 가장 조용한 날
- upper right chart: 하루가 시작될 때 트래픽이 증가
    - 가장 바쁜 시간: 13~17시
    - 밤이 되면 congestion 감소
- middle graph: 주마다 강한 계절성
    - trend: 거의 일정하게 유지, 시간이 지남에 따라 미미하게 증가
- 아래 그래프: congestion dirtribution이 normally distributed(정상 분포) / 대부분의 congestion value 비율은 40~60

<br>

## 2.3 **Daily Congestion Analysis**

요일별 교통량 분석 결과이다.

```python
# 결측치 제거
df_graph = df_data[df_data['congestion'].isnull()==False]

# chart
fig = make_subplots(rows=2, cols=3, 
                    specs=[[{"type": "scatter"}, {"type": "scatter"}, {'type':'scatter'}], [{"type": "scatter"}, {"type": "scatter"}, {'type':'scatter'}]],
                    column_widths=[0.33, 0.33, 0.34], vertical_spacing=0.125, horizontal_spacing=0.1,
                    subplot_titles=("Monday Congestion", "Tuesday Congestion", "Wednesday Congestion",'Thursday Congestion','Friday Congestion','Weekend Congestion'))

# Upper Left chart
df_monday = df_graph[df_graph.day_of_week == 'Monday'].groupby(['hour']).agg({"congestion" : "mean"}).reset_index()
fig.add_trace(go.Scatter(x=df_monday['hour'], y=df_monday['congestion'], mode='lines+markers',
                 marker = dict(color = px.colors.sequential.Viridis[0]), name='Monday Congestion'), row = 1, col = 1)

fig.update_xaxes(showgrid = False, linecolor='gray', linewidth = 2, zeroline = False, row=1, col=1)
fig.update_yaxes(showgrid = False, linecolor='gray',linewidth=2, row=1, col=1)

# Upper Medium chart
df_tuesday = df_graph[df_graph.day_of_week == 'Tuesday'].groupby(['hour']).agg({"congestion" : "mean"}).reset_index()
fig.add_trace(go.Scatter(x=df_tuesday['hour'], y=df_tuesday['congestion'], mode='lines+markers',
                 marker = dict(color = px.colors.sequential.Viridis[2]), name='Monday Congestion'), row = 1, col = 2)

fig.update_xaxes(showgrid = False, linecolor='gray', linewidth = 2, zeroline = False, row=1, col=2)
fig.update_yaxes(showgrid = False, linecolor='gray',linewidth=2, row=1, col=2)

# Upper Right chart
df_wednesday = df_graph[df_graph.day_of_week == 'Wednesday'].groupby(['hour']).agg({"congestion" : "mean"}).reset_index()
fig.add_trace(go.Scatter(x=df_wednesday['hour'], y=df_wednesday['congestion'], mode='lines+markers',
                 marker = dict(color = px.colors.sequential.Viridis[4]), name='Monday Congestion'), row = 1, col = 3)

fig.update_xaxes(showgrid = False, linecolor='gray', linewidth = 2, zeroline = False, row=1, col=3)
fig.update_yaxes(showgrid = False, linecolor='gray',linewidth=2, row=1, col=3)

# Bottom Left chart
df_thursday = df_graph[df_graph.day_of_week == 'Thursday'].groupby(['hour']).agg({"congestion" : "mean"}).reset_index()
fig.add_trace(go.Scatter(x=df_thursday['hour'], y=df_thursday['congestion'], mode='lines+markers',
                 marker = dict(color = px.colors.sequential.Viridis[6]), name='Monday Congestion'), row = 2, col = 1)

fig.update_xaxes(showgrid = False, linecolor='gray', linewidth = 2, zeroline = False, row=2, col=1)
fig.update_yaxes(showgrid = False, linecolor='gray',linewidth=2, row=2, col=1)

# Bottom Medium chart
df_friday = df_graph[df_graph.day_of_week == 'Friday'].groupby(['hour']).agg({"congestion" : "mean"}).reset_index()
fig.add_trace(go.Scatter(x=df_friday['hour'], y=df_friday['congestion'], mode='lines+markers',
                 marker = dict(color = px.colors.sequential.Viridis[9]), name='Monday Congestion'), row = 2, col = 2)

fig.update_xaxes(showgrid = False, linecolor='gray', linewidth = 2, zeroline = False, row=2, col=2)
fig.update_yaxes(showgrid = False,linecolor='gray', linewidth=2, row=2, col=2)

# Bottom Right chart
df_weekend = df_graph[df_graph.is_weekend == True].groupby(['hour']).agg({"congestion" : "mean"}).reset_index()
df_saturday = df_graph[df_graph.day_of_week == 'Saturday'].groupby(['hour']).agg({"congestion" : "mean"}).reset_index()
df_sunday = df_graph[df_graph.day_of_week == 'Sunday'].groupby(['hour']).agg({"congestion" : "mean"}).reset_index()

fig.add_trace(go.Scatter(x=df_weekend['hour'], y=df_weekend['congestion'], mode='lines+markers',
                 marker = dict(color = px.colors.sequential.Viridis[6]), name='Weekend Average Congestion'), row = 2, col = 3)
fig.add_trace(go.Scatter(x=df_saturday['hour'], y=df_saturday['congestion'], mode='lines+markers',
                 marker = dict(color = px.colors.sequential.Viridis[3]), name='Saturday Congestion'), row = 2, col = 3)
fig.add_trace(go.Scatter(x=df_sunday['hour'], y=df_sunday['congestion'], mode='lines+markers',
                 marker = dict(color = px.colors.sequential.Viridis[9]), name='Sunday Congestion'), row = 2, col = 3)

fig.update_xaxes(showgrid = False, linecolor='gray', linewidth = 2, zeroline = False, row=2, col=3)
fig.update_yaxes(showgrid = False, linecolor='gray',linewidth=2, row=2, col=3)

# General Styling
fig.update_layout(height=750, bargap=0.2,
                  margin=dict(b=50,r=30,l=100),
                  title = "<span style='font-size:36px; font-family:Times New Roman'>Daily Congestion Analysis</span>",                  
                  plot_bgcolor='rgb(242,242,242)',
                  paper_bgcolor = 'rgb(242,242,242)',
                  font=dict(family="Times New Roman", size= 14),
                  hoverlabel=dict(font_color="floralwhite"),
                  showlegend=False)
```

![Untitled 5](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/b926e624-c1cf-4046-9c6f-ee339bf3c289)

<br>

**Interpret**

- 평일 간 congestion 비슷한 양상, **주말**과는 차이 존재
    - 대체적으로 낮은 congestion (일을 하지 않기 때문)
    - 상대적으로 기복이 낮음

<br>

## 2.4 **Direction and Location Analysis**

방향과 도로의 위치에 따른 교통량 분석이다.

```python
palette = cycle(px.colors.sequential.Sunsetdark)
df_graph = df_data[df_data['congestion'].isnull() == False]

# chart
fig = make_subplots(rows=1, cols=3, 
                    specs=[[{"type": "bar"}, {"type": "bar"}, {'type':'bar'}]],
                    column_widths=[0.33, 0.34, 0.33], vertical_spacing=0.1, horizontal_spacing=0.1,
                    subplot_titles=("Mean Congestion per Direction", "Mean Congestion Per X Location", "Mean Congestion per Y Location"))

# Left chart
df_direction = df_graph.groupby(['direction']).agg({"congestion" : "mean"})
values = list(range(8))
fig.add_trace(go.Bar(x=df_direction.index, y=df_direction['congestion'], marker = dict(color=values, colorscale="Viridis"), name = 'Day of Week'),
                      row=1, col=1)

fig.update_xaxes(showgrid = False, linecolor='gray', linewidth = 2, zeroline = False, row=1, col=1)
fig.update_yaxes(showgrid = False, linecolor='gray',linewidth=2, zeroline = False, row=1, col=1)

# Middle chart
df_x = df_graph.groupby(['x']).agg({"congestion" : "mean"})
values = list(range(3))
fig.add_trace(go.Bar(x=df_x.index, y=df_x['congestion'], marker = dict(color=values, colorscale="Viridis"), name = 'Day of Week'),
                      row=1, col=2)

fig.update_xaxes(showgrid = False, linecolor='gray', linewidth = 2, zeroline = False, row=1, col=2)
fig.update_yaxes(showgrid = False, linecolor='gray',linewidth=2, zeroline = False, row=1, col=2)

# Right chart
df_y = df_graph.groupby(['y']).agg({"congestion" : "mean"})
values = list(range(4))
fig.add_trace(go.Bar(x=df_y.index, y=df_y['congestion'], marker = dict(color=values, colorscale="Viridis"), name = 'Day of Week'),
                      row=1, col=3)

fig.update_xaxes(showgrid = False, linecolor='gray', linewidth = 2, zeroline = False, row=1, col=3)
fig.update_yaxes(showgrid = False, linecolor='gray',linewidth=2, zeroline = False, row=1, col=3)

# General Styling
fig.update_layout(height=400, bargap=0.2,
                  margin=dict(b=50,r=30,l=100),
                  title = "<span style='font-size:36px; font-family:Times New Roman'>Direction and Location Analysis</span>",                  
                  plot_bgcolor='rgb(242,242,242)',
                  paper_bgcolor = 'rgb(242,242,242)',
                  font=dict(family="Times New Roman", size= 14),
                  hoverlabel=dict(font_color="floralwhite"),
                  showlegend=False)
```

![Untitled 6](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/aeae8afb-60b5-4c37-bd39-006e4d015f10)

<br>

**Interpret**

- 남쪽 방향으로 가장 congestion 높음
- x=1 위치가 가장 붐빔
- y=0 & y=2 위치가 붐빔

<br>

## 2.5 Outliers

### 2.5.1 Grubbs’ test

**Grubbs' test(그럽스 검정)**: dataset 안의 이상치 판별 시 사용

![Untitled 7](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/aafa8e2a-f82b-4262-9e7a-0c760ca2e963)

Grubbs’ test 관련 추가적인 구글링 필요

위 수식을 구현한 코드이다.

```python
import scipy.stats as stats
def grubbs_test(x):
    n = len(x)
    mean_x = np.mean(x)
    sd_x = np.std(x)
    numerator = max(abs(x-mean_x))
    g_calculated = numerator/sd_x
    print("Grubbs Calculated Value:", g_calculated)
    t_value = stats.t.ppf(1-0.05 / (2*n), n-2)
    g_critical = ((n-1) * np.sqrt(np.square(t_value))) / (np.sqrt(n) * np.sqrt(n-2 + np.square(t_value)))
    print("Grubbs Critical Value:", g_critical)
    if g_critical > g_calculated:
        print("From grubbs_test we observe that calculated value is lesser than critical value, Accept null hypothesis and conclude that there is no outliers\n")
    else:
        print("From grubbs_test we observe that calculated value is greater than critical value, Reject null hypothesis and conclude that there is an outliers\n")

grubbs_test(df_data[df_data.congestion.isnull()==False]['congestion'])
```

![Untitled 8](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/4eaf1d88-2b46-4034-bdec-37c95998f383)

<br>

### 2.5.2 ****Z-score method****

Z score method 사용하여 평균에서 얼마나 많은 표준 편차 값이 있는지 확인할 수 있다. (추가적인 구글링 필요)

```python
df_outlier = df_data.reset_index().set_index('row_id').copy()
out = []
def Zscore_outlier(df):
    m = np.mean(df)
    sd = np.std(df)
    row = 0
    for iin df:
        z = (i-m)/sd
        if np.abs(z) > 3:
            out.append(row)
        row += 1
    return out

outliers_index = Zscore_outlier(df_outlier[df_outlier.congestion.isnull()==False]['congestion'])
```

```python
df_outlier['outlier'] = 0
df_outlier.loc[outliers_index, 'outlier'] = 1

fig = px.scatter(df_outlier, x=df_outlier.index, y='congestion', color='outlier')
fig.update_xaxes(visible=False, zeroline=False)
fig.update_yaxes(showgrid=False, gridcolor='gray', gridwidth=.5, zeroline=False)

#General Stylingfig.update_layout(height=400, bargap=0.2,
                  margin=dict(b=50,r=30,l=100),
                  title = "<span style='font-size:36px; font-family:Times New Roman'>Congestion Outliers Analysis</span>",
                  plot_bgcolor='rgb(242,242,242)',
                  paper_bgcolor = 'rgb(242,242,242)',
                  font=dict(family="Times New Roman", size= 14),
                  hoverlabel=dict(font_color="floralwhite"),
                  showlegend=False)
```

![Untitled 9](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/6b07f386-c43a-4fa9-8e49-9f8b08e84de9)

노란색이 outliers

<br><br>

# 3. Feature Engineering

## 3.1 Dataset Score

baseline score: 수정된 set of features 실제로 문제 개선으로 이어진건지 확인하는 용도

첫 번째 단계: outlier 삭제 (이후에 알겠지만, outlier 삭제한 것이 결과가 더 좋다(?))

```python
def score_dataset(X, y, model=XGBRegressor(tree_method='gpu_hist', predictor='gpu_predictor'), model_2 = CatBoostRegressor(task_type='GPU', silent=True)):
# def score_dataset(X, y, model=XGBRegressor(), model_2=CatBoostRegressor(silent=True)):
    # Label encoding is good for XGBRegressor() and RandomForest, but one-hot would be better for models like Lasso or Ridge.
    # The 'cat.codes' attribute holds the category levels.
    for colname in X.select_dtypes(["object"]).columns:
        X[colname] = LabelEncoder().fit_transform(X[colname])
    print(X)
    X['week'] = X['week'].astype(int)
    X = X.drop('row_id', axis=1)
    # Metric for TPS Mar22 competition is MAE (Mean Absolute Error)
    score_xgb = cross_val_score(
    model, X, y, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1
    )
    
    score_cat = cross_val_score(
    model_2, X, y, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1
    )
    
    score = -0.5 * (score_xgb.mean() + score_cat.mean())
    return score

# with outliers: 6.86624 MAE
# without outliers: 6.83095 MAE
# df_data = df_data.reset_index().set_index('row_id')
# df_data = df_data.drop(outliers_index, axis=0)
# df_data = df_data.reset_index().set_index('time')

x = df_data[df_data['congestion'].isnull()==False].copy()
y = pd.DataFrame(x.pop('congestion'))

baseline_score = score_dataset(x, y)
print(f"Baseline score: {baseline_score:.5f} MAE")
```

![Untitled 10](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/7455b257-57ca-432d-b0d5-11c44b1a45ff)

![Untitled 11](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/de321021-54d9-4ff8-8bff-985197675b5e)

첫 Baseline model에서 사용된 데이터 columns: 총 12개

- row_id
- x, y
- direction
- year, month, week, hour, minute
- day_of_week, day_of_year, is_weekend

<br>

## 3.2 Cyclical Features

direction을 sin & cos 구성 요소로 분해하여 캡처

- 부동 소수점 노이즈 피하기 위해 일부 값을 직접 코딩

```python
from math import sin, cos, pi, exp
sin_vals = {
    'NB': 0.0,
    'NE': sin(1 * pi/4),
    'EB': 1.0,
    'SE': sin(3 * pi/4),
    'SB': 0.0,
    'SW': sin(5 * pi/4),
    'WB': -1.0,
    'NW': sin(7 * pi/4),
}

cos_vals = {
    'NB': 1.0,
    'NE': cos(1 * pi/4),
    'EB': 0.0,
    'SE': cos(3 * pi/4),
    'SB': -1.0,
    'SW': cos(5 * pi/4),
    'WB': 0.0,
    'NW': cos(7 * pi/4),
}

df_data['sin'] = df_data['direction'].map(sin_vals)
df_data['cos'] = df_data['direction'].map(cos_vals)
```


<br>

## 3.3 Geography and Direction

각 도로 간의 상관관계

- 비교적 가까운 도로간의 상관관계에 따라 미래 예측 가능

### 3.3.1 Roadway

```python
x = df_data[df_data['congestion'].isnull()==False].copy()
y = pd.DataFrame(x.pop('congestion'))
baseline_score = score_dataset(x, y)
print(f"Baseline score:{baseline_score:.5f} MAE")
```

![Untitled 12](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/1e6f2cc6-5e69-41ef-82a6-aa7c3b614c2e)

![Untitled 13](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/473aa356-5db4-42c1-ac5c-77a627ecedcc)

두 번째 Baseline model에서 사용된 데이터 columns: 총 14개 

- 추가된 columns: sin, cos


```python
# px: plotly.express
df_data['roadway'] = df_data.x.astype(str) + df_data.y.astype(str) + df_data.direction.astype(str)
px.box(df_data[df_data.congestion.isnull()==False], x="roadway", y="congestion", color='roadway')
```

![Untitled 14](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/210386ab-f1f5-4dc2-a559-9d75089d86a1)

위에서 만든 feature를 활용하면 각 위치의 congestion으로 잘 구분할 수 있어서 사용 가능한 feature라고 하는데, 고르게 분포하고 있어서 그렇다고 일단 이해해놓긴 함.

<br>

### 3.3.2 Mathematical Transformations

각 sin/cos 함수로 위치 좌표 곱하기

```python
df_data['x_cos_hour'] = df_data.x * df_data.cos * df_data.hour
df_data['y_sen_hour'] = df_data.y * df_data.sin * df_data.hour

df_data = df_data.drop(['year','x','y','direction'], axis=1)

x = df_data[df_data['congestion'].isnull() == False].copy()
y = pd.DataFrame(x.pop('congestion'))
baseline_score = score_dataset(x, y)
print(f"Baseline score: {baseline_score:.5f} MAE")
```

![Untitled 15](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/88d2cc94-1dcc-4114-bef9-5c869be20d0d)

여기서 궁금한 점: hour를 왜 곱하지?

- hour를 곱하지 않고 돌렸을 때 baseline score가 좋지 않긴 함. 왜지???

![Untitled 16](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/9afdc3b7-be34-4631-a415-de1dfcd52e4b)

세 번째 Baseline model에서 사용된 데이터 columns: 총 13개

- 추가된 columns: x_cos_hour, y_sin_hour, roadway
- 삭제된 columns: year, x, y, direction

<br>

### 3.3.3 Mean, median, maximum, minimum congestion per roadway / time

각 도로, 시간별로 congestion target value 관련 statistical features 생성

총 4개 features

- Mean congestion
- Median congestion
- Minimum congestion
- Maximum congestion

```python
df_data = df_data.reset_index()
keys = ['roadway', 'day_of_week','hour', 'minute']

df = df_data.groupby(by=keys).mean().reset_index().set_index(keys)
df['mean congestion'] = df['congestion']
df_data = df_data.merge(df['mean congestion'], how='left', left_on=keys, right_on=keys)

df = df_data.groupby(by=keys).median().reset_index().set_index(keys)
df['median congestion'] = df['congestion']
df_data = df_data.merge(df['median congestion'], how='left', left_on=keys, right_on=keys)

df = df_data.groupby(by=keys).min().reset_index().set_index(keys)
df['min congestion'] = df['congestion']
df_data = df_data.merge(df['min congestion'], how='left', left_on=keys, right_on=keys)

df = df_data.groupby(by=keys).max().reset_index().set_index(keys)
df['max congestion'] = df['congestion']
df_data = df_data.merge(df['max congestion'], how='left', left_on=keys, right_on=keys)

df_data = df_data.set_index('time')

x = df_data[df_data['congestion'].isnull() == False].copy()
y = pd.DataFrame(x.pop('congestion'))
baseline_score = score_dataset(x, y)
print(f"Baseline score: {baseline_score:.5f} MAE")
```

![Untitled 17](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/5abfc5aa-cc8f-4c58-9e55-f77eabab6a87)

![Untitled 18](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/7f59645e-4853-4f1c-a015-1ead9889cc39)

네 번째 Baseline model에서 사용된 데이터 columns: 총 17개

- 추가된 columns: mean congestion, median congestion, min congestion, max congestion

<br>

## 3.4 Mutual Information (상호의존정보)

mutual information(MI): 한 knowledge of one quantity가 다른 knowledge에 대한 불확실성을 줄이는 정도를 측정

- feature에 대한 가치를 알 수 있다?

```python
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(["object"]):
        X[colname], _ = X[colname].factorize() # _: 값을 무시할 때 사용
    # All discrete features should now have integer dtypes
    # discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.types]
    mi_scores = mutual_info_regression(X, y, random_state=0) # 두 random variable들이 얼마나 mutual dependence한지(?) measure
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

y = df_data[df_data['congestion'].isnull()==False]['congestion']
x = df_data[df_data['congestion'].isnull()==False].drop('congestion', axis=1)
mi_scores = make_mi_scores(x, y)
mi_scores = pd.DataFrame(mi_scores).reset_index().rename(columns={'index':'Feature'})
mi_scores
```

<br>

mi_scores 시각화

```python
fig = px.bar(mi_scores, x='MI Scores', y='Feature', color="MI Scores", color_continuous_scale='darkmint')
fig.update_layout(height=750, title_text='Mutual Information Scores',
                 title_font=dict(size=29, family="Lato, sans-serif"), xaxis={'categoryorder':'category ascending'}, margin=dict(t=80))
```

![Untitled 19](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/a02f9ec4-99e9-459b-99ea-df5e7cda27f8)

<br>

## 3.5 training 전 마지막 데이터 전처리

datatype이 ‘object’인 feature에 대해 label encoding한 후 메모리 축소 작업을 수행한다. object가 확실히 메모리를 많이 잡아먹는 것을 알 수 있다.

```python
qualitative = [col for col in df_data if df_data[col].dtype=='object']
for feature in qualitative:
    df_data[feature] = LabelEncoder().fit_transform(df_data[feature])
df_data = reduce_mem_usage(df_data)

df_data = df_data.drop(['month','minute','week','day_of_week','is_weekend','day_of_year','cos','sin'], axis=1)

df_train = df_data[df_data.congestion.isnull()==False]
df_test = df_data[df_data.congestion.isnull()==True]

from sklearn.model_selection import train_test_split
X = df_train.drop(['congestion', 'row_id'], axis=1)
y = df_train['congestion']
```

![Untitled 20](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/d41d9574-942f-4357-a109-36b6df148f10)

여기서 ‘roadway’ feature에 대해서도 label encoding을 해주었는데, 이게 가장 베스트인지는 잘 모르겠다. 그렇다고 one-hot encoding하기에는 64개나 되기에 많은 column 수를 차지하게 될 것이고, 그렇다고 도로 간에 순서가 있는 것도 아니라서… 고민을 좀 더 해봐야할 것으로 보인다.

- 당장의 생각에는 roadway에 도로의 위치 정보까지 포함해서 feature engineering을 한 것으로 보이는데, 이를 위치 정보 / 도로 방향으로 나누어 label encoding해 주는 것이 어떨까 싶다.

<br><br>

# 4. Modeling

## 4.1 Catboost

### 4.1.1 Hyperparameter Tuning - Optuna

Catboost에 대해서만 Optuna로 튜닝

- Optuna: 하이퍼파라미터 최적화 프레임워크

~~Optuna 어떤 프레임워크인지 추후 정리~~

```python
from sklearn.metrics import mean_absolute_error as mae

def objective(trial):
    params = {
        "random_state": trial.suggest_categorical("random_state", [2022]),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.0001, 0.3),
        'bagging_temperature': trial.suggest_loguniform('bagging_temperature', 0.01, 100.00), # Bayesian bootstrap(?) 설정 정
        "n_estimators": 1000, # 생성할 tree 개수
        "max_depth": trial.suggest_int('max_depth', 4, 16), # 트리의 최대 깊이
        'random_strength': trial.suggest_int('random_strength', 0, 100), # scoring splits에 사용할 randomness의 양(모르겠다...)
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 3e-5), # l2 regulation
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100), # 최종 결정 클래스인 Leaf Node가 되기 위해 필요한 최소한의 데이터 개체수
        "max_bin": trial.suggest_int("max_bin", 200, 500), # 히스토그램 빈 개수(?)
        "od_type": trial.suggest_categorical('od_type', ['IncToDec', 'Iter']), # 과적합 검출기 유형
        'task_type': trial.suggest_categorical('task_type', ['GPU']), # CPU -> GPU
        'loss_function': trial.suggest_categorical('loss_function', ['MAE']),  # (Maybe) 훈련에 사용되는 함수 정의
        'eval_metric': trial.suggest_categorical('eval_metric', ['MAE']) # 검증에 사용되는 함수 정의
    }
    
    model = CatBoostRegressor(**params) # **params: 여러 파라미터 받을 수 있음. 파라미터 명과 함께 전달.
    X_train_tmp, X_valid_tmp, y_train_tmp, y_valid_tmp = train_test_split(X, y, test_size=0.3, random_state=42)
    model.fit(
        X_train_tmp, y_train_tmp,
        eval_set = [(X_valid_tmp, y_valid_tmp)],
        early_stopping_rounds=35, verbose=0
    )
    
    y_train_pred = model.predict(X_train_tmp)
    y_valid_pred = model.predict(X_valid_tmp)
    train_mae = mae(y_train_tmp, y_train_pred)
    valid_mae = mae(y_valid_tmp, y_valid_pred)
    
    print(f'MAE of Train: {train_mae}')
    print(f'MAE of Validation: {valid_mae}')
    
    return valid_mae

allow_optimize = 1
```

```python
TRIALS = 100
TIMEOUT = 3600

if allow_optimize:
    sampler = TPESampler(seed=42) # Sampler using TPE(Tree-structured Parzen Estimator) algorithm (어려운 개념, 공부할 필요 O)
    
    study = optuna.create_study( # optuna.create_study(): 새로운 Study를 만든다?
        study_name = 'cat_parameter_opt',
        direction = 'minimize',
        sampler = sampler,
    )
    study.optimize(objective, n_trials=TRIALS)
    print("Best Score: ", study.best_value)
    print("Best Trial", study.best_trial.params)
    
    best_params = study.best_params
    
    X_train_tmp, X_valid_tmp, y_train_tmp, y_valid_tmp = train_test_split(X, y, test_size=0.3, random_state=42)
    model_tmp = CatBoostRegressor(**best_params, n_estimators=30000, verbose=1000).fit(X_train_tmp, y_train_tmp,
            eval_set=[(X_valid_tmp, y_valid_tmp)], early_stopping_rounds=35)
```

![Untitled 21](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/c8a89987-1270-42ac-a5cc-b83ad0532a12)

![Untitled 22](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/666df8ac-01dd-414a-b0b2-4f7219814225)

Optuna 프레임워크를 활용하여 반복하면서 최적의 hyperparameter 찾는 과정

<br>

### 4.1.2 Fitting - Feature Importance

찾은 hyperparameter 토대로 학습 수행

```python
if allow_optimize:
    model = CatBoostRegressor(**best_params, n_estimators=model_tmp.get_best_iteration(), verbose=1000).fit(X, y) # get_best_iteration(): 마지막 검증 세트에서 evaluation metric or loss function에 대해 최상의 결과로 iteration 반환
else:
    model = CatBoostRegressor(
    verbose=1000,
    early_stopping_rounds=10,
    # iterators=5000,
    random_state = 2022, learning_rate = 0.0824038781081412, bagging_temperature = 0.03568558360430449, max_depth = 16,
    random_strength=47, l2_leaf_reg = 7.459775961819184e-06, min_child_samples = 49, max_bin = 320, od_type = 'Iter',
    task_type='GPU', loss_function='MAE', eval_metric='MAE').fit(X, y)
```

![Untitled 23](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/13333c6c-d730-4984-bbfb-92fa51f94a3b)

```python
plot_feature_importance(model.get_feature_importance(), X.columns, 'CatBoost')
```

![Untitled 24](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/254b83f1-0bdb-4dfa-b81c-6a49b5dd7cc6)

<br>

### 4.1.3 Making Prediction

```python
x_test = df_test.drop(['congestion', 'row_id'], axis=1).copy()
predictions = model.predict(x_test)
submit_cat = pd.DataFrame({'row_id': df_test.row_id, 'congestion': predictions})
submit_cat = submit_cat.reset_index().drop('time', axis=1).set_index('row_id')
submit_cat
```

![Untitled 25](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/b3c0fdb7-b364-4a25-91fd-e1ea39a0d20b)

<br>

## 4.2 XGBoost

글쓴이는 XGBoost에서는 그렇게 좋은 결과를 보진 못한 것 같다.

<br>

### 4.2.1 Hyperparameter Tuning - GridSearch / RandomizedSearch

상대적으로 적은 수의 조합 탐색 시 grid search 적합. hyperparameter 탐색 공간이 큰 경우 RandomizedSearchCV 사용이 더 나은 경우가 많다.

**RandomizedSearchCV**: GridSearchCV와 거의 같은 방식으로 사용 가능

- 가능한 모든 combinations 시도, every iteration(반복)에서 각 hyperparameter에 대해 임의의 값을 선택하여 주어진 수의 임의 조합 평가
- 좀 더 찾아보자

RandomizedSearchCV 두 가지 이점

1. 예를 들어 1,000번의 반복에 대해 임의 검색 실행 시 각 hyperparameter에 대해 1,000개의 다른 값 탐색
2. iterations 횟수 설정만 하면 hyperparameter search에 할당하려는 컴퓨팅 예산을 더 많이 제어할 수 있다

```python
from sklearn.model_selection import GridSearchCV

allow_optimize = 1

if allow_optimize:
    param_grid={'max_depth': [4,5,6,7,8,9],
            #'n_estimators': [100,200,300,400,500,600,700,800,900,1000],
            'min_child_weight' : [1,2,3,4,5,6],
            'gpu_id' : [0]
        }

    regressor = XGBRegressor(tree_method = 'gpu_hist', predictor = 'gpu_predictor')
    CV_regressor = GridSearchCV(regressor, param_grid, cv=3, scoring="neg_mean_absolute_error", n_jobs= -1, return_train_score = True, verbose = 0)
    CV_regressor.fit(X, y)
    
    print("The best hyperparameters are : ","\n")
    print(CV_regressor.best_params_)
```

![Untitled 26](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/a36b50f1-2628-4804-8fae-1ec250847975)

<br>

### 4.2.2 Fitting - Feature Importance

```python
if allow_optimize:
    CV_regressor = CV_regressor.best_estimator_
else:
    CV_regressor = XGBRegressor(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0, max_depth=4, n_estimators=100)
CV_regressor.fit(X, y)
```

![Untitled 27](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/1b0508de-43c7-4a03-9fd1-f866f6e4da9b)

```python
plot_feature_importance(CV_regressor.feature_importances_, X.columns, 'XGBoost')
```

![Untitled 28](https://github.com/SuhwanMylife/SuhwanMylife.github.io/assets/70688382/6941fb98-accc-45ea-8d40-70b3a9235220)

<br>

### 4.2.3 Making Predictions

```python
predictions = CV_regressor.predict(x_test)
submit_xgb = pd.DataFrame({'row_id':df_test.row_id, 'congestion':predictions})
submit_xgb = submit_xgb.reset_index().drop('time',axis=1).set_index('row_id')
```

<br>

## 4.3 Blending and Submitting

Blending: 앙상블의 한 종류, validation set의 예측 결과들을 원래 feature에 붙여 새로운 dataset을 만든 후 Final prediction model을 적합시켜 test_set에 대한 예측 수
- 실제로 다른 모델은 성능이 그닥이라 블렌딩하지 않고 제출

```python
submit = pd.DataFrame({'congestion': submit_cat['congestion']+0*submit_xgb['congestion']})
special = pd.read_csv('../input/tps-mar-22-special-values/special v2.csv', index_col="row_id")
special = special[['congestion']].rename(columns={'congestion':'special'})
submit = submit.merge(special, left_index=True, right_index=True, how='left')
submit['special'] = submit['special'].fillna(submit['congestion']).round().astype(int)
submit = submit.drop(['congestion'], axis=1).rename(columns={'special':'congestion'})
submit['congestion'] = round(submit['congestion'])
submit.to_csv('./submission.csv')
```