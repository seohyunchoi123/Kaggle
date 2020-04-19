import pandas as pd
import numpy as np
import os

import lightgbm as lgb
from sklearn.metrics import mean_squared_error

# 파일 불러오기
if 'Mission_AIFrenz_Season1' not in os.getcwd():
    os.chdir(os.path.join(os.getcwd(), 'Mission_AIFrenz_Season1'))# 디렉토리 변경
data_path =  os.path.join(os.getcwd(), 'data')

train = pd.read_csv(os.path.join(data_path, 'train.csv'))
test = pd.read_csv(os.path.join(data_path, 'test.csv'))


# 데이터 설명
# -     대전지역에서 측정한 실내외 19곳의 센서데이터와, 주변 지역의 기상청 공공데이터를 semi-비식별화하여 제공합니다.
# -     센서는 온도를 측정하였습니다.
# -     모든 데이터는 시간 순으로 정렬 되어 있으며 10분 단위 데이터 입니다.
# -     예측 대상(target variable)은 Y18입니다.
#
#
# train.csv
# -     30일 간의 기상청 데이터 (X00~X39) 및 센서데이터 (Y00~Y17)
# -     이후 3일 간의 기상청 데이터 (X00~X39) 및 센서데이터 (Y18)
#
# test.csv
# -     train.csv 기간 이후 80일 간의 기상청 데이터 (X00~X39)


# 변수 설명
temperature_name = ["X00","X07","X28","X31","X32"] #기온
localpress_name  = ["X01","X06","X22","X27","X29"] #현지기압
speed_name       = ["X02","X03","X18","X24","X26"] #풍속
water_name       = ["X04","X10","X21","X36","X39"] #일일 누적강수량
press_name       = ["X05","X08","X09","X23","X33"] #해면기압
sun_name         = ["X11","X14","X16","X19","X34"] #일일 누적일사량
humidity_name    = ["X12","X20","X30","X37","X38"] #습도
direction_name   = ["X13","X15","X17","X25","X35"] #풍향

# Y18번의 데이터를 나머지들의 평균값으로 채운다. (GeonwooKim 님 코드)
t = train["Y18"].isna()
null_index = t[t==True].index
train.loc[null_index, "Y18"] = train.loc[null_index, "Y00":"Y17"].mean(axis=1)

# null값을 채워주고, 데이터의 구조를 살핀다. (뚱냥이 님 코드)
print('total number of sample in train :',train.shape[0])
print('total number of Y18 in train :',train.shape[0] - train['Y18'].isnull().sum())


# id 변수를 삼각함수를 이용해 시간 변수 추가 (26님 코드 -- 기상 캐스터 잔나)
def make_minute(df):
    minute = (df.id % 144).astype(int)
    hour = pd.Series((df.index % 144 / 6).astype(int))

    min_in_day = 24 * 6
    hour_in_day = 24

    minute_sin = np.sin(np.pi * minute / min_in_day)
    hour_sin = np.sin(np.pi * hour / hour_in_day)

    return minute_sin, hour_sin

#### 시간변수 추가
minute_sin, hour_sin = make_minute(train)
train['minute_sin'] = minute_sin
train['hour_sin'] = hour_sin

minute_sin_test, hour_sin_test = make_minute(test)
test['minute_sin'] = minute_sin_test
test['hour_sin'] = hour_sin_test


#################################
date = 0 # 날자 데이터 추가
for i in range(0, train.shape[0], 144):
    train.loc[i:(i+144),'datekey'] = date
    train.loc[i, 'date_start'] = 1
    train.loc[i+143, 'date_end'] = 1
    date += 1
train = train.fillna(0)

date = 0 # 날자 데이터 추가 - test
for i in range(0, test.shape[0], 144):
    test.loc[i:(i+144),'datekey'] = date
    test.loc[i, 'date_start'] = 1
    test.loc[i + 143, 'date_end'] = 1
    date += 1
test = test.fillna(0)




# data-set 구조 바꾸기 (뚱냥이 님 코드)
# panel data

# sensor list
sensor_list = list(train.columns[41:60]) # Y00 ~ Y17

# panel dataset
df = pd.melt(train,
        id_vars='id',
        value_vars=list(train.columns[41:60]), var_name='sensor')

df = pd.merge(df,train.drop(columns=sensor_list), on='id' ) # 합치기.
df = df.dropna() # 결측제거.
idx = df.value > 0
df = df.loc[idx]


# 전날 최고, 최저 기온 만들기
idx = df.datekey > 0

tt = df.loc[idx,].groupby(['datekey', 'sensor'])['value'].agg({ 'min_value':np.min, 'max_value':np.max})
tt = tt.reset_index()

df2 = pd.merge(df.loc[idx], tt, how = 'left', on =  ['datekey', 'sensor'])

# 이전 데이터가 없을 때는 0으로 채워 넣는다
df.loc[~idx, 'min_value'] = 0
df.loc[~idx, 'max_value'] = 0

# 두 데이터 프레임 합치기
df3 = pd.concat([df.loc[~idx], df2], axis = 0)

# 필요없는 컬럼 제거
df3 = df3.drop(['datekey', 'date_start', 'date_end'], axis=1)

####################
    # 모델링 데이터 준
X_train = df3.loc[:, df3.columns[3:]]
y_train = df3.loc[:, 'value']

X_test = test.iloc[:, 1:]

lgb_train = lgb.Dataset(X_train, label=y_train)

# custom metric -- 대회 목적에 맞는 방법으로 변경 (자카종신 님 코드)
def mse1(y_pred, dataset):
    y_true = dataset.get_label()

    diff = abs(y_true - y_pred)
    less_then_one = np.array([0 if x < 1 else 1 for x in diff])

    y_pred = less_then_one * y_pred
    y_true = less_then_one * y_true

    score = mean_squared_error(y_true, y_pred)

    return 'score', score, False

# GeonwooKim님 코드
lgb_param = {
    "objective":"regression",
    "learning_rate":0.01
}

print("cv start")
cv_result = lgb.cv(
    lgb_param,
    lgb_train,
    feval=mse1,
    num_boost_round=3000,
    nfold=5,
    early_stopping_rounds=10,
    stratified=False,
    verbose_eval=10,
    seed= 1
)
print('Current parameters:\n', lgb_param)
print('\nBest num_boost_round:', len(cv_result['l2-mean']))
print('Best CV score:', cv_result['l2-mean'][-1])

print("train start")
lgb_model = lgb.train(
    lgb_param,
    lgb_train,
    num_boost_round=len(cv_result["l2-mean"])
)

# 변수 중요도 확인
# lgb.plot_importance(lgb_model, max_num_features=30)



#######
    # test 제출 파일 만들기
# 제출파일 불러오기
submission = pd.read_csv(os.path.join(data_path, 'sample_submission.csv'))
# 일별 예측을 위해
submission2 = pd.merge(submission, test[['id', 'datekey']], how = 'left', on = 'id')


# 변수 추가
X_test['min_value'] = 0
X_test['max_value'] = 0

# test파일의 첫 날은 train의 마지막날과 이어진다.
X_test.loc[X_test.datekey == 0, 'min_value'] = df.loc[(df.sensor == 'Y18') & (df.datekey == 32), 'value'].min()
X_test.loc[X_test.datekey == 0, 'max_value'] = df.loc[(df.sensor == 'Y18') & (df.datekey == 32), 'value'].max()

# 필요 없는 변수 제거
X_test = X_test.drop(['date_start', 'date_end'], axis=1)

# 일별 예측을 시작한다.
    # 첫날은 반복문에 들어가지 않고 한다. -- 코딩 귀찮
X_test_hat = X_test[X_test.datekey == 0].drop('datekey', axis = 1)
pred = lgb_model.predict(X_test_hat)
submission2.loc[submission2.datekey == 0, 'Y18'] = pred

# 2일차 부터 반복문으로 예측
X_test.loc[X_test.datekey == 1, 'min_value'] = pred.min()
X_test.loc[X_test.datekey == 1, 'max_value'] = pred.max()

for day in range(1, 80):
    X_test_hat = X_test[X_test.datekey == day].drop('datekey', axis = 1)
    pred = lgb_model.predict(X_test_hat)
    submission2.loc[submission2.datekey == day, 'Y18'] = pred

# 제출전, datekey 변수 제거
submission2 = submission2.drop('datekey', axis = 1)
submission2.to_csv('submit/result2.csv',index = False)