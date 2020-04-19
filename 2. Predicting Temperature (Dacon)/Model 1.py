import numpy as np
import seaborn as sns
import lightgbm as lgb
import xgboost as xgb

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.ensemble import  GradientBoostingRegressor
from catboost import CatBoostRegressor, CatBoostClassifier

sns.set_style('darkgrid')
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

train = pd.read_csv("/kaggle/input/dacon/train.csv")
test = pd.read_csv("/kaggle/input/dacon/test.csv")
print(train.shape, test.shape)

is_test = False

if is_test:
    train=train.loc[:80]
    test=test.loc[:80]

# Y18 FIlling in
# best_weight = [1/6]*6
# best_weight = [-0.18333333333333338, 0.7566666666666672, 0.3366666666666668, 0.04666666666666665, -0.013333333333333358, 0.05666666666666665] #EN
best_weight = [0.18666666666666665, 0.2766666666666667, 0.08666666666666659, -0.08333333333333343, 0.22666666666666668, 0.30666666666666675] # LGBM

indexes = [1,2,13,14,15,16]
train.loc[:4319, 'Y18'] = 0

for j in range(len(indexes)):
    idx = str(indexes[j])
    if len(idx) ==1:
        idx = '0' + idx
    col = 'Y' + idx
    train.loc[:4319, 'Y18'] += best_weight[j] * train.loc[:4319, col]

cols_igr = ['id'] + [col for col in train.columns if col.startswith('Y')]

# is_밤
for data in [train,test]:
    data['time'] = data['id']%144
    data['is_night'] = data['time'].apply(lambda x: 1 if x>=60 and x<=110 else 0)
cols_igr.append('time')

# lag 함수 만들기 > Y18, 기온, 일사량, 습도 등 각각 실험해보기
def making_lag(data, n_lags, vars):
    for n_lag in n_lags:
        for var in vars:
            value = data[var]
            value = value[:-n_lag]
            data[str(var)+'_Lag_' +str(n_lag)] = [-1]*n_lag + value.tolist()
    return data

data = pd.concat([train,test], axis=0)
data = making_lag(data,np.arange(3,43,1),['X02', 'X03', 'X18', 'X24', 'X26'])
data = making_lag(data,np.arange(3,60,1),['X11', 'X34'])
train = data[data['id']<len(train)]
test = data[data['id']>=len(train)]

# sin, cos, tan  변수 만들기
for data in [train,test]:
    for time in range(1,41):
        data['Sin_{}'.format(time)] = np.sin(data['time']*time)
        data['Cos_{}'.format(time)] = np.cos(data['time']*time)
        data['Tan_{}'.format(time)] = np.tan(data['time']*time)
    for time in range(2,41):
        data['Sin_/{}'.format(time)] = np.sin(data['time']/time)
        data['Cos_/{}'.format(time)] = np.cos(data['time']/time)
        data['Tan_/{}'.format(time)] = np.tan(data['time']/time)
#         data['Sin_Cos_'.format(time)] = np.sin(data['time']*time) * np.cos(data['time']*time)

# 기여도0 변수 제거
for data in [train,test]:
    data.drop(['X14', 'X16', 'X19'], axis=1, inplace=True)

# 통계량
for data in [train,test]:
    data['day'] = data['id']//144
    for col in ['X00', 'X07', 'X31', 'X32', 'X02', 'X03', 'X18', 'X24', 'X26', 'X11', 'X34']:
        for method in ['mean', 'std', 'min', 'max']:
            tmp = data.groupby('day')[col].agg(method)
            dic = dict(zip(tmp.index, tmp))
            data['{}_day_{}'.format(col,method)] = data['day'].map(dic)


def mse_AIFrenz(y_true, y_pred):
    diff = abs(y_true - y_pred)
    less_then_one = np.where(diff < 1, 0, diff)
    # multi-column일 경우에도 계산 할 수 있도록 np.average를 한번 더 씌움
    score = np.average(np.average(np.square(less_then_one), axis=0))
    return -score

my_scoring = make_scorer(mse_AIFrenz)
def best_param(model, grid_param, n_fold, train_X, train_y):
    grid_search = GridSearchCV(estimator=model, param_grid=grid_param,
                               scoring=my_scoring, cv=n_fold, refit=True)
    grid_search.fit(X=train_X, y=train_y)
    tmp = grid_search.cv_results_

    return sorted(list(zip(-tmp['mean_test_score'], tmp['params'])), key=lambda x: x[0]), grid_search
# Scoring Options:
# ['explained_variance', 'r2', 'max_error', 'neg_median_absolute_error', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_mean_squared_log_error',
#  'accuracy', 'roc_auc', 'balanced_accuracy', 'average_precision', 'neg_log_loss', 'brier_score_loss', 'adjusted_rand_score', 'homogeneity_score',
#  'completeness_score', 'v_measure_score', 'mutual_info_score', 'adjusted_mutual_info_score', 'normalized_mutual_info_score', 'fowlkes_mallows_score',
#  'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'recall', 'recall_macro', 'recall_micro', 'recall_samples',
#  'recall_weighted', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'jaccard', 'jaccard_macro', 'jaccard_micro', 'jaccard_samples', 'jaccard_weighted']

X = train.columns[~pd.Series(train.columns).isin(cols_igr)]
y = 'Y18'

print(X)
print(train.shape, test.shape)
print(train.isnull().sum().sum(), test.isnull().sum().sum())
print('# of NA in Test Data should be {}'.format(19*len(test)))

# ElasticNet
EN = make_pipeline(RobustScaler(), ElasticNet(alpha=0.01, l1_ratio=0.9, max_iter=1e+3))
grid_param = [{'elasticnet__alpha': np.arange(0.01, 0.07, 0.01), 'elasticnet__l1_ratio': np.arange(0.5, 1, 0.1)}]
param_result, grid_model = best_param(EN, grid_param, 4, train[X], train[y])
for t in param_result:
    print(t)
EN = grid_model.best_estimator_

#lasso
lasso = make_pipeline(StandardScaler(), Lasso(alpha= 0.09099999999999998))
grid_param=[{'lasso__alpha':np.arange(0,1,0.05)}]
param_result, grid_model = best_param(lasso, grid_param, 4, train[X], train[y])
for t in param_result:
    print(t)
lasso = grid_model.best_estimator_

# KRR
KRR = KernelRidge(alpha=100, kernel='polynomial', degree=2, coef0=0) # kernel: 'linear', 'laplacian', 'rbf'
# grid_param=[{'alpha':[0.1,0,1,10,100], 'coef0':[-20,-5,0,5]}]
grid_param = {}
param_result, grid_model = best_param(KRR, grid_param, 4, train[X], train[y])
for t in param_result:
    print(t)
KRR = grid_model.best_estimator_

# lgb
lgb_model = lgb.LGBMRegressor( n_estimators=2048,
                               num_leaves=16,
                               bagging_fraction = 0.9,
                               feature_fraction = 0.3,
                               feature_fraction_seed=9,
                               bagging_seed=9,
                               learning_rate=0.05,
                               bagging_freq = 20,
                               max_bin = 256,
                               min_data_in_leaf = 2,
                               min_child_samples=20,
                               min_sum_hessian_in_leaf = 1,
                               n_jobs = -1,
                               random_state=5)

# xgb
xgb_model = xgb.XGBRegressor(colsample_bytree=0.4,
                             gamma=0.0468,
                             learning_rate=0.05,
                             max_depth=10,
                             min_child_weight=1.3,
                             n_estimators=2048,
                             reg_alpha=0.3,
                             reg_lambda=0.4,
                             subsample=0.8,
                             silent=1,
                             random_state =7,
                             nthread = -1)

# gb
GBoost = GradientBoostingRegressor(n_estimators=2048,
                                   learning_rate=0.05,
                                   max_depth=8,
                                   max_features='sqrt',
                                   min_samples_leaf=30,
                                   min_samples_split=10,
                                   loss='huber',
                                   subsample = 0.4,
                                   random_state =5)

# catboost
cat_model = CatBoostRegressor(loss_function = 'RMSE',
                               eval_metric = 'RMSE',
                               custom_metric = ['RMSE'],
                               logging_level = 'Silent',
                               early_stopping_rounds = 100,
                               n_estimators = 2048,
                               learning_rate = 0.05,
                               depth = 10,
                               subsample = 0.8,
                               bootstrap_type = 'Bernoulli',
                               max_bin = 256,
                              colsample_bylevel=0.4,
                               random_seed = 42)


class Meta_Regressor(BaseEstimator):
    def __init__(self, base_models, meta_models):
        self.base_models = base_models  # self.A = B 에서 A와 B가 이름이 같아야한다.. 뭐지
        self.meta_models = meta_models

    def fit(self, X, y):
        self.base_models_ = [[] for _ in self.base_models]
        self.meta_models_ = clone(self.meta_models)

        Kf = KFold(n_splits=5, shuffle=True, random_state=5)
        out_fold_pred = np.zeros((len(X), len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_idx, val_idx in Kf.split(X):
                model = clone(self.base_models[i])
                model.fit(X[train_idx], y[train_idx])
                pred = model.predict(X[val_idx])
                out_fold_pred[val_idx, i] = pred
                self.base_models_[i].append(model)

        self.meta_models_.fit(X=out_fold_pred, y=y)

    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in sub_models]).mean(axis=1)
            for sub_models in self.base_models_])
        scores = self.meta_models_.predict(meta_features)
        return scores

class Weighted_Ensemble(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        self.models_ = [clone(model) for model in self.models]
        for model in self.models_:
            model.fit(X, y)

    def predict(self, x):
        results = np.zeros(len(x))
        scores = [model.predict(x) for model in self.models_]
        weights = [1 / 8] * 8
        for i, model in enumerate(scores):
            results += scores[i] * weights[i]
        return results

# Meta Ensemble Training & Opitmizing ensemble weights
meta_regressor = Meta_Regressor(base_models= [EN, lasso, KRR, lgb_model, xgb_model, GBoost, cat_model], meta_models = EN)
weighted_ensemble = Weighted_Ensemble(models= [EN, lasso, KRR, lgb_model, xgb_model, GBoost, cat_model, meta_regressor])

weighted_ensemble = grid_model.best_estimator_
score = weighted_ensemble.predict(test[X].values)

# submisison
submission = pd.DataFrame({
    'id': list(range(4752, 16272)),
    'Y18': score
})
submission = submission[['id', 'Y18']]
submission.to_csv('result1.csv', index=False)