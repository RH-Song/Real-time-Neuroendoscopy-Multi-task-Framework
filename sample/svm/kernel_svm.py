import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

tool_data = pd.read_csv('data/tip_data.csv')
#bankdata = pd.read_csv('/home/raphael/Desktop/data/surgery/test_data/bill_authentication.csv')
print(tool_data.shape)
print(tool_data.head())

X = tool_data.drop('label', axis=1)
y = tool_data['label']
#std = StandardScaler()
#std = MinMaxScaler(feature_range=(-1, 1))
#std = MinMaxScaler(feature_range=(0, 1))
#X_std = std.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

TRAIN = True

parameters = {'gamma': [50, 100, 150, 200, 500, 1000], 'C':[50, 100, 150, 200]}
#n_jobs =-1 #使用全部CPU并行多线程搜索
gs = GridSearchCV(SVC(), parameters, refit = True, cv = 5, verbose = 1, n_jobs = -1)

if TRAIN:
    gs.fit(X_train,y_train) #Run fit with all sets of parameters.
    print('最优参数: ',gs.best_params_)
    print('最佳性能: ', gs.best_score_)
    joblib.dump(gs, 'save/gs.m')
else:
    svclassifier = joblib.load("save/gs.m")


# svclassifier = SVC(kernel='poly', degree=1)
# #svclassifier = SVC(kernel='sigmoid')
# svclassifier.fit(X_train, y_train)
y_pred = gs.predict(X_test)
#
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#最优参数:  {'C': 10, 'gamma': 1}
#最佳性能:  0.9420489172936386

# with data standard scaler
#最优参数:  {'C': 10, 'gamma': 1}
#最佳性能:  0.9532130625893267

# MM(-1,1)
# 最优参数:  {'C': 10, 'gamma': 10}
# 最佳性能:  0.9534483920986988
# [[4464  275]
#  [  79 2599]]
#               precision    recall  f1-score   support
#
#            0       0.98      0.94      0.96      4739
#            1       0.90      0.97      0.94      2678
#
#     accuracy                           0.95      7417
#    macro avg       0.94      0.96      0.95      7417
# weighted avg       0.95      0.95      0.95      7417

# MM(0, -1)
# 最优参数:  {'C': 100, 'gamma': 10}
# 最佳性能:  0.9522685903054001
# [[4422  266]
#  [  81 2648]]
#               precision    recall  f1-score   support
#
#            0       0.98      0.94      0.96      4688
#            1       0.91      0.97      0.94      2729
#
#     accuracy                           0.95      7417
#    macro avg       0.95      0.96      0.95      7417
# weighted avg       0.96      0.95      0.95      7417

# 最优参数:  {'C': 100, 'gamma': 100}
# 最佳性能:  0.9532124317400391

#  [  77 2648]]
#               precision    recall  f1-score   support
#
#            0       0.98      0.94      0.96      4692
#            1       0.91      0.97      0.94      2725
#
#     accuracy                           0.95      7417
#    macro avg       0.94      0.96      0.95      7417
# weighted avg       0.95      0.95      0.95      7417

# 最优参数:  {'C': 150, 'gamma': 100}
# 最佳性能:  0.9533135576080362
# [[4491  275]
#  [  68 2583]]
#               precision    recall  f1-score   support
#
#            0       0.99      0.94      0.96      4766
#            1       0.90      0.97      0.94      2651
#
#     accuracy                           0.95      7417
#    macro avg       0.94      0.96      0.95      7417
# weighted avg       0.96      0.95      0.95      7417