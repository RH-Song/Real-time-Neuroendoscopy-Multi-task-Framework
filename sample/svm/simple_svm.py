import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib

tool_data = pd.read_csv('data/tip_data.csv')
#bankdata = pd.read_csv('/home/raphael/Desktop/data/surgery/test_data/bill_authentication.csv')
print(tool_data.shape)
print(tool_data.head())

X = tool_data.drop('label', axis=1)
y = tool_data['label']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
print(X_test)

from sklearn.svm import SVC

TRAIN = True

if TRAIN:
   svclassifier = SVC(kernel='linear')
   svclassifier.fit(X_train, y_train)
   joblib.dump(svclassifier, 'save/linear_svc.m')
else:
   svclassifier = joblib.load("save/linear_svc.m")

y_pred = svclassifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
