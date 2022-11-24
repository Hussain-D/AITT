"""SOURCE CODE"""

# IMPORTING REQUIRED MODULES

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import collections
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix , precision_score, recall_score, \
f1_score, roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# DATA

df = pd.read_csv('/content/drive/MyDrive/AITTA PBL/Dataset/creditcard.csv')
df.head()

#TRAIN - TEST SPLIT

df_train, df_test = train_test_split(df, test_size=0.2,random_state=123,stratify=df["Class"])
df_train, df_val = train_test_split(df_train, 
test_size=0.25,random_state=123,stratify=df_train["Class"])

#SCALING the columns that are left to scale (AMOUNT AND TIME)

from sklearn.preprocessing import StandardScaler, RobustScaler

# RobustScaler is less prone to outliers.

std_scaler = StandardScaler()
rob_scaler = RobustScaler()
df_train['scaled_amount'] = rob_scaler.fit_transform(df_train['Amount'].values.reshape(-
1,1))
df_train['scaled_time'] = rob_scaler.fit_transform(df_train['Time'].values.reshape(-1,1))
df_train.drop(['Time','Amount'], axis=1, inplace=True)
X_train = df_train.drop(["Class"], axis = 1)
y_train = df_train["Class"]
df_val['scaled_amount'] = rob_scaler.fit_transform(df_val['Amount'].values.reshape(-1,1))
df_val['scaled_time'] = rob_scaler.fit_transform(df_val['Time'].values.reshape(-1,1))
df_val.drop(['Time','Amount'], axis=1, inplace=True)
X_val = df_val.drop(["Class"], axis = 1)
y_val = df_val["Class"]

#TRAIN

dtc_cfl = DecisionTreeClassifier(random_state=1,max_depth=2)
dtc_cfl.fit(X_train, y_train)
y_predict = dtc_cfl.predict(X_train)
# evaluate the model
print(classification_report(y_train, y_predict))
print(confusion_matrix(y_train, y_predict))

#VAL

dtc_cfl = DecisionTreeClassifier(random_state=1,max_depth=2)
dtc_cfl.fit(X_train, y_train)
y_predict = dtc_cfl.predict(X_val)

# evaluate the model

print(classification_report(y_val, y_predict))
print(confusion_matrix(y_val, y_predict))

#Accuracy

print(“Accuracy: “, accuracy_score(y_val,y_predict))
