#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_numeric_dtype
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("../data/diabetes_prediction_dataset.csv")

def standardize_column_names(col_lst):
    return [col.lower().replace(' ', '_') for col in col_lst]

# Standardizing column names
df.columns = standardize_column_names(df.columns)

# Seperate lists for numerical and categorical columns
num_col = [col for col in df.columns if is_numeric_dtype(df[col])]
cat_col = [col for col in df.columns if col not in num_col]

df = df.drop_duplicates()

def linear_model_accuracy(data, col_removed=None, C=1.0):
    
    df_copy = data.copy()
    
    df_full_train, df_test = train_test_split(df_copy, test_size=0.2, random_state=1)
    df_train, df_val = train_test_split(df_full_train, test_size=0.2, random_state=1)

    y_train = df_train['diabetes']
    y_val = df_val['diabetes']
    y_test = df_test['diabetes']
    y_full_train = df_full_train['diabetes']

    del df_train['diabetes']
    del df_val['diabetes']
    del df_test['diabetes']
    del df_full_train['diabetes']
    
    if col_removed!=None:
        del df_train[col_removed]
        del df_val[col_removed]
        del df_test[col_removed]
        del df_full_train[col_removed]

    df_train_emb = df_train.to_dict(orient="records")
    df_val_emb = df_val.to_dict(orient="records")

    df_full_train_emb = df_full_train.to_dict(orient="records")
    df_test_emb = df_test.to_dict(orient="records")

    dv = DictVectorizer(sparse=False)

    X_train = dv.fit_transform(df_train_emb)
    X_val = dv.transform(df_val_emb)

    X_full_train = dv.fit_transform(df_full_train_emb)
    X_test = dv.transform(df_test_emb)

    model = LogisticRegression(solver='liblinear', C=C, max_iter=1000, random_state=1)

    model.fit(X_train, y_train)
    pred_val = model.predict(X_val)

    lr_lst.append((col, C, roc_auc_score(y_val, pred_val)))

lr_lst = []

ind_cols = df.columns.tolist()
ind_cols.remove('diabetes')

for col in ind_cols+[None]:
    for c in [0.01, 0.1, 1, 10, 100]:
        linear_model_accuracy(data=df, col_removed=col, C = c)

# FIND THE MOST OPTIMAL VALUE OF C based on the ROC-AUC Score
col_to_remove, opt_c, auc_score = lr_lst[np.argmax(np.array(lr_lst)[:, 2])]
print("optimal C: %s" %opt_c)

df_copy = df.copy()

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.2, random_state=1)

y_train = df_train['diabetes']
y_val = df_val['diabetes']
y_test = df_test['diabetes']
y_full_train = df_full_train['diabetes']

del df_train['diabetes']
del df_val['diabetes']
del df_test['diabetes']
del df_full_train['diabetes']

if col_to_remove!=None:
    del df_train[col_to_remove]
    del df_val[col_to_remove]
    del df_test[col_to_remove]
    del df_full_train[col_to_remove]    

df_train_emb = df_train.to_dict(orient="records")
df_val_emb = df_val.to_dict(orient="records")

df_full_train_emb = df_full_train.to_dict(orient="records")
df_test_emb = df_test.to_dict(orient="records")

dv = DictVectorizer(sparse=False)

X_train = dv.fit_transform(df_train_emb)
X_val = dv.transform(df_val_emb)

X_full_train = dv.fit_transform(df_full_train_emb)
X_test = dv.transform(df_test_emb)

lr = LogisticRegression(solver='liblinear', C=opt_c, max_iter=1000, random_state=42)

lr.fit(X_train, y_train)
pred_val = lr.predict(X_val)

print(classification_report(y_val, pred_val))
print(roc_auc_score(y_val, pred_val))


# ### ENSEMBLED DECISION TREES Algorithms

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.2, random_state=1)

y_train = df_train['diabetes']
y_val = df_val['diabetes']
y_test = df_test['diabetes']
y_full_train = df_full_train['diabetes']

del df_train['diabetes']
del df_val['diabetes']
del df_test['diabetes']
del df_full_train['diabetes']

df_train_emb = df_train.to_dict(orient="records")
df_val_emb = df_val.to_dict(orient="records")

df_full_train_emb = df_full_train.to_dict(orient="records")
df_test_emb = df_test.to_dict(orient="records")

dv = DictVectorizer(sparse=False)

X_train = dv.fit_transform(df_train_emb)
X_val = dv.transform(df_val_emb)

X_full_train = dv.fit_transform(df_full_train_emb)
X_test = dv.transform(df_test_emb)

# #### RANDOM FOREST
rf_lst = []

for depth in [10, 15, 20, 25]:
    for est in range(10, 210, 10):
        for n_split in [2, 5, 10]:
            rf = RandomForestClassifier(n_estimators=est, max_depth=depth, min_samples_split=n_split, 
                                        random_state=1, n_jobs=-1)
            rf.fit(X_train, y_train)
            pred = rf.predict(X_val)
            eval_score = roc_auc_score(y_val, pred)
            rf_lst.append((depth, est, n_split, round(eval_score, 4)))

est, depth, n_split, score = rf_lst[np.argmax(np.array(rf_lst)[:, 3])]

rf = RandomForestClassifier(n_estimators=est, max_depth=depth, min_samples_split=n_split, 
                            random_state=1, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_val)


print(classification_report(y_val, rf_pred))
print(roc_auc_score(y_val, rf_pred))

######## XGBOOST
features = dv.feature_names_
dtrain = xgb.DMatrix(X_train, y_train, feature_names=features)
dval = xgb.DMatrix(X_val, y_val, feature_names=features)
dtest = xgb.DMatrix(X_test, y_test, feature_names=features)

watchlist = [(dtrain, 'train'), (dval, 'val')]

xgb_lst = []

for depth in [10, 15, 20, 25]:
    for eta in [0.01, 0.03, 0.1, 0.3, 1, 10]:
            xgb_params = {
                'eta': eta, 
                'max_depth': depth,
                'min_child_weight': 1,
                'early_stopping_rounds': 10,

                'objective': 'binary:logistic',
                'nthread': 8,

                'seed': 1,
                'verbosity': 1,
            }

            model = xgb.train(xgb_params, dtrain, num_boost_round=200,
                              verbose_eval=5,
                              evals=watchlist)

            y_pred = model.predict(dval)
            predictions = [round(value) for value in y_pred]
            eval_score = roc_auc_score(y_val, predictions)
            
            xgb_lst.append((depth, eta, round(eval_score, 4)))


depth, eta, score = xgb_lst[np.argmax(np.array(xgb_lst)[:, 2])]

xgb_params = {
    'eta': eta, 
    'max_depth': depth,
    'min_child_weight': 1,
    'early_stopping_rounds': 10,

    'objective': 'binary:logistic',
    'nthread': 8,

    'seed': 1,
    'verbosity': 1,
}

xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=200,
                  verbose_eval=5,
                  evals=watchlist)

y_pred = xgb_model.predict(dval)
xgb_pred = [round(value) for value in y_pred]
eval_score = roc_auc_score(y_val, xgb_pred)

print(classification_report(y_val, xgb_pred))
print(eval_score)

val_pred_df = pd.DataFrame({'LOGISTIC':pred_val, 'RF':rf_pred, 'XGB':xgb_pred})
val_pred_df['Ensembled'] = val_pred_df.mode(axis=1)

print(roc_auc_score(y_val, val_pred_df['Ensembled']))
print(classification_report(y_val, val_pred_df['Ensembled']))


####### TEST SET EVALUATION

# LR
model = LogisticRegression(solver='liblinear', C=opt_c, max_iter=1000, random_state=1)
model.fit(X_full_train, y_full_train)

lr_test_pred = model.predict(X_test)
print(classification_report(y_test, lr_test_pred))
print(roc_auc_score(y_test, lr_test_pred))

# RF
rf = RandomForestClassifier(n_estimators=est, max_depth=depth, min_samples_split=n_split, 
                            random_state=1, n_jobs=-1)
rf.fit(X_full_train, y_full_train)

rf_test_pred = rf.predict(X_test)
print(classification_report(y_test, rf_test_pred))
print(roc_auc_score(y_test, rf_test_pred))

# XGB
xgb_model = xgb.train(xgb_params, dfull_train, num_boost_round=200,
                  verbose_eval=5,
                  evals=watchlist)

y_pred = xgb_model.predict(dtest)
xgb_test_pred = [round(value) for value in y_pred]

print(classification_report(y_test, xgb_test_pred))
print(roc_auc_score(y_test, xgb_test_pred))


with open("../models/dv.bin", "wb") as file_in:
    pickle.dump(dv, file_in)
    
with open("../models/lr.bin", "wb") as file_in:
    pickle.dump(model, file_in)

with open("../models/rf.bin", "wb") as file_in:
    pickle.dump(rf, file_in)

with open("../models/xgb.bin", "wb") as file_in:
    pickle.dump(xgb_model, file_in)