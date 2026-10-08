"""Train, compare and persist an end-to-end pipeline."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score
from sklearn.metrics import roc_auc_score,average_precision_score,accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,brier_score_loss
from config.settings import DATA,MODEL,METRICS,SEED
from src.generate_data import generate
from src.data_preprocessing import clean,FEATURES,CATEGORICAL,NUMERIC
from src.feature_engineering import EngagementFeatures

def build(estimator):
    nums=NUMERIC+['engagement_score','recency_score','customer_value']
    prep=ColumnTransformer([('cat',Pipeline([('fill',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),CATEGORICAL),('num',Pipeline([('fill',SimpleImputer(strategy='median')),('scale',StandardScaler())]),nums)])
    return Pipeline([('features',EngagementFeatures()),('preprocess',prep),('classifier',estimator)])

def train():
    if not DATA.exists():
        DATA.parent.mkdir(parents=True,exist_ok=True)
        generate().to_csv(DATA,index=False)
    df=clean(pd.read_csv(DATA),training=True)
    X=df[FEATURES];y=df['converted']
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,random_state=SEED,stratify=y)
    candidates={'Logistic Regression':LogisticRegression(max_iter=1500,random_state=SEED),'Random Forest':RandomForestClassifier(n_estimators=160,min_samples_leaf=8,n_jobs=-1,random_state=SEED),'Gradient Boosting':HistGradientBoostingClassifier(max_iter=120,random_state=SEED)}
    results={};cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=SEED)
    for name,estimator in candidates.items():
        pipe=build(estimator)
        scores=cross_val_score(pipe,X_train,y_train,cv=cv,scoring='roc_auc',n_jobs=1)
        results[name]=round(float(scores.mean()),4)
    # Select by training-only CV; small tolerance favors interpretable logistic regression.
    best=max(results,key=results.get)
    if results[best]-results['Logistic Regression']<=.015:best='Logistic Regression'
    model=build(candidates[best]).fit(X_train,y_train)
    proba=model.predict_proba(X_test)[:,1];pred=(proba>=.5).astype(int)
    metrics={'dataset':'Synthetic illustrative B2B leads','rows':len(df),'train_rows':len(X_train),'test_rows':len(X_test),'conversion_rate':round(float(y.mean()),4),'cv_roc_auc':results,'selected_model':best,'test':{'roc_auc':round(float(roc_auc_score(y_test,proba)),4),'average_precision':round(float(average_precision_score(y_test,proba)),4),'accuracy':round(float(accuracy_score(y_test,pred)),4),'precision':round(float(precision_score(y_test,pred,zero_division=0)),4),'recall':round(float(recall_score(y_test,pred,zero_division=0)),4),'f1':round(float(f1_score(y_test,pred,zero_division=0)),4),'brier':round(float(brier_score_loss(y_test,proba)),4),'confusion_matrix':confusion_matrix(y_test,pred).tolist()}}
    MODEL.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,MODEL);METRICS.write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2));return metrics
if __name__=='__main__':train()
