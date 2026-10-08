import joblib
import numpy as np
import pandas as pd
from config.settings import MODEL,HIGH,MEDIUM
from src.data_preprocessing import clean,FEATURES

def classify(prob):return 'High Potential' if prob>=HIGH else 'Medium Potential' if prob>=MEDIUM else 'Low Potential'
def recommend(prob,row):
    if prob>=HIGH:
        if float(row.get('days_since_last_interaction',0) or 0)>21:return 'Re-engage urgently: call and send a tailored recap.'
        return 'Call within 24 hours; offer a tailored demo.'
    if prob>=MEDIUM:return 'Enroll in a targeted nurture sequence; review in 7 days.'
    return 'Use low-cost automated follow-up; reassess after new engagement.'
def score(df,model=None):
    model=model or load_model()
    data=clean(df)
    probabilities=model.predict_proba(data[FEATURES])[:,1]
    result=data.copy();result['conversion_probability']=probabilities
    result['priority']=list(map(classify,probabilities))
    result['recommended_action']=[recommend(p,row) for p,(_,row) in zip(probabilities,data.iterrows())]
    return result.sort_values('conversion_probability',ascending=False).reset_index(drop=True)
def load_model():
    if not MODEL.exists():raise FileNotFoundError('Model missing. Run: python -m src.train_model')
    return joblib.load(MODEL) # Load only trusted local model files.
def explain_lead(model,row):
    """Local perturbation explanation: probability change when a feature is replaced by baseline."""
    data=clean(pd.DataFrame([row]));base=float(model.predict_proba(data[FEATURES])[0,1]);imp=[]
    defaults={'website_visits':5,'sales_calls':2,'email_engagement':.3,'demo_requested':0,'days_since_last_interaction':30,'previous_purchases':0,'previous_revenue':0,'campaign_response':0,'lead_age_days':90,'company_size':100,'lead_source':'Outbound','industry':'Retail','customer_segment':'SMB','geography':'APAC','product_interest':'Core'}
    for field,value in defaults.items():
        alt=data[FEATURES].copy();alt[field]=value
        change=base-float(model.predict_proba(alt)[0,1]);imp.append({'factor':field.replace('_',' ').title(),'impact':change})
    return pd.DataFrame(imp).assign(magnitude=lambda d:d.impact.abs()).sort_values('magnitude',ascending=False).head(6)
