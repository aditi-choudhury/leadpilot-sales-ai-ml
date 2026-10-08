import pandas as pd
TARGET='converted'
CATEGORICAL=['lead_source','industry','customer_segment','geography','product_interest']
NUMERIC=['company_size','website_visits','sales_calls','email_engagement','demo_requested','days_since_last_interaction','previous_purchases','previous_revenue','campaign_response','lead_age_days']
FEATURES=CATEGORICAL+NUMERIC

def clean(df:pd.DataFrame, training=False)->pd.DataFrame:
    if df is None or df.empty: raise ValueError('The uploaded dataset is empty.')
    missing=set(FEATURES)-set(df.columns)
    if missing: raise ValueError('Missing required columns: '+', '.join(sorted(missing)))
    out=df.copy().drop_duplicates(subset=['lead_id'] if 'lead_id' in df else None)
    for col in CATEGORICAL:
        out[col]=out[col].astype('string').str.strip().replace('',pd.NA).fillna('Unknown').astype(str)
    for col in NUMERIC:
        out[col]=pd.to_numeric(out[col],errors='coerce')
        out[col]=out[col].clip(lower=0)
    out['email_engagement']=out['email_engagement'].clip(upper=1)
    for col in ['demo_requested','campaign_response']:
        out[col]=out[col].clip(upper=1)
    if training:
        if TARGET not in out: raise ValueError('Training data must contain converted.')
        out[TARGET]=pd.to_numeric(out[TARGET],errors='coerce')
        out=out[out[TARGET].isin([0,1])].copy()
        if out.empty or out[TARGET].nunique()<2: raise ValueError('Training requires both conversion classes.')
        out[TARGET]=out[TARGET].astype(int)
    return out
