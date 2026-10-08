from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import pandas as pd
import plotly.express as px
import streamlit as st
from config.settings import DATA,METRICS
from src.data_preprocessing import FEATURES,CATEGORICAL,NUMERIC
from src.predict import load_model,score,explain_lead
from src.database import store_predictions
st.set_page_config(page_title='LeadPilot | Sales Intelligence',page_icon='📈',layout='wide')
st.markdown("<style>.block-container{padding-top:2rem;max-width:1250px}</style>",unsafe_allow_html=True)
st.title('LeadPilot')
st.caption('Sales intelligence • Lead scoring and conversion prioritisation')
st.info('Demo environment: all sample leads are synthetic. Predictions are estimates, not guaranteed outcomes.')
@st.cache_resource
def model():
    return load_model()

try:
    m = model()
except (FileNotFoundError, ValueError, ImportError) as exc:
    st.error(f"Unable to load the trained model: {exc}")
    st.stop()

st.markdown("<style>.block-container{padding-top:2rem;max-width:1250px}h1{letter-spacing:-1.5px}div[data-testid='stMetric']{background:#f4f7fb;padding:18px;border-radius:12px;border:1px solid #e1e7ef}</style>", unsafe_allow_html=True)
page = st.sidebar.radio('Workspace', ['Overview', 'Score a lead', 'Prioritise CSV', 'Model insights'])
if page=='Overview':
    try:
        df=pd.read_csv(DATA)
        ranked=score(df,m)
        a,b,c,d,e=st.columns(5)
        a.metric('Total leads',f'{len(ranked):,}')
        b.metric('Observed conversion',f'{df.converted.mean():.1%}' if 'converted' in df else 'N/A')
        c.metric('High-potential',f'{(ranked.priority=="High Potential").sum():,}')
        d.metric('Avg. predicted',f'{ranked.conversion_probability.mean():.1%}')
        e.metric('Expected conversions',f'{ranked.conversion_probability.sum():.0f}')
        left,right=st.columns(2)
        with left:
            st.plotly_chart(px.histogram(ranked,x='conversion_probability',nbins=25,title='Predicted conversion distribution'),use_container_width=True)
            st.plotly_chart(px.bar(ranked.groupby('lead_source',as_index=False).conversion_probability.mean(),x='lead_source',y='conversion_probability',title='Mean predicted probability by source'),use_container_width=True)
        with right:
            st.plotly_chart(px.pie(ranked,names='priority',title='Lead potential mix',hole=.55),use_container_width=True)
            st.plotly_chart(px.bar(ranked.groupby('industry',as_index=False).conversion_probability.mean(),x='industry',y='conversion_probability',title='Mean predicted probability by industry'),use_container_width=True)
        st.subheader('Highest-priority opportunities')
        st.dataframe(ranked[['lead_id','lead_source','industry','conversion_probability','priority','recommended_action']].head(20),use_container_width=True,hide_index=True)
    except (OSError,ValueError,KeyError) as exc:st.error(f'Could not load demo leads: {exc}')
elif page=='Score a lead':
    st.subheader('New opportunity assessment')
    defaults={'lead_source':'Referral','industry':'SaaS','customer_segment':'Mid-market','geography':'APAC','product_interest':'Analytics','company_size':250,'website_visits':12,'sales_calls':3,'email_engagement':.6,'demo_requested':1,'days_since_last_interaction':5,'previous_purchases':0,'previous_revenue':0,'campaign_response':1,'lead_age_days':20}
    with st.form('newlead'):
        cols=st.columns(3);values={}
        choices={'lead_source':['Referral','Organic','Paid search','Event','Outbound'],'industry':['SaaS','Manufacturing','Retail','Finance','Healthcare'],'customer_segment':['SMB','Mid-market','Enterprise'],'geography':['North America','Europe','APAC','LATAM'],'product_interest':['Core','Analytics','Enterprise Suite']}
        for i,col in enumerate(FEATURES):
            with cols[i%3]:
                if col in CATEGORICAL:values[col]=st.selectbox(col.replace('_',' ').title(),choices[col],index=choices[col].index(defaults[col]))
                elif col in ['demo_requested','campaign_response']:values[col]=int(st.checkbox(col.replace('_',' ').title(),value=bool(defaults[col])))
                elif col=='email_engagement':values[col]=st.slider('Email engagement (0-1)',0.,1.,float(defaults[col]),.05)
                else:values[col]=st.number_input(col.replace('_',' ').title(),min_value=0.0,value=float(defaults[col]),step=1.0)
        submit=st.form_submit_button('Predict Lead Potential',type='primary')
    if submit:
        try:
            result=score(pd.DataFrame([values]),m).iloc[0]
            a,b=st.columns(2);a.metric('Conversion probability',f'{result.conversion_probability:.1%}');b.metric('Lead potential',result.priority)
            st.success(result.recommended_action)
            st.caption('Factor impacts are local what-if changes against fixed reference values; they are not causal explanations.')
            st.plotly_chart(px.bar(explain_lead(m,values),x='impact',y='factor',orientation='h',title='Main prediction drivers (probability-point difference)'),use_container_width=True)
        except (ValueError,KeyError) as exc:st.error(f'Unable to score lead: {exc}')
elif page=='Prioritise CSV':
    st.subheader('Bulk lead prioritisation')
    st.write('Upload a CSV with the same feature columns as the demo dataset. The conversion label is optional.')
    st.download_button('Download example CSV template',pd.read_csv(DATA).drop(columns=['converted'],errors='ignore').head(5).to_csv(index=False),'lead_template.csv','text/csv')
    upload=st.file_uploader('Upload lead CSV',type='csv')
    if upload:
        try:
            df=pd.read_csv(upload)
            if len(df)>50000:raise ValueError('Upload limit is 50,000 rows.')
            ranked=score(df,m)
            st.metric('Scored leads',len(ranked))
            st.dataframe(ranked,hide_index=True,use_container_width=True)
            st.download_button('Download ranked leads',ranked.to_csv(index=False),'ranked_leads.csv','text/csv',type='primary')
            if st.button('Save predictions to local SQLite'):
                store_predictions(ranked);st.success('Predictions saved locally.')
        except (ValueError,KeyError,pd.errors.ParserError,UnicodeError,OSError) as exc:st.error(f'Unable to process CSV: {exc}')
else:
    st.subheader('Model governance & evaluation')
    if METRICS.exists():
        info=json.loads(METRICS.read_text());st.write('**Selected model:**',info['selected_model']);st.write('**Dataset:**',info['dataset']);st.write('**Training-only CV ROC-AUC:**',info['cv_roc_auc']);st.write('**Held-out test metrics:**');st.json(info['test'])
        st.caption('ROC-AUC measures ranking quality; average precision evaluates precision across recall levels; Brier score measures probability error. Threshold metrics use 0.50.')
    else:st.warning('Metrics missing; run training.')
    st.markdown('**Model selection:** Highest training cross-validation ROC-AUC, with a 0.015 tolerance favoring interpretable logistic regression. Test data is not used for selection.')
    st.markdown('**Recommendation engine:** Independent rules: high ≥70%, medium ≥40%; high + inactive over 21 days triggers re-engagement. Edit config/settings.py and src/predict.py.')
    st.warning('Synthetic relationships are illustrative; retrain, calibrate, and validate on real, consented CRM outcomes before operational use.')
