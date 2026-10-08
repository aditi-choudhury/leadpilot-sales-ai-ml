"""Reproducible illustrative B2B lead data; not actual customers."""
import numpy as np
import pandas as pd
from config.settings import DATA, SEED

def generate(n=3500, seed=SEED):
    r=np.random.default_rng(seed)
    source=r.choice(['Referral','Organic','Paid search','Event','Outbound'],n,p=[.15,.26,.23,.14,.22])
    industry=r.choice(['SaaS','Manufacturing','Retail','Finance','Healthcare'],n)
    segment=r.choice(['SMB','Mid-market','Enterprise'],n,p=[.5,.32,.18])
    visits=r.poisson(5,n)+r.binomial(1,.25,n)*r.poisson(8,n)
    calls=r.poisson(2,n)
    email=np.clip(r.beta(2,4,n)+.025*visits,0,1)
    demo=r.binomial(1,np.clip(.08+.025*visits+.12*(source=='Referral'),0,.85))
    recency=r.integers(0,90,n)
    purchases=r.poisson(np.where(segment=='Enterprise',.7,.3))
    revenue=np.round(purchases*r.lognormal(8,1,n),2)
    campaign=r.binomial(1,np.clip(.1+.4*email,0,.9))
    age=r.integers(1,181,n)
    size=np.where(segment=='SMB',r.integers(5,100,n),np.where(segment=='Mid-market',r.integers(100,1000,n),r.integers(1000,10000,n)))
    score=-2.65+.09*visits+.25*calls+1.0*demo+1.15*email+.7*campaign+.6*(source=='Referral')+.4*(industry=='SaaS')+.35*(segment=='Enterprise')+.3*(purchases>0)-.016*recency-.003*age
    probability=1/(1+np.exp(-score))
    df=pd.DataFrame({'lead_id':[f'L{i:05d}' for i in range(1,n+1)],'lead_source':source,'industry':industry,'customer_segment':segment,'company_size':size,'geography':r.choice(['North America','Europe','APAC','LATAM'],n),'website_visits':visits,'sales_calls':calls,'email_engagement':email.round(3),'demo_requested':demo,'days_since_last_interaction':recency,'previous_purchases':purchases,'previous_revenue':revenue,'campaign_response':campaign,'lead_age_days':age,'product_interest':r.choice(['Core','Analytics','Enterprise Suite'],n),'converted':r.binomial(1,probability)})
    return df

if __name__=='__main__':
    DATA.parent.mkdir(parents=True,exist_ok=True)
    generate().to_csv(DATA,index=False)
    print(f'Generated {DATA}')
