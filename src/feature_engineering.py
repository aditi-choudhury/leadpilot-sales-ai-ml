from sklearn.base import BaseEstimator, TransformerMixin
class EngagementFeatures(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):return self
    def transform(self,X):
        out=X.copy()
        out['engagement_score']=out['website_visits'].fillna(0)+2*out['sales_calls'].fillna(0)+8*out['email_engagement'].fillna(0)
        out['recency_score']=1/(1+out['days_since_last_interaction'].fillna(30))
        out['customer_value']=out['previous_revenue'].fillna(0)/(1+out['previous_purchases'].fillna(0))
        return out
