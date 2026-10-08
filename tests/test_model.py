import numpy as np
import pandas as pd
import pytest
from src.generate_data import generate
from src.data_preprocessing import clean,FEATURES
from src.feature_engineering import EngagementFeatures
from src.predict import classify,score
from src.train_model import build
from sklearn.linear_model import LogisticRegression

def test_generation_is_reproducible():
    assert generate(30).equals(generate(30))
def test_clean_and_validation():
    df=generate(20);assert len(clean(df,training=True))==20
    with pytest.raises(ValueError,match='Missing required'):clean(df.drop(columns=['industry']))
def test_feature_engineering():
    d=EngagementFeatures().fit_transform(clean(generate(5)))
    assert (d.engagement_score>=0).all()
def test_classification():
    assert [classify(x) for x in [.1,.5,.9]]==['Low Potential','Medium Potential','High Potential']
def test_probability_and_sorting():
    df=clean(generate(160),training=True)
    m=build(LogisticRegression(max_iter=1000)).fit(df[FEATURES],df.converted)
    ranked=score(df.head(12),m)
    assert len(ranked)==12
    assert ranked.conversion_probability.between(0,1).all()
    assert ranked.conversion_probability.is_monotonic_decreasing
