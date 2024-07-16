
import pandas as pd
import numpy as np
from IPFInitFit import IPF

def generate_sample_data():
    np.random.seed(42)
    data = pd.DataFrame({
        'Gender': np.random.choice(['Male', 'Female'], size=1000),
        'AgeGroup': np.random.choice(['0-18', '19-35', '36-60', '60+'], size=1000),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], size=1000),
        'weight': np.ones(1000)
    })
    return data

def test_apply_weighting():
    ipf = IPF()
    data = generate_sample_data()
    constraints = [
        (['Gender'], {'Male': 520, 'Female': 480}),
        (['AgeGroup'], {'0-18': 200, '19-35': 300, '36-60': 300, '60+': 200}),
        (['Region'], {'North': 250, 'South': 250, 'East': 250, 'West': 250})
    ]

    data_weighted = ipf.apply_weighting(data, constraints)
    assert np.isclose(data_weighted['weight'].sum(), 1000), "Total weight should be 1000"

def test_check_results():
    ipf = IPF()
    data_clean = generate_sample_data()
    data_weighted = pd.DataFrame({
        'Gender': np.random.choice(['Male', 'Female'], size=1000),
        'AgeGroup': np.random.choice(['0-18', '19-35', '36-60', '60+'], size=1000),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], size=1000),
        'weight': np.random.rand(1000) * 10
    })
    constraints = [
        (['Gender'], {'Male': 520, 'Female': 480}),
        (['AgeGroup'], {'0-18': 200, '19-35': 300, '36-60': 300, '60+': 200}),
        (['Region'], {'North': 250, 'South': 250, 'East': 250, 'West': 250})
    ]

    ipf.check_results(data_clean, data_weighted, constraints)

test_apply_weighting()
test_check_results()
