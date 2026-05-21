import pytest
import json
import pandas as pd


def test_taxonomy_loaded():
    with open('data/processed/taxonomy.json') as f:
        tax = json.load(f)
    assert len(tax['terms']) >= 200
    assert all('id' in t and 'term' in t for t in tax['terms'])


def test_sample_data_quality():
    df = pd.read_csv('data/processed/sample_listing.csv')
    assert len(df) >= 500
    assert df['remarks'].str.len().min() > 50
