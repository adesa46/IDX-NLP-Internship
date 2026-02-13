import json
import pandas as pd
import pytest

def get_taxonomy_terms():
    try:
        with open('data/processed/taxonomy.json', 'r') as f:
            taxonomy = json.load(f)
            
        all_terms = []
        for item in taxonomy.get('terms', []):
            all_terms.append(item['term'])
        return set(all_terms)
    except Exception as e:
        pytest.fail(f"Could not load taxonomy: {e}")

def get_remarks():
    try:
        df = pd.read_csv('data/processed/sample_listing.csv')
        # Drop nan and lowercase
        remarks = df['remarks'].dropna().astype(str).str.lower().tolist()
        return remarks
    except Exception as e:
        pytest.fail(f"Could not load sample data: {e}")

def test_taxonomy_coverage():
    terms = get_taxonomy_terms()
    remarks = get_remarks()
    
    if not remarks:
        pytest.skip("No remarks to test coverage against.")
        
    covered_remarks_count = 0
    
    for remark in remarks:
        # Check if any taxonomy term exists in the remark
        # (A real implementation might tokenize first, but simple string matching is fine for a basic assertion)
        if any(term in remark for term in terms):
            covered_remarks_count += 1
            
    coverage_percentage = (covered_remarks_count / len(remarks)) * 100
    
    print(f"\nTotal Remarks: {len(remarks)}")
    print(f"Covered Remarks: {covered_remarks_count}")
    print(f"Coverage: {coverage_percentage:.2f}%")
    
    assert coverage_percentage >= 30.0, f"Taxonomy coverage is too low: {coverage_percentage:.2f}% (Expected >= 30%)"
