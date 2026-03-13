import pytest
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.text_cleaning import TextCleaner

@pytest.fixture
def cleaner():
    return TextCleaner()

@pytest.mark.parametrize("input_text, expected", [
    ("priced at 450k", "priced at 450000"),
    ("$1.2m home", "$1200000 home"),
    ("100K", "100000"),
    ("1.75M", "1750000"),
    ("Asking 850k completely firm", "Asking 850000 completely firm"),
    ("2m", "2000000"),
    ("0.5m", "500000"),
])
def test_normalize_prices(cleaner, input_text, expected):
    assert cleaner.normalize_prices(input_text) == expected

@pytest.mark.parametrize("input_text, expected", [
    ("2,000 sqft", "2000 square feet"),
    ("2000SqFt", "2000 square feet"),
    ("1,500 sq. ft.", "1500 square feet"),
    ("3000 sq ft", "3000 square feet"),
    ("1,234,567 sqft", "1234567 square feet"),
    ("sqft", "sqft"),  # No numbers
])
def test_normalize_measurements(cleaner, input_text, expected):
    assert cleaner.normalize_measurements(input_text) == expected

@pytest.mark.parametrize("input_text, expected", [
    ("3 br, 2 ba", "3 bedroom, 2 bathroom"),
    ("New ss appl in kitch", "New stainless steel appliances in kitchen"),
    ("1st flr apt", "1st floor apartment"),
    ("w/ w/o", "with without"),
    ("gar and bsmt", "garage and basement"),
    ("hw flr throughout", "hardwood floor throughout"),
    ("fab lux cnd", "fabulous luxury condo"),
    ("This is a mbr", "This is a master bedroom"),
    ("ac and cac", "air conditioning and central air conditioning"),
    ("br. 1 ba.", "bedroom. 1 bathroom."),
    ("BR", "bedroom"),  # Case insensitive
    ("brand new", "brand new"),  # Partial match should not trigger "br"
    ("baker", "baker"),  # Partial match should not trigger "ba"
])
def test_expand_abbreviations(cleaner, input_text, expected):
    assert cleaner.expand_abbreviations(input_text).lower() == expected.lower()

@pytest.mark.parametrize("input_text, expected", [
    ("hello\xa0world", "hello world"),
    ("“smart” ‘quotes’", "\"smart\" 'quotes'"),
    ("multiple    spaces", "multiple spaces"),
    ("tab\tspace", "tab space"),
])
def test_normalize_unicode(cleaner, input_text, expected):
    assert cleaner.normalize_unicode(input_text) == expected

@pytest.mark.parametrize("input_text, expected", [
    ("hello <br> world", "hello  world"),
    ("<b>bold</b>", "bold"),
    ("missing < tag >", "missing "),
    ("<div><p>text</p></div>", "text"),
    ("<img src='foo'>", ""),
])
def test_remove_html(cleaner, input_text, expected):
    assert cleaner.remove_html(input_text) == expected

@pytest.mark.parametrize("input_text, expected", [
    ("<b>450k</b> 3 br w/ 2,000 sqft", "450000 3 bedroom with 2000 square feet"),
    ("‘lux’ apt for $1.5m", "'luxury' apartment for $1500000"),
])
def test_clean_text_combo(cleaner, input_text, expected):
    assert cleaner.clean_text(input_text) == expected

def test_profiling(cleaner):
    df = pd.DataFrame({
        'remarks': [
            'beautiful 3 br house for 450k <br>',
            'lux apt w/ hw flr',
            'missing $1.2m',
            None
        ]
    })
    profile = cleaner.profile_column(df, 'remarks')
    
    assert 'null_rate' in profile
    assert profile['null_rate'] == 0.25
    
    assert 'avg_length' in profile
    assert profile['avg_length'] > 0
    
    assert 'price_mentions' in profile
    assert profile['price_mentions'] == 1
    
    assert 'has_html' in profile
    assert profile['has_html'] == 1
    
    assert 'common_terms' in profile
    assert 'common_abbreviations' in profile
