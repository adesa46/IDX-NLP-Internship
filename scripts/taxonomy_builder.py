import os
import json
import pandas as pd
import nltk
from collections import Counter
from nltk.util import ngrams
from nltk.corpus import stopwords
import re

def ensure_nltk_downloads():
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('stopwords')
        nltk.download('punkt')

def build_taxonomy(input_file, output_file):
    print(f"Loading data from {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        return

    if 'remarks' not in df.columns or df['remarks'].empty:
        print("No 'remarks' column found or data is empty.")
        return

    ensure_nltk_downloads()
    stop_words = set(stopwords.words('english'))
    # Add common useless real estate words
    stop_words.update(['home', 'house', 'property', 'room', 'features', 'great', 'beautiful', 
                       'perfect', 'must', 'see', 'one', 'two', 'new', 'large', 'spacious', 
                       'close', 'located', 'like'])

    print("Processing language...")
    all_text = ' '.join(df['remarks'].dropna().astype(str).str.lower())
    
    # Remove punctuation
    all_text = re.sub(r'[^\w\s]', '', all_text)
    
    tokens = nltk.word_tokenize(all_text)
    # Filter stopwords and short words
    cleaned_tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    print("Extracting terms...")
    # Get unigrams, bigrams, and trigrams
    unigrams = cleaned_tokens
    bigrams = [' '.join(b) for b in ngrams(cleaned_tokens, 2)]
    trigrams = [' '.join(t) for t in ngrams(cleaned_tokens, 3)]
    
    # Combine
    all_terms = unigrams + bigrams + trigrams
    freq = Counter(all_terms)
    
    top_terms = [term for term, count in freq.most_common(5000)]
    
    print("Categorizing terms...")
    taxonomy = {
        "Amenities": [],
        "Location": [],
        "Condition": [],
        "Architecture_Style": [],
        "Property_Type": [],
        "Financial_Legal": [],
        "Exterior_Lot": [],
        "Rooms_Layout": []
    }
    
    # Simple keyword-based categorization rules
    categories_keywords = {
        "Amenities": ['pool', 'spa', 'garage', 'parking', 'fireplace', 'laundry', 'ac', 'air conditioning', 'heater', 'gym', 'clubhouse'],
        "Location": ['school', 'park', 'shopping', 'freeway', 'downtown', 'neighborhood', 'street', 'quiet', 'cul de sac', 'views'],
        "Condition": ['updated', 'remodeled', 'turnkey', 'tlc', 'fresh', 'paint', 'carpet', 'newly', 'fixed', 'maintained'],
        "Architecture_Style": ['ranch', 'custom', 'modern', 'traditional', 'colonial', 'craftsman', 'contemporary', 'style'],
        "Property_Type": ['condo', 'townhouse', 'single family', 'apartment', 'duplex', 'estate', 'pud'],
        "Financial_Legal": ['hoa', 'fee', 'taxes', 'mello', 'roos', 'financing', 'fha', 'va', 'lease', 'sold', 'sale'],
        "Exterior_Lot": ['yard', 'patio', 'deck', 'balcony', 'acres', 'lot', 'landscaping', 'fence', 'sprinklers', 'grass'],
        "Rooms_Layout": ['kitchen', 'living', 'dining', 'master', 'bedroom', 'bathroom', 'suite', 'floor plan', 'open', 'concept', 'closet', 'bath']
    }
    
    # Assign terms to categories
    for term in top_terms:
        assigned = False
        for cat, keywords in categories_keywords.items():
            if any(keyword in term for keyword in keywords):
                if term not in taxonomy[cat]:
                    taxonomy[cat].append(term)
                assigned = True
                break
        
        # We only really want to keep the categorized ones to form a clean taxonomy
        # until we hit ~200 terms total.
        if sum(len(terms) for terms in taxonomy.values()) > 250:
            break
        
    # Trim to make sure it's clean and around 200 items in total
    total_assigned = sum(len(terms) for terms in taxonomy.values())
    print(f"Categorized {total_assigned} terms in total.")
    
    # Map to the format expected by tests
    final_terms_list = []
    term_id_counter = 1
    for category, terms in taxonomy.items():
        for term in terms:
            final_terms_list.append({
                "id": str(term_id_counter),
                "term": term,
                "category": category
            })
            term_id_counter += 1

    final_taxonomy_structure = {
        "terms": final_terms_list
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"Saving taxonomy to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_taxonomy_structure, f, indent=4)
        
    print("Done!")

if __name__ == '__main__':
    build_taxonomy('data/processed/sample_listing.csv', 'data/processed/taxonomy.json')
