import json
from entity_extractor import EntityExtractor
import re

def evaluate(dataset_path, taxonomy_path):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        datasets = json.load(f)

    extractor = EntityExtractor(taxonomy_path)

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    errors = []

    print(f"Evaluating on {len(datasets)} examples...")

    for data in datasets:
        text = data['text']
        true_entities = data['entities']
        
        extracted = extractor.extract_all(text)
        
        # We need to compute spans for the extracted entities to compare with true_entities
        pred_entities = []
        
        def add_span(entity_type, val):
            if val is None: return
            if entity_type == 'BEDROOMS':
                patterns = []
                for w in ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]: patterns.append(r'\b' + w + r'\s*(?:bed|br|bedroom)s?\b')
                patterns.extend([r'(\d+)\s*(?:bed|br|bedroom)s?\b', r'\b(\d+)\s*bd\b'])
            elif entity_type == 'BATHROOMS':
                patterns = []
                for w in ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]: patterns.append(r'\b' + w + r'\s*(?:bath|ba|bathroom)s?\b')
                patterns.append(r'(\d+(?:\.\d+)?)\s*(?:bath|ba|bathroom)s?\b')
            elif entity_type == 'PRICE':
                patterns = [r'\$\s*(\d+(?:[,\.]\d{3})*)', r'\b(\d{5,})\b']
            elif entity_type == 'SQFT':
                patterns = [r'(\d+(?:,\d{3})*)\s*(?:sq\s*ft|sqft|square\s*feet|sf)\b']
            
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.I):
                    pred_entities.append((match.start(), match.end(), entity_type))
                    return

        add_span('BEDROOMS', extracted['bedrooms'])
        add_span('BATHROOMS', extracted['bathrooms'])
        add_span('PRICE', extracted['price'])
        add_span('SQFT', extracted['sqft'])
        
        for amenity in extracted['amenities']:
            pattern = r'\b' + re.escape(amenity) + r'\b'
            for match in re.finditer(pattern, text, re.I):
                pred_entities.append((match.start(), match.end(), "AMENITY"))
                
        # To avoid duplicates in predictions matching
        pred_entities = list(set(pred_entities))
        true_entities_tuples = [(e[0], e[1], e[2]) for e in true_entities]
        
        for p in pred_entities:
            if p in true_entities_tuples:
                true_positives += 1
            else:
                false_positives += 1
                errors.append({"type": "FP", "text": text[p[0]:p[1]], "label": p[2], "context": text[max(0, p[0]-20):min(len(text), p[1]+20)]})
                
        for t in true_entities_tuples:
            if t not in pred_entities:
                false_negatives += 1
                errors.append({"type": "FN", "text": text[t[0]:t[1]], "label": t[2], "context": text[max(0, t[0]-20):min(len(text), t[1]+20)]})

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    print(f"\nTotal Errors: {len(errors)}")
    if len(errors) > 0:
        print("Top Failure Patterns (Sample):")
        for e in errors[:10]:
            print(f" - {e['type']}: '{e['text']}' [{e['label']}] -> context: '...{e['context']}...'")

if __name__ == "__main__":
    evaluate("data/labeled_dataset.json", "data/processed/taxonomy.json")
