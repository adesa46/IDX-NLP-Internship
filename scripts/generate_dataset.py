import csv
import json
import random
import re
import os
from entity_extractor import EntityExtractor

def create_dataset(input_csv, output_json, num_samples=300):
    extractor = EntityExtractor("data/processed/taxonomy.json")
    
    with open(input_csv, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        header = next(reader)
        all_remarks = [row[0] for row in reader if len(row) > 0 and ' ' in row[0]]
        
    # filter out rows that are just comma separated lists (fewer than 5 spaces)
    valid_remarks = [r for r in all_remarks if r.count(' ') > 5]
    
    random.seed(42)
    sample_remarks = random.sample(valid_remarks, min(num_samples, len(valid_remarks)))
    
    dataset = []

    def find_number_span(text, entity_type):
        if entity_type == 'BEDROOMS':
            patterns = [r'(\d+)\s*(?:bed|br|bedroom)s?\b', r'\b(\d+)\s*bd\b']
            words = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
            for w in words:
                patterns.append(r'\b' + w + r'\s*(?:bed|br|bedroom)s?\b')
        elif entity_type == 'BATHROOMS':
            patterns = [r'(\d+(?:\.\d+)?)\s*(?:bath|ba|bathroom)s?\b']
            words = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
            for w in words:
                patterns.append(r'\b' + w + r'\s*(?:bath|ba|bathroom)s?\b')
        elif entity_type == 'PRICE':
            patterns = [r'\$\s*(\d+(?:[,\.]\d{3})*)', r'\b(\d{5,})\b']
        elif entity_type == 'SQFT':
            patterns = [r'(\d+(?:,\d{3})*)\s*(?:sq\s*ft|sqft|square\s*feet|sf)\b']
        else:
            patterns = []
            
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.I):
                return (match.start(), match.end())
        return None

    for text in sample_remarks:
        extracted = extractor.extract_all(text)
        entities = []
        
        if extracted.get('bedrooms') is not None:
            span = find_number_span(text, 'BEDROOMS')
            if span: entities.append([span[0], span[1], "BEDROOMS"])
            
        if extracted.get('bathrooms') is not None:
            span = find_number_span(text, 'BATHROOMS')
            if span: entities.append([span[0], span[1], "BATHROOMS"])
            
        if extracted.get('price') is not None:
            span = find_number_span(text, 'PRICE')
            if span: entities.append([span[0], span[1], "PRICE"])
            
        if extracted.get('sqft') is not None:
            span = find_number_span(text, 'SQFT')
            if span: entities.append([span[0], span[1], "SQFT"])
            
        for amenity in extracted.get('amenities', []):
            pattern = r'\b' + re.escape(amenity) + r'\b'
            for match in re.finditer(pattern, text, re.I):
                overlap = False
                new_start, new_end = match.start(), match.end()
                for e_start, e_end, _ in entities:
                    if max(new_start, e_start) < min(new_end, e_end):
                        overlap = True
                        break
                if not overlap:
                    entities.append([new_start, new_end, "AMENITY"])
                    
        # Filter overlaps within amenities
        entities = sorted(entities, key=lambda x: (x[0], -x[1]))
        filtered_entities = []
        last_end = -1
        for start, end, label in entities:
            if start >= last_end:
                filtered_entities.append([start, end, label])
                last_end = end
        
        dataset.append({
            "text": text,
            "entities": filtered_entities
        })
        
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2)
    print(f"Generated {len(dataset)} labeled examples at {output_json}")

if __name__ == "__main__":
    create_dataset("data/processed/sample_listing.csv", "data/labeled_dataset.json")
