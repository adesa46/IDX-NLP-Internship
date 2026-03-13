import re
import json
import os
import argparse

class EntityExtractor:
    def __init__(self, taxonomy_path="data/processed/taxonomy.json"):
        self.amenities_terms = []
        if os.path.exists(taxonomy_path):
            with open(taxonomy_path, 'r', encoding='utf-8') as f:
                taxonomy_data = json.load(f)
                for item in taxonomy_data.get('terms', []):
                    if item.get('category') == 'Amenities':
                        self.amenities_terms.append(item.get('term').lower())
        
        # Sort amenities by length descending to match longest phrases first
        self.amenities_terms.sort(key=len, reverse=True)

    def extract_bedrooms(self, text):
        patterns = [
            r'(\d+)\s*(?:bed|br|bedroom)s?\b',
            r'\b(\d+)\s*bd\b'
        ]
        text_lower = text.lower()
        
        # Handle written numbers
        number_map = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
        for word, num in number_map.items():
            if re.search(r'\b' + word + r'\s*(?:bed|br|bedroom)s?\b', text_lower):
                return num

        for pattern in patterns:
            match = re.search(pattern, text, re.I)
            if match:
                return int(match.group(1))
        return None

    def extract_bathrooms(self, text):
        patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:bath|ba|bathroom)s?\b'
        ]
        text_lower = text.lower()
        
        # Handle written numbers
        number_map = {"one": 1.0, "two": 2.0, "three": 3.0, "four": 4.0, "five": 5.0, "six": 6.0, "seven": 7.0, "eight": 8.0, "nine": 9.0, "ten": 10.0}
        for word, num in number_map.items():
            if re.search(r'\b' + word + r'\s*(?:bath|ba|bathroom)s?\b', text_lower):
                return num

        for pattern in patterns:
            match = re.search(pattern, text, re.I)
            if match:
                return float(match.group(1))
        return None

    def extract_price(self, text):
        # Match $1,200,000 or 1200000 etc.
        patterns = [
            r'\$\s*(\d+(?:[,\.]\d{3})*)',
            r'\b(\d{5,})\b'
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.I)
            if match:
                price_str = match.group(1).replace(',', '').replace('.', '')
                try:
                    return int(price_str)
                except ValueError:
                    continue
        return None

    def extract_sqft(self, text):
        patterns = [
            r'(\d+(?:,\d{3})*)\s*(?:sq\s*ft|sqft|square\s*feet|sf)\b'
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.I)
            if match:
                sqft_str = match.group(1).replace(',', '')
                try:
                    return int(sqft_str)
                except ValueError:
                    continue
        return None

    def extract_amenities(self, text):
        found_amenities = []
        text_lower = text.lower()
        for term in self.amenities_terms:
            # Use regex to find whole words only
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, text_lower):
                found_amenities.append(term)
        return found_amenities

    def extract_all(self, text):
        return {
            'bedrooms': self.extract_bedrooms(text),
            'bathrooms': self.extract_bathrooms(text),
            'price': self.extract_price(text),
            'sqft': self.extract_sqft(text),
            'amenities': self.extract_amenities(text)
        }

    def process_file(self, input_path, output_path):
        if not os.path.exists(input_path):
            print(f"Error: Input file {input_path} not found.")
            return

        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        results = []
        for item in data:
            # Handle different dataset formats that might use 'text' or 'remarks'
            text = item.get('text', item.get('remarks', ''))
            extracted = self.extract_all(text)
            
            result_item = item.copy()
            result_item['extracted_entities'] = extracted
            results.append(result_item)

        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        print(f"Extraction complete. Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract entities from real estate listings.")
    parser.add_argument("--input", type=str, help="Path to input JSON file", default=None)
    parser.add_argument("--output", type=str, help="Path to output JSON file", default="extracted_entities.json")
    parser.add_argument("--taxonomy", type=str, help="Path to taxonomy JSON file", default="../data/processed/taxonomy.json")
    args = parser.parse_args()

    extractor = EntityExtractor(args.taxonomy)
    
    if args.input:
        extractor.process_file(args.input, args.output)
    else:
        sample_text = "Beautiful 3 bedroom, 2.5 bath home with 2,500 sqft of living space. Features a sparkling pool and a two-car garage. Over $500,000 in upgrades! Priced at $1,250,000."
        print("Sample Extraction:")
        print(json.dumps(extractor.extract_all(sample_text), indent=2))
        print("\nTo process a file, run: python entity_extractor.py --input path/to/dataset.json")