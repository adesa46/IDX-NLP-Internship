import re
from collections import Counter

class TextCleaner:
    def __init__(self):
        self.abbrev_map = {
            'br': 'bedroom', 'ba': 'bathroom', 'sqft': 'square feet',
            'w/': 'with', 'w/o': 'without', 'mbr': 'master bedroom',
            'hw': 'hardwood', 'ss': 'stainless steel', 'gar': 'garage',
            'bsmt': 'basement', 'fba': 'full bathroom', 'hba': 'half bathroom',
            'ac': 'air conditioning', 'appl': 'appliances', 'apt': 'apartment',
            'bldg': 'building', 'blt': 'built', 'bthu': 'bath', 'cbd': 'central business district',
            'cac': 'central air conditioning', 'cen': 'central', 'clt': 'closet', 'cnd': 'condo',
            'comp': 'completely', 'cp': 'carport', 'crpt': 'carpet', 'ctv': 'cable tv',
            'c/w': 'complete with', 'dbl': 'double', 'det': 'detached', 'dk': 'deck',
            'dr': 'dining room', 'dw': 'dishwasher', 'eff': 'efficiency', 'elev': 'elevator',
            'excl': 'excellent', 'f/p': 'fireplace', 'fab': 'fabulous', 'flr': 'floor',
            'frm': 'formal', 'fr': 'family room', 'hdwd': 'hardwood', 'htg': 'heating',
            'kitch': 'kitchen', 'ldry': 'laundry', 'lr': 'living room', 'lux': 'luxury',
            'occ': 'occupancy', 'pkg': 'parking', 'rm': 'room', 'renov': 'renovated',
            's/s': 'stainless steel', 'spec': 'spectacular', 'sq ft': 'square feet', 'updtd': 'updated',
        }

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        text = self.normalize_unicode(text)
        text = self.remove_html(text)
        text = self.normalize_prices(text)
        text = self.normalize_measurements(text)
        text = self.expand_abbreviations(text)
        return text.strip()

    def normalize_unicode(self, text):
        # Replace non-breaking spaces
        text = text.replace('\xa0', ' ')
        # Replace curly quotes and apostrophes
        text = text.replace('‘', "'").replace('’', "'")
        text = text.replace('“', '"').replace('”', '"')
        # Combine multiple spaces into one
        text = re.sub(r'\s+', ' ', text)
        return text

    def remove_html(self, text):
        # Remove anything looking like an HTML tag
        return re.sub(r'<[^>]*>', '', text)

    def normalize_prices(self, text):
        # 450k → 450000
        text = re.sub(r'(\d+)k', lambda m: str(int(m.group(1))*1000), text, flags=re.I)
        # 1.2m → 1200000
        text = re.sub(r'(\d+\.?\d*)m', lambda m: str(int(float(m.group(1))*1000000)), text, flags=re.I)
        return text

    def normalize_measurements(self, text):
        # e.g., 2,000 sqft or 2000SqFt or 2000 sq. ft. -> 2000 square feet
        # First remove commas in numbers related to sqft to simplify matching
        # We can just remove all commas that are strictly between digits
        text = re.sub(r'(?<=\d),(?=\d)', '', text)
        # Now convert the measurement format
        text = re.sub(r'(\d+)\s*(?:sqft|sq\.?\s*ft\.?)', r'\1 square feet', text, flags=re.I)
        return text

    def expand_abbreviations(self, text):
        # Sort abbreviations by length descending to replace longer ones first (e.g., 'w/o' before 'w/')
        sorted_abbrevs = sorted(self.abbrev_map.keys(), key=len, reverse=True)
        
        for abbrev in sorted_abbrevs:
            full_form = self.abbrev_map[abbrev]
            
            # Create a pattern that handles word boundaries.
            # If the abbreviation ends/starts with non-word char (like '/'), \b might fail.
            # So we use negative lookahead/lookbehind for word chars instead of strict \b
            # We also capture the optional period so we can restore it if it existed.
            pattern = r'(?<!\w)' + re.escape(abbrev) + r'(?!\w)(\.?)'
            
            # Replace with the full form, and put the period back if it was captured
            text = re.sub(pattern, lambda m: full_form + m.group(1), text, flags=re.I)
            
        return text

    def profile_column(self, df, column_name):
        """Analyze what's actually in L_Remarks"""
        if column_name not in df.columns:
            return {}
        
        column_data = df[column_name].dropna().astype(str)
        
        return {
            'null_rate': df[column_name].isnull().mean(),
            'avg_length': column_data.str.len().mean() if not column_data.empty else 0,
            'common_terms': self._extract_top_ngrams(column_data),
            'price_mentions': column_data.str.contains(r'\$\d').sum(),
            'has_html': column_data.str.contains('<[^>]*>').sum(),
            'common_abbreviations': self._detect_abbreviations(column_data)
        }

    def _extract_top_ngrams(self, series, n=20):
        # Basic word count for simplicity
        words = ' '.join(series.tolist()).lower().split()
        # Filter out very short words
        words = [w for w in words if len(w) > 3]
        return dict(Counter(words).most_common(n))

    def _detect_abbreviations(self, series, n=10):
        words = ' '.join(series.tolist()).lower().split()
        # Clean words from punctuation for abbreviation detection
        words = [re.sub(r'[^\w/]', '', w) for w in words]
        
        found_abbrevs = [w for w in words if w in self.abbrev_map]
        return dict(Counter(found_abbrevs).most_common(n))