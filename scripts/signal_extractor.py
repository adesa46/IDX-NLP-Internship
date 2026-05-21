"""
Week 6: Listing Signal Extraction

SignalExtractor combines Week 3 EntityExtractor with taxonomy-based
pattern matching to extract comprehensive structured signals from
each listing: amenities, condition, financing, and location features.
"""

import re
import json
import os
import pandas as pd

from entity_extractor import EntityExtractor


class SignalExtractor:
    """Extract structured signals from real estate listing records."""

    def __init__(self, taxonomy_path="data/processed/taxonomy.json", entity_extractor=None):
        # Load taxonomy grouped by category
        self.taxonomy = {}
        if os.path.exists(taxonomy_path):
            with open(taxonomy_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data.get('terms', []):
                    cat = item.get('category', 'Other')
                    self.taxonomy.setdefault(cat, []).append(item.get('term', '').lower())

        # Use provided extractor or create one
        self.extractor = entity_extractor or EntityExtractor(taxonomy_path)

        # ----- Condition patterns -----
        self.condition_patterns = [
            r'\b(updated)\b', r'\b(remodeled)\b', r'\b(renovated)\b',
            r'\b(turnkey)\b', r'\b(move[\s-]?in\s*ready)\b',
            r'\b(new\s*construction)\b', r'\b(newly\s*built)\b',
            r'\b(brand\s*new)\b', r'\b(fixer[\s-]?upper)\b',
            r'\b(fixer)\b', r'\b(tlc)\b', r'\b(needs\s*work)\b',
            r'\b(well[\s-]?maintained)\b', r'\b(freshly\s*painted)\b',
            r'\b(fresh\s*paint)\b', r'\b(new\s*roof)\b',
            r'\b(new\s*carpet)\b', r'\b(new\s*flooring)\b',
            r'\b(new\s*appliances)\b', r'\b(upgraded)\b',
            r'\b(remod[a-z]*)\b', r'\b(refurbished)\b',
            r'\b(restored)\b', r'\b(custom[\s-]?built)\b',
            r'\b(original\s*condition)\b', r'\b(as[\s-]?is)\b',
            r'\b(new\s*windows)\b', r'\b(new\s*hvac)\b',
            r'\b(new\s*plumbing)\b', r'\b(new\s*electrical)\b',
        ]

        # ----- Financing patterns -----
        self.financing_patterns = [
            r'\b(seller\s*financing)\b', r'\b(owner\s*financing)\b',
            r'\b(owner\s*carry)\b', r'\b(owner\s*will\s*carry)\b',
            r'\b(assumable)\b', r'\b(assumable\s*loan)\b',
            r'\b(fha)\b', r'\b(va)\b', r'\b(conventional)\b',
            r'\b(cash\s*only)\b', r'\b(cash)\b',
            r'\b(lease\s*option)\b', r'\b(rent[\s-]?to[\s-]?own)\b',
            r'\b(1031\s*exchange)\b', r'\b(exchange\s*1031)\b',
            r'\b(hoa)\b', r'\b(mello[\s-]?roos)\b',
            r'\b(no\s*hoa)\b', r'\b(low\s*hoa)\b',
            r'\b(fannie\s*mae)\b', r'\b(freddie\s*mac)\b',
            r'\b(usda)\b', r'\b(motivated\s*seller)\b',
            r'\b(price\s*reduced)\b', r'\b(below\s*market)\b',
        ]

        # ----- Location feature patterns -----
        self.location_patterns = [
            r'\b(cul[\s-]?de[\s-]?sac)\b', r'\b(corner\s*lot)\b',
            r'\b(gated)\b', r'\b(gated\s*community)\b',
            r'\b(golf\s*course)\b', r'\b(lake\s*front)\b',
            r'\b(lakefront)\b', r'\b(waterfront)\b',
            r'\b(ocean\s*view)\b', r'\b(mountain\s*view)\b',
            r'\b(city\s*view)\b', r'\b(panoramic\s*view)\b',
            r'\b(views?)\b', r'\b(near\s*schools?)\b',
            r'\b(close\s*to\s*schools?)\b', r'\b(school\s*district)\b',
            r'\b(walking\s*distance)\b', r'\b(walk\s*to)\b',
            r'\b(near\s*(?:park|shopping|freeway|beach|downtown))\b',
            r'\b(close\s*to\s*(?:park|shopping|freeway|beach|downtown))\b',
            r'\b(quiet\s*(?:street|neighborhood|area|cul))\b',
            r'\b(private\s*(?:lot|street|road|setting))\b',
            r'\b(dead[\s-]?end\s*street)\b',
            r'\b(greenbelt)\b', r'\b(no\s*neighbors\s*behind)\b',
            r'\b(backs?\s*to\s*(?:park|open\s*space|greenbelt))\b',
            r'\b(hoa\s*community)\b',
        ]

    def extract_signals(self, listing_record):
        """Extract all signals from a single listing record.

        Args:
            listing_record: dict with at least a text field ('L_Remarks' or 'remarks')

        Returns:
            dict with listing_id, entities, amenities, condition_keywords,
            financing_terms, and location_features
        """
        remarks = listing_record.get('L_Remarks',
                    listing_record.get('remarks',
                    listing_record.get('cleaned_remarks', '')))

        if not isinstance(remarks, str):
            remarks = ''

        # Get entities from Week 3 EntityExtractor
        entities = self.extractor.extract_all(remarks)

        # Get amenities from taxonomy
        amenities = self._match_amenities(remarks)

        # Detect condition, financing, location signals
        return {
            'listing_id': listing_record.get('L_ListingID',
                           listing_record.get('listing_id', None)),
            'entities': entities,
            'amenities': amenities,
            'condition_keywords': self._extract_condition(remarks),
            'financing_terms': self._extract_financing(remarks),
            'location_features': self._extract_location(remarks),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _match_amenities(self, text):
        """Match amenities from taxonomy Amenities + Exterior_Lot categories."""
        if not text:
            return []

        text_lower = text.lower()
        found = set()

        # Search in Amenities, Exterior_Lot, and Rooms_Layout taxonomy categories
        for cat in ('Amenities', 'Exterior_Lot', 'Rooms_Layout'):
            for term in self.taxonomy.get(cat, []):
                pattern = r'\b' + re.escape(term) + r'\b'
                if re.search(pattern, text_lower):
                    found.add(term)

        return sorted(found)

    def _extract_condition(self, text):
        """Extract condition/renovation keywords from text."""
        if not text:
            return []
        text_lower = text.lower()
        found = set()
        for pattern in self.condition_patterns:
            matches = re.findall(pattern, text_lower)
            for m in matches:
                cleaned = re.sub(r'\s+', ' ', m).strip()
                if cleaned:
                    found.add(cleaned)
        return sorted(found)

    def _extract_financing(self, text):
        """Extract financing-related terms from text."""
        if not text:
            return []
        text_lower = text.lower()
        found = set()
        for pattern in self.financing_patterns:
            matches = re.findall(pattern, text_lower)
            for m in matches:
                cleaned = re.sub(r'\s+', ' ', m).strip()
                if cleaned:
                    found.add(cleaned)
        return sorted(found)

    def _extract_location(self, text):
        """Extract location feature keywords from text."""
        if not text:
            return []
        text_lower = text.lower()
        found = set()
        for pattern in self.location_patterns:
            matches = re.findall(pattern, text_lower)
            for m in matches:
                cleaned = re.sub(r'\s+', ' ', m).strip()
                if cleaned:
                    found.add(cleaned)
        return sorted(found)

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def process_dataset(self, csv_path, output_path):
        """Process entire CSV dataset and save extracted signals to JSON.

        Args:
            csv_path: path to CSV with a 'remarks' (or 'cleaned_remarks') column
            output_path: path for the output JSON file
        """
        print(f"Loading dataset from {csv_path}...")
        df = pd.read_csv(csv_path)

        # Determine text column
        text_col = None
        for col in ('cleaned_remarks', 'remarks', 'L_Remarks'):
            if col in df.columns:
                text_col = col
                break

        if text_col is None:
            raise ValueError(f"No remarks column found in {csv_path}. "
                             f"Columns: {df.columns.tolist()}")

        print(f"Using column '{text_col}' — processing {len(df)} listings...")

        results = []
        for idx, row in df.iterrows():
            record = {
                'listing_id': idx,
                text_col: row[text_col] if pd.notna(row[text_col]) else '',
            }
            # Map column to a standard key the extractor understands
            if text_col != 'remarks':
                record['remarks'] = record[text_col]

            signals = self.extract_signals(record)
            signals['listing_id'] = idx  # ensure row index is used
            results.append(signals)

            if (idx + 1) % 200 == 0:
                print(f"  Processed {idx + 1}/{len(df)} listings...")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

        print(f"Done! Saved {len(results)} records to {output_path}")
        return results

    def compute_coverage_stats(self, results):
        """Compute extraction coverage statistics.

        Returns dict with percentage of listings that have each signal type.
        """
        total = len(results)
        if total == 0:
            return {}

        has_bedrooms = sum(1 for r in results if r['entities'].get('bedrooms') is not None)
        has_bathrooms = sum(1 for r in results if r['entities'].get('bathrooms') is not None)
        has_price = sum(1 for r in results if r['entities'].get('price') is not None)
        has_sqft = sum(1 for r in results if r['entities'].get('sqft') is not None)
        has_amenities = sum(1 for r in results if len(r.get('amenities', [])) > 0)
        has_condition = sum(1 for r in results if len(r.get('condition_keywords', [])) > 0)
        has_financing = sum(1 for r in results if len(r.get('financing_terms', [])) > 0)
        has_location = sum(1 for r in results if len(r.get('location_features', [])) > 0)
        has_any_entity = sum(1 for r in results
                            if r['entities'].get('bedrooms') is not None
                            or r['entities'].get('bathrooms') is not None
                            or r['entities'].get('price') is not None
                            or r['entities'].get('sqft') is not None)
        has_any_signal = sum(1 for r in results
                            if len(r.get('amenities', [])) > 0
                            or len(r.get('condition_keywords', [])) > 0
                            or len(r.get('financing_terms', [])) > 0
                            or len(r.get('location_features', [])) > 0)

        return {
            'total_listings': total,
            'bedrooms_pct': round(has_bedrooms / total * 100, 1),
            'bathrooms_pct': round(has_bathrooms / total * 100, 1),
            'price_pct': round(has_price / total * 100, 1),
            'sqft_pct': round(has_sqft / total * 100, 1),
            'amenities_pct': round(has_amenities / total * 100, 1),
            'condition_pct': round(has_condition / total * 100, 1),
            'financing_pct': round(has_financing / total * 100, 1),
            'location_pct': round(has_location / total * 100, 1),
            'any_entity_pct': round(has_any_entity / total * 100, 1),
            'any_signal_pct': round(has_any_signal / total * 100, 1),
        }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Extract signals from real estate listings.")
    parser.add_argument('--input', type=str, default='data/processed/sample_listing.csv',
                        help='Path to input CSV file')
    parser.add_argument('--output', type=str, default='data/processed/week6_signals.json',
                        help='Path to output JSON file')
    parser.add_argument('--taxonomy', type=str, default='data/processed/taxonomy.json',
                        help='Path to taxonomy JSON file')
    args = parser.parse_args()

    se = SignalExtractor(taxonomy_path=args.taxonomy)
    results = se.process_dataset(args.input, args.output)

    stats = se.compute_coverage_stats(results)
    print("\n=== Coverage Stats ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
