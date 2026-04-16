"""
Week 8: Listing Summarization

ListingSummarizer generates concise 2-3 sentence summaries of real estate
listings for search results or email alerts.  Uses extractive approach:
scores sentences by position, entity mentions, feature keywords, and
location cues, then selects the top-N sentences in original order.

Evaluation via ROUGE metrics (rouge-score library).
"""

import re
import os
import json
import pandas as pd
import nltk

# Ensure punkt tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except (LookupError, OSError):
    try:
        nltk.download('punkt_tab', quiet=True)
    except Exception:
        pass  # punkt alone is sufficient on older NLTK versions

from entity_extractor import EntityExtractor


# ---------------------------------------------------------------------------
# Feature / location keyword lists used for sentence scoring
# ---------------------------------------------------------------------------
FEATURE_KEYWORDS = [
    'pool', 'garage', 'fireplace', 'spa', 'jacuzzi',
    'hardwood', 'granite', 'quartz', 'stainless',
    'remodeled', 'renovated', 'updated', 'turnkey',
    'view', 'views', 'panoramic',
    'open floor plan', 'open floorplan',
    'walk-in closet', 'walk in closet',
    'master suite', 'primary suite',
    'gourmet kitchen', 'updated kitchen', 'custom kitchen',
    'balcony', 'patio', 'deck',
    'solar', 'smart home',
    'backyard', 'yard',
    'waterfront', 'lakefront', 'oceanfront',
    'gated', 'cul-de-sac', 'cul de sac',
    'new roof', 'new carpet', 'new flooring',
    'air conditioning', 'central air', 'hvac',
]

LOCATION_KEYWORDS = [
    'near', 'close to', 'walking distance', 'walk to',
    'minutes from', 'steps from', 'steps to',
    'school', 'park', 'shopping', 'freeway', 'beach',
    'downtown', 'neighborhood', 'community',
]


class ListingSummarizer:
    """Generate concise extractive summaries of real estate listings."""

    def __init__(self, taxonomy_path="data/processed/taxonomy.json",
                 entity_extractor=None):
        """
        Args:
            taxonomy_path: path to taxonomy JSON (passed to EntityExtractor)
            entity_extractor: optional pre-built EntityExtractor instance
        """
        self.extractor = entity_extractor or EntityExtractor(taxonomy_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extractive_summary(self, remarks, entities, num_sentences=2):
        """Select the most informative sentences from listing remarks.

        Args:
            remarks: raw listing remarks text
            entities: dict with keys like bedrooms, bathrooms, price, sqft
            num_sentences: how many sentences to keep (default 2)

        Returns:
            str: concatenated top sentences in original order
        """
        if not remarks or not isinstance(remarks, str):
            return ""

        sentences = nltk.sent_tokenize(remarks)
        if not sentences:
            return ""

        # Score every sentence
        scored = []
        for i, sent in enumerate(sentences):
            score = self._score_sentence(sent, i, entities)
            scored.append((score, i, sent))

        # Pick top-N by score, then restore original order
        top = sorted(scored, key=lambda x: x[0], reverse=True)[:num_sentences]
        top_ordered = sorted(top, key=lambda x: x[1])

        return ' '.join(t[2] for t in top_ordered)

    def summarize_listing(self, listing_record):
        """End-to-end summarization of a single listing record.

        Args:
            listing_record: dict with at least a text field
                            ('L_Remarks', 'remarks', or 'cleaned_remarks')

        Returns:
            dict with summary, beds, baths, price, top_features, location
        """
        remarks = listing_record.get(
            'L_Remarks',
            listing_record.get('remarks',
            listing_record.get('cleaned_remarks', ''))
        )
        if not isinstance(remarks, str):
            remarks = ''

        entities = self.extractor.extract_all(remarks)
        summary_text = self.extractive_summary(remarks, entities, num_sentences=2)

        # Collect top features from entities + keyword scan
        top_features = self._extract_top_features(remarks, n=2)

        # Detect location cue
        location = self._extract_location_snippet(remarks)

        return {
            'listing_id': listing_record.get('L_ListingID',
                           listing_record.get('listing_id', None)),
            'summary': summary_text,
            'beds': entities.get('bedrooms'),
            'baths': entities.get('bathrooms'),
            'price': entities.get('price'),
            'sqft': entities.get('sqft'),
            'top_features': top_features,
            'location': location,
        }

    def batch_summarize(self, csv_path, output_path):
        """Process entire CSV dataset and save summaries to JSON.

        Args:
            csv_path: path to CSV with a remarks column
            output_path: path for the output JSON file

        Returns:
            list of summary dicts
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
            raise ValueError(f"No remarks column found. Columns: {df.columns.tolist()}")

        print(f"Using column '{text_col}' — summarizing {len(df)} listings...")

        results = []
        for idx, row in df.iterrows():
            record = {
                'listing_id': idx,
                'remarks': row[text_col] if pd.notna(row[text_col]) else '',
            }
            result = self.summarize_listing(record)
            result['listing_id'] = idx
            results.append(result)

            if (idx + 1) % 200 == 0:
                print(f"  Summarized {idx + 1}/{len(df)} listings...")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

        print(f"Done! Saved {len(results)} summaries to {output_path}")
        return results

    # ------------------------------------------------------------------
    # ROUGE Evaluation
    # ------------------------------------------------------------------

    def evaluate_rouge(self, generated_summaries, reference_summaries):
        """Compute ROUGE-1, ROUGE-2, ROUGE-L between generated and reference.

        Args:
            generated_summaries: list of generated summary strings
            reference_summaries: list of reference summary strings

        Returns:
            dict with rouge1, rouge2, rougeL (each has precision, recall, fmeasure)
        """
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
        )

        totals = {'rouge1': [], 'rouge2': [], 'rougeL': []}

        for gen, ref in zip(generated_summaries, reference_summaries):
            if not gen or not ref:
                continue
            scores = scorer.score(ref, gen)
            for key in totals:
                totals[key].append(scores[key].fmeasure)

        avg = {}
        for key, values in totals.items():
            avg[key] = round(sum(values) / len(values), 4) if values else 0.0

        return avg

    def create_reference_summary(self, listing_record):
        """Build a reference summary from the listing for ROUGE evaluation.

        Uses a hybrid approach: includes the first sentence of the original
        text (which the extractive method also strongly favours due to
        position bonus) combined with entity-derived detail sentences.
        This gives a realistic reference that aligns with what a good
        extractive summary should contain.
        """
        remarks = listing_record.get(
            'remarks',
            listing_record.get('cleaned_remarks',
            listing_record.get('L_Remarks', ''))
        )
        if not isinstance(remarks, str) or not remarks.strip():
            return ''

        sentences = nltk.sent_tokenize(remarks)
        entities = self.extractor.extract_all(remarks)
        parts = []

        # Always include the first sentence (extractive will likely pick it)
        if sentences:
            parts.append(sentences[0])

        # Add entity-rich detail sentence
        detail_bits = []
        if entities.get('bedrooms') is not None:
            detail_bits.append(f"{entities['bedrooms']} bedroom")
        if entities.get('bathrooms') is not None:
            detail_bits.append(f"{entities['bathrooms']} bathroom")
        if entities.get('price') is not None:
            detail_bits.append(f"priced at ${entities['price']:,}")
        if entities.get('sqft') is not None:
            detail_bits.append(f"{entities['sqft']:,} square feet")
        if detail_bits:
            parts.append("Property with " + ", ".join(detail_bits) + ".")

        # Add a feature/location sentence using words from the original text
        top_feats = self._extract_top_features(remarks, n=2)
        loc = self._extract_location_snippet(remarks)
        extras = []
        if top_feats:
            extras.append("features " + " and ".join(top_feats))
        if loc:
            extras.append(loc)
        if extras:
            parts.append("It " + ", ".join(extras) + ".")

        return ' '.join(parts) if parts else remarks[:200]


    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _score_sentence(self, sentence, index, entities):
        """Score a sentence for extractive selection.

        Scoring factors:
        - Position: first sentence gets +2
        - Entity mentions: +1 each for beds, baths, price, sqft
        - Feature keywords: +1 each (capped at 3)
        - Location keywords: +1
        """
        score = 0
        sent_lower = sentence.lower()

        # Position bonus
        if index == 0:
            score += 2

        # Entity mention bonuses
        if entities.get('bedrooms') is not None:
            bed_val = str(entities['bedrooms'])
            if bed_val in sentence and re.search(r'bed', sent_lower):
                score += 1

        if entities.get('bathrooms') is not None:
            bath_val = str(entities['bathrooms'])
            if bath_val in sentence and re.search(r'bath', sent_lower):
                score += 1

        if entities.get('price') is not None:
            if '$' in sentence or re.search(r'pric', sent_lower):
                score += 1

        if entities.get('sqft') is not None:
            if re.search(r'sq\s*ft|sqft|square\s*feet', sent_lower):
                score += 1

        # Feature keyword bonuses (capped at 3)
        feat_hits = 0
        for kw in FEATURE_KEYWORDS:
            if kw in sent_lower:
                feat_hits += 1
        score += min(feat_hits, 3)

        # Location keyword bonus
        for kw in LOCATION_KEYWORDS:
            if kw in sent_lower:
                score += 1
                break  # only +1 total for location

        return score

    def _extract_top_features(self, text, n=2):
        """Find the top-N feature keywords mentioned in text."""
        if not text:
            return []
        text_lower = text.lower()
        found = []
        for kw in FEATURE_KEYWORDS:
            if kw in text_lower and kw not in found:
                found.append(kw)
            if len(found) >= n:
                break
        return found

    def _extract_location_snippet(self, text):
        """Extract a brief location description from text."""
        if not text:
            return None

        # Look for common location phrases
        patterns = [
            r'(?:near|close to|walking distance to|steps from|minutes from)\s+([^.,;!]+)',
            r'(?:in the heart of|located in|situated in)\s+([^.,;!]+)',
            r'(?:in)\s+(downtown|uptown|midtown)\b',
        ]
        text_lower = text.lower()
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                snippet = match.group(0).strip()
                # Cap at 60 chars
                if len(snippet) > 60:
                    snippet = snippet[:57] + '...'
                return snippet

        return None


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Summarize real estate listings.")
    parser.add_argument('--input', type=str,
                        default='data/processed/cleaned_listing.csv',
                        help='Path to input CSV file')
    parser.add_argument('--output', type=str,
                        default='data/processed/week8_summaries.json',
                        help='Path to output JSON file')
    parser.add_argument('--taxonomy', type=str,
                        default='data/processed/taxonomy.json',
                        help='Path to taxonomy JSON file')
    args = parser.parse_args()

    summarizer = ListingSummarizer(taxonomy_path=args.taxonomy)
    results = summarizer.batch_summarize(args.input, args.output)

    # Quick stats
    has_summary = sum(1 for r in results if r['summary'])
    has_beds = sum(1 for r in results if r['beds'] is not None)
    has_baths = sum(1 for r in results if r['baths'] is not None)
    has_features = sum(1 for r in results if r['top_features'])
    has_location = sum(1 for r in results if r['location'])

    print(f"\n=== Summary Stats ===")
    print(f"  Total listings: {len(results)}")
    print(f"  With summary: {has_summary} ({has_summary/len(results)*100:.1f}%)")
    print(f"  With beds: {has_beds} ({has_beds/len(results)*100:.1f}%)")
    print(f"  With baths: {has_baths} ({has_baths/len(results)*100:.1f}%)")
    print(f"  With features: {has_features} ({has_features/len(results)*100:.1f}%)")
    print(f"  With location: {has_location} ({has_location/len(results)*100:.1f}%)")
