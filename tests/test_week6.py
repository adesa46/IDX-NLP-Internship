"""
Week 6 Tests: Listing Signal Extraction

Tests for the SignalExtractor class covering:
  - Single-listing signal extraction
  - Amenity matching from taxonomy
  - Condition keyword detection
  - Financing term detection
  - Location feature detection
  - Output schema validation
  - Batch processing
  - Coverage statistics
"""

import pytest
import sys
import os
import json
import tempfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))
from signal_extractor import SignalExtractor


@pytest.fixture(scope="module")
def extractor():
    """Build a SignalExtractor with the project taxonomy."""
    taxonomy_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'taxonomy.json')
    return SignalExtractor(taxonomy_path=taxonomy_path)


# ── Schema / Structure ────────────────────────────────────────────────

class TestOutputSchema:
    """Verify the output dict has all required keys."""

    def test_all_keys_present(self, extractor):
        record = {'remarks': 'Nice 3 bed 2 bath home with pool.', 'listing_id': '001'}
        result = extractor.extract_signals(record)
        for key in ('listing_id', 'entities', 'amenities',
                     'condition_keywords', 'financing_terms', 'location_features'):
            assert key in result, f"Missing key: {key}"

    def test_entities_sub_keys(self, extractor):
        record = {'remarks': '4 bedroom 3 bathroom 2500 sqft'}
        result = extractor.extract_signals(record)
        for key in ('bedrooms', 'bathrooms', 'sqft'):
            assert key in result['entities'], f"Missing entity key: {key}"

    def test_empty_remarks_returns_defaults(self, extractor):
        record = {'remarks': ''}
        result = extractor.extract_signals(record)
        assert result['amenities'] == []
        assert result['condition_keywords'] == []
        assert result['financing_terms'] == []
        assert result['location_features'] == []

    def test_none_remarks_returns_defaults(self, extractor):
        record = {'remarks': None}
        result = extractor.extract_signals(record)
        assert result['amenities'] == []

    def test_missing_remarks_returns_defaults(self, extractor):
        record = {}
        result = extractor.extract_signals(record)
        assert result['entities']['bedrooms'] is None


# ── Entity Extraction (from Week 3) ──────────────────────────────────

class TestEntityExtraction:
    """Verify Week 3 entity extraction still works through SignalExtractor."""

    def test_bedrooms(self, extractor):
        record = {'remarks': 'Spacious 4 bedroom home with views.'}
        result = extractor.extract_signals(record)
        assert result['entities']['bedrooms'] == 4

    def test_bathrooms(self, extractor):
        record = {'remarks': 'Features 2.5 bath and granite counters.'}
        result = extractor.extract_signals(record)
        assert result['entities']['bathrooms'] == 2.5

    def test_sqft(self, extractor):
        record = {'remarks': 'Over 3,200 sqft of living space.'}
        result = extractor.extract_signals(record)
        assert result['entities']['sqft'] == 3200

    def test_price(self, extractor):
        record = {'remarks': 'Listed at $750,000.'}
        result = extractor.extract_signals(record)
        assert result['entities']['price'] == 750000


# ── Amenity Matching ─────────────────────────────────────────────────

class TestAmenityMatching:
    """Verify taxonomy-based amenity matching."""

    def test_pool_detected(self, extractor):
        record = {'remarks': 'Large backyard with pool and spa.'}
        result = extractor.extract_signals(record)
        assert 'pool' in result['amenities']

    def test_garage_detected(self, extractor):
        record = {'remarks': 'Two-car garage with direct access.'}
        result = extractor.extract_signals(record)
        assert 'garage' in result['amenities']

    def test_fireplace_detected(self, extractor):
        record = {'remarks': 'Cozy living room with fireplace.'}
        result = extractor.extract_signals(record)
        assert 'fireplace' in result['amenities']

    def test_multiple_amenities(self, extractor):
        record = {'remarks': 'This home has a pool, fireplace, and garage.'}
        result = extractor.extract_signals(record)
        assert len(result['amenities']) >= 3

    def test_no_false_positives_on_plain_text(self, extractor):
        record = {'remarks': 'A nice day in the city.'}
        result = extractor.extract_signals(record)
        # Should not randomly detect amenities from unrelated text
        assert len(result['amenities']) <= 1


# ── Condition Keywords ───────────────────────────────────────────────

class TestConditionExtraction:
    """Detect condition / renovation keywords."""

    def test_updated(self, extractor):
        record = {'remarks': 'Fully updated kitchen and bathrooms.'}
        result = extractor.extract_signals(record)
        assert 'updated' in result['condition_keywords']

    def test_remodeled(self, extractor):
        record = {'remarks': 'Recently remodeled with new floors.'}
        result = extractor.extract_signals(record)
        assert any('remodel' in k for k in result['condition_keywords'])

    def test_turnkey(self, extractor):
        record = {'remarks': 'Turnkey condition, just move in!'}
        result = extractor.extract_signals(record)
        assert 'turnkey' in result['condition_keywords']

    def test_fixer(self, extractor):
        record = {'remarks': 'Great fixer-upper opportunity in prime area.'}
        result = extractor.extract_signals(record)
        assert any('fixer' in k for k in result['condition_keywords'])

    def test_move_in_ready(self, extractor):
        record = {'remarks': 'Move-in ready! Nothing to do.'}
        result = extractor.extract_signals(record)
        assert any('move' in k for k in result['condition_keywords'])

    def test_new_construction(self, extractor):
        record = {'remarks': 'Brand new construction, never lived in.'}
        result = extractor.extract_signals(record)
        assert any('new' in k for k in result['condition_keywords'])


# ── Financing Terms ──────────────────────────────────────────────────

class TestFinancingExtraction:
    """Detect financing-related terms."""

    def test_seller_financing(self, extractor):
        record = {'remarks': 'Seller financing available.'}
        result = extractor.extract_signals(record)
        assert 'seller financing' in result['financing_terms']

    def test_fha(self, extractor):
        record = {'remarks': 'FHA and VA approved complex.'}
        result = extractor.extract_signals(record)
        assert 'fha' in result['financing_terms']
        assert 'va' in result['financing_terms']

    def test_hoa(self, extractor):
        record = {'remarks': 'Low HOA of $150/month.'}
        result = extractor.extract_signals(record)
        assert 'hoa' in result['financing_terms']

    def test_cash_only(self, extractor):
        record = {'remarks': 'Cash only, no contingencies.'}
        result = extractor.extract_signals(record)
        assert any('cash' in t for t in result['financing_terms'])

    def test_1031_exchange(self, extractor):
        record = {'remarks': '1031 exchange opportunity.'}
        result = extractor.extract_signals(record)
        assert any('1031' in t for t in result['financing_terms'])


# ── Location Features ────────────────────────────────────────────────

class TestLocationExtraction:
    """Detect location-related features."""

    def test_cul_de_sac(self, extractor):
        record = {'remarks': 'Quiet cul-de-sac location.'}
        result = extractor.extract_signals(record)
        assert any('cul' in f for f in result['location_features'])

    def test_gated_community(self, extractor):
        record = {'remarks': 'Located in a gated community.'}
        result = extractor.extract_signals(record)
        assert any('gated' in f for f in result['location_features'])

    def test_views(self, extractor):
        record = {'remarks': 'Stunning ocean view from the living room.'}
        result = extractor.extract_signals(record)
        assert any('view' in f for f in result['location_features'])

    def test_near_schools(self, extractor):
        record = {'remarks': 'Close to schools and shopping.'}
        result = extractor.extract_signals(record)
        assert any('school' in f for f in result['location_features'])

    def test_walking_distance(self, extractor):
        record = {'remarks': 'Walking distance to parks and restaurants.'}
        result = extractor.extract_signals(record)
        assert 'walking distance' in result['location_features']

    def test_corner_lot(self, extractor):
        record = {'remarks': 'Beautiful corner lot with mature trees.'}
        result = extractor.extract_signals(record)
        assert 'corner lot' in result['location_features']


# ── Batch Processing ─────────────────────────────────────────────────

class TestBatchProcessing:
    """Test processing an entire CSV dataset."""

    def test_process_sample_dataset(self, extractor):
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'sample_listing.csv')
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            results = extractor.process_dataset(csv_path, tmp_path)
            assert len(results) == 1000
            # Verify output file exists and is valid JSON
            with open(tmp_path, 'r') as f:
                loaded = json.load(f)
            assert len(loaded) == 1000
        finally:
            os.unlink(tmp_path)

    def test_each_record_has_schema(self, extractor):
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'sample_listing.csv')
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            results = extractor.process_dataset(csv_path, tmp_path)
            for r in results[:10]:
                assert 'listing_id' in r
                assert 'entities' in r
                assert 'amenities' in r
                assert 'condition_keywords' in r
                assert 'financing_terms' in r
                assert 'location_features' in r
        finally:
            os.unlink(tmp_path)


# ── Coverage Statistics ──────────────────────────────────────────────

class TestCoverageStats:
    """Test coverage computation."""

    def test_stats_keys(self, extractor):
        results = [
            extractor.extract_signals({'remarks': '3 bed 2 bath home with pool, updated kitchen.'}),
            extractor.extract_signals({'remarks': 'FHA approved condo near schools.'}),
        ]
        stats = extractor.compute_coverage_stats(results)
        assert 'total_listings' in stats
        assert 'amenities_pct' in stats
        assert 'any_signal_pct' in stats
        assert stats['total_listings'] == 2

    def test_empty_results(self, extractor):
        stats = extractor.compute_coverage_stats([])
        assert stats == {}
