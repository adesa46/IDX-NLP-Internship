"""
Week 8 Tests: Listing Summarization & Answerability

Tests for ListingSummarizer (extractive summaries, ROUGE evaluation,
summary content requirements) and AnswerabilityChecker (pre/post-query).
"""

import pytest
import sys
import os
import json
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))
from listing_summarizer import ListingSummarizer
from answerability_checker import AnswerabilityChecker
from query_parser import QueryParser, SchemaValidator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def summarizer():
    taxonomy_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'taxonomy.json')
    return ListingSummarizer(taxonomy_path=taxonomy_path)


@pytest.fixture(scope="module")
def checker():
    schema_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'schema.json')
    return AnswerabilityChecker(schema_path=schema_path)


@pytest.fixture
def sample_listing():
    return {
        'listing_id': 1,
        'remarks': (
            "Beautiful 3 bedroom, 2.5 bath home located in the heart of Portland. "
            "This stunning property features an updated kitchen with granite countertops "
            "and stainless steel appliances. The spacious backyard includes a sparkling pool "
            "perfect for entertaining. Hardwood floors throughout the main living areas. "
            "Walking distance to top-rated schools and just minutes from downtown shopping. "
            "Priced at $525,000 with 2,100 sqft of living space."
        )
    }


@pytest.fixture
def sample_listings():
    """Several listings to test batch and ROUGE."""
    return [
        {
            'listing_id': 1,
            'remarks': (
                "Charming 4 bedroom, 3 bath home with pool and spa. "
                "Remodeled kitchen with quartz counters. Near great schools. "
                "Priced at $750,000."
            ),
        },
        {
            'listing_id': 2,
            'remarks': (
                "Modern 2 bed, 2 bath condo in downtown with city views. "
                "Open floor plan, balcony, and garage parking. "
                "Walking distance to restaurants and nightlife. $450,000."
            ),
        },
        {
            'listing_id': 3,
            'remarks': (
                "Spacious 5 bedroom estate on a quiet cul-de-sac. "
                "Features a gourmet kitchen, home theater, and 3-car garage. "
                "Large backyard with pool. Close to parks and hiking trails. "
                "Listed at $1,200,000 with 4,500 sqft."
            ),
        },
    ]


# ── ListingSummarizer ─────────────────────────────────────────────────


class TestExtractiveMethod:
    """Test the extractive_summary method."""

    def test_returns_string(self, summarizer, sample_listing):
        entities = summarizer.extractor.extract_all(sample_listing['remarks'])
        result = summarizer.extractive_summary(
            sample_listing['remarks'], entities, num_sentences=2
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_respects_num_sentences(self, summarizer, sample_listing):
        entities = summarizer.extractor.extract_all(sample_listing['remarks'])
        for n in (1, 2, 3):
            result = summarizer.extractive_summary(
                sample_listing['remarks'], entities, num_sentences=n
            )
            # The result should not have more sentences than requested
            # (may have fewer if input has fewer sentences)
            import nltk
            sents = nltk.sent_tokenize(result)
            assert len(sents) <= n

    def test_handles_empty_input(self, summarizer):
        assert summarizer.extractive_summary("", {}) == ""
        assert summarizer.extractive_summary(None, {}) == ""

    def test_single_sentence_input(self, summarizer):
        text = "Beautiful 3 bedroom home with pool."
        entities = summarizer.extractor.extract_all(text)
        result = summarizer.extractive_summary(text, entities, num_sentences=2)
        assert result == text

    def test_preserves_original_order(self, summarizer):
        text = (
            "First sentence about the home. "
            "Second sentence mentions pool. "
            "Third sentence about the neighborhood."
        )
        entities = {}
        result = summarizer.extractive_summary(text, entities, num_sentences=2)
        # First sentence should appear before any subsequent sentence
        if "First" in result and "Third" in result:
            assert result.index("First") < result.index("Third")


class TestSummarizeListing:
    """Test the full summarize_listing method."""

    def test_returns_dict_with_required_keys(self, summarizer, sample_listing):
        result = summarizer.summarize_listing(sample_listing)
        for key in ('summary', 'beds', 'baths', 'price', 'top_features', 'location'):
            assert key in result, f"Missing key: {key}"

    def test_extracts_beds(self, summarizer, sample_listing):
        result = summarizer.summarize_listing(sample_listing)
        assert result['beds'] == 3

    def test_extracts_baths(self, summarizer, sample_listing):
        result = summarizer.summarize_listing(sample_listing)
        assert result['baths'] == 2.5

    def test_extracts_price(self, summarizer, sample_listing):
        result = summarizer.summarize_listing(sample_listing)
        assert result['price'] is not None
        assert result['price'] > 0

    def test_extracts_top_features(self, summarizer, sample_listing):
        result = summarizer.summarize_listing(sample_listing)
        assert isinstance(result['top_features'], list)
        assert len(result['top_features']) <= 2

    def test_extracts_location(self, summarizer, sample_listing):
        result = summarizer.summarize_listing(sample_listing)
        # Sample listing has "in the heart of Portland" and "walking distance"
        assert result['location'] is not None

    def test_handles_empty_remarks(self, summarizer):
        result = summarizer.summarize_listing({'listing_id': 99, 'remarks': ''})
        assert result['summary'] == ''
        assert result['beds'] is None


# ── Summary Content Requirements ──────────────────────────────────────


class TestSummaryContent:
    """Summaries should include beds/baths, price, top 2 features, location."""

    def test_summaries_include_key_info(self, summarizer, sample_listings):
        """At least some summaries should include entities when they exist."""
        results = [summarizer.summarize_listing(lst) for lst in sample_listings]

        # At least 2 of 3 should have beds extracted
        have_beds = sum(1 for r in results if r['beds'] is not None)
        assert have_beds >= 2

        # At least 2 of 3 should have features
        have_feats = sum(1 for r in results if len(r['top_features']) > 0)
        assert have_feats >= 2

    def test_summary_text_not_too_long(self, summarizer, sample_listing):
        """Summary should be concise (not the entire remarks)."""
        result = summarizer.summarize_listing(sample_listing)
        # Summary should be shorter than full remarks
        assert len(result['summary']) < len(sample_listing['remarks'])


# ── ROUGE Evaluation ──────────────────────────────────────────────────


class TestRougeEvaluation:
    """Test ROUGE metric computation."""

    def test_rouge_returns_all_metrics(self, summarizer):
        gen = ["This is a 3 bedroom home with pool."]
        ref = ["A beautiful 3 bedroom property featuring a pool."]
        scores = summarizer.evaluate_rouge(gen, ref)
        assert 'rouge1' in scores
        assert 'rouge2' in scores
        assert 'rougeL' in scores

    def test_rouge_scores_in_range(self, summarizer):
        gen = ["Beautiful home with pool and garden."]
        ref = ["A beautiful home featuring a pool and lovely garden."]
        scores = summarizer.evaluate_rouge(gen, ref)
        for key in ('rouge1', 'rouge2', 'rougeL'):
            assert 0.0 <= scores[key] <= 1.0

    def test_identical_text_gets_perfect_score(self, summarizer):
        text = ["3 bedroom 2 bath home with pool near downtown."]
        scores = summarizer.evaluate_rouge(text, text)
        assert scores['rougeL'] == 1.0

    def test_rouge_l_above_threshold_on_samples(self, summarizer, sample_listings):
        """ROUGE-L should be > 0.4 when comparing extractive summaries
        to entity-based reference summaries on realistic data."""
        generated = []
        references = []
        for lst in sample_listings:
            result = summarizer.summarize_listing(lst)
            generated.append(result['summary'])
            ref = summarizer.create_reference_summary(lst)
            references.append(ref)

        scores = summarizer.evaluate_rouge(generated, references)
        assert scores['rougeL'] > 0.4, f"ROUGE-L {scores['rougeL']} <= 0.4"


# ── AnswerabilityChecker ──────────────────────────────────────────────


class TestAnswerabilityPreQuery:
    """Test pre-query validation."""

    def test_rejects_non_real_estate(self, checker):
        ok, msg = checker.check_pre_query("What's the weather like today?")
        assert ok is False
        assert "real estate" in msg.lower()

    def test_rejects_random_query(self, checker):
        ok, msg = checker.check_pre_query("Tell me a joke about cats")
        assert ok is False

    def test_accepts_valid_re_query(self, checker):
        ok, msg = checker.check_pre_query("3 bed house in Portland under 500k")
        assert ok is True

    def test_accepts_amenity_query(self, checker):
        ok, msg = checker.check_pre_query("homes with pool")
        assert ok is True

    def test_rejects_nonsense_query(self, checker):
        ok, msg = checker.check_pre_query("what color is the sky on mars")
        assert ok is False

    def test_returns_tuple(self, checker):
        result = checker.check_pre_query("any query about a house")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)


class TestAnswerabilityPostQuery:
    """Test post-query validation."""

    def test_rejects_empty_results(self, checker):
        ok, msg = checker.check_post_query("test", pd.DataFrame())
        assert ok is False
        assert "no listings" in msg.lower()

    def test_rejects_all_null_results(self, checker):
        df = pd.DataFrame({'a': [None, None], 'b': [None, None]})
        ok, msg = checker.check_post_query("test", df)
        assert ok is False
        assert "no meaningful" in msg.lower()

    def test_accepts_valid_results(self, checker):
        df = pd.DataFrame({'price': [500000], 'beds': [3]})
        ok, msg = checker.check_post_query("test", df)
        assert ok is True

    def test_reports_result_count(self, checker):
        df = pd.DataFrame({'price': [500000, 600000, 700000]})
        ok, msg = checker.check_post_query("test", df)
        assert ok is True
        assert "3" in msg


class TestAnswerabilityFull:
    """Test the convenience check_full method."""

    def test_full_check_non_re(self, checker):
        result = checker.check_full("weather forecast")
        assert result['answerable'] is False
        assert result['stage'] == 'pre_query'

    def test_full_check_valid_no_results(self, checker):
        result = checker.check_full(
            "3 bed house in Portland",
            results_df=pd.DataFrame()
        )
        assert result['answerable'] is False
        assert result['stage'] == 'post_query'

    def test_full_check_valid_with_results(self, checker):
        df = pd.DataFrame({'price': [500000]})
        result = checker.check_full("homes with pool", results_df=df)
        assert result['answerable'] is True
        assert result['stage'] == 'complete'


# ── Human Evaluation ──────────────────────────────────────────────────


class TestHumanEvaluation:
    """Verify human evaluation form generation."""

    def test_human_eval_form_structure(self, summarizer, sample_listings):
        """Generate a mini eval form and check structure."""
        form = []
        for lst in sample_listings:
            result = summarizer.summarize_listing(lst)
            form.append({
                'listing_id': result['listing_id'],
                'original_remarks': lst['remarks'][:200],
                'generated_summary': result['summary'],
                'rating': None,       # 1-5 scale (to be filled)
                'feedback': None,     # free text (to be filled)
            })

        assert len(form) >= 3
        for entry in form:
            assert 'listing_id' in entry
            assert 'original_remarks' in entry
            assert 'generated_summary' in entry
            assert 'rating' in entry
            assert 'feedback' in entry
