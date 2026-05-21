"""
Week 7 Tests: Buyer Intent Classification

Tests for IntentClassifier, IntentQueryParser, dataset quality,
and confidence scores.
"""

import pytest
import sys
import os
import json
import tempfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))
from intent_classifier import IntentClassifier, IntentQueryParser
from query_parser import QueryParser


@pytest.fixture(scope="module")
def dataset():
    """Load the intent dataset."""
    path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'intent_dataset.json')
    with open(path, 'r') as f:
        return json.load(f)


@pytest.fixture(scope="module")
def trained_classifier(dataset):
    """Train a classifier on the full dataset for testing."""
    queries = [d['query'] for d in dataset]
    labels = [d['intent'] for d in dataset]
    clf = IntentClassifier()
    clf.train(queries, labels)
    return clf


# ── Dataset Quality ──────────────────────────────────────────────────

class TestDatasetQuality:
    """Verify the dataset meets requirements."""

    def test_minimum_200_queries(self, dataset):
        assert len(dataset) >= 200, f"Dataset has only {len(dataset)} queries"

    def test_all_three_labels_present(self, dataset):
        labels = set(d['intent'] for d in dataset)
        assert 'browsing' in labels
        assert 'researching' in labels
        assert 'ready_to_buy' in labels

    def test_no_empty_queries(self, dataset):
        for d in dataset:
            assert d['query'].strip(), "Empty query found"

    def test_reasonable_balance(self, dataset):
        from collections import Counter
        counts = Counter(d['intent'] for d in dataset)
        # Each class should have at least 40 samples
        for label, count in counts.items():
            assert count >= 40, f"{label} has only {count} queries"

    def test_valid_json_structure(self, dataset):
        for d in dataset:
            assert 'query' in d
            assert 'intent' in d
            assert d['intent'] in ('browsing', 'researching', 'ready_to_buy')


# ── IntentClassifier ─────────────────────────────────────────────────

class TestIntentClassifier:
    """Test the classifier training, prediction, and evaluation."""

    def test_predict_returns_tuple(self, trained_classifier):
        intent, conf = trained_classifier.predict("Show me homes in Portland")
        assert isinstance(intent, str)
        assert isinstance(conf, float)

    def test_predict_valid_intent(self, trained_classifier):
        intent, _ = trained_classifier.predict("3 bed 2 bath under 500k")
        assert intent in ('browsing', 'researching', 'ready_to_buy')

    def test_predict_confidence_range(self, trained_classifier):
        _, conf = trained_classifier.predict("looking around at homes")
        assert 0.0 <= conf <= 1.0

    def test_predict_proba_sums_to_one(self, trained_classifier):
        probas = trained_classifier.predict_proba("Compare condos vs townhouses")
        total = sum(probas.values())
        assert abs(total - 1.0) < 0.01

    def test_predict_proba_all_labels(self, trained_classifier):
        probas = trained_classifier.predict_proba("any query")
        for label in ('browsing', 'researching', 'ready_to_buy'):
            assert label in probas

    def test_predict_batch(self, trained_classifier):
        queries = ["Show me homes", "Compare prices", "3 bed under 500k"]
        results = trained_classifier.predict_batch(queries)
        assert len(results) == 3
        for intent, conf in results:
            assert isinstance(intent, str)
            assert 0.0 <= conf <= 1.0

    def test_accuracy_above_80_percent(self, dataset):
        """Train/test split and verify ≥80% accuracy."""
        from sklearn.model_selection import train_test_split
        queries = [d['query'] for d in dataset]
        labels = [d['intent'] for d in dataset]
        train_q, test_q, train_l, test_l = train_test_split(
            queries, labels, test_size=0.2, random_state=42, stratify=labels
        )
        clf = IntentClassifier()
        clf.train(train_q, train_l)
        results = clf.evaluate(test_q, test_l)
        assert results['accuracy'] >= 0.80, f"Accuracy {results['accuracy']:.2f} < 0.80"

    def test_evaluate_returns_correct_keys(self, trained_classifier):
        results = trained_classifier.evaluate(
            ["show me homes", "compare prices"],
            ["browsing", "researching"]
        )
        assert 'accuracy' in results
        assert 'classification_report' in results
        assert 'confusion_matrix' in results

    def test_save_and_load_model(self, trained_classifier):
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            trained_classifier.save_model(tmp_path)
            assert os.path.exists(tmp_path)

            new_clf = IntentClassifier()
            new_clf.load_model(tmp_path)
            intent1, conf1 = trained_classifier.predict("3 bed house in Portland")
            intent2, conf2 = new_clf.predict("3 bed house in Portland")
            assert intent1 == intent2
            assert abs(conf1 - conf2) < 0.001
        finally:
            os.unlink(tmp_path)

    def test_untrained_raises_error(self):
        clf = IntentClassifier()
        with pytest.raises(RuntimeError):
            clf.predict("some query")


# ── Confidence Scores ────────────────────────────────────────────────

class TestConfidenceScores:
    """Test that confidence scores behave correctly."""

    def test_high_confidence_on_clear_browsing(self, trained_classifier):
        intent, conf = trained_classifier.predict("Just looking around at homes")
        assert conf >= 0.5

    def test_high_confidence_on_clear_ready_to_buy(self, trained_classifier):
        intent, conf = trained_classifier.predict(
            "3 bed 2 bath under $500000 in Portland with pool"
        )
        assert conf >= 0.5

    def test_all_confidences_valid(self, trained_classifier, dataset):
        queries = [d['query'] for d in dataset[:20]]
        results = trained_classifier.predict_batch(queries)
        for intent, conf in results:
            assert 0.0 <= conf <= 1.0


# ── IntentQueryParser Integration ────────────────────────────────────

class TestIntentQueryParser:
    """Test integration of IntentClassifier with QueryParser."""

    def test_analyze_returns_all_keys(self, trained_classifier):
        parser = QueryParser()
        iqp = IntentQueryParser(trained_classifier, parser)
        result = iqp.analyze("3 bed 2 bath under 500k in Portland with pool")
        assert 'query' in result
        assert 'intent' in result
        assert 'confidence' in result
        assert 'probabilities' in result
        assert 'filters' in result
        assert 'sql' in result
        assert 'params' in result

    def test_analyze_extracts_filters(self, trained_classifier):
        parser = QueryParser()
        iqp = IntentQueryParser(trained_classifier, parser)
        result = iqp.analyze("3 bed in Portland under 500k")
        assert result['filters'].get('bedrooms') == 3 or result['filters'].get('bedrooms_min') == 3

    def test_analyze_produces_valid_sql(self, trained_classifier):
        parser = QueryParser()
        iqp = IntentQueryParser(trained_classifier, parser)
        result = iqp.analyze("Show me houses under $500k")
        assert 'SELECT' in result['sql']

    def test_analyze_intent_is_valid(self, trained_classifier):
        parser = QueryParser()
        iqp = IntentQueryParser(trained_classifier, parser)
        result = iqp.analyze("Compare condos vs townhouses in Portland")
        assert result['intent'] in ('browsing', 'researching', 'ready_to_buy')

    def test_analyze_confidence_is_float(self, trained_classifier):
        parser = QueryParser()
        iqp = IntentQueryParser(trained_classifier, parser)
        result = iqp.analyze("Any homes available?")
        assert isinstance(result['confidence'], float)
