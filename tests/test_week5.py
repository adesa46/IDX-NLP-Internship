"""
Week 5 Tests: Semantic Search with Embeddings

Tests for the SemanticSearcher class covering:
  - FAISS index creation
  - Semantic search results
  - BM25 search results
  - Search latency < 100ms
  - Relevance evaluation
  - Comparison between methods
"""

import pytest
import sys
import os
import time
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))
from semantic_search import SemanticSearcher, generate_evaluation_pairs


@pytest.fixture(scope="module")
def searcher():
    """Build a SemanticSearcher with cleaned listing data (module-scoped for speed)."""
    s = SemanticSearcher()

    # Load data
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'cleaned_listing.csv')
    df = pd.read_csv(csv_path)
    remarks = df['cleaned_remarks'].dropna().astype(str).tolist()
    # Filter out non-descriptive remarks
    remarks = [r for r in remarks if len(r) > 50 and ' ' in r]

    s.build_index(remarks)
    return s


class TestIndexCreation:
    """Tests for index building."""

    def test_faiss_index_exists(self, searcher):
        assert searcher.index is not None

    def test_faiss_index_count(self, searcher):
        assert searcher.index.ntotal > 0

    def test_embedding_dimensions(self, searcher):
        """all-MiniLM-L6-v2 produces 384-dim embeddings."""
        assert searcher.embedding_dim == 384

    def test_bm25_index_exists(self, searcher):
        assert searcher.bm25 is not None

    def test_listings_stored(self, searcher):
        assert searcher.listings is not None
        assert len(searcher.listings) > 0


class TestSemanticSearch:
    """Tests for FAISS-based semantic search."""

    def test_returns_results(self, searcher):
        results = searcher.search("home with pool", top_k=5)
        assert len(results) == 5

    def test_results_are_tuples(self, searcher):
        results = searcher.search("spacious kitchen", top_k=3)
        for item in results:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_scores_are_valid(self, searcher):
        results = searcher.search("modern condo", top_k=5)
        for _, score in results:
            # Cosine similarity on normalized vectors: between -1 and 1
            assert -1.0 <= score <= 1.0

    def test_results_ordered_by_score(self, searcher):
        results = searcher.search("large backyard", top_k=5)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_parameter(self, searcher):
        for k in [1, 3, 5, 10]:
            results = searcher.search("ocean view", top_k=k)
            assert len(results) <= k


class TestBM25Search:
    """Tests for BM25 keyword search."""

    def test_returns_results(self, searcher):
        results = searcher.bm25_search("home with pool", top_k=5)
        assert len(results) == 5

    def test_results_are_tuples(self, searcher):
        results = searcher.bm25_search("granite countertops", top_k=3)
        for item in results:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_scores_are_non_negative(self, searcher):
        results = searcher.bm25_search("fireplace", top_k=5)
        for _, score in results:
            assert score >= 0.0

    def test_top_k_parameter(self, searcher):
        for k in [1, 3, 5, 10]:
            results = searcher.bm25_search("garage", top_k=k)
            assert len(results) == k


class TestCompareSearch:
    """Tests for the comparison method."""

    def test_returns_both_methods(self, searcher):
        results = searcher.compare_search("3 bedroom house")
        assert 'semantic' in results
        assert 'bm25' in results

    def test_each_method_returns_results(self, searcher):
        results = searcher.compare_search("modern kitchen", top_k=5)
        assert len(results['semantic']) == 5
        assert len(results['bm25']) == 5


class TestLatency:
    """Tests for search latency requirements."""

    def test_semantic_search_under_100ms(self, searcher):
        """Latency requirement: < 100ms per query."""
        queries = [
            "home with pool",
            "modern condo downtown",
            "spacious kitchen",
            "ocean view property",
            "quiet neighborhood",
        ]
        for query in queries:
            start = time.perf_counter()
            searcher.search(query, top_k=10)
            elapsed_ms = (time.perf_counter() - start) * 1000
            assert elapsed_ms < 100, f"Query '{query}' took {elapsed_ms:.1f}ms (>100ms)"

    def test_bm25_search_under_100ms(self, searcher):
        queries = [
            "home with pool",
            "modern condo downtown",
            "spacious kitchen",
        ]
        for query in queries:
            start = time.perf_counter()
            searcher.bm25_search(query, top_k=10)
            elapsed_ms = (time.perf_counter() - start) * 1000
            assert elapsed_ms < 100, f"Query '{query}' took {elapsed_ms:.1f}ms (>100ms)"


class TestRelevanceEvaluation:
    """Tests for the relevance evaluation on 50 query-result pairs."""

    def test_evaluation_pair_count(self):
        pairs = generate_evaluation_pairs()
        assert len(pairs) == 50

    def test_evaluation_runs(self, searcher):
        pairs = generate_evaluation_pairs()
        results = searcher.evaluate_relevance(pairs, top_k=5)
        assert 'semantic_precision_at_k' in results
        assert 'bm25_precision_at_k' in results
        assert 'semantic_mrr' in results
        assert 'bm25_mrr' in results
        assert 'avg_overlap' in results
        assert results['num_queries'] == 50

    def test_precision_values_valid(self, searcher):
        pairs = generate_evaluation_pairs()
        results = searcher.evaluate_relevance(pairs, top_k=5)
        assert 0.0 <= results['semantic_precision_at_k'] <= 1.0
        assert 0.0 <= results['bm25_precision_at_k'] <= 1.0

    def test_mrr_values_valid(self, searcher):
        pairs = generate_evaluation_pairs()
        results = searcher.evaluate_relevance(pairs, top_k=5)
        assert 0.0 <= results['semantic_mrr'] <= 1.0
        assert 0.0 <= results['bm25_mrr'] <= 1.0


class TestBenchmarkLatency:
    """Tests for the latency benchmarking method."""

    def test_benchmark_runs(self, searcher):
        queries = ["home with pool", "modern condo"]
        results = searcher.benchmark_latency(queries, num_runs=2)
        assert 'semantic_avg_ms' in results
        assert 'bm25_avg_ms' in results
        assert results['semantic_avg_ms'] > 0
        assert results['bm25_avg_ms'] > 0
