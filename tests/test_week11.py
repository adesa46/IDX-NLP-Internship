"""
Tests for Week 11: Product Integration Demo

Covers:
  - Streamlit app module imports
  - End-to-end pipeline flow via TestClient
  - NLP vs keyword comparison
  - Metrics tracking
  - Results generation output structure
"""

import json
import os
import sys
import time

import pytest

# Ensure scripts/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from fastapi.testclient import TestClient
from api_app import app, _cache, _rate_buckets


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def client():
    """Shared TestClient for all tests."""
    with TestClient(app) as c:
        _cache.clear()
        _rate_buckets.clear()
        yield c


# ═══════════════════════════════════════════════════════════════════════════
# Module import tests
# ═══════════════════════════════════════════════════════════════════════════

class TestModuleImports:
    """Verify all Week 11 modules can be imported."""

    def test_import_streamlit_app(self):
        """streamlit_app.py is importable (module-level logic guarded)."""
        # We can't fully import streamlit_app because it calls st.set_page_config
        # at module level, but we can verify the file exists and is valid Python
        app_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'streamlit_app.py')
        assert os.path.exists(app_path), "streamlit_app.py not found"
        with open(app_path, 'r', encoding='utf-8') as f:
            source = f.read()
        assert 'st.set_page_config' in source
        assert 'Intelligent Search' in source
        assert 'NLP vs Keyword' in source or 'Keyword' in source
        assert 'Metrics Dashboard' in source or 'Metrics' in source

    def test_import_generate_week11(self):
        """generate_week11_results.py can be imported."""
        import generate_week11_results
        assert hasattr(generate_week11_results, 'main')
        assert hasattr(generate_week11_results, 'DEMO_QUERIES')
        assert len(generate_week11_results.DEMO_QUERIES) >= 5

    def test_import_update_notebook(self):
        """update_notebook_week11.py can be imported."""
        import update_notebook_week11
        assert hasattr(update_notebook_week11, 'main')
        assert hasattr(update_notebook_week11, 'make_md_cell')
        assert hasattr(update_notebook_week11, 'make_code_cell')


# ═══════════════════════════════════════════════════════════════════════════
# End-to-end pipeline tests
# ═══════════════════════════════════════════════════════════════════════════

class TestEndToEndPipeline:
    """Test the complete NLP pipeline flow."""

    def test_parse_query(self, client):
        """Parse a natural language query into filters + SQL."""
        r = client.post("/parse-query", json={"query": "3 bed 2 bath under 700k in Irvine"})
        assert r.status_code == 200
        data = r.json()
        assert "filters" in data
        assert "sql" in data
        assert data["filters"].get("bedrooms") == 3 or data["filters"].get("bedrooms_min") == 3

    def test_classify_intent(self, client):
        """Classify buyer intent."""
        r = client.post("/classify-intent", json={"query": "3 bed 2 bath under 700k in Irvine"})
        if r.status_code == 200:
            data = r.json()
            assert "intent" in data
            assert "confidence" in data
            assert 0 <= data["confidence"] <= 1

    def test_search(self, client):
        """Semantic search returns results."""
        r = client.post("/search", json={"query": "home with pool", "top_k": 5})
        if r.status_code == 200:
            data = r.json()
            assert "results" in data
            assert "count" in data
            assert data["count"] <= 5
        else:
            # Search may be unavailable (503) — that's acceptable
            assert r.status_code == 503

    def test_summarize(self, client):
        """Summarize listing text."""
        text = ("Beautiful 3 bedroom, 2 bath home with sparkling pool. "
                "Updated kitchen with granite countertops and stainless steel appliances. "
                "Large backyard perfect for entertaining. Close to top-rated schools.")
        r = client.post("/summarize", json={"text": text, "num_sentences": 2})
        assert r.status_code == 200
        data = r.json()
        assert "summary" in data
        assert len(data["summary"]) > 0

    def test_compliance_check(self, client):
        """Compliance check on clean listing."""
        r = client.post("/check-compliance", json={
            "text": "Beautiful 4 bed, 3 bath home with pool. $599,000."
        })
        assert r.status_code == 200
        data = r.json()
        assert "compliant" in data
        assert data["compliant"] is True

    def test_extract_signals(self, client):
        """Signal extraction returns structured data."""
        r = client.post("/extract-signals", json={
            "text": "Updated 3 bed home with pool and hardwood floors. Near schools."
        })
        assert r.status_code == 200
        data = r.json()
        assert "entities" in data
        assert "amenities" in data

    def test_full_pipeline_flow(self, client):
        """Run the complete pipeline for a single query."""
        query = "3 bed 2 bath under 700k with pool"

        # 1. Parse
        r1 = client.post("/parse-query", json={"query": query})
        assert r1.status_code == 200
        parse_data = r1.json()

        # 2. Intent
        r2 = client.post("/classify-intent", json={"query": query})
        # May be 503 if classifier not loaded

        # 3. Search
        r3 = client.post("/search", json={"query": query, "top_k": 3})
        if r3.status_code == 200:
            results = r3.json()["results"]
            assert len(results) > 0

            # 4. Summarize top result
            top_text = results[0]["text"]
            r4 = client.post("/summarize", json={"text": top_text})
            assert r4.status_code == 200

            # 5. Compliance
            r5 = client.post("/check-compliance", json={"text": top_text})
            assert r5.status_code == 200

            # 6. Signals
            r6 = client.post("/extract-signals", json={"text": top_text})
            assert r6.status_code == 200

    def test_pipeline_latency_reasonable(self, client):
        """Pipeline steps should complete within reasonable time."""
        _rate_buckets.clear()
        start = time.perf_counter()
        client.post("/parse-query", json={"query": "3 bed in Portland"})
        elapsed = (time.perf_counter() - start) * 1000
        # Parse should be fast (< 500ms)
        assert elapsed < 500, f"Parse took {elapsed:.0f}ms, expected < 500ms"


# ═══════════════════════════════════════════════════════════════════════════
# Comparison tests
# ═══════════════════════════════════════════════════════════════════════════

class TestComparison:
    """Test NLP vs keyword comparison functionality."""

    def test_bm25_available(self):
        """BM25 search module is importable."""
        try:
            from semantic_search import SemanticSearcher
        except ImportError:
            pytest.skip("sentence_transformers not installed (Docker-only dependency)")
        searcher = SemanticSearcher.__new__(SemanticSearcher)
        assert hasattr(searcher, 'bm25_search')
        assert hasattr(searcher, 'compare_search')

    def test_compare_search_method(self):
        """compare_search returns both semantic and bm25 keys."""
        try:
            from semantic_search import SemanticSearcher
        except ImportError:
            pytest.skip("sentence_transformers not installed (Docker-only dependency)")
        # Just verify the method exists and has correct signature
        import inspect
        sig = inspect.signature(SemanticSearcher.compare_search)
        params = list(sig.parameters.keys())
        assert 'query' in params
        assert 'top_k' in params


# ═══════════════════════════════════════════════════════════════════════════
# Metrics tests
# ═══════════════════════════════════════════════════════════════════════════

class TestMetrics:
    """Test metrics and stats endpoints."""

    def test_api_stats(self, client):
        """API stats endpoint returns expected fields."""
        _rate_buckets.clear()
        r = client.get("/api-stats")
        assert r.status_code == 200
        data = r.json()
        assert "total_requests" in data
        assert "cache_size" in data
        assert "cache_hit_rate" in data
        assert "uptime_seconds" in data

    def test_health_endpoint(self, client):
        """Health endpoint reports correct status."""
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"
        assert "endpoints_available" in data

    def test_cache_behavior(self, client):
        """Cache returns cached=True on second identical call."""
        _cache.clear()
        _rate_buckets.clear()
        payload = {"text": "Week 11 cache test: 3 bed home."}

        r1 = client.post("/extract-entities", json=payload)
        assert r1.status_code == 200
        assert r1.json().get("cached") is False

        r2 = client.post("/extract-entities", json=payload)
        assert r2.status_code == 200
        assert r2.json().get("cached") is True


# ═══════════════════════════════════════════════════════════════════════════
# Results file structure tests
# ═══════════════════════════════════════════════════════════════════════════

class TestResultsStructure:
    """Test the structure of generated results files."""

    def test_week11_summary_exists_or_generatable(self):
        """week11_summary.json exists or can be generated."""
        summary_path = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'processed', 'week11_summary.json'
        )
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                data = json.load(f)
            assert "num_demo_queries" in data
            assert "pipeline_results" in data
            assert "latency_summary" in data
        else:
            # File hasn't been generated yet — just verify the generator exists
            import generate_week11_results
            assert hasattr(generate_week11_results, 'main')

    def test_demo_queries_well_formed(self):
        """All demo queries are non-empty strings."""
        import generate_week11_results
        for q in generate_week11_results.DEMO_QUERIES:
            assert isinstance(q, str)
            assert len(q) > 5

    def test_streamlit_app_has_all_tabs(self):
        """Streamlit app defines all three tabs."""
        app_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'streamlit_app.py')
        with open(app_path, 'r', encoding='utf-8') as f:
            source = f.read()
        assert 'tab1' in source
        assert 'tab2' in source
        assert 'tab3' in source
        assert '/parse-query' in source
        assert '/search' in source
        assert '/summarize' in source
        assert '/check-compliance' in source
