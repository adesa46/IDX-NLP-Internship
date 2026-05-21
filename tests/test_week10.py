"""
Week 10 Tests: REST API with FastAPI

Tests for the FastAPI application covering:
- All 9 endpoints return correct status codes
- Request validation (invalid payloads → 422)
- Response schema validation via Pydantic models
- Entity extraction accuracy
- Compliance checking (violations detected / clean passes)
- Query parsing → correct filters and SQL
- Summarization returns non-empty summaries
- Signal extraction returns structured signals
- Rate limiting (429 after exceeding limit)
- Caching (repeated requests served from cache)
- Health check returns status and uptime
"""

import pytest
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

from fastapi.testclient import TestClient
from api_app import app, _cache, _rate_buckets


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """Create a TestClient for the FastAPI app."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def clear_rate_limits():
    """Clear rate limit buckets between tests."""
    _rate_buckets.clear()
    yield


# ═══════════════════════════════════════════════════════════════════════════
# Health & Stats
# ═══════════════════════════════════════════════════════════════════════════

class TestHealth:
    """Test the /health endpoint."""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_status(self, client):
        data = client.get("/health").json()
        assert data["status"] == "healthy"

    def test_health_has_uptime(self, client):
        data = client.get("/health").json()
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0

    def test_health_lists_endpoints(self, client):
        data = client.get("/health").json()
        assert "endpoints_available" in data
        assert "/health" in data["endpoints_available"]
        assert "/check-compliance" in data["endpoints_available"]

    def test_health_shows_search_status(self, client):
        data = client.get("/health").json()
        assert "search_available" in data
        assert isinstance(data["search_available"], bool)


class TestApiStats:
    """Test the /api-stats endpoint."""

    def test_stats_returns_200(self, client):
        response = client.get("/api-stats")
        assert response.status_code == 200

    def test_stats_has_cache_info(self, client):
        data = client.get("/api-stats").json()
        assert "cache_hits" in data
        assert "cache_misses" in data
        assert "cache_hit_rate" in data

    def test_stats_has_request_count(self, client):
        data = client.get("/api-stats").json()
        assert "total_requests" in data
        assert data["total_requests"] > 0


# ═══════════════════════════════════════════════════════════════════════════
# Parse Query
# ═══════════════════════════════════════════════════════════════════════════

class TestParseQuery:
    """Test the /parse-query endpoint."""

    def test_parse_query_returns_200(self, client):
        response = client.post("/parse-query", json={"query": "3 bed in Portland under 500k"})
        assert response.status_code == 200

    def test_parse_query_extracts_filters(self, client):
        data = client.post("/parse-query", json={"query": "3 bed in Portland under 500k"}).json()
        assert "filters" in data
        filters = data["filters"]
        assert filters.get("bedrooms") == 3 or filters.get("bedrooms_min") == 3
        assert "price_max" in filters

    def test_parse_query_generates_sql(self, client):
        data = client.post("/parse-query", json={"query": "3 bed under 500k"}).json()
        assert "sql" in data
        assert "SELECT" in data["sql"]

    def test_parse_query_returns_validation(self, client):
        data = client.post("/parse-query", json={"query": "3 bed house"}).json()
        assert "valid" in data
        assert isinstance(data["valid"], bool)

    def test_parse_query_empty_string_rejected(self, client):
        response = client.post("/parse-query", json={"query": ""})
        assert response.status_code == 422

    def test_parse_query_missing_field(self, client):
        response = client.post("/parse-query", json={})
        assert response.status_code == 422


# ═══════════════════════════════════════════════════════════════════════════
# Extract Entities
# ═══════════════════════════════════════════════════════════════════════════

class TestExtractEntities:
    """Test the /extract-entities endpoint."""

    def test_extract_entities_returns_200(self, client):
        response = client.post("/extract-entities", json={
            "text": "Beautiful 3 bedroom, 2 bath home with 2,500 sqft. $500,000."
        })
        assert response.status_code == 200

    def test_extract_entities_finds_bedrooms(self, client):
        data = client.post("/extract-entities", json={
            "text": "3 bedroom home"
        }).json()
        assert data["entities"]["bedrooms"] == 3

    def test_extract_entities_finds_bathrooms(self, client):
        data = client.post("/extract-entities", json={
            "text": "2.5 bath home"
        }).json()
        assert data["entities"]["bathrooms"] == 2.5

    def test_extract_entities_finds_price(self, client):
        data = client.post("/extract-entities", json={
            "text": "Priced at $1,250,000"
        }).json()
        assert data["entities"]["price"] is not None

    def test_extract_entities_finds_sqft(self, client):
        data = client.post("/extract-entities", json={
            "text": "2,500 sqft living space"
        }).json()
        assert data["entities"]["sqft"] == 2500

    def test_extract_entities_no_entities(self, client):
        data = client.post("/extract-entities", json={
            "text": "A nice place to live."
        }).json()
        assert "entities" in data

    def test_extract_entities_empty_rejected(self, client):
        response = client.post("/extract-entities", json={"text": ""})
        assert response.status_code == 422


# ═══════════════════════════════════════════════════════════════════════════
# Summarize
# ═══════════════════════════════════════════════════════════════════════════

class TestSummarize:
    """Test the /summarize endpoint."""

    def test_summarize_returns_200(self, client):
        response = client.post("/summarize", json={
            "text": (
                "Beautiful 3 bedroom, 2 bath home with sparkling pool. "
                "Updated kitchen with granite countertops. "
                "Large backyard perfect for entertaining. "
                "Close to top-rated schools and parks."
            )
        })
        assert response.status_code == 200

    def test_summarize_returns_summary(self, client):
        data = client.post("/summarize", json={
            "text": (
                "Beautiful 3 bedroom, 2 bath home with sparkling pool. "
                "Updated kitchen with granite countertops. "
                "Large backyard perfect for entertaining."
            )
        }).json()
        assert "summary" in data
        assert len(data["summary"]) > 0

    def test_summarize_returns_features(self, client):
        data = client.post("/summarize", json={
            "text": (
                "Beautiful 3 bedroom, 2 bath home with pool. "
                "Hardwood floors throughout. Near top schools."
            )
        }).json()
        assert "top_features" in data

    def test_summarize_custom_sentences(self, client):
        data = client.post("/summarize", json={
            "text": (
                "Sentence one about the home. "
                "Sentence two about the kitchen. "
                "Sentence three about the yard."
            ),
            "num_sentences": 1,
        }).json()
        assert "summary" in data

    def test_summarize_empty_rejected(self, client):
        response = client.post("/summarize", json={"text": ""})
        assert response.status_code == 422


# ═══════════════════════════════════════════════════════════════════════════
# Check Compliance
# ═══════════════════════════════════════════════════════════════════════════

class TestCheckCompliance:
    """Test the /check-compliance endpoint."""

    def test_compliance_returns_200(self, client):
        response = client.post("/check-compliance", json={
            "text": "Beautiful 3 bed, 2 bath home."
        })
        assert response.status_code == 200

    def test_clean_listing_compliant(self, client):
        data = client.post("/check-compliance", json={
            "text": "Beautiful 3 bed, 2 bath home with pool and garage."
        }).json()
        assert data["compliant"] is True
        assert len(data["violations"]) == 0

    def test_violation_detected(self, client):
        data = client.post("/check-compliance", json={
            "text": "No children allowed. Adults only."
        }).json()
        assert data["compliant"] is False
        assert len(data["violations"]) > 0

    def test_violation_has_category(self, client):
        data = client.post("/check-compliance", json={
            "text": "No children allowed."
        }).json()
        categories = [v["category"] for v in data["violations"]]
        assert "familial_status" in categories

    def test_violation_has_severity(self, client):
        data = client.post("/check-compliance", json={
            "text": "No children allowed."
        }).json()
        severities = [v["severity"] for v in data["violations"]]
        assert "error" in severities

    def test_stats_included(self, client):
        data = client.post("/check-compliance", json={
            "text": "No children. Must be able-bodied."
        }).json()
        assert "stats" in data
        assert data["stats"]["error"] >= 2

    def test_multiple_categories(self, client):
        data = client.post("/check-compliance", json={
            "text": "No children allowed. Must be able-bodied. White neighborhood."
        }).json()
        categories = set(v["category"] for v in data["violations"])
        assert len(categories) >= 3

    def test_compliance_empty_rejected(self, client):
        response = client.post("/check-compliance", json={"text": ""})
        assert response.status_code == 422


# ═══════════════════════════════════════════════════════════════════════════
# Classify Intent
# ═══════════════════════════════════════════════════════════════════════════

class TestClassifyIntent:
    """Test the /classify-intent endpoint."""

    def test_classify_intent_returns_200(self, client):
        response = client.post("/classify-intent", json={
            "query": "3 bed 2 bath under 500k in Portland"
        })
        # May be 503 if classifier not trained (no dataset)
        assert response.status_code in (200, 503)

    def test_classify_intent_has_label(self, client):
        response = client.post("/classify-intent", json={
            "query": "Show me homes in Portland"
        })
        if response.status_code == 200:
            data = response.json()
            assert data["intent"] in ("browsing", "researching", "ready_to_buy")

    def test_classify_intent_has_confidence(self, client):
        response = client.post("/classify-intent", json={
            "query": "I want to buy a 3 bed home immediately"
        })
        if response.status_code == 200:
            data = response.json()
            assert 0.0 <= data["confidence"] <= 1.0

    def test_classify_intent_has_probabilities(self, client):
        response = client.post("/classify-intent", json={
            "query": "What areas have good schools?"
        })
        if response.status_code == 200:
            data = response.json()
            assert "probabilities" in data
            probs = data["probabilities"]
            assert abs(sum(probs.values()) - 1.0) < 0.01


# ═══════════════════════════════════════════════════════════════════════════
# Extract Signals
# ═══════════════════════════════════════════════════════════════════════════

class TestExtractSignals:
    """Test the /extract-signals endpoint."""

    def test_extract_signals_returns_200(self, client):
        response = client.post("/extract-signals", json={
            "text": "Updated 3 bed home with pool. Near schools. FHA approved. Corner lot."
        })
        assert response.status_code == 200

    def test_extract_signals_has_entities(self, client):
        data = client.post("/extract-signals", json={
            "text": "3 bedroom, 2 bath home with 2,000 sqft."
        }).json()
        assert "entities" in data
        assert data["entities"]["bedrooms"] == 3

    def test_extract_signals_has_condition(self, client):
        data = client.post("/extract-signals", json={
            "text": "Newly renovated with updated kitchen and new roof."
        }).json()
        assert "condition_keywords" in data
        assert len(data["condition_keywords"]) > 0

    def test_extract_signals_has_financing(self, client):
        data = client.post("/extract-signals", json={
            "text": "Seller financing available. FHA approved."
        }).json()
        assert "financing_terms" in data
        assert len(data["financing_terms"]) > 0

    def test_extract_signals_has_location(self, client):
        data = client.post("/extract-signals", json={
            "text": "Quiet corner lot near park. Cul-de-sac location."
        }).json()
        assert "location_features" in data
        assert len(data["location_features"]) > 0

    def test_extract_signals_empty_rejected(self, client):
        response = client.post("/extract-signals", json={"text": ""})
        assert response.status_code == 422


# ═══════════════════════════════════════════════════════════════════════════
# Caching
# ═══════════════════════════════════════════════════════════════════════════

class TestCaching:
    """Verify response caching works."""

    def test_repeated_request_cached(self, client):
        payload = {"text": "No children allowed in this caching test listing."}
        # First request
        r1 = client.post("/check-compliance", json=payload)
        assert r1.status_code == 200
        assert r1.json()["cached"] is False

        # Second request — should be cached
        r2 = client.post("/check-compliance", json=payload)
        assert r2.status_code == 200
        assert r2.json()["cached"] is True

    def test_different_requests_not_cached(self, client):
        r1 = client.post("/extract-entities", json={
            "text": "3 bedroom home cache test A"
        })
        r2 = client.post("/extract-entities", json={
            "text": "5 bedroom home cache test B"
        })
        assert r1.status_code == 200
        assert r2.status_code == 200
        # Both should be cache misses (different text)
        assert r1.json()["cached"] is False
        assert r2.json()["cached"] is False


# ═══════════════════════════════════════════════════════════════════════════
# Rate Limiting
# ═══════════════════════════════════════════════════════════════════════════

class TestRateLimiting:
    """Verify per-IP rate limiting."""

    def test_rate_limit_triggers_429(self, client):
        _rate_buckets.clear()
        payload = {"text": "Rate limit test listing text."}

        # Send 10 requests (should all pass)
        for i in range(10):
            r = client.post("/check-compliance", json={
                "text": f"Rate limit test {i} unique text to avoid cache."
            })
            assert r.status_code == 200, f"Request {i+1} failed with {r.status_code}"

        # 11th request should be rate-limited
        r = client.post("/check-compliance", json={
            "text": "Rate limit test 11 unique text to avoid cache."
        })
        assert r.status_code == 429


# ═══════════════════════════════════════════════════════════════════════════
# Request Validation
# ═══════════════════════════════════════════════════════════════════════════

class TestRequestValidation:
    """Verify Pydantic validation rejects bad inputs."""

    def test_missing_body(self, client):
        response = client.post("/parse-query")
        assert response.status_code == 422

    def test_wrong_type(self, client):
        response = client.post("/parse-query", json={"query": 123})
        # FastAPI coerces int → str in some cases, so check it's handled
        assert response.status_code in (200, 422)

    def test_empty_string(self, client):
        response = client.post("/extract-entities", json={"text": ""})
        assert response.status_code == 422

    def test_top_k_out_of_range(self, client):
        response = client.post("/search", json={"query": "test", "top_k": 500})
        # Will be 422 (validation) or 503 (search not available)
        assert response.status_code in (422, 503)


# ═══════════════════════════════════════════════════════════════════════════
# Response Structure
# ═══════════════════════════════════════════════════════════════════════════

class TestResponseStructure:
    """Verify response structures match expected schemas."""

    def test_parse_query_structure(self, client):
        data = client.post("/parse-query", json={"query": "3 bed house"}).json()
        assert "query" in data
        assert "filters" in data
        assert "sql" in data
        assert "params" in data
        assert "valid" in data
        assert "cached" in data

    def test_extract_entities_structure(self, client):
        data = client.post("/extract-entities", json={"text": "3 bed 2 bath"}).json()
        assert "text" in data
        assert "entities" in data
        assert "cached" in data

    def test_compliance_structure(self, client):
        data = client.post("/check-compliance", json={"text": "Nice home."}).json()
        assert "text" in data
        assert "compliant" in data
        assert "violations" in data
        assert "stats" in data
        assert "cached" in data

    def test_summarize_structure(self, client):
        data = client.post("/summarize", json={
            "text": "Beautiful home with pool. Great kitchen."
        }).json()
        assert "text" in data
        assert "summary" in data
        assert "top_features" in data
        assert "cached" in data

    def test_signals_structure(self, client):
        data = client.post("/extract-signals", json={
            "text": "3 bed home with pool."
        }).json()
        assert "entities" in data
        assert "amenities" in data
        assert "condition_keywords" in data
        assert "financing_terms" in data
        assert "location_features" in data
        assert "cached" in data

    def test_health_structure(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "uptime_seconds" in data
        assert "search_available" in data
        assert "endpoints_available" in data

    def test_stats_structure(self, client):
        data = client.get("/api-stats").json()
        assert "uptime_seconds" in data
        assert "total_requests" in data
        assert "cache_size" in data
        assert "cache_hits" in data
        assert "cache_misses" in data
        assert "cache_hit_rate" in data


# ═══════════════════════════════════════════════════════════════════════════
# OpenAPI Docs
# ═══════════════════════════════════════════════════════════════════════════

class TestOpenAPIDocs:
    """Verify OpenAPI documentation is auto-generated."""

    def test_openapi_json(self, client):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert data["info"]["title"] == "Real Estate NLP API"

    def test_openapi_has_all_paths(self, client):
        data = client.get("/openapi.json").json()
        paths = list(data["paths"].keys())
        assert "/health" in paths
        assert "/parse-query" in paths
        assert "/extract-entities" in paths
        assert "/summarize" in paths
        assert "/check-compliance" in paths
        assert "/classify-intent" in paths
        assert "/extract-signals" in paths
        assert "/search" in paths
        assert "/api-stats" in paths

    def test_swagger_ui_accessible(self, client):
        response = client.get("/docs")
        assert response.status_code == 200
