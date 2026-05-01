"""
Week 10: REST API with FastAPI

Production-ready REST API exposing all NLP capabilities:
  - /search           — Semantic search over listing remarks
  - /parse-query      — Parse NL query → structured filters + SQL
  - /extract-entities — Extract bed/bath/price/sqft/amenities
  - /summarize        — Extractive listing summarization
  - /check-compliance — Fair Housing compliance check
  - /classify-intent  — Buyer intent classification
  - /extract-signals  — Full signal extraction
  - /health           — Health / liveness probe
  - /api-stats        — Cache & request statistics

Features:
  - Pydantic request/response models with validation
  - In-memory TTL caching (cachetools)
  - Per-IP rate limiting (10 req/s)
  - Structured logging
  - Auto-generated OpenAPI docs at /docs
"""

import os
import sys
import time
import json
import hashlib
import logging
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from cachetools import TTLCache

# ---------------------------------------------------------------------------
# Ensure scripts/ is on sys.path so sibling modules can be imported
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# Project root (one level up from scripts/)
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..'))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("nlp_api")

# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated @app.on_event)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(application: FastAPI):
    """Startup / shutdown lifecycle hook."""
    await _load_models()
    yield
    # Shutdown: nothing special needed


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Real Estate NLP API",
    description=(
        "Production-ready REST API exposing search, entity extraction, "
        "query parsing, summarization, compliance checking, intent "
        "classification, and signal extraction for real estate listings."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_startup_time: float = time.time()
_request_count: int = 0

# Cache: 512 entries, 5-minute TTL
_cache = TTLCache(maxsize=512, ttl=300)
_cache_hits: int = 0
_cache_misses: int = 0

# Rate limiting: per-IP tracking
_rate_limit_window = 1.0       # seconds
_rate_limit_max = 10           # requests per window per IP
_rate_buckets: dict = defaultdict(list)  # IP -> list of timestamps

# Module holders (lazily loaded)
_query_parser = None
_schema_validator = None
_entity_extractor = None
_listing_summarizer = None
_compliance_checker = None
_intent_classifier = None
_signal_extractor = None
_semantic_searcher = None
_search_available = False


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic Models
# ═══════════════════════════════════════════════════════════════════════════

# --- Search ---
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query text")
    top_k: int = Field(10, ge=1, le=100, description="Number of results to return")

class SearchResult(BaseModel):
    text: str
    score: float

class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    count: int
    cached: bool = False

# --- Parse Query ---
class ParseQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language query")

class ParseQueryResponse(BaseModel):
    query: str
    filters: dict
    sql: str
    params: list
    valid: bool
    validation_errors: list[str] = []
    cached: bool = False

# --- Extract Entities ---
class ExtractEntitiesRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Listing text to extract entities from")

class ExtractEntitiesResponse(BaseModel):
    text: str
    entities: dict
    cached: bool = False

# --- Summarize ---
class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Listing remarks text")
    num_sentences: int = Field(2, ge=1, le=10, description="Number of sentences in summary")

class SummarizeResponse(BaseModel):
    text: str
    summary: str
    beds: Optional[int] = None
    baths: Optional[float] = None
    price: Optional[int] = None
    sqft: Optional[int] = None
    top_features: list[str] = []
    location: Optional[str] = None
    cached: bool = False

# --- Check Compliance ---
class CheckComplianceRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Listing text to check for compliance")

class ComplianceViolation(BaseModel):
    category: str
    pattern: str
    severity: str
    message: str
    position: int
    matched_text: str

class CheckComplianceResponse(BaseModel):
    text: str
    compliant: bool
    violations: list[ComplianceViolation]
    stats: dict
    cached: bool = False

# --- Classify Intent ---
class ClassifyIntentRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Buyer query to classify")

class ClassifyIntentResponse(BaseModel):
    query: str
    intent: str
    confidence: float
    probabilities: dict
    cached: bool = False

# --- Extract Signals ---
class ExtractSignalsRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Listing text for signal extraction")

class ExtractSignalsResponse(BaseModel):
    text: str
    entities: dict
    amenities: list[str]
    condition_keywords: list[str]
    financing_terms: list[str]
    location_features: list[str]
    cached: bool = False

# --- Health ---
class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    search_available: bool
    endpoints_available: list[str]

# --- API Stats ---
class ApiStatsResponse(BaseModel):
    uptime_seconds: float
    total_requests: int
    cache_size: int
    cache_hits: int
    cache_misses: int
    cache_hit_rate: float
    search_available: bool


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _cache_key(prefix: str, data: str) -> str:
    """Generate a deterministic cache key."""
    h = hashlib.md5(data.encode()).hexdigest()
    return f"{prefix}:{h}"


def _check_rate_limit(client_ip: str) -> bool:
    """Return True if request is allowed, False if rate-limited."""
    now = time.time()
    # Prune old timestamps
    _rate_buckets[client_ip] = [
        t for t in _rate_buckets[client_ip]
        if now - t < _rate_limit_window
    ]
    if len(_rate_buckets[client_ip]) >= _rate_limit_max:
        return False
    _rate_buckets[client_ip].append(now)
    return True


# ═══════════════════════════════════════════════════════════════════════════
# Middleware
# ═══════════════════════════════════════════════════════════════════════════

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Per-IP rate limiting middleware."""
    global _request_count
    _request_count += 1

    client_ip = request.client.host if request.client else "unknown"

    # Skip rate limiting for docs and health
    if request.url.path in ("/docs", "/redoc", "/openapi.json", "/health"):
        return await call_next(request)

    if not _check_rate_limit(client_ip):
        logger.warning(f"Rate limit exceeded for {client_ip}")
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Max 10 requests per second."},
        )

    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000

    logger.info(
        f"{request.method} {request.url.path} → {response.status_code} "
        f"({elapsed:.1f}ms) [IP: {client_ip}]"
    )
    return response


# ═══════════════════════════════════════════════════════════════════════════
# Startup
# ═══════════════════════════════════════════════════════════════════════════

async def _load_models():
    """Load all NLP modules on startup."""
    global _query_parser, _schema_validator, _entity_extractor
    global _listing_summarizer, _compliance_checker
    global _intent_classifier, _signal_extractor
    global _semantic_searcher, _search_available, _startup_time

    _startup_time = time.time()
    logger.info("Loading NLP modules...")

    # ── Lightweight modules (always available) ──────────────────
    try:
        from query_parser import QueryParser, SchemaValidator
        _query_parser = QueryParser()
        schema_path = os.path.join(_PROJECT_ROOT, 'data', 'schema.json')
        if os.path.exists(schema_path):
            _schema_validator = SchemaValidator(schema_path)
        else:
            _schema_validator = None
        logger.info("  ✓ QueryParser loaded")
    except Exception as e:
        logger.error(f"  ✗ QueryParser: {e}")

    try:
        taxonomy_path = os.path.join(_PROJECT_ROOT, 'data', 'processed', 'taxonomy.json')
        from entity_extractor import EntityExtractor
        _entity_extractor = EntityExtractor(taxonomy_path)
        logger.info("  ✓ EntityExtractor loaded")
    except Exception as e:
        logger.error(f"  ✗ EntityExtractor: {e}")

    try:
        from listing_summarizer import ListingSummarizer
        taxonomy_path = os.path.join(_PROJECT_ROOT, 'data', 'processed', 'taxonomy.json')
        _listing_summarizer = ListingSummarizer(
            taxonomy_path=taxonomy_path,
            entity_extractor=_entity_extractor,
        )
        logger.info("  ✓ ListingSummarizer loaded")
    except Exception as e:
        logger.error(f"  ✗ ListingSummarizer: {e}")

    try:
        from compliance_checker import ComplianceChecker
        _compliance_checker = ComplianceChecker()
        logger.info("  ✓ ComplianceChecker loaded")
    except Exception as e:
        logger.error(f"  ✗ ComplianceChecker: {e}")

    try:
        from signal_extractor import SignalExtractor
        taxonomy_path = os.path.join(_PROJECT_ROOT, 'data', 'processed', 'taxonomy.json')
        _signal_extractor = SignalExtractor(
            taxonomy_path=taxonomy_path,
            entity_extractor=_entity_extractor,
        )
        logger.info("  ✓ SignalExtractor loaded")
    except Exception as e:
        logger.error(f"  ✗ SignalExtractor: {e}")

    # ── Intent classifier (needs training data) ────────────────
    try:
        from intent_classifier import IntentClassifier
        dataset_path = os.path.join(_PROJECT_ROOT, 'data', 'processed', 'intent_dataset.json')
        if os.path.exists(dataset_path):
            with open(dataset_path, 'r') as f:
                dataset = json.load(f)
            queries = [d['query'] for d in dataset]
            labels = [d['intent'] for d in dataset]
            _intent_classifier = IntentClassifier()
            _intent_classifier.train(queries, labels)
            logger.info("  ✓ IntentClassifier loaded & trained")
        else:
            logger.warning("  ⚠ IntentClassifier: no training data found")
    except Exception as e:
        logger.error(f"  ✗ IntentClassifier: {e}")

    # ── Semantic search (heavy — may not be available) ─────────
    try:
        from semantic_search import SemanticSearcher
        import pandas as pd
        csv_path = os.path.join(_PROJECT_ROOT, 'data', 'processed', 'cleaned_listing.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            text_col = None
            for col in ('cleaned_remarks', 'remarks', 'L_Remarks'):
                if col in df.columns:
                    text_col = col
                    break
            if text_col:
                remarks = df[text_col].dropna().astype(str).tolist()
                remarks = [r for r in remarks if len(r) > 50 and ' ' in r]
                _semantic_searcher = SemanticSearcher()
                _semantic_searcher.build_index(remarks)
                _search_available = True
                logger.info(f"  ✓ SemanticSearcher loaded ({len(remarks)} listings indexed)")
            else:
                logger.warning("  ⚠ SemanticSearcher: no text column found")
        else:
            logger.warning("  ⚠ SemanticSearcher: no dataset CSV found")
    except Exception as e:
        logger.warning(f"  ⚠ SemanticSearcher: {e} (search disabled)")

    logger.info("Startup complete.")


# ═══════════════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════════════

# ── 1. Health ──────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check / liveness probe."""
    endpoints = ["/health", "/parse-query", "/extract-entities",
                 "/summarize", "/check-compliance", "/extract-signals",
                 "/classify-intent", "/api-stats"]
    if _search_available:
        endpoints.append("/search")

    return HealthResponse(
        status="healthy",
        uptime_seconds=round(time.time() - _startup_time, 2),
        search_available=_search_available,
        endpoints_available=sorted(endpoints),
    )


# ── 2. Search ─────────────────────────────────────────────────────────────

@app.post("/search", response_model=SearchResponse, tags=["NLP"])
async def search_listings(request: SearchRequest):
    """Semantic search over listing remarks using FAISS embeddings."""
    if not _search_available or _semantic_searcher is None:
        raise HTTPException(
            status_code=503,
            detail="Search is not available. Semantic index not loaded.",
        )

    global _cache_hits, _cache_misses
    key = _cache_key("search", f"{request.query}:{request.top_k}")

    if key in _cache:
        _cache_hits += 1
        cached = _cache[key]
        cached["cached"] = True
        return SearchResponse(**cached)

    _cache_misses += 1
    results = _semantic_searcher.search(request.query, request.top_k)
    result_list = [SearchResult(text=text, score=round(score, 4)) for text, score in results]

    response_data = {
        "query": request.query,
        "results": result_list,
        "count": len(result_list),
        "cached": False,
    }
    _cache[key] = response_data
    return SearchResponse(**response_data)


# ── 3. Parse Query ────────────────────────────────────────────────────────

@app.post("/parse-query", response_model=ParseQueryResponse, tags=["NLP"])
async def parse_query(request: ParseQueryRequest):
    """Parse a natural language query into structured filters and SQL."""
    if _query_parser is None:
        raise HTTPException(status_code=503, detail="QueryParser not loaded.")

    global _cache_hits, _cache_misses
    key = _cache_key("parse", request.query)

    if key in _cache:
        _cache_hits += 1
        cached = _cache[key]
        cached["cached"] = True
        return ParseQueryResponse(**cached)

    _cache_misses += 1
    filters = _query_parser.parse(request.query)
    sql, params = _query_parser.to_sql(filters)

    valid = True
    validation_errors = []
    if _schema_validator:
        valid, validation_errors = _schema_validator.validate_query(filters)

    response_data = {
        "query": request.query,
        "filters": filters,
        "sql": sql,
        "params": params,
        "valid": valid,
        "validation_errors": validation_errors,
        "cached": False,
    }
    _cache[key] = response_data
    return ParseQueryResponse(**response_data)


# ── 4. Extract Entities ──────────────────────────────────────────────────

@app.post("/extract-entities", response_model=ExtractEntitiesResponse, tags=["NLP"])
async def extract_entities(request: ExtractEntitiesRequest):
    """Extract structured entities (beds, baths, price, sqft, amenities)."""
    if _entity_extractor is None:
        raise HTTPException(status_code=503, detail="EntityExtractor not loaded.")

    global _cache_hits, _cache_misses
    key = _cache_key("entities", request.text)

    if key in _cache:
        _cache_hits += 1
        cached = _cache[key]
        cached["cached"] = True
        return ExtractEntitiesResponse(**cached)

    _cache_misses += 1
    entities = _entity_extractor.extract_all(request.text)

    response_data = {
        "text": request.text,
        "entities": entities,
        "cached": False,
    }
    _cache[key] = response_data
    return ExtractEntitiesResponse(**response_data)


# ── 5. Summarize ─────────────────────────────────────────────────────────

@app.post("/summarize", response_model=SummarizeResponse, tags=["NLP"])
async def summarize_listing(request: SummarizeRequest):
    """Generate an extractive summary of listing remarks."""
    if _listing_summarizer is None:
        raise HTTPException(status_code=503, detail="ListingSummarizer not loaded.")

    global _cache_hits, _cache_misses
    key = _cache_key("summarize", f"{request.text}:{request.num_sentences}")

    if key in _cache:
        _cache_hits += 1
        cached = _cache[key]
        cached["cached"] = True
        return SummarizeResponse(**cached)

    _cache_misses += 1
    result = _listing_summarizer.summarize_listing({"remarks": request.text})

    response_data = {
        "text": request.text,
        "summary": result.get("summary", ""),
        "beds": result.get("beds"),
        "baths": result.get("baths"),
        "price": result.get("price"),
        "sqft": result.get("sqft"),
        "top_features": result.get("top_features", []),
        "location": result.get("location"),
        "cached": False,
    }
    _cache[key] = response_data
    return SummarizeResponse(**response_data)


# ── 6. Check Compliance ──────────────────────────────────────────────────

@app.post("/check-compliance", response_model=CheckComplianceResponse, tags=["NLP"])
async def check_compliance(request: CheckComplianceRequest):
    """Check listing text for Fair Housing Act compliance violations."""
    if _compliance_checker is None:
        raise HTTPException(status_code=503, detail="ComplianceChecker not loaded.")

    global _cache_hits, _cache_misses
    key = _cache_key("compliance", request.text)

    if key in _cache:
        _cache_hits += 1
        cached = _cache[key]
        cached["cached"] = True
        return CheckComplianceResponse(**cached)

    _cache_misses += 1
    result = _compliance_checker.check_listing(request.text)

    violations = [
        ComplianceViolation(**v) for v in result["violations"]
    ]

    response_data = {
        "text": request.text,
        "compliant": result["compliant"],
        "violations": violations,
        "stats": result["stats"],
        "cached": False,
    }
    _cache[key] = response_data
    return CheckComplianceResponse(**response_data)


# ── 7. Classify Intent ───────────────────────────────────────────────────

@app.post("/classify-intent", response_model=ClassifyIntentResponse, tags=["NLP"])
async def classify_intent(request: ClassifyIntentRequest):
    """Classify buyer intent (browsing, researching, ready_to_buy)."""
    if _intent_classifier is None:
        raise HTTPException(status_code=503, detail="IntentClassifier not loaded.")

    global _cache_hits, _cache_misses
    key = _cache_key("intent", request.query)

    if key in _cache:
        _cache_hits += 1
        cached = _cache[key]
        cached["cached"] = True
        return ClassifyIntentResponse(**cached)

    _cache_misses += 1
    intent, confidence = _intent_classifier.predict(request.query)
    probas = _intent_classifier.predict_proba(request.query)

    response_data = {
        "query": request.query,
        "intent": intent,
        "confidence": round(confidence, 4),
        "probabilities": {k: round(v, 4) for k, v in probas.items()},
        "cached": False,
    }
    _cache[key] = response_data
    return ClassifyIntentResponse(**response_data)


# ── 8. Extract Signals ───────────────────────────────────────────────────

@app.post("/extract-signals", response_model=ExtractSignalsResponse, tags=["NLP"])
async def extract_signals(request: ExtractSignalsRequest):
    """Extract comprehensive structured signals from listing text."""
    if _signal_extractor is None:
        raise HTTPException(status_code=503, detail="SignalExtractor not loaded.")

    global _cache_hits, _cache_misses
    key = _cache_key("signals", request.text)

    if key in _cache:
        _cache_hits += 1
        cached = _cache[key]
        cached["cached"] = True
        return ExtractSignalsResponse(**cached)

    _cache_misses += 1
    result = _signal_extractor.extract_signals({"remarks": request.text})

    response_data = {
        "text": request.text,
        "entities": result.get("entities", {}),
        "amenities": result.get("amenities", []),
        "condition_keywords": result.get("condition_keywords", []),
        "financing_terms": result.get("financing_terms", []),
        "location_features": result.get("location_features", []),
        "cached": False,
    }
    _cache[key] = response_data
    return ExtractSignalsResponse(**response_data)


# ── 9. API Stats ─────────────────────────────────────────────────────────

@app.get("/api-stats", response_model=ApiStatsResponse, tags=["System"])
async def api_stats():
    """Return API usage statistics: cache, requests, uptime."""
    total_cache_ops = _cache_hits + _cache_misses
    hit_rate = (_cache_hits / total_cache_ops) if total_cache_ops > 0 else 0.0

    return ApiStatsResponse(
        uptime_seconds=round(time.time() - _startup_time, 2),
        total_requests=_request_count,
        cache_size=len(_cache),
        cache_hits=_cache_hits,
        cache_misses=_cache_misses,
        cache_hit_rate=round(hit_rate, 4),
        search_available=_search_available,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_app:app", host="0.0.0.0", port=8000, reload=True)
