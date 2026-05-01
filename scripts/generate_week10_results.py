"""
Generate Week 10 results summary for notebook visualization.

Runs all API endpoints via TestClient and produces
data/processed/week10_summary.json with endpoint inventory,
response times, cache performance, and sample responses.
"""

import json
import os
import sys
import time

# Ensure scripts/ is on sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from fastapi.testclient import TestClient
from api_app import app, _cache, _rate_buckets


def main():
    out_path = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'processed', 'week10_summary.json'
    )

    print("Generating Week 10 results...")

    with TestClient(app) as client:

        # ── Endpoint inventory & response times ────────────────────
        endpoints = [
            {"method": "GET",  "path": "/health",           "payload": None},
            {"method": "POST", "path": "/parse-query",      "payload": {"query": "3 bed in Portland under 500k"}},
            {"method": "POST", "path": "/extract-entities",  "payload": {"text": "Beautiful 3 bedroom, 2.5 bath home with 2,500 sqft. $1,250,000."}},
            {"method": "POST", "path": "/summarize",         "payload": {"text": "Beautiful 3 bedroom, 2 bath home with sparkling pool. Updated kitchen with granite countertops and stainless steel appliances. Large backyard perfect for entertaining friends and family. Close to top-rated schools and parks."}},
            {"method": "POST", "path": "/check-compliance",  "payload": {"text": "Beautiful 4 bed, 3 bath home with pool. Near top schools. $599,000."}},
            {"method": "POST", "path": "/classify-intent",   "payload": {"query": "3 bed 2 bath under 500k in Portland with pool"}},
            {"method": "POST", "path": "/extract-signals",   "payload": {"text": "Updated 3 bed home with pool and hardwood floors. Near schools. FHA approved. Corner lot in quiet cul-de-sac."}},
            {"method": "GET",  "path": "/api-stats",         "payload": None},
        ]

        endpoint_results = []
        sample_responses = {}

        for ep in endpoints:
            start = time.perf_counter()
            if ep["method"] == "GET":
                r = client.get(ep["path"])
            else:
                r = client.post(ep["path"], json=ep["payload"])
            elapsed = (time.perf_counter() - start) * 1000

            endpoint_results.append({
                "method": ep["method"],
                "path": ep["path"],
                "status_code": r.status_code,
                "response_time_ms": round(elapsed, 1),
            })

            if r.status_code == 200:
                sample_responses[ep["path"]] = r.json()

            print(f"  {ep['method']:4} {ep['path']:<25} -> {r.status_code} ({elapsed:.1f}ms)")

        # ── Cache performance test ────────────────────────────────
        print("\nTesting cache performance...")
        _cache.clear()
        cache_test_payload = {"text": "Cache test: 3 bed home with pool."}

        # First call — cache miss
        r1 = client.post("/extract-entities", json=cache_test_payload)
        r1_cached = r1.json().get("cached", False)

        # Second call — cache hit
        r2 = client.post("/extract-entities", json=cache_test_payload)
        r2_cached = r2.json().get("cached", False)

        cache_results = {
            "first_call_cached": r1_cached,
            "second_call_cached": r2_cached,
            "cache_working": (not r1_cached) and r2_cached,
        }
        print(f"  Cache working: {cache_results['cache_working']}")

        # ── Rate limiting test ────────────────────────────────────
        print("Testing rate limiting...")
        _rate_buckets.clear()

        rate_limit_passed = 0
        rate_limit_blocked = 0
        for i in range(15):
            r = client.post("/extract-entities", json={
                "text": f"Rate limit test {i} unique text."
            })
            if r.status_code == 200:
                rate_limit_passed += 1
            elif r.status_code == 429:
                rate_limit_blocked += 1

        rate_limit_results = {
            "requests_sent": 15,
            "requests_passed": rate_limit_passed,
            "requests_blocked": rate_limit_blocked,
            "rate_limiting_working": rate_limit_blocked > 0,
        }
        print(f"  Rate limiting working: {rate_limit_results['rate_limiting_working']}")
        print(f"  Passed: {rate_limit_passed}, Blocked: {rate_limit_blocked}")

        # ── Compliance demo ───────────────────────────────────────
        print("Running compliance demos...")
        _rate_buckets.clear()  # Reset rate limits after the rate limit test
        compliance_demos = [
            {"label": "Clean listing", "text": "Beautiful 4 bed, 3 bath home with pool. Near top schools. $599,000."},
            {"label": "Familial violation", "text": "No children allowed. Adults only community."},
            {"label": "Disability violation", "text": "Must be able-bodied. No wheelchairs."},
            {"label": "Race violation", "text": "Located in a white neighborhood."},
            {"label": "Multi-violation", "text": "No children. Must be able-bodied. English only. White community."},
        ]

        compliance_results = []
        for demo in compliance_demos:
            r = client.post("/check-compliance", json={"text": demo["text"]})
            if r.status_code == 200:
                data = r.json()
                compliance_results.append({
                    "label": demo["label"],
                    "text": demo["text"],
                    "compliant": data["compliant"],
                    "num_violations": len(data["violations"]),
                    "stats": data["stats"],
                })

        # ── Final stats ───────────────────────────────────────────
        _rate_buckets.clear()
        stats_r = client.get("/api-stats")
        final_stats = stats_r.json() if stats_r.status_code == 200 else {}

        # ── Get OpenAPI info ──────────────────────────────────────
        openapi_r = client.get("/openapi.json")
        openapi_paths = list(openapi_r.json().get("paths", {}).keys()) if openapi_r.status_code == 200 else []

    # ── Assemble summary ──────────────────────────────────────────
    summary = {
        "total_endpoints": len(endpoint_results),
        "all_endpoints_ok": all(e["status_code"] in (200, 503) for e in endpoint_results),
        "endpoints": endpoint_results,
        "sample_responses": sample_responses,
        "cache_results": cache_results,
        "rate_limit_results": rate_limit_results,
        "compliance_demos": compliance_results,
        "final_stats": final_stats,
        "openapi_paths": openapi_paths,
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nSaved summary to {out_path}")
    print(f"  Endpoints tested: {summary['total_endpoints']}")
    print(f"  All OK: {summary['all_endpoints_ok']}")
    print(f"  Cache working: {cache_results['cache_working']}")
    print(f"  Rate limiting working: {rate_limit_results['rate_limiting_working']}")


if __name__ == '__main__':
    main()
