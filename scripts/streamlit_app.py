"""
Week 11: Product Integration Demo — Streamlit Web UI

Three-tab interface consuming the Week 10 FastAPI backend:
  Tab 1 — Intelligent Search: full NLP pipeline demo
  Tab 2 — NLP vs Keyword Comparison: semantic vs BM25 side-by-side
  Tab 3 — Metrics Dashboard: latency, cache, query volume
"""

import os
import sys
import time
import json

import streamlit as st
import requests as http_requests

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Real Estate Intelligent Search",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "latency_log" not in st.session_state:
    st.session_state.latency_log = []
if "query_count" not in st.session_state:
    st.session_state.query_count = 0

# ---------------------------------------------------------------------------
# API helper
# ---------------------------------------------------------------------------
API_URL = st.sidebar.text_input("API URL", value="http://localhost:8000")


def _use_testclient():
    """Return True when the live API is unreachable."""
    try:
        r = http_requests.get(f"{API_URL}/health", timeout=2)
        return r.status_code != 200
    except Exception:
        return True


_FALLBACK = _use_testclient()

_tc = None  # lazily loaded TestClient


def _get_testclient():
    global _tc
    if _tc is None:
        from fastapi.testclient import TestClient
        from api_app import app
        _tc = TestClient(app)
        _tc.__enter__()
    return _tc


def api_get(path: str) -> dict:
    if _FALLBACK:
        r = _get_testclient().get(path)
    else:
        r = http_requests.get(f"{API_URL}{path}", timeout=10)
    return r.json() if r.status_code == 200 else {"error": r.text, "status": r.status_code}


def api_post(path: str, payload: dict) -> dict:
    start = time.perf_counter()
    if _FALLBACK:
        r = _get_testclient().post(path, json=payload)
    else:
        r = http_requests.post(f"{API_URL}{path}", json=payload, timeout=30)
    elapsed = (time.perf_counter() - start) * 1000
    st.session_state.latency_log.append({"path": path, "ms": round(elapsed, 1)})
    if r.status_code == 200:
        return r.json()
    return {"error": r.text, "status": r.status_code}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown("## 🏠 Real Estate NLP")
health = api_get("/health")
if "error" not in health:
    st.sidebar.success(f"API: **{health['status']}**")
    st.sidebar.caption(f"Uptime: {health.get('uptime_seconds', 0):.0f}s")
    st.sidebar.caption(f"Search: {'✅' if health.get('search_available') else '❌'}")
    st.sidebar.caption(f"Endpoints: {len(health.get('endpoints_available', []))}")
else:
    st.sidebar.warning("API offline — using local fallback")

mode_label = "TestClient (local)" if _FALLBACK else f"HTTP → {API_URL}"
st.sidebar.caption(f"Mode: {mode_label}")

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Queries this session:** {st.session_state.query_count}")

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .result-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px; padding: 20px; margin-bottom: 16px;
        border-left: 4px solid #0f3460; color: #e0e0e0;
    }
    .result-card h4 { color: #e94560; margin: 0 0 8px 0; }
    .result-card .price { color: #0ff; font-size: 1.3em; font-weight: bold; }
    .metric-badge {
        display: inline-block; padding: 3px 10px; border-radius: 20px;
        font-size: 0.85em; margin-right: 6px; font-weight: 600;
    }
    .badge-green { background: #0d7337; color: #fff; }
    .badge-red   { background: #c0392b; color: #fff; }
    .badge-blue  { background: #2980b9; color: #fff; }
    .badge-purple { background: #8e44ad; color: #fff; }
    .tag { display: inline-block; background: #2c3e50; color: #ecf0f1;
           padding: 2px 8px; border-radius: 10px; font-size: 0.8em;
           margin: 2px 4px 2px 0; }
    .compare-header { text-align: center; padding: 8px;
                      border-radius: 8px; margin-bottom: 12px; font-weight: bold; }
    .sem-header { background: linear-gradient(135deg, #0f3460, #533483); color: white; }
    .bm25-header { background: linear-gradient(135deg, #b33939, #cd6133); color: white; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "🔍 Intelligent Search",
    "⚖️ NLP vs Keyword",
    "📊 Metrics Dashboard",
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: Intelligent Search
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("🔍 Real Estate Intelligent Search")
    st.caption("Natural language → parsed filters → semantic results → summaries")

    examples = [
        "3 bed 2 bath under 700k in Irvine",
        "modern condo with pool near schools",
        "spacious home with large backyard and garage",
        "waterfront property with ocean views",
        "4 bed house with updated kitchen under 500k",
    ]
    st.markdown("**Try:** " + " · ".join(f"`{e}`" for e in examples))

    query = st.text_input("What are you looking for?", value=examples[0], key="main_query")

    if st.button("🚀 Search", type="primary", key="search_btn"):
        st.session_state.query_count += 1
        st.session_state.query_history.append(query)

        # ── Step 1: Parse query ──────────────────────────────────
        with st.spinner("Parsing query..."):
            parse = api_post("/parse-query", {"query": query})

        if "error" not in parse:
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("📋 Parsed Filters")
                st.json(parse.get("filters", {}))
            with col_b:
                st.subheader("🗃️ Generated SQL")
                st.code(parse.get("sql", ""), language="sql")
                if not parse.get("valid", True):
                    for err in parse.get("validation_errors", []):
                        st.warning(err)

        # ── Step 2: Classify intent ──────────────────────────────
        with st.spinner("Classifying intent..."):
            intent = api_post("/classify-intent", {"query": query})

        if "error" not in intent:
            label = intent.get("intent", "unknown")
            conf = intent.get("confidence", 0)
            badge_cls = "badge-green" if conf > 0.7 else "badge-blue"
            st.markdown(
                f'**Buyer Intent:** <span class="metric-badge {badge_cls}">'
                f'{label} ({conf:.0%})</span>',
                unsafe_allow_html=True,
            )

        # ── Step 3: Semantic search ──────────────────────────────
        with st.spinner("Searching listings..."):
            search = api_post("/search", {"query": query, "top_k": 10})

        if "error" in search:
            st.error(f"Search unavailable: {search.get('error', 'unknown')}")
        else:
            results = search.get("results", [])
            st.success(f"Found **{search.get('count', len(results))}** results")

            for idx, item in enumerate(results):
                text = item.get("text", "")
                score = item.get("score", 0)

                # ── Step 4: Summarize ────────────────────────────
                summ = api_post("/summarize", {"text": text, "num_sentences": 2})
                summary = summ.get("summary", text[:200]) if "error" not in summ else text[:200]
                beds = summ.get("beds") if "error" not in summ else None
                baths = summ.get("baths") if "error" not in summ else None
                price = summ.get("price") if "error" not in summ else None
                features = summ.get("top_features", []) if "error" not in summ else []

                # ── Step 5: Compliance ───────────────────────────
                comp = api_post("/check-compliance", {"text": text})
                compliant = comp.get("compliant", True) if "error" not in comp else True

                # ── Render card ──────────────────────────────────
                comp_badge = ('<span class="metric-badge badge-green">✓ Compliant</span>'
                              if compliant else
                              '<span class="metric-badge badge-red">⚠ Violation</span>')

                price_str = f"${price:,}" if price else "N/A"
                beds_str = f"{beds} bd" if beds else ""
                baths_str = f"{baths} ba" if baths else ""
                detail_parts = [p for p in [beds_str, baths_str] if p]
                detail_line = " · ".join(detail_parts) if detail_parts else ""

                tags_html = "".join(f'<span class="tag">{f}</span>' for f in features)

                st.markdown(f"""
                <div class="result-card">
                    <h4>Result #{idx+1} <small>(score: {score:.4f})</small></h4>
                    <span class="price">{price_str}</span>
                    &nbsp; {detail_line} &nbsp; {comp_badge}
                    <p style="margin-top:8px;">{summary}</p>
                    {tags_html}
                </div>
                """, unsafe_allow_html=True)

                # Expandable full text
                with st.expander(f"Full listing text #{idx+1}"):
                    st.write(text)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: NLP vs Keyword Comparison
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("⚖️ Semantic Search vs Keyword Search")
    st.caption("Side-by-side comparison of FAISS embeddings vs BM25")

    cmp_query = st.text_input(
        "Compare query:", value="home with pool and large backyard", key="cmp_query"
    )
    top_k = st.slider("Results to compare", 3, 10, 5, key="cmp_k")

    if st.button("⚡ Compare", type="primary", key="cmp_btn"):
        st.session_state.query_count += 1

        # Semantic search
        with st.spinner("Running semantic search..."):
            sem = api_post("/search", {"query": cmp_query, "top_k": top_k})

        # For BM25 we use the module directly (API only exposes semantic)
        bm25_results = []
        try:
            with st.spinner("Running BM25 keyword search..."):
                import pandas as pd
                from semantic_search import SemanticSearcher

                if "bm25_searcher" not in st.session_state:
                    csv_path = os.path.join(_PROJECT_ROOT, "data", "processed", "cleaned_listing.csv")
                    df = pd.read_csv(csv_path)
                    col = next((c for c in ("cleaned_remarks", "remarks", "L_Remarks") if c in df.columns), None)
                    if col:
                        remarks = [r for r in df[col].dropna().astype(str).tolist() if len(r) > 50 and " " in r]
                        searcher = SemanticSearcher()
                        searcher.build_index(remarks)
                        st.session_state.bm25_searcher = searcher

                if "bm25_searcher" in st.session_state:
                    bm25_results = st.session_state.bm25_searcher.bm25_search(cmp_query, top_k)
        except Exception as e:
            st.warning(f"BM25 comparison unavailable: {e}")

        # Side-by-side display
        col_sem, col_bm25 = st.columns(2)

        with col_sem:
            st.markdown('<div class="compare-header sem-header">🧠 Semantic (FAISS)</div>',
                        unsafe_allow_html=True)
            if "error" not in sem:
                for i, item in enumerate(sem.get("results", [])):
                    with st.container():
                        st.markdown(f"**#{i+1}** (score: {item['score']:.4f})")
                        st.write(item["text"][:250] + "..." if len(item["text"]) > 250 else item["text"])
                        st.markdown("---")
            else:
                st.error("Semantic search unavailable")

        with col_bm25:
            st.markdown('<div class="compare-header bm25-header">🔤 Keyword (BM25)</div>',
                        unsafe_allow_html=True)
            if bm25_results:
                for i, (text, score) in enumerate(bm25_results):
                    with st.container():
                        st.markdown(f"**#{i+1}** (score: {score:.4f})")
                        st.write(text[:250] + "..." if len(text) > 250 else text)
                        st.markdown("---")
            else:
                st.info("No BM25 results")

        # Overlap analysis
        if "error" not in sem and bm25_results:
            sem_texts = set(r["text"] for r in sem.get("results", []))
            bm25_texts = set(t for t, _ in bm25_results)
            overlap = sem_texts & bm25_texts
            only_sem = sem_texts - bm25_texts
            only_bm25 = bm25_texts - sem_texts

            st.subheader("📊 Overlap Analysis")
            m1, m2, m3 = st.columns(3)
            m1.metric("Shared Results", len(overlap))
            m2.metric("Semantic Only", len(only_sem))
            m3.metric("Keyword Only", len(only_bm25))

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 3))
            cats = ["Shared", "Semantic Only", "Keyword Only"]
            vals = [len(overlap), len(only_sem), len(only_bm25)]
            colors = ["#2ecc71", "#3498db", "#e74c3c"]
            ax.barh(cats, vals, color=colors, edgecolor="white", height=0.5)
            ax.set_xlabel("Count")
            ax.set_title("Result Overlap: Semantic vs Keyword", fontweight="bold")
            for bar, val in zip(ax.patches, vals):
                ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                        str(val), va="center", fontsize=11, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: Metrics Dashboard
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("📊 Metrics Dashboard")

    # Live stats
    stats = api_get("/api-stats")
    if "error" not in stats:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Requests", stats.get("total_requests", 0))
        c2.metric("Cache Entries", stats.get("cache_size", 0))
        c3.metric("Cache Hit Rate", f"{stats.get('cache_hit_rate', 0):.1%}")
        c4.metric("Uptime", f"{stats.get('uptime_seconds', 0):.0f}s")

    # Session metrics
    st.subheader("Session Metrics")
    s1, s2 = st.columns(2)
    s1.metric("Queries This Session", st.session_state.query_count)
    latencies = st.session_state.latency_log
    avg_lat = sum(l["ms"] for l in latencies) / len(latencies) if latencies else 0
    s2.metric("Avg Latency", f"{avg_lat:.1f} ms")

    # Latency chart
    if latencies:
        st.subheader("Latency per API Call")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4))
        xs = list(range(len(latencies)))
        ys = [l["ms"] for l in latencies]
        labels = [l["path"] for l in latencies]
        colors_map = {
            "/search": "#3498db", "/parse-query": "#2ecc71",
            "/classify-intent": "#9b59b6", "/summarize": "#e67e22",
            "/check-compliance": "#e74c3c", "/extract-signals": "#1abc9c",
        }
        cs = [colors_map.get(l, "#95a5a6") for l in labels]
        ax.bar(xs, ys, color=cs, edgecolor="white", width=0.7)
        ax.set_ylabel("Latency (ms)")
        ax.set_title("Per-Call Latency", fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels([l.split("/")[-1] for l in labels], rotation=45, ha="right", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)

        # Latency by endpoint
        st.subheader("Avg Latency by Endpoint")
        from collections import defaultdict
        ep_lats = defaultdict(list)
        for l in latencies:
            ep_lats[l["path"]].append(l["ms"])

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ep_names = sorted(ep_lats.keys())
        ep_avgs = [sum(ep_lats[e]) / len(ep_lats[e]) for e in ep_names]
        ax2.barh(ep_names, ep_avgs, color="#3498db", edgecolor="white", height=0.5)
        ax2.set_xlabel("Avg Latency (ms)")
        ax2.set_title("Average Latency by Endpoint", fontweight="bold")
        for bar, val in zip(ax2.patches, ep_avgs):
            ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                     f"{val:.1f}ms", va="center", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig2)

    # Query history
    if st.session_state.query_history:
        st.subheader("Query History")
        for i, q in enumerate(reversed(st.session_state.query_history), 1):
            st.text(f"{i}. {q}")
