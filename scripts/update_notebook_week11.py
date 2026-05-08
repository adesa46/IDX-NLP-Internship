"""
Update data_exploration.ipynb with Week 11 Product Integration Demo visuals.
All graphs are rendered directly in the notebook — no separate image files.
"""
import json
import os


def make_md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source if isinstance(source, list) else [source]
    }


def make_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source if isinstance(source, list) else [source]
    }


def main():
    nb_path = os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'data_exploration.ipynb')
    summary_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'week11_summary.json')

    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)

    # Pre-compute values for markdown
    n_queries = summary.get("num_demo_queries", 0)
    avg_pipeline = summary.get("latency_summary", {}).get("total_pipeline", {}).get("avg_ms", 0)
    n_comparisons = len(summary.get("comparison_results", []))

    week11_cells = [
        # ── Title ────────────────────────────────────────────────
        make_md_cell([
            "# Week 11: Product Integration Demo\n",
            "\n",
            "End-to-end integration of all NLP components into a unified system.\n",
            "Demonstrates the complete user journey: **natural language query →\n",
            "parsed filters → semantic results → summaries → details**.\n",
            "\n",
            "**Components integrated:**\n",
            "- Query Parser (Week 4)\n",
            "- Semantic Search with FAISS (Week 5)\n",
            "- Signal Extraction (Week 6)\n",
            "- Listing Summarization (Week 8)\n",
            "- Buyer Intent Classification (Week 9)\n",
            "- Fair Housing Compliance (Week 7)\n",
            "- REST API (Week 10)\n",
            "- Streamlit Web UI (Week 11)\n",
        ]),

        # ── Load data ───────────────────────────────────────────
        make_code_cell([
            "import sys, os, json\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import matplotlib\n",
            "matplotlib.rcParams['figure.dpi'] = 120\n",
            "matplotlib.rcParams['figure.figsize'] = (10, 6)\n",
            "\n",
            "sys.path.insert(0, os.path.abspath('../scripts'))\n",
            "\n",
            "with open('../data/processed/week11_summary.json', 'r') as f:\n",
            "    w11 = json.load(f)\n",
            "\n",
            "print(f\"Demo queries: {w11['num_demo_queries']}\")\n",
            "print(f\"Pipeline results: {len(w11['pipeline_results'])}\")\n",
            "print(f\"Comparisons: {len(w11['comparison_results'])}\")",
        ]),

        # ── End-to-End Pipeline Demo ──────────────────────────
        make_md_cell([
            "## End-to-End Pipeline Demo\n",
            "\n",
            f"Each of the {n_queries} demo queries is processed through the full\n",
            "NLP pipeline: parse → intent → search → summarize → compliance → signals.\n",
        ]),

        make_code_cell([
            "# Show pipeline results for each query\n",
            "for i, entry in enumerate(w11['pipeline_results']):\n",
            "    print(f\"\\n{'='*60}\")\n",
            "    print(f\"Query {i+1}: {entry['query']}\")\n",
            "    print(f\"{'='*60}\")\n",
            "\n",
            "    # Parse\n",
            "    parse = entry['steps'].get('parse')\n",
            "    if parse:\n",
            "        print(f\"  Filters:    {parse.get('filters', {})}\")\n",
            "        print(f\"  SQL:        {parse.get('sql', 'N/A')[:80]}\")\n",
            "\n",
            "    # Intent\n",
            "    intent = entry['steps'].get('intent')\n",
            "    if intent:\n",
            "        print(f\"  Intent:     {intent.get('intent', 'N/A')} \"\n",
            "              f\"(confidence: {intent.get('confidence', 0):.2f})\")\n",
            "\n",
            "    # Search\n",
            "    search = entry['steps'].get('search')\n",
            "    if search:\n",
            "        print(f\"  Results:    {search.get('count', 0)} listings found\")\n",
            "\n",
            "    # Summary\n",
            "    summ = entry['steps'].get('summarize')\n",
            "    if summ:\n",
            "        print(f\"  Summary:    {summ.get('summary', '')[:100]}...\")\n",
            "\n",
            "    # Compliance\n",
            "    comp = entry['steps'].get('compliance')\n",
            "    if comp:\n",
            "        status = '✓ Compliant' if comp.get('compliant') else '✗ Non-compliant'\n",
            "        print(f\"  Compliance: {status}\")\n",
            "\n",
            "    # Latencies\n",
            "    print(f\"  Total time: {entry['total_latency_ms']:.1f}ms\")",
        ]),

        # ── Pipeline Latency Breakdown ────────────────────────
        make_md_cell([
            "## Pipeline Latency Breakdown\n",
            "\n",
            "Per-step latency for each query through the NLP pipeline.\n",
        ]),

        make_code_cell([
            "lat = w11['latency_summary']\n",
            "\n",
            "# Bar chart of average latency per step\n",
            "steps = [s for s in lat if s != 'total_pipeline']\n",
            "avgs = [lat[s]['avg_ms'] for s in steps]\n",
            "p95s = [lat[s]['p95_ms'] for s in steps]\n",
            "\n",
            "fig, ax = plt.subplots(figsize=(10, 5))\n",
            "x = range(len(steps))\n",
            "w = 0.35\n",
            "bars1 = ax.bar([i - w/2 for i in x], avgs, w, label='Avg', color='#3498db', edgecolor='white')\n",
            "bars2 = ax.bar([i + w/2 for i in x], p95s, w, label='P95', color='#e74c3c', edgecolor='white')\n",
            "ax.set_xlabel('Pipeline Step', fontsize=12)\n",
            "ax.set_ylabel('Latency (ms)', fontsize=12)\n",
            "ax.set_title('Per-Step Latency (Avg vs P95)', fontsize=14, fontweight='bold')\n",
            "ax.set_xticks(list(x))\n",
            "ax.set_xticklabels(steps, rotation=30, ha='right')\n",
            "ax.legend()\n",
            "\n",
            "for bar in bars1:\n",
            "    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,\n",
            "            f'{bar.get_height():.0f}', ha='center', fontsize=9)\n",
            "for bar in bars2:\n",
            "    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,\n",
            "            f'{bar.get_height():.0f}', ha='center', fontsize=9)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "\n",
            "# Total pipeline stats\n",
            "tp = lat.get('total_pipeline', {})\n",
            "print(f\"\\nTotal Pipeline: avg={tp.get('avg_ms',0):.1f}ms, \"\n",
            "      f\"p95={tp.get('p95_ms',0):.1f}ms, max={tp.get('max_ms',0):.1f}ms\")",
        ]),

        # ── Stacked latency per query ─────────────────────────
        make_code_cell([
            "# Stacked bar: latency breakdown per query\n",
            "fig, ax = plt.subplots(figsize=(12, 5))\n",
            "steps_order = ['parse', 'intent', 'search', 'summarize', 'compliance', 'signals']\n",
            "colors = ['#2ecc71', '#9b59b6', '#3498db', '#e67e22', '#e74c3c', '#1abc9c']\n",
            "queries = [e['query'][:30] + '...' for e in w11['pipeline_results']]\n",
            "x = range(len(queries))\n",
            "\n",
            "bottom = [0] * len(queries)\n",
            "for step, color in zip(steps_order, colors):\n",
            "    vals = [e['latencies'].get(step, 0) for e in w11['pipeline_results']]\n",
            "    ax.bar(x, vals, bottom=bottom, label=step, color=color, edgecolor='white', width=0.6)\n",
            "    bottom = [b + v for b, v in zip(bottom, vals)]\n",
            "\n",
            "ax.set_ylabel('Latency (ms)', fontsize=12)\n",
            "ax.set_title('Pipeline Latency Breakdown per Query', fontsize=14, fontweight='bold')\n",
            "ax.set_xticks(list(x))\n",
            "ax.set_xticklabels(queries, rotation=35, ha='right', fontsize=8)\n",
            "ax.legend(loc='upper right', fontsize=9)\n",
            "plt.tight_layout()\n",
            "plt.show()",
        ]),

        # ── NLP vs Keyword Comparison ─────────────────────────
        make_md_cell([
            "## NLP Search vs Keyword Search\n",
            "\n",
            "Side-by-side comparison of Semantic (FAISS + sentence-transformers)\n",
            "vs BM25 keyword search. Shows score distributions and result overlap.\n",
        ]),

        make_code_cell([
            "comparisons = w11.get('comparison_results', [])\n",
            "if comparisons:\n",
            "    # Score comparison\n",
            "    queries = [c['query'][:25] + '...' for c in comparisons]\n",
            "    sem_avgs = [c['semantic_avg_score'] for c in comparisons]\n",
            "    bm25_avgs = [c['bm25_avg_score'] for c in comparisons]\n",
            "    overlaps = [c['overlap_pct'] for c in comparisons]\n",
            "\n",
            "    fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
            "\n",
            "    # Score comparison\n",
            "    x = range(len(queries))\n",
            "    w = 0.35\n",
            "    axes[0].bar([i - w/2 for i in x], sem_avgs, w, label='Semantic', color='#3498db')\n",
            "    axes[0].bar([i + w/2 for i in x], bm25_avgs, w, label='BM25', color='#e74c3c')\n",
            "    axes[0].set_ylabel('Avg Score')\n",
            "    axes[0].set_title('Avg Result Score: Semantic vs BM25', fontweight='bold')\n",
            "    axes[0].set_xticks(list(x))\n",
            "    axes[0].set_xticklabels(queries, rotation=35, ha='right', fontsize=8)\n",
            "    axes[0].legend()\n",
            "\n",
            "    # Overlap\n",
            "    axes[1].bar(queries, overlaps, color='#2ecc71', edgecolor='white', width=0.5)\n",
            "    axes[1].set_ylabel('Overlap %')\n",
            "    axes[1].set_title('Result Overlap (Top 5)', fontweight='bold')\n",
            "    axes[1].set_xticklabels(queries, rotation=35, ha='right', fontsize=8)\n",
            "    for bar, val in zip(axes[1].patches, overlaps):\n",
            "        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,\n",
            "                     f'{val:.0f}%', ha='center', fontsize=10)\n",
            "\n",
            "    plt.tight_layout()\n",
            "    plt.show()\n",
            "else:\n",
            "    print('No comparison data available.')",
        ]),

        # ── Metrics Summary ───────────────────────────────────
        make_md_cell([
            "## API Performance Metrics\n",
            "\n",
            "Aggregate statistics from the integration demo run.\n",
        ]),

        make_code_cell([
            "stats = w11.get('api_stats', {})\n",
            "lat = w11.get('latency_summary', {})\n",
            "\n",
            "rows = []\n",
            "for step in sorted(lat.keys()):\n",
            "    s = lat[step]\n",
            "    rows.append({\n",
            "        'Step': step,\n",
            "        'Avg (ms)': s['avg_ms'],\n",
            "        'P95 (ms)': s['p95_ms'],\n",
            "        'Max (ms)': s['max_ms'],\n",
            "        'Calls': s['count'],\n",
            "    })\n",
            "\n",
            "df_lat = pd.DataFrame(rows)\n",
            "display(df_lat.style.set_caption('Latency Summary by Pipeline Step'))\n",
            "\n",
            "if stats:\n",
            "    print(f\"\\nAPI Stats:\")\n",
            "    print(f\"  Total requests:  {stats.get('total_requests', 'N/A')}\")\n",
            "    print(f\"  Cache hit rate:  {stats.get('cache_hit_rate', 0):.1%}\")\n",
            "    print(f\"  Cache entries:   {stats.get('cache_size', 'N/A')}\")",
        ]),

        # ── Streamlit Web UI ──────────────────────────────────
        make_md_cell([
            "## Streamlit Web UI\n",
            "\n",
            "A Streamlit web application provides an interactive demo interface\n",
            "with three tabs:\n",
            "\n",
            "1. **Intelligent Search** — Full NLP pipeline: parse → intent → search → summarize\n",
            "2. **NLP vs Keyword** — Side-by-side semantic vs BM25 comparison\n",
            "3. **Metrics Dashboard** — Real-time latency, cache, and query stats\n",
            "\n",
            "**Run locally:**\n",
            "```bash\n",
            "# Start the API backend\n",
            "uvicorn scripts.api_app:app --reload\n",
            "\n",
            "# In another terminal, start the Streamlit UI\n",
            "streamlit run scripts/streamlit_app.py\n",
            "```\n",
            "\n",
            "The Streamlit app auto-detects the API and falls back to a local\n",
            "TestClient mode when the backend is unavailable.\n",
        ]),

        # ── Project Summary ───────────────────────────────────
        make_md_cell([
            "## Project Summary — All 11 Weeks\n",
            "\n",
            "| Week | Component | Key Deliverable |\n",
            "|------|-----------|----------------|\n",
            "| 1 | Environment Setup | MySQL + Python + Docker |\n",
            "| 2 | Data Pipeline | Text cleaning, CSV export |\n",
            "| 3 | Entity Extraction | Regex bed/bath/price/sqft extraction |\n",
            "| 4 | Query Parser | NL → SQL with schema validation |\n",
            "| 5 | Semantic Search | FAISS embeddings + BM25 comparison |\n",
            "| 6 | Signal Extraction | Taxonomy-based amenity/condition/finance signals |\n",
            "| 7 | Fair Housing Compliance | Pattern-based compliance checker |\n",
            "| 8 | Listing Summarization | Extractive 2-sentence summaries |\n",
            "| 9 | Buyer Intent Classification | TF-IDF + Logistic Regression classifier |\n",
            "| 10 | REST API | FastAPI with caching, rate limiting, OpenAPI |\n",
            "| **11** | **Integration Demo** | **Streamlit UI + end-to-end pipeline** |\n",
            "\n",
            f"**Final pipeline latency:** avg {avg_pipeline:.0f}ms per query\n",
            f"across {n_queries} demo queries with {n_comparisons} NLP vs keyword comparisons.\n",
            "\n",
            "All components are containerized via Docker and exposed through\n",
            "a production-ready REST API with auto-generated documentation.\n",
        ]),
    ]

    nb['cells'].extend(week11_cells)

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"Added {len(week11_cells)} cells to notebook. Total cells: {len(nb['cells'])}")


if __name__ == '__main__':
    main()
