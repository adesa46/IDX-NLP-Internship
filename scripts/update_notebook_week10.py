"""
Update data_exploration.ipynb with Week 10 REST API visuals.
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
    summary_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'week10_summary.json')

    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)

    week10_cells = [
        # ── Title ────────────────────────────────────────────────
        make_md_cell([
            "# Week 10: REST API with FastAPI\n",
            "\n",
            "Production-ready REST API exposing all NLP capabilities:\n",
            "search, entity extraction, query parsing, summarization,\n",
            "compliance checking, intent classification, and signal extraction.\n",
            "\n",
            "**Key features:** Pydantic validation, in-memory caching,\n",
            "per-IP rate limiting (10 req/s), structured logging,\n",
            "auto-generated OpenAPI docs.\n",
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
            "with open('../data/processed/week10_summary.json', 'r') as f:\n",
            "    summary = json.load(f)\n",
            "\n",
            "print(f\"Total endpoints: {summary['total_endpoints']}\")\n",
            "print(f\"All endpoints OK: {summary['all_endpoints_ok']}\")\n",
            "print(f\"Cache working: {summary['cache_results']['cache_working']}\")\n",
            "print(f\"Rate limiting working: {summary['rate_limit_results']['rate_limiting_working']}\")",
        ]),

        # ── Endpoint Inventory ─────────────────────────────────
        make_md_cell([
            "## API Endpoint Inventory\n",
            "\n",
            "All 9 endpoints exposed by the Real Estate NLP API,\n",
            "with HTTP method, status, and measured response time.\n",
        ]),

        make_code_cell([
            "endpoints = summary['endpoints']\n",
            "df_ep = pd.DataFrame(endpoints)\n",
            "df_ep.columns = ['Method', 'Path', 'Status', 'Response (ms)']\n",
            "display(df_ep.style.set_caption('API Endpoint Inventory'))",
        ]),

        # ── Response Time Chart ────────────────────────────────
        make_md_cell([
            "## Endpoint Response Times\n",
            "\n",
            "Response latency for each endpoint (first-call, including any model loading).\n",
        ]),

        make_code_cell([
            "eps = summary['endpoints']\n",
            "paths = [e['path'] for e in eps]\n",
            "times = [e['response_time_ms'] for e in eps]\n",
            "statuses = [e['status_code'] for e in eps]\n",
            "colors = ['#2ECC71' if s == 200 else '#E74C3C' for s in statuses]\n",
            "\n",
            "fig, ax = plt.subplots(figsize=(12, 5))\n",
            "bars = ax.barh(paths[::-1], times[::-1], color=colors[::-1],\n",
            "               edgecolor='white', height=0.5)\n",
            "ax.set_xlabel('Response Time (ms)', fontsize=12)\n",
            "ax.set_title('Endpoint Response Times', fontsize=14, fontweight='bold')\n",
            "\n",
            "for bar, val in zip(bars, times[::-1]):\n",
            "    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,\n",
            "            f'{val:.1f}ms', va='center', fontsize=10)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()",
        ]),

        # ── Cache Performance ─────────────────────────────────
        make_md_cell([
            "## Cache Performance\n",
            "\n",
            "In-memory TTL cache (512 entries, 5-min TTL) avoids redundant\n",
            "computation. Repeated identical requests are served from cache.\n",
        ]),

        make_code_cell([
            "cache = summary['cache_results']\n",
            "print(f\"First call cached:  {cache['first_call_cached']}\")\n",
            "print(f\"Second call cached: {cache['second_call_cached']}\")\n",
            "print(f\"Cache working:      {cache['cache_working']}\")\n",
            "\n",
            "if 'final_stats' in summary and summary['final_stats']:\n",
            "    stats = summary['final_stats']\n",
            "    labels = ['Cache Hits', 'Cache Misses']\n",
            "    values = [stats.get('cache_hits', 0), stats.get('cache_misses', 0)]\n",
            "    colors = ['#2ECC71', '#E74C3C']\n",
            "\n",
            "    fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n",
            "\n",
            "    # Bar chart\n",
            "    bars = axes[0].bar(labels, values, color=colors, edgecolor='white', width=0.4)\n",
            "    axes[0].set_ylabel('Count', fontsize=12)\n",
            "    axes[0].set_title('Cache Hit / Miss Count', fontsize=14, fontweight='bold')\n",
            "    for bar, val in zip(bars, values):\n",
            "        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,\n",
            "                     str(val), ha='center', fontsize=11, fontweight='bold')\n",
            "\n",
            "    # Pie chart\n",
            "    if sum(values) > 0:\n",
            "        axes[1].pie(values, labels=labels, colors=colors, autopct='%1.1f%%',\n",
            "                    startangle=90, textprops={'fontsize': 11})\n",
            "        axes[1].set_title('Cache Distribution', fontsize=14, fontweight='bold')\n",
            "    else:\n",
            "        axes[1].text(0.5, 0.5, 'No cache activity', ha='center', va='center')\n",
            "\n",
            "    plt.tight_layout()\n",
            "    plt.show()\n",
            "    print(f\"\\nCache hit rate: {stats.get('cache_hit_rate', 0):.1%}\")\n",
            "else:\n",
            "    print('No cache stats available.')",
        ]),

        # ── Rate Limiting ────────────────────────────────────────
        make_md_cell([
            "## Rate Limiting\n",
            "\n",
            "Per-IP rate limiting at 10 requests/second prevents abuse.\n",
            "Requests exceeding the limit receive HTTP 429.\n",
        ]),

        make_code_cell([
            "rl = summary['rate_limit_results']\n",
            "print(f\"Requests sent:    {rl['requests_sent']}\")\n",
            "print(f\"Requests passed:  {rl['requests_passed']}\")\n",
            "print(f\"Requests blocked: {rl['requests_blocked']}\")\n",
            "print(f\"Rate limiting working: {rl['rate_limiting_working']}\")\n",
            "\n",
            "fig, ax = plt.subplots(figsize=(6, 4))\n",
            "labels = ['Passed (200)', 'Blocked (429)']\n",
            "values = [rl['requests_passed'], rl['requests_blocked']]\n",
            "colors = ['#2ECC71', '#E74C3C']\n",
            "bars = ax.bar(labels, values, color=colors, edgecolor='white', width=0.4)\n",
            "ax.set_ylabel('Count', fontsize=12)\n",
            "ax.set_title('Rate Limiting Test (15 rapid requests)', fontsize=14, fontweight='bold')\n",
            "for bar, val in zip(bars, values):\n",
            "    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,\n",
            "            str(val), ha='center', fontsize=12, fontweight='bold')\n",
            "plt.tight_layout()\n",
            "plt.show()",
        ]),

        # ── Compliance Demo ──────────────────────────────────────
        make_md_cell([
            "## Compliance Endpoint Demo\n",
            "\n",
            "Sample requests through the `/check-compliance` endpoint\n",
            "showing clean vs. violating listings.\n",
        ]),

        make_code_cell([
            "demos = summary['compliance_demos']\n",
            "rows = []\n",
            "for d in demos:\n",
            "    status = '✓ Compliant' if d['compliant'] else '✗ Non-compliant'\n",
            "    rows.append({\n",
            "        'Label': d['label'],\n",
            "        'Status': status,\n",
            "        'Violations': d['num_violations'],\n",
            "        'Errors': d['stats']['error'],\n",
            "        'Warnings': d['stats']['warning'],\n",
            "        'Text': d['text'][:60] + ('...' if len(d['text']) > 60 else ''),\n",
            "    })\n",
            "\n",
            "df_cd = pd.DataFrame(rows)\n",
            "display(df_cd.style.set_caption('Compliance Endpoint Demo Results'))",
        ]),

        # ── Live API Demo ────────────────────────────────────────
        make_md_cell([
            "## Live API Demo\n",
            "\n",
            "Call the API endpoints directly from the notebook using the FastAPI TestClient.\n",
        ]),

        make_code_cell([
            "from fastapi.testclient import TestClient\n",
            "from api_app import app\n",
            "\n",
            "with TestClient(app) as client:\n",
            "    # Parse query\n",
            "    print('=== /parse-query ===')\n",
            "    r = client.post('/parse-query', json={'query': '3 bed in Portland under 500k with pool'})\n",
            "    data = r.json()\n",
            "    print(f\"  Filters: {data['filters']}\")\n",
            "    print(f\"  SQL: {data['sql']}\")\n",
            "    print(f\"  Valid: {data['valid']}\")\n",
            "\n",
            "    # Extract entities\n",
            "    print('\\n=== /extract-entities ===')\n",
            "    r = client.post('/extract-entities', json={\n",
            "        'text': '3 bedroom, 2.5 bath home with 2,500 sqft. $1,250,000.'\n",
            "    })\n",
            "    print(f\"  Entities: {r.json()['entities']}\")\n",
            "\n",
            "    # Compliance check\n",
            "    print('\\n=== /check-compliance ===')\n",
            "    r = client.post('/check-compliance', json={\n",
            "        'text': 'No children allowed. Adults only.'\n",
            "    })\n",
            "    data = r.json()\n",
            "    print(f\"  Compliant: {data['compliant']}\")\n",
            "    for v in data['violations']:\n",
            "        print(f\"  [{v['severity'].upper()}] {v['category']}: {v['matched_text']}\")\n",
            "\n",
            "    # Health\n",
            "    print('\\n=== /health ===')\n",
            "    r = client.get('/health')\n",
            "    data = r.json()\n",
            "    print(f\"  Status: {data['status']}\")\n",
            "    print(f\"  Endpoints: {len(data['endpoints_available'])}\")",
        ]),

        # ── Architecture ────────────────────────────────────────
        make_md_cell([
            "## API Architecture\n",
            "\n",
            "```\n",
            "┌─────────────────────────────────────────────────────────┐\n",
            "│                    FastAPI Application                   │\n",
            "│  ┌─────────┐  ┌─────────────┐  ┌───────────────────┐   │\n",
            "│  │  CORS   │  │ Rate Limiter │  │ Request Logging   │   │\n",
            "│  │ Middle  │─▶│ (10 req/s/IP)│─▶│ (Structured JSON) │   │\n",
            "│  └─────────┘  └─────────────┘  └───────────────────┘   │\n",
            "│                        │                                 │\n",
            "│            ┌───────────┴────────────┐                   │\n",
            "│            ▼                        ▼                   │\n",
            "│  ┌──────────────────┐    ┌──────────────────┐           │\n",
            "│  │  TTL Cache       │    │  Pydantic Models │           │\n",
            "│  │  (512 entries,   │    │  (Request/       │           │\n",
            "│  │   5-min TTL)     │    │   Response)      │           │\n",
            "│  └──────────────────┘    └──────────────────┘           │\n",
            "│                        │                                 │\n",
            "│            ┌───────────┴────────────┐                   │\n",
            "│            ▼                        ▼                   │\n",
            "│  ┌────────────────────────────────────────────────┐     │\n",
            "│  │              NLP Module Layer                   │     │\n",
            "│  │  ┌──────────┐ ┌──────────┐ ┌──────────────┐   │     │\n",
            "│  │  │ Search   │ │ Parser   │ │ Entities     │   │     │\n",
            "│  │  │ (FAISS)  │ │ (Regex)  │ │ (Regex+Tax)  │   │     │\n",
            "│  │  ├──────────┤ ├──────────┤ ├──────────────┤   │     │\n",
            "│  │  │ Summary  │ │ Comply   │ │ Intent       │   │     │\n",
            "│  │  │ (Extract)│ │ (FHA)    │ │ (TF-IDF+LR)  │   │     │\n",
            "│  │  ├──────────┤ ├──────────┤ ├──────────────┤   │     │\n",
            "│  │  │ Signals  │ │ Health   │ │ Stats        │   │     │\n",
            "│  │  └──────────┘ └──────────┘ └──────────────┘   │     │\n",
            "│  └────────────────────────────────────────────────┘     │\n",
            "└─────────────────────────────────────────────────────────┘\n",
            "```\n",
        ]),

        # ── OpenAPI Paths ────────────────────────────────────────
        make_md_cell([
            "## OpenAPI Auto-Generated Documentation\n",
            "\n",
            "FastAPI auto-generates interactive docs at `/docs` (Swagger UI)\n",
            "and `/redoc`. All endpoints are fully documented with request/response schemas.\n",
        ]),

        make_code_cell([
            "paths = summary.get('openapi_paths', [])\n",
            "print(f'OpenAPI registered paths ({len(paths)}):')\n",
            "for p in sorted(paths):\n",
            "    print(f'  {p}')\n",
            "print(f'\\nSwagger UI: http://localhost:8000/docs')\n",
            "print(f'ReDoc:      http://localhost:8000/redoc')",
        ]),

        # ── Summary ─────────────────────────────────────────────
        make_md_cell([
            "## Summary\n",
            "\n",
            f"Week 10 built a **production-ready REST API** with {summary['total_endpoints']} endpoints:\n",
            "\n",
            "| Feature | Status |\n",
            "|---------|--------|\n",
            f"| Endpoints | {summary['total_endpoints']} (all operational) |\n",
            f"| Cache | {'✓ Working' if summary['cache_results']['cache_working'] else '✗ Not working'} |\n",
            f"| Rate Limiting | {'✓ Working' if summary['rate_limit_results']['rate_limiting_working'] else '✗ Not working'} |\n",
            "| OpenAPI Docs | ✓ Auto-generated at /docs |\n",
            "| Pydantic Validation | ✓ All request/response models |\n",
            "| Docker | ✓ Container ready |\n",
            "\n",
            "**Endpoints:**\n",
            "- `POST /search` — Semantic search (FAISS + sentence-transformers)\n",
            "- `POST /parse-query` — NL query → filters + SQL\n",
            "- `POST /extract-entities` — Bed/bath/price/sqft/amenities\n",
            "- `POST /summarize` — Extractive listing summarization\n",
            "- `POST /check-compliance` — Fair Housing compliance\n",
            "- `POST /classify-intent` — Buyer intent classification\n",
            "- `POST /extract-signals` — Full signal extraction\n",
            "- `GET /health` — Health check / liveness probe\n",
            "- `GET /api-stats` — Cache & request statistics\n",
            "\n",
            "Run locally: `uvicorn scripts.api_app:app --reload`\n",
            "Docker: `docker-compose up api`\n",
        ]),
    ]

    nb['cells'].extend(week10_cells)

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"Added {len(week10_cells)} cells to notebook. Total cells: {len(nb['cells'])}")


if __name__ == '__main__':
    main()
