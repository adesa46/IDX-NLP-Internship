"""
Update data_exploration.ipynb with Week 8 listing summarization visuals.
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
    summary_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'week8_summary.json')

    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)

    week8_cells = [
        # ── Title ────────────────────────────────────────────────
        make_md_cell([
            "# Week 8: Listing Summarization & Answerability\n",
            "\n",
            "Generate concise 2-3 sentence summaries of listings for search results\n",
            "or email alerts. Includes an **extractive summarization** approach,\n",
            "**ROUGE metric evaluation**, a **human evaluation form**, and a new\n",
            "**AnswerabilityChecker** layer.\n",
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
            "from listing_summarizer import ListingSummarizer\n",
            "from answerability_checker import AnswerabilityChecker\n",
            "\n",
            "# Load summary\n",
            "with open('../data/processed/week8_summary.json', 'r') as f:\n",
            "    summary = json.load(f)\n",
            "\n",
            "print(f\"Listings summarized: {summary['total_listings']}\")\n",
            "print(f\"ROUGE eval count: {summary['rouge_eval_count']}\")\n",
            "print(f\"ROUGE-L: {summary['rouge_scores']['rougeL']:.4f}\")\n",
            "print(f\"ROUGE-L > 0.4: {'PASS' if summary['rouge_l_target_met'] else 'FAIL'}\")",
        ]),

        # ── Coverage Stats ──────────────────────────────────────
        make_md_cell([
            "## Summary Coverage\n",
            "\n",
            "Percentage of listings where each field was successfully extracted.",
        ]),

        make_code_cell([
            "cov = summary['coverage']\n",
            "fields = ['with_summary_pct', 'with_beds_pct', 'with_baths_pct',\n",
            "          'with_price_pct', 'with_features_pct', 'with_location_pct']\n",
            "labels = ['Summary', 'Beds', 'Baths', 'Price', 'Features', 'Location']\n",
            "values = [cov[f] for f in fields]\n",
            "colors = ['#3498DB', '#2ECC71', '#E74C3C', '#F39C12', '#9B59B6', '#1ABC9C']\n",
            "\n",
            "fig, ax = plt.subplots(figsize=(9, 5))\n",
            "bars = ax.barh(labels, values, color=colors, edgecolor='white', height=0.5)\n",
            "ax.set_xlabel('Coverage (%)', fontsize=12)\n",
            "ax.set_title('Summary Field Coverage', fontsize=14, fontweight='bold')\n",
            "ax.set_xlim(0, 110)\n",
            "\n",
            "for bar, val in zip(bars, values):\n",
            "    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,\n",
            "            f'{val:.1f}%', va='center', fontsize=11, fontweight='bold')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()",
        ]),

        # ── ROUGE Scores ────────────────────────────────────────
        make_md_cell([
            "## ROUGE Evaluation Scores\n",
            "\n",
            "ROUGE-1, ROUGE-2, and ROUGE-L F-measure comparing extractive summaries\n",
            "to entity-based reference summaries.",
        ]),

        make_code_cell([
            "rouge = summary['rouge_scores']\n",
            "r_labels = list(rouge.keys())\n",
            "r_values = list(rouge.values())\n",
            "colors = ['#3498DB', '#E74C3C', '#2ECC71']\n",
            "\n",
            "fig, ax = plt.subplots(figsize=(8, 5))\n",
            "bars = ax.bar(r_labels, r_values, color=colors, edgecolor='white', width=0.4)\n",
            "ax.set_ylabel('F-Measure', fontsize=12)\n",
            "ax.set_title('ROUGE Scores', fontsize=14, fontweight='bold')\n",
            "ax.set_ylim(0, 1.1)\n",
            "\n",
            "# Threshold line\n",
            "ax.axhline(y=0.4, color='#E74C3C', linestyle='--', alpha=0.6, label='Target (0.4)')\n",
            "\n",
            "for bar, val in zip(bars, r_values):\n",
            "    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,\n",
            "            f'{val:.4f}', ha='center', fontsize=11, fontweight='bold')\n",
            "\n",
            "ax.legend(fontsize=10)\n",
            "plt.tight_layout()\n",
            "plt.show()",
        ]),

        # ── Sample Summaries ────────────────────────────────────
        make_md_cell([
            "## Sample Generated Summaries\n",
            "\n",
            "Examples of extractive summaries with extracted metadata.",
        ]),

        make_code_cell([
            "samples = summary['sample_summaries']\n",
            "rows = []\n",
            "for s in samples:\n",
            "    rows.append({\n",
            "        'ID': s['listing_id'],\n",
            "        'Beds': s['beds'],\n",
            "        'Baths': s['baths'],\n",
            "        'Price': f\"${s['price']:,}\" if s['price'] else 'N/A',\n",
            "        'Features': ', '.join(s['top_features']) if s['top_features'] else 'N/A',\n",
            "        'Summary': s['summary'][:80] + ('...' if len(s['summary']) > 80 else ''),\n",
            "    })\n",
            "\n",
            "df = pd.DataFrame(rows)\n",
            "display(df.style.set_caption('Sample Listing Summaries'))",
        ]),

        # ── Answerability Demo ──────────────────────────────────
        make_md_cell([
            "## Answerability Checker\n",
            "\n",
            "Pre-query validation determines if user queries can be answered\n",
            "by the system before executing any SQL.",
        ]),

        make_code_cell([
            "ans_results = summary['answerability_demo']\n",
            "rows = []\n",
            "for r in ans_results:\n",
            "    rows.append({\n",
            "        'Query': r['query'],\n",
            "        'Answerable': '✓' if r['answerable'] else '✗',\n",
            "        'Message': r['message'],\n",
            "        'Correct': '✓' if r['correct'] else '✗',\n",
            "    })\n",
            "\n",
            "df = pd.DataFrame(rows)\n",
            "display(df.style.set_caption('Answerability Checker Results'))\n",
            "print(f\"\\nAccuracy: {summary['answerability_accuracy']:.0%}\")",
        ]),

        # ── Live Demo ───────────────────────────────────────────
        make_md_cell([
            "## Live Summarization Demo\n",
            "\n",
            "Run the summarizer on a sample listing in real time.",
        ]),

        make_code_cell([
            "summarizer = ListingSummarizer(taxonomy_path='../data/processed/taxonomy.json')\n",
            "\n",
            "demo_listing = {\n",
            "    'listing_id': 'demo',\n",
            "    'remarks': (\n",
            "        'Stunning 4 bedroom, 3 bathroom home in a quiet cul-de-sac. '\n",
            "        'Features a gourmet kitchen with granite countertops and stainless '\n",
            "        'steel appliances. Spacious backyard with sparkling pool and spa. '\n",
            "        'Hardwood floors and recessed lighting throughout. '\n",
            "        'Walking distance to award-winning schools and shopping centers. '\n",
            "        'Priced at $825,000 with 2,800 sqft of living space.'\n",
            "    )\n",
            "}\n",
            "\n",
            "result = summarizer.summarize_listing(demo_listing)\n",
            "print('Summary:', result['summary'])\n",
            "print(f\"Beds: {result['beds']}, Baths: {result['baths']}\")\n",
            "print(f\"Price: ${result['price']:,}\" if result['price'] else 'Price: N/A')\n",
            "print(f\"Features: {result['top_features']}\")\n",
            "print(f\"Location: {result['location']}\")",
        ]),

        # ── Summary ─────────────────────────────────────────────
        make_md_cell([
            "## Summary\n",
            "\n",
            f"Week 8 summarized **{summary['total_listings']}** listings using extractive methods:\n",
            "\n",
            f"- **ROUGE-L**: {summary['rouge_scores']['rougeL']:.4f} ",
            f"({'above' if summary['rouge_l_target_met'] else 'below'} 0.4 target)\n",
            f"- **ROUGE-1**: {summary['rouge_scores']['rouge1']:.4f}\n",
            f"- **Human eval form**: {summary['human_eval_count']} summaries prepared for review\n",
            f"- **Answerability checker**: {summary['answerability_accuracy']:.0%} accuracy on test queries\n",
            "\n",
            "Summaries include beds/baths, price, top features, and location when available.\n",
            "The `AnswerabilityChecker` validates queries pre- and post-SQL execution.",
        ]),
    ]

    nb['cells'].extend(week8_cells)

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"Added {len(week8_cells)} cells to notebook. Total cells: {len(nb['cells'])}")


if __name__ == '__main__':
    main()
