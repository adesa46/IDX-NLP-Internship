"""
Update data_exploration.ipynb with Week 9 Fair Housing Compliance visuals.
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
    summary_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'week9_summary.json')

    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)

    week9_cells = [
        # ── Title ────────────────────────────────────────────────
        make_md_cell([
            "# Week 9: Fair Housing Compliance\n",
            "\n",
            "Implement Fair Housing Act compliance checking to detect prohibited\n",
            "language related to protected classes (race, religion, familial status,\n",
            "disability, etc.) and flag biased descriptions before publication.\n",
            "\n",
            "**Key goals:** 100% recall on known violations, precision > 80%,\n",
            "multi-level severity (error / warning / info).\n",
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
            "from compliance_checker import ComplianceChecker\n",
            "\n",
            "# Load summary\n",
            "with open('../data/processed/week9_summary.json', 'r') as f:\n",
            "    summary = json.load(f)\n",
            "\n",
            "print(f\"Listings scanned: {summary['total_listings_scanned']}\")\n",
            "print(f\"Flagged: {summary['flagged_listings']} ({summary['flagged_pct']}%)\")\n",
            "print(f\"Total violations: {summary['total_violations']}\")\n",
            "print(f\"Recall: {summary['recall']:.2%}\")\n",
            "print(f\"Precision: {summary['precision']:.2%}\")",
        ]),

        # ── Severity Breakdown ──────────────────────────────────
        make_md_cell([
            "## Violation Severity Breakdown\n",
            "\n",
            "Distribution of detected violations by severity level:\n",
            "- **Error**: Must fix before publication\n",
            "- **Warning**: Should be reviewed by compliance officer\n",
            "- **Info**: Informational — no action required\n",
        ]),

        make_code_cell([
            "sev = summary['severity_breakdown']\n",
            "labels = list(sev.keys())\n",
            "values = list(sev.values())\n",
            "colors = ['#E74C3C', '#F39C12', '#3498DB']\n",
            "\n",
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
            "\n",
            "# Bar chart\n",
            "bars = axes[0].bar(labels, values, color=colors, edgecolor='white', width=0.5)\n",
            "axes[0].set_ylabel('Count', fontsize=12)\n",
            "axes[0].set_title('Violations by Severity', fontsize=14, fontweight='bold')\n",
            "for bar, val in zip(bars, values):\n",
            "    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,\n",
            "                 str(val), ha='center', fontsize=11, fontweight='bold')\n",
            "\n",
            "# Pie chart\n",
            "if sum(values) > 0:\n",
            "    axes[1].pie(values, labels=labels, colors=colors, autopct='%1.1f%%',\n",
            "                startangle=90, textprops={'fontsize': 11})\n",
            "    axes[1].set_title('Severity Distribution', fontsize=14, fontweight='bold')\n",
            "else:\n",
            "    axes[1].text(0.5, 0.5, 'No violations found', ha='center', va='center',\n",
            "                 fontsize=14)\n",
            "    axes[1].set_title('Severity Distribution', fontsize=14, fontweight='bold')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()",
        ]),

        # ── Category Breakdown ──────────────────────────────────
        make_md_cell([
            "## Violations by Protected Class\n",
            "\n",
            "Fair Housing Act protects seven classes. This chart shows which\n",
            "categories have the most violations in the dataset.\n",
        ]),

        make_code_cell([
            "cat = summary['category_breakdown']\n",
            "if cat:\n",
            "    cats = sorted(cat.items(), key=lambda x: x[1], reverse=True)\n",
            "    labels = [c[0].replace('_', ' ').title() for c in cats]\n",
            "    values = [c[1] for c in cats]\n",
            "    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))\n",
            "\n",
            "    fig, ax = plt.subplots(figsize=(10, 5))\n",
            "    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1],\n",
            "                   edgecolor='white', height=0.5)\n",
            "    ax.set_xlabel('Violation Count', fontsize=12)\n",
            "    ax.set_title('Violations by Protected Class', fontsize=14, fontweight='bold')\n",
            "\n",
            "    for bar, val in zip(bars, values[::-1]):\n",
            "        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,\n",
            "                str(val), va='center', fontsize=11, fontweight='bold')\n",
            "\n",
            "    plt.tight_layout()\n",
            "    plt.show()\n",
            "else:\n",
            "    print('No violations found in dataset.')",
        ]),

        # ── Recall & Precision ──────────────────────────────────
        make_md_cell([
            "## Recall & Precision Evaluation\n",
            "\n",
            "- **Recall** (on known violations): must be 100% — no false negatives\n",
            "- **Precision** (on clean listings): must be > 80% — minimal false positives\n",
        ]),

        make_code_cell([
            "metrics = {\n",
            "    'Recall': summary['recall'],\n",
            "    'Precision': summary['precision'],\n",
            "}\n",
            "thresholds = {'Recall': 1.0, 'Precision': 0.80}\n",
            "\n",
            "fig, ax = plt.subplots(figsize=(8, 5))\n",
            "x = list(metrics.keys())\n",
            "y = list(metrics.values())\n",
            "colors = ['#2ECC71' if metrics[k] >= thresholds[k] else '#E74C3C' for k in x]\n",
            "\n",
            "bars = ax.bar(x, y, color=colors, edgecolor='white', width=0.4)\n",
            "ax.set_ylabel('Score', fontsize=12)\n",
            "ax.set_title('Recall & Precision', fontsize=14, fontweight='bold')\n",
            "ax.set_ylim(0, 1.15)\n",
            "\n",
            "# Threshold lines\n",
            "ax.axhline(y=1.0, color='#E74C3C', linestyle='--', alpha=0.4, label='Recall target (1.0)')\n",
            "ax.axhline(y=0.8, color='#F39C12', linestyle='--', alpha=0.4, label='Precision target (0.8)')\n",
            "\n",
            "for bar, val in zip(bars, y):\n",
            "    status = '✓ PASS' if val >= list(thresholds.values())[list(thresholds.keys()).index(bar.get_x())] else '✗ FAIL'\n",
            "    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,\n",
            "            f'{val:.2%}', ha='center', fontsize=12, fontweight='bold')\n",
            "\n",
            "ax.legend(fontsize=10)\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "\n",
            "print(f\"Recall:    {summary['recall']:.2%} ({summary['recall_detected']}/{summary['recall_total']})\")\n",
            "print(f\"Precision: {summary['precision']:.2%} ({summary['precision_correct']}/{summary['precision_total']})\")",
        ]),

        # ── Submission Workflow Demo ────────────────────────────
        make_md_cell([
            "## Listing Submission Workflow Demo\n",
            "\n",
            "Integration example showing how the ComplianceChecker fits into\n",
            "a listing submission pipeline with approve / review / block decisions.\n",
        ]),

        make_code_cell([
            "workflow = summary['workflow_demo']\n",
            "rows = []\n",
            "for w in workflow:\n",
            "    status = '✓ Approved' if w['approved'] else ('⚠ Review' if w['needs_review'] else '✗ Blocked')\n",
            "    rows.append({\n",
            "        'Listing': w['listing_id'],\n",
            "        'Status': status,\n",
            "        'Errors': w['errors'],\n",
            "        'Warnings': w['warnings'],\n",
            "        'Info': w['infos'],\n",
            "        'Text': w['text'][:60] + ('...' if len(w['text']) > 60 else ''),\n",
            "    })\n",
            "\n",
            "df_wf = pd.DataFrame(rows)\n",
            "display(df_wf.style.set_caption('Listing Submission Workflow Results'))",
        ]),

        # ── Live Demo ───────────────────────────────────────────
        make_md_cell([
            "## Live Compliance Check Demo\n",
            "\n",
            "Run the checker on sample listings in real time.\n",
        ]),

        make_code_cell([
            "checker = ComplianceChecker()\n",
            "\n",
            "# Clean listing\n",
            "clean = checker.check_listing(\n",
            "    'Beautiful 4 bed, 3 bath home with pool. Near top schools. $599,000.'\n",
            ")\n",
            "print('Clean listing:')\n",
            "print(f\"  Compliant: {clean['compliant']}\")\n",
            "print(f\"  Violations: {len(clean['violations'])}\")\n",
            "\n",
            "# Violation listing\n",
            "print('\\nViolation listing:')\n",
            "bad = checker.check_listing(\n",
            "    'No children allowed. Must be able-bodied. Adults only building.'\n",
            ")\n",
            "print(f\"  Compliant: {bad['compliant']}\")\n",
            "for v in bad['violations']:\n",
            "    print(f\"  [{v['severity'].upper()}] {v['category']}: {v['message'][:70]}\")\n",
            "    fix = checker.suggest_fix(v)\n",
            "    print(f\"    Fix: {fix[:70]}\")",
        ]),

        # ── Sample Violations from Dataset ──────────────────────
        make_md_cell([
            "## Sample Violations from Dataset\n",
            "\n",
            "Examples of flagged listings from the actual dataset.\n",
        ]),

        make_code_cell([
            "samples = summary.get('sample_violations', [])\n",
            "if samples:\n",
            "    rows = []\n",
            "    for s in samples[:8]:\n",
            "        for v in s['violations'][:2]:\n",
            "            rows.append({\n",
            "                'Listing': s['listing_id'],\n",
            "                'Category': v['category'].replace('_', ' ').title(),\n",
            "                'Severity': v['severity'].upper(),\n",
            "                'Pattern': v['pattern'],\n",
            "                'Text': s['text'][:60] + '...',\n",
            "            })\n",
            "    df_v = pd.DataFrame(rows)\n",
            "    display(df_v.style.set_caption('Sample Violations from Dataset'))\n",
            "else:\n",
            "    print('No violations found in dataset — listings are all compliant!')",
        ]),

        # ── Summary ─────────────────────────────────────────────
        make_md_cell([
            "## Summary\n",
            "\n",
            f"Week 9 scanned **{summary['total_listings_scanned']}** listings for Fair Housing compliance:\n",
            "\n",
            f"- **Flagged listings**: {summary['flagged_listings']} ({summary['flagged_pct']}%)\n",
            f"- **Total violations**: {summary['total_violations']}\n",
            f"  - Errors: {summary['severity_breakdown']['error']}\n",
            f"  - Warnings: {summary['severity_breakdown']['warning']}\n",
            f"  - Info: {summary['severity_breakdown']['info']}\n",
            f"- **Recall**: {summary['recall']:.2%} ",
            f"({'PASS' if summary['recall'] >= 1.0 else 'FAIL'})\n",
            f"- **Precision**: {summary['precision']:.2%} ",
            f"({'PASS' if summary['precision'] > 0.80 else 'FAIL'})\n",
            "\n",
            "The `ComplianceChecker` integrates with the listing submission workflow\n",
            "to automatically block, review, or approve listings based on Fair Housing\n",
            "compliance. See the team documentation via `checker.get_fair_housing_summary()`.\n",
        ]),
    ]

    nb['cells'].extend(week9_cells)

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"Added {len(week9_cells)} cells to notebook. Total cells: {len(nb['cells'])}")


if __name__ == '__main__':
    main()
