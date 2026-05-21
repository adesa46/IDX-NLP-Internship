"""
Update data_exploration.ipynb with Week 6 signal extraction visuals.
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
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    week6_cells = [
        make_md_cell([
            "# Week 6: Listing Signal Extraction\n",
            "\n",
            "Extract comprehensive structured signals from each listing by combining\n",
            "Week 3 entity extraction with taxonomy-based pattern matching.\n",
            "Signals include: amenities, condition, financing, and location features."
        ]),

        make_code_cell([
            "import sys, os, json\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import matplotlib\n",
            "matplotlib.rcParams['figure.dpi'] = 120\n",
            "matplotlib.rcParams['figure.figsize'] = (10, 6)\n",
            "\n",
            "sys.path.insert(0, os.path.abspath('../scripts'))\n",
            "from signal_extractor import SignalExtractor\n",
            "\n",
            "# Load pre-computed results\n",
            "with open('../data/processed/week6_signals.json', 'r') as f:\n",
            "    signals = json.load(f)\n",
            "with open('../data/processed/week6_summary.json', 'r') as f:\n",
            "    summary = json.load(f)\n",
            "\n",
            "print(f'Loaded {len(signals)} listing signals')\n",
            "print(f'Summary keys: {list(summary.keys())}')"
        ]),

        make_md_cell([
            "## Coverage Statistics\n",
            "\n",
            "Percentage of listings where each signal type was successfully extracted."
        ]),

        make_code_cell([
            "coverage = summary['coverage']\n",
            "\n",
            "# Filter to percentage fields only\n",
            "pct_keys = [k for k in coverage if k.endswith('_pct')]\n",
            "labels = [k.replace('_pct','').replace('_',' ').title() for k in pct_keys]\n",
            "values = [coverage[k] for k in pct_keys]\n",
            "\n",
            "colors = ['#4C72B0','#55A868','#C44E52','#8172B2',\n",
            "          '#CCB974','#64B5CD','#E5AE38','#6D904F','#8B8B8B','#D65F5F']\n",
            "\n",
            "fig, ax = plt.subplots(figsize=(12, 6))\n",
            "bars = ax.barh(labels, values, color=colors[:len(labels)], edgecolor='white', height=0.6)\n",
            "ax.set_xlabel('Coverage (%)', fontsize=12)\n",
            "ax.set_title('Signal Extraction Coverage by Type', fontsize=14, fontweight='bold')\n",
            "ax.set_xlim(0, 100)\n",
            "\n",
            "for bar, val in zip(bars, values):\n",
            "    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,\n",
            "            f'{val:.1f}%', va='center', fontsize=10)\n",
            "\n",
            "ax.invert_yaxis()\n",
            "plt.tight_layout()\n",
            "plt.savefig('../data/processed/week6_coverage.png', bbox_inches='tight')\n",
            "plt.show()"
        ]),

        make_md_cell([
            "## Top 15 Amenities by Frequency\n",
            "\n",
            "Most commonly detected amenities across all 1,000 listings."
        ]),

        make_code_cell([
            "amenities = summary['top_amenities']\n",
            "terms = list(amenities.keys())[:15]\n",
            "counts = list(amenities.values())[:15]\n",
            "\n",
            "fig, ax = plt.subplots(figsize=(12, 7))\n",
            "bars = ax.barh(terms[::-1], counts[::-1],\n",
            "               color=plt.cm.viridis([i/15 for i in range(15)]),\n",
            "               edgecolor='white', height=0.65)\n",
            "ax.set_xlabel('Frequency', fontsize=12)\n",
            "ax.set_title('Top 15 Amenities Detected in Listings', fontsize=14, fontweight='bold')\n",
            "\n",
            "for bar, val in zip(bars, counts[::-1]):\n",
            "    ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,\n",
            "            str(val), va='center', fontsize=10)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.savefig('../data/processed/week6_amenities.png', bbox_inches='tight')\n",
            "plt.show()"
        ]),

        make_md_cell([
            "## Condition Keywords Distribution\n",
            "\n",
            "How often different condition/renovation keywords appear in listing remarks."
        ]),

        make_code_cell([
            "condition = summary['top_condition']\n",
            "c_terms = list(condition.keys())\n",
            "c_counts = list(condition.values())\n",
            "\n",
            "fig, ax = plt.subplots(figsize=(10, 6))\n",
            "bars = ax.bar(c_terms, c_counts,\n",
            "              color=['#E74C3C','#E67E22','#F1C40F','#2ECC71','#1ABC9C',\n",
            "                     '#3498DB','#9B59B6','#34495E','#95A5A6','#D35400'],\n",
            "              edgecolor='white')\n",
            "ax.set_ylabel('Frequency', fontsize=12)\n",
            "ax.set_title('Condition Keywords in Listings', fontsize=14, fontweight='bold')\n",
            "plt.xticks(rotation=45, ha='right', fontsize=10)\n",
            "\n",
            "for bar, val in zip(bars, c_counts):\n",
            "    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,\n",
            "            str(val), ha='center', fontsize=9)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.savefig('../data/processed/week6_condition.png', bbox_inches='tight')\n",
            "plt.show()"
        ]),

        make_md_cell([
            "## Signal Category Distribution (Pie Chart)\n",
            "\n",
            "Distribution of listings by which signal categories were extracted."
        ]),

        make_code_cell([
            "from collections import Counter\n",
            "\n",
            "# Count how many listings have each signal type\n",
            "cat_counts = Counter()\n",
            "for s in signals:\n",
            "    if s.get('amenities'):      cat_counts['Amenities'] += 1\n",
            "    if s.get('condition_keywords'): cat_counts['Condition'] += 1\n",
            "    if s.get('financing_terms'):    cat_counts['Financing'] += 1\n",
            "    if s.get('location_features'):  cat_counts['Location'] += 1\n",
            "    entities = s.get('entities', {})\n",
            "    if any(entities.get(k) is not None for k in ['bedrooms','bathrooms','price','sqft']):\n",
            "        cat_counts['Structured Entities'] += 1\n",
            "\n",
            "fig, ax = plt.subplots(figsize=(8, 8))\n",
            "pie_colors = ['#3498DB','#E74C3C','#2ECC71','#F39C12','#9B59B6']\n",
            "wedges, texts, autotexts = ax.pie(\n",
            "    cat_counts.values(), labels=cat_counts.keys(),\n",
            "    autopct='%1.1f%%', startangle=140, colors=pie_colors,\n",
            "    textprops={'fontsize': 11},\n",
            "    wedgeprops={'edgecolor': 'white', 'linewidth': 2})\n",
            "ax.set_title('Signal Category Distribution', fontsize=14, fontweight='bold')\n",
            "plt.tight_layout()\n",
            "plt.savefig('../data/processed/week6_pie.png', bbox_inches='tight')\n",
            "plt.show()"
        ]),

        make_md_cell([
            "## Sample Extracted Signals\n",
            "\n",
            "A table showing extracted signals for 5 sample listings."
        ]),

        make_code_cell([
            "# Build a summary table\n",
            "rows = []\n",
            "for s in signals[:10]:\n",
            "    ents = s.get('entities', {})\n",
            "    rows.append({\n",
            "        'ID': s['listing_id'],\n",
            "        'Beds': ents.get('bedrooms', '-'),\n",
            "        'Baths': ents.get('bathrooms', '-'),\n",
            "        'SqFt': ents.get('sqft', '-'),\n",
            "        'Amenities': ', '.join(s.get('amenities', [])[:4]) or '-',\n",
            "        'Condition': ', '.join(s.get('condition_keywords', [])[:3]) or '-',\n",
            "        'Financing': ', '.join(s.get('financing_terms', [])[:3]) or '-',\n",
            "        'Location': ', '.join(s.get('location_features', [])[:3]) or '-',\n",
            "    })\n",
            "\n",
            "df_table = pd.DataFrame(rows)\n",
            "display(df_table.style.set_caption('Sample Extracted Signals (first 10 listings)'))"
        ]),

        make_md_cell([
            "## Financing & Location Features\n",
            "\n",
            "Top financing terms and location features detected across listings."
        ]),

        make_code_cell([
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))\n",
            "\n",
            "# Financing\n",
            "fin = summary['top_financing']\n",
            "f_terms = list(fin.keys())[:8]\n",
            "f_counts = list(fin.values())[:8]\n",
            "ax1.barh(f_terms[::-1], f_counts[::-1],\n",
            "         color=plt.cm.Oranges([0.3 + i*0.08 for i in range(len(f_terms))]),\n",
            "         edgecolor='white')\n",
            "ax1.set_xlabel('Frequency')\n",
            "ax1.set_title('Top Financing Terms', fontsize=13, fontweight='bold')\n",
            "for i, (t, c) in enumerate(zip(f_terms[::-1], f_counts[::-1])):\n",
            "    ax1.text(c + 0.3, i, str(c), va='center', fontsize=9)\n",
            "\n",
            "# Location\n",
            "loc = summary['top_location']\n",
            "l_terms = list(loc.keys())[:8]\n",
            "l_counts = list(loc.values())[:8]\n",
            "ax2.barh(l_terms[::-1], l_counts[::-1],\n",
            "         color=plt.cm.Greens([0.3 + i*0.08 for i in range(len(l_terms))]),\n",
            "         edgecolor='white')\n",
            "ax2.set_xlabel('Frequency')\n",
            "ax2.set_title('Top Location Features', fontsize=13, fontweight='bold')\n",
            "for i, (t, c) in enumerate(zip(l_terms[::-1], l_counts[::-1])):\n",
            "    ax2.text(c + 0.3, i, str(c), va='center', fontsize=9)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.savefig('../data/processed/week6_fin_loc.png', bbox_inches='tight')\n",
            "plt.show()"
        ]),

        make_md_cell([
            "## Summary\n",
            "\n",
            "Week 6 signal extraction processed **1,000 listings** and extracted:\n",
            "- **Amenities** from 51.6% of listings\n",
            "- **Condition keywords** from 21.1% of listings\n",
            "- **Location features** from 23.9% of listings\n",
            "- **54.3%** of listings had at least one signal extracted\n",
            "\n",
            "The output JSON is suitable for search indexing and filtering."
        ]),
    ]

    nb['cells'].extend(week6_cells)

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"Added {len(week6_cells)} cells to notebook. Total cells: {len(nb['cells'])}")

if __name__ == '__main__':
    main()
