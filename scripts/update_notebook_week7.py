"""
Update data_exploration.ipynb with Week 7 intent classification visuals.
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
    summary_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'week7_summary.json')

    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Load summary for actual values in markdown
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)

    week7_cells = [
        # ── Title ────────────────────────────────────────────────
        make_md_cell([
            "# Week 7: Buyer Intent Classification\n",
            "\n",
            "Classify user queries into intent categories:\n",
            "- **Browsing** — casual look, window shopping\n",
            "- **Researching** — gathering info, comparing options\n",
            "- **Ready to buy** — specific criteria, action-oriented\n",
            "\n",
            "Uses TF-IDF + Logistic Regression with confidence scores.\n",
            "Integrated with Week 4 QueryParser for richer output."
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
            "from intent_classifier import IntentClassifier, IntentQueryParser\n",
            "from query_parser import QueryParser\n",
            "\n",
            "# Load summary\n",
            "with open('../data/processed/week7_summary.json', 'r') as f:\n",
            "    summary = json.load(f)\n",
            "\n",
            "# Load dataset\n",
            "with open('../data/processed/intent_dataset.json', 'r') as f:\n",
            "    dataset = json.load(f)\n",
            "\n",
            "print(f\"Dataset: {summary['total_queries']} queries\")\n",
            "print(f\"Train: {summary['train_size']}, Test: {summary['test_size']}\")\n",
            "print(f\"Accuracy: {summary['accuracy']:.2%}\")"
        ]),

        # ── Intent Distribution ──────────────────────────────────
        make_md_cell([
            "## Dataset Intent Distribution\n",
            "\n",
            "Class balance across the three intent categories."
        ]),

        make_code_cell([
            "dist = summary['dataset_distribution']\n",
            "labels = list(dist.keys())\n",
            "values = list(dist.values())\n",
            "colors = ['#3498DB', '#E74C3C', '#2ECC71']\n",
            "\n",
            "fig, ax = plt.subplots(figsize=(8, 5))\n",
            "bars = ax.bar(labels, values, color=colors, edgecolor='white', width=0.5)\n",
            "ax.set_ylabel('Number of Queries', fontsize=12)\n",
            "ax.set_title('Intent Distribution in Dataset', fontsize=14, fontweight='bold')\n",
            "\n",
            "for bar, val in zip(bars, values):\n",
            "    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,\n",
            "            str(val), ha='center', fontsize=11, fontweight='bold')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]),

        # ── Confusion Matrix ─────────────────────────────────────
        make_md_cell([
            "## Confusion Matrix\n",
            "\n",
            "Actual vs predicted intent on the held-out test set."
        ]),

        make_code_cell([
            "cm = np.array(summary['confusion_matrix'])\n",
            "cm_labels = summary['labels']\n",
            "\n",
            "fig, ax = plt.subplots(figsize=(7, 6))\n",
            "im = ax.imshow(cm, interpolation='nearest', cmap='Blues')\n",
            "ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')\n",
            "plt.colorbar(im, ax=ax)\n",
            "\n",
            "ax.set_xticks(range(len(cm_labels)))\n",
            "ax.set_yticks(range(len(cm_labels)))\n",
            "ax.set_xticklabels(cm_labels, fontsize=11)\n",
            "ax.set_yticklabels(cm_labels, fontsize=11)\n",
            "ax.set_xlabel('Predicted', fontsize=12)\n",
            "ax.set_ylabel('Actual', fontsize=12)\n",
            "\n",
            "# Annotate cells\n",
            "for i in range(len(cm_labels)):\n",
            "    for j in range(len(cm_labels)):\n",
            "        color = 'white' if cm[i, j] > cm.max()/2 else 'black'\n",
            "        ax.text(j, i, str(cm[i, j]), ha='center', va='center',\n",
            "                fontsize=14, fontweight='bold', color=color)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]),

        # ── Precision / Recall / F1 ─────────────────────────────
        make_md_cell([
            "## Per-Class Precision, Recall & F1\n",
            "\n",
            "Performance breakdown for each intent category."
        ]),

        make_code_cell([
            "metrics = summary['per_class_metrics']\n",
            "m_labels = list(metrics.keys())\n",
            "precision = [metrics[l]['precision'] for l in m_labels]\n",
            "recall = [metrics[l]['recall'] for l in m_labels]\n",
            "f1 = [metrics[l]['f1'] for l in m_labels]\n",
            "\n",
            "x = np.arange(len(m_labels))\n",
            "width = 0.25\n",
            "\n",
            "fig, ax = plt.subplots(figsize=(9, 5))\n",
            "b1 = ax.bar(x - width, precision, width, label='Precision', color='#3498DB', edgecolor='white')\n",
            "b2 = ax.bar(x, recall, width, label='Recall', color='#E74C3C', edgecolor='white')\n",
            "b3 = ax.bar(x + width, f1, width, label='F1-Score', color='#2ECC71', edgecolor='white')\n",
            "\n",
            "ax.set_ylabel('Score', fontsize=12)\n",
            "ax.set_title('Per-Class Classification Metrics', fontsize=14, fontweight='bold')\n",
            "ax.set_xticks(x)\n",
            "ax.set_xticklabels(m_labels, fontsize=11)\n",
            "ax.set_ylim(0, 1.15)\n",
            "ax.legend(fontsize=10)\n",
            "\n",
            "for bars in [b1, b2, b3]:\n",
            "    for bar in bars:\n",
            "        h = bar.get_height()\n",
            "        ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,\n",
            "                f'{h:.2f}', ha='center', fontsize=9)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]),

        # ── Confidence Distribution ──────────────────────────────
        make_md_cell([
            "## Confidence Score Distribution\n",
            "\n",
            "Average model confidence when predicting each intent class."
        ]),

        make_code_cell([
            "avg_conf = summary['avg_confidence_by_intent']\n",
            "c_labels = list(avg_conf.keys())\n",
            "c_values = list(avg_conf.values())\n",
            "colors = ['#3498DB', '#E74C3C', '#2ECC71']\n",
            "\n",
            "fig, ax = plt.subplots(figsize=(8, 5))\n",
            "bars = ax.bar(c_labels, c_values, color=colors, edgecolor='white', width=0.5)\n",
            "ax.set_ylabel('Average Confidence', fontsize=12)\n",
            "ax.set_title('Average Prediction Confidence by Intent', fontsize=14, fontweight='bold')\n",
            "ax.set_ylim(0, 1.1)\n",
            "\n",
            "for bar, val in zip(bars, c_values):\n",
            "    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,\n",
            "            f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]),

        # ── Sample Predictions Table ─────────────────────────────
        make_md_cell([
            "## Sample Predictions\n",
            "\n",
            "Model predictions with confidence scores on representative queries."
        ]),

        make_code_cell([
            "samples = summary['sample_predictions']\n",
            "rows = []\n",
            "for s in samples:\n",
            "    rows.append({\n",
            "        'Query': s['query'][:55] + ('...' if len(s['query']) > 55 else ''),\n",
            "        'Predicted Intent': s['predicted_intent'],\n",
            "        'Confidence': f\"{s['confidence']:.3f}\",\n",
            "    })\n",
            "\n",
            "df = pd.DataFrame(rows)\n",
            "display(df.style.set_caption('Sample Predictions with Confidence Scores'))"
        ]),

        # ── Integration Demo ─────────────────────────────────────
        make_md_cell([
            "## Integration with QueryParser (Week 4)\n",
            "\n",
            "Combined output showing intent classification alongside\n",
            "structured query parsing and SQL generation."
        ]),

        make_code_cell([
            "# Load trained model\n",
            "clf = IntentClassifier()\n",
            "clf.load_model('../data/models/intent_model.pkl')\n",
            "parser = QueryParser()\n",
            "iqp = IntentQueryParser(clf, parser)\n",
            "\n",
            "demo_queries = [\n",
            "    \"What's available in Portland?\",\n",
            "    \"Compare condos vs townhouses in Irvine\",\n",
            "    \"3 bed 2 bath under 500k in Portland with pool\",\n",
            "    \"Just browsing homes\",\n",
            "    \"How much do houses cost in the suburbs?\",\n",
            "]\n",
            "\n",
            "for q in demo_queries:\n",
            "    r = iqp.analyze(q)\n",
            "    print(f\"Query:      {r['query']}\")\n",
            "    print(f\"Intent:     {r['intent']} (confidence: {r['confidence']:.3f})\")\n",
            "    print(f\"Filters:    {r['filters']}\")\n",
            "    print(f\"SQL:        {r['sql']}\")\n",
            "    print(f\"Params:     {r['params']}\")\n",
            "    print('-' * 60)"
        ]),

        # ── Summary ──────────────────────────────────────────────
        make_md_cell([
            "## Summary\n",
            "\n",
            f"Week 7 intent classification trained on **{summary['train_size']}** queries ",
            f"and tested on **{summary['test_size']}** queries, achieving:\n",
            "\n",
            f"- **Accuracy**: {summary['accuracy']:.2%}  \n",
            "- **3 intent classes**: browsing, researching, ready_to_buy\n",
            "- **Confidence scores** for every prediction\n",
            "- **Full integration** with Week 4 QueryParser\n",
            "\n",
            "The model is saved and can be loaded for inference with `IntentClassifier.load_model()`."
        ]),
    ]

    nb['cells'].extend(week7_cells)

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"Added {len(week7_cells)} cells to notebook. Total cells: {len(nb['cells'])}")


if __name__ == '__main__':
    main()
