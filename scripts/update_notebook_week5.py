"""Update the Jupyter notebook with Week 5 cells."""
import json
import sys

def main():
    nb_path = 'notebooks/data_exploration.ipynb'
    
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    new_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Week 5: Semantic Search with Embeddings\n",
                "\n",
                "In Week 5, we implemented embedding-based semantic search using `sentence-transformers` and `FAISS`, and compared it with BM25 keyword search.\n",
                "\n",
                "**Key Components:**\n",
                "- `SemanticSearcher` class with FAISS index (384-dim embeddings from `all-MiniLM-L6-v2`)\n",
                "- BM25 keyword search for comparison (using `rank_bm25`)\n",
                "- Relevance evaluation on 50 query-result pairs\n",
                "- Latency benchmarking\n",
                "\n",
                "All heavy ML dependencies run inside Docker to keep the local environment clean."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import json\n",
                "from IPython.display import Image, display\n",
                "\n",
                "# Load pre-computed results (generated inside Docker)\n",
                "with open('../data/processed/week5_results.json', 'r') as f:\n",
                "    week5_results = json.load(f)\n",
                "\n",
                "print('=== Week 5 Evaluation Summary ===')\n",
                "print(f'Number of listings indexed: {week5_results[\"num_listings\"]}')\n",
                "print(f'Embedding dimensions: {week5_results[\"embedding_dim\"]}')\n",
                "print(f'Number of evaluation queries: {week5_results[\"num_queries\"]}')\n",
                "print(f'Top-k evaluated: {week5_results[\"top_k\"]}')\n",
                "print()\n",
                "print('--- Precision@5 ---')\n",
                "print(f'  Semantic (FAISS): {week5_results[\"semantic_precision_at_k\"]:.3f}')\n",
                "print(f'  BM25 (Keyword):   {week5_results[\"bm25_precision_at_k\"]:.3f}')\n",
                "print()\n",
                "print('--- Mean Reciprocal Rank ---')\n",
                "print(f'  Semantic (FAISS): {week5_results[\"semantic_mrr\"]:.3f}')\n",
                "print(f'  BM25 (Keyword):   {week5_results[\"bm25_mrr\"]:.3f}')\n",
                "print()\n",
                "print('--- Latency (avg) ---')\n",
                "print(f'  Semantic: {week5_results[\"semantic_avg_latency_ms\"]:.1f}ms')\n",
                "print(f'  BM25:     {week5_results[\"bm25_avg_latency_ms\"]:.1f}ms')\n",
                "print()\n",
                "print(f'Average result overlap: {week5_results[\"avg_overlap\"]:.1%}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Precision@5: Semantic vs BM25\n",
                "Fraction of top-5 results containing relevant terms for each of 50 queries."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "display(Image(filename='../data/processed/week5_graphs/precision_comparison.png'))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Mean Reciprocal Rank: Semantic vs BM25\n",
                "MRR measures how quickly the first relevant result appears in the ranking."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "display(Image(filename='../data/processed/week5_graphs/mrr_comparison.png'))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Search Latency: Semantic vs BM25\n",
                "Both methods achieve latency well under the 100ms target."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "display(Image(filename='../data/processed/week5_graphs/latency_comparison.png'))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Per-Query Precision@5: Semantic vs BM25\n",
                "Shows how precision varies across all 50 evaluation queries for each method."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "display(Image(filename='../data/processed/week5_graphs/per_query_precision.png'))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Result Overlap Between Methods\n",
                "Shows what fraction of top-5 results are shared between semantic and BM25 search.\n",
                "Low overlap indicates the methods find fundamentally different relevant listings."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "display(Image(filename='../data/processed/week5_graphs/result_overlap.png'))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Week 5 Conclusions\n",
                "\n",
                "1. **BM25 outperforms semantic search on term-matching precision** (0.968 vs 0.784), which is expected since our evaluation uses keyword-based relevance labels.\n",
                "2. **Semantic search captures meaning beyond exact keywords** \u2014 it finds conceptually related listings even when query terms don\u2019t appear verbatim.\n",
                "3. **Very low result overlap (15.2%)** between methods shows they are complementary \u2014 combining both would improve coverage.\n",
                "4. **Both methods meet the latency requirement** with avg latency well under 100ms.\n",
                "5. **Recommendation**: Use a hybrid approach combining BM25 for keyword precision and semantic search for conceptual relevance."
            ]
        }
    ]

    nb['cells'].extend(new_cells)

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f'Notebook updated: added {len(new_cells)} cells for Week 5')

if __name__ == '__main__':
    main()
