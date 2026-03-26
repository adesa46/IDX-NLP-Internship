"""
Generate Week 5 notebook cells with graphs.

Runs inside Docker to produce the notebook update for data_exploration.ipynb.
Outputs a Python script that can be used to inject cells into the notebook,
or prints results for manual insertion.
"""

import os
import sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from scripts.semantic_search import SemanticSearcher, generate_evaluation_pairs


def main():
    # Set style
    sns.set_theme(style="whitegrid")

    # Load data
    df = pd.read_csv('data/processed/cleaned_listing.csv')
    remarks = df['cleaned_remarks'].dropna().astype(str).tolist()
    remarks = [r for r in remarks if len(r) > 50 and ' ' in r]
    print(f"Using {len(remarks)} cleaned listing remarks")

    # Build searcher
    searcher = SemanticSearcher()
    searcher.build_index(remarks)

    # === 1. Example search results ===
    example_query = "modern home with pool and spacious kitchen"
    comparison = searcher.compare_search(example_query, top_k=5)

    print(f"\n{'='*60}")
    print(f"Example Query: '{example_query}'")
    print(f"{'='*60}")
    print("\nSemantic Search Results:")
    for i, (text, score) in enumerate(comparison['semantic'], 1):
        print(f"  #{i} (score: {score:.4f}) {text[:120]}...")
    print("\nBM25 Search Results:")
    for i, (text, score) in enumerate(comparison['bm25'], 1):
        print(f"  #{i} (score: {score:.4f}) {text[:120]}...")

    # === 2. Relevance Evaluation ===
    pairs = generate_evaluation_pairs()
    eval_results = searcher.evaluate_relevance(pairs, top_k=5)

    print(f"\n{'='*60}")
    print("Relevance Evaluation (50 queries, top-5)")
    print(f"{'='*60}")
    print(f"  Semantic Precision@5: {eval_results['semantic_precision_at_k']:.3f}")
    print(f"  BM25 Precision@5:     {eval_results['bm25_precision_at_k']:.3f}")
    print(f"  Semantic MRR:         {eval_results['semantic_mrr']:.3f}")
    print(f"  BM25 MRR:             {eval_results['bm25_mrr']:.3f}")
    print(f"  Avg Overlap:          {eval_results['avg_overlap']:.3f}")

    # === 3. Latency Benchmark ===
    sample_queries = [p['query'] for p in pairs[:20]]
    latency = searcher.benchmark_latency(sample_queries, num_runs=3)

    print(f"\n{'='*60}")
    print("Latency Benchmark (20 queries, 3 runs each)")
    print(f"{'='*60}")
    print(f"  Semantic avg: {latency['semantic_avg_ms']:.1f}ms, p95: {latency['semantic_p95_ms']:.1f}ms")
    print(f"  BM25 avg:     {latency['bm25_avg_ms']:.1f}ms, p95: {latency['bm25_p95_ms']:.1f}ms")

    # === Generate Graphs ===
    os.makedirs('data/processed/week5_graphs', exist_ok=True)

    # Graph 1: Precision@5 comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    methods = ['Semantic\n(FAISS)', 'BM25\n(Keyword)']
    precisions = [eval_results['semantic_precision_at_k'],
                  eval_results['bm25_precision_at_k']]
    colors = ['#4C72B0', '#DD8452']
    bars = ax.bar(methods, precisions, color=colors, width=0.5, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Precision@5')
    ax.set_title('Semantic vs BM25: Precision@5 on 50 Queries')
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, precisions):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig('data/processed/week5_graphs/precision_comparison.png', dpi=150)
    plt.close()

    # Graph 2: MRR comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    mrrs = [eval_results['semantic_mrr'], eval_results['bm25_mrr']]
    bars = ax.bar(methods, mrrs, color=colors, width=0.5, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Mean Reciprocal Rank')
    ax.set_title('Semantic vs BM25: MRR on 50 Queries')
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, mrrs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig('data/processed/week5_graphs/mrr_comparison.png', dpi=150)
    plt.close()

    # Graph 3: Latency comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    latencies = [latency['semantic_avg_ms'], latency['bm25_avg_ms']]
    bars = ax.bar(methods, latencies, color=colors, width=0.5, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Average Latency (ms)')
    ax.set_title('Semantic vs BM25: Search Latency')
    ax.axhline(y=100, color='red', linestyle='--', linewidth=1, label='100ms target')
    ax.legend()
    for bar, val in zip(bars, latencies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}ms', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig('data/processed/week5_graphs/latency_comparison.png', dpi=150)
    plt.close()

    # Graph 4: Per-query precision scatter
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(pairs))
    ax.scatter(x, eval_results['per_query_semantic_precision'], label='Semantic', alpha=0.7, s=30, color=colors[0])
    ax.scatter(x, eval_results['per_query_bm25_precision'], label='BM25', alpha=0.7, s=30, color=colors[1])
    ax.set_xlabel('Query Index')
    ax.set_ylabel('Precision@5')
    ax.set_title('Per-Query Precision@5: Semantic vs BM25')
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig('data/processed/week5_graphs/per_query_precision.png', dpi=150)
    plt.close()

    # Graph 5: Per-query overlap
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, eval_results['per_query_overlap'], color='#55A868', alpha=0.8)
    ax.set_xlabel('Query Index')
    ax.set_ylabel('Result Overlap (fraction)')
    ax.set_title('Result Overlap Between Semantic and BM25 (per query)')
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig('data/processed/week5_graphs/result_overlap.png', dpi=150)
    plt.close()

    print(f"\nGraphs saved to data/processed/week5_graphs/")

    # === Save evaluation results as JSON ===
    save_results = {
        'semantic_precision_at_k': float(eval_results['semantic_precision_at_k']),
        'bm25_precision_at_k': float(eval_results['bm25_precision_at_k']),
        'semantic_mrr': float(eval_results['semantic_mrr']),
        'bm25_mrr': float(eval_results['bm25_mrr']),
        'avg_overlap': float(eval_results['avg_overlap']),
        'num_queries': eval_results['num_queries'],
        'top_k': eval_results['top_k'],
        'semantic_avg_latency_ms': float(latency['semantic_avg_ms']),
        'bm25_avg_latency_ms': float(latency['bm25_avg_ms']),
        'semantic_p95_latency_ms': float(latency['semantic_p95_ms']),
        'bm25_p95_latency_ms': float(latency['bm25_p95_ms']),
        'num_listings': len(remarks),
        'embedding_dim': searcher.embedding_dim,
    }
    with open('data/processed/week5_results.json', 'w') as f:
        json.dump(save_results, f, indent=2)

    print("Results saved to data/processed/week5_results.json")


if __name__ == '__main__':
    main()
