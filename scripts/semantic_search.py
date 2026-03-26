"""
Week 5: Semantic Search with Embeddings

SemanticSearcher class that provides:
  - Sentence embeddings for listing remarks using sentence-transformers (384 dims)
  - FAISS index for efficient similarity search
  - BM25 keyword search for comparison
  - Comparison study and relevance evaluation
"""

import time
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi


class SemanticSearcher:
    """Embedding-based semantic search with FAISS index and BM25 comparison."""

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the searcher with a sentence-transformer model.

        Args:
            model_name: Name of the sentence-transformers model to use.
                        'all-MiniLM-L6-v2' produces 384-dim embeddings.
        """
        print(f"Loading model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.bm25 = None
        self.listings = None
        self.embeddings = None
        self.tokenized_listings = None
        self.embedding_dim = None

    def build_index(self, remarks_list):
        """
        Build both FAISS and BM25 indices from a list of listing remarks.

        Args:
            remarks_list: List of strings (listing remarks).
        """
        self.listings = list(remarks_list)
        print(f"Encoding {len(self.listings)} listings...")

        # Generate sentence embeddings
        self.embeddings = self.model.encode(self.listings, show_progress_bar=True)
        self.embeddings = np.array(self.embeddings, dtype='float32')
        self.embedding_dim = self.embeddings.shape[1]

        # Build FAISS index (Inner Product for cosine similarity on normalized vectors)
        faiss.normalize_L2(self.embeddings)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(self.embeddings)
        print(f"FAISS index built: {self.index.ntotal} vectors, {self.embedding_dim} dimensions")

        # Build BM25 index
        self.tokenized_listings = [doc.lower().split() for doc in self.listings]
        self.bm25 = BM25Okapi(self.tokenized_listings)
        print("BM25 index built.")

    def search(self, query, top_k=10):
        """
        Semantic search using FAISS.

        Args:
            query: Search query string.
            top_k: Number of results to return.

        Returns:
            List of (listing_text, score) tuples, sorted by score descending.
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        query_emb = self.model.encode([query])
        query_emb = np.array(query_emb, dtype='float32')
        faiss.normalize_L2(query_emb)

        scores, indices = self.index.search(query_emb, top_k)
        results = []
        for j, i in enumerate(indices[0]):
            if i >= 0:  # FAISS returns -1 for missing results
                results.append((self.listings[i], float(scores[0][j])))
        return results

    def bm25_search(self, query, top_k=10):
        """
        Keyword search using BM25.

        Args:
            query: Search query string.
            top_k: Number of results to return.

        Returns:
            List of (listing_text, score) tuples, sorted by score descending.
        """
        if self.bm25 is None:
            raise ValueError("Index not built. Call build_index() first.")

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top_k indices sorted by score
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(self.listings[i], float(scores[i])) for i in top_indices]
        return results

    def compare_search(self, query, top_k=10):
        """
        Run both semantic and BM25 search and return results side-by-side.

        Args:
            query: Search query string.
            top_k: Number of results to return per method.

        Returns:
            Dict with 'semantic' and 'bm25' keys, each containing a list of
            (listing_text, score) tuples.
        """
        return {
            'semantic': self.search(query, top_k),
            'bm25': self.bm25_search(query, top_k)
        }

    def evaluate_relevance(self, query_result_pairs, top_k=5):
        """
        Evaluate search relevance on a set of query-result pairs.

        Each pair is a dict with:
            - 'query': the search query
            - 'relevant_terms': list of terms that should appear in good results

        Computes:
            - Precision@k for both semantic and BM25
            - Mean Reciprocal Rank (MRR) for both
            - Result overlap between the two methods

        Args:
            query_result_pairs: List of dicts with 'query' and 'relevant_terms'.
            top_k: Number of results to evaluate.

        Returns:
            Dict with evaluation metrics.
        """
        semantic_precisions = []
        bm25_precisions = []
        semantic_mrrs = []
        bm25_mrrs = []
        overlaps = []

        for pair in query_result_pairs:
            query = pair['query']
            relevant_terms = [t.lower() for t in pair['relevant_terms']]

            sem_results = self.search(query, top_k)
            bm25_results = self.bm25_search(query, top_k)

            # Precision@k: fraction of results containing any relevant term
            sem_hits = sum(1 for text, _ in sem_results
                          if any(term in text.lower() for term in relevant_terms))
            bm25_hits = sum(1 for text, _ in bm25_results
                           if any(term in text.lower() for term in relevant_terms))

            semantic_precisions.append(sem_hits / top_k if top_k > 0 else 0)
            bm25_precisions.append(bm25_hits / top_k if top_k > 0 else 0)

            # MRR: reciprocal rank of first relevant result
            sem_rr = 0.0
            for rank, (text, _) in enumerate(sem_results, 1):
                if any(term in text.lower() for term in relevant_terms):
                    sem_rr = 1.0 / rank
                    break

            bm25_rr = 0.0
            for rank, (text, _) in enumerate(bm25_results, 1):
                if any(term in text.lower() for term in relevant_terms):
                    bm25_rr = 1.0 / rank
                    break

            semantic_mrrs.append(sem_rr)
            bm25_mrrs.append(bm25_rr)

            # Overlap: how many results appear in both top-k
            sem_texts = set(text for text, _ in sem_results)
            bm25_texts = set(text for text, _ in bm25_results)
            if len(sem_texts | bm25_texts) > 0:
                overlap = len(sem_texts & bm25_texts) / top_k
            else:
                overlap = 0.0
            overlaps.append(overlap)

        return {
            'semantic_precision_at_k': np.mean(semantic_precisions),
            'bm25_precision_at_k': np.mean(bm25_precisions),
            'semantic_mrr': np.mean(semantic_mrrs),
            'bm25_mrr': np.mean(bm25_mrrs),
            'avg_overlap': np.mean(overlaps),
            'num_queries': len(query_result_pairs),
            'top_k': top_k,
            # Per-query details for graphing
            'per_query_semantic_precision': semantic_precisions,
            'per_query_bm25_precision': bm25_precisions,
            'per_query_semantic_mrr': semantic_mrrs,
            'per_query_bm25_mrr': bm25_mrrs,
            'per_query_overlap': overlaps,
        }

    def benchmark_latency(self, queries, num_runs=3):
        """
        Benchmark search latency for both methods.

        Args:
            queries: List of query strings to benchmark.
            num_runs: Number of times to repeat for averaging.

        Returns:
            Dict with latency stats in milliseconds.
        """
        semantic_times = []
        bm25_times = []

        for query in queries:
            # Semantic search timing
            times = []
            for _ in range(num_runs):
                start = time.perf_counter()
                self.search(query, top_k=10)
                end = time.perf_counter()
                times.append((end - start) * 1000)  # ms
            semantic_times.append(np.mean(times))

            # BM25 search timing
            times = []
            for _ in range(num_runs):
                start = time.perf_counter()
                self.bm25_search(query, top_k=10)
                end = time.perf_counter()
                times.append((end - start) * 1000)  # ms
            bm25_times.append(np.mean(times))

        return {
            'semantic_avg_ms': np.mean(semantic_times),
            'semantic_p95_ms': np.percentile(semantic_times, 95),
            'semantic_max_ms': np.max(semantic_times),
            'bm25_avg_ms': np.mean(bm25_times),
            'bm25_p95_ms': np.percentile(bm25_times, 95),
            'bm25_max_ms': np.max(bm25_times),
            'per_query_semantic_ms': semantic_times,
            'per_query_bm25_ms': bm25_times,
        }


def generate_evaluation_pairs():
    """
    Generate 50 query-result pairs for relevance evaluation.
    Each pair has a query and a list of relevant terms that should appear
    in good search results.
    """
    pairs = [
        {"query": "home with a pool", "relevant_terms": ["pool", "swimming"]},
        {"query": "spacious kitchen with granite countertops", "relevant_terms": ["kitchen", "granite"]},
        {"query": "3 bedroom house with garage", "relevant_terms": ["bedroom", "garage"]},
        {"query": "waterfront property with ocean views", "relevant_terms": ["ocean", "water", "view"]},
        {"query": "modern condo downtown", "relevant_terms": ["condo", "modern", "downtown"]},
        {"query": "quiet neighborhood for families", "relevant_terms": ["quiet", "family", "neighborhood"]},
        {"query": "hardwood floors throughout", "relevant_terms": ["hardwood", "floor"]},
        {"query": "newly renovated bathroom", "relevant_terms": ["renovated", "bathroom", "remodel"]},
        {"query": "large backyard for entertaining", "relevant_terms": ["backyard", "yard", "entertain"]},
        {"query": "walk-in closet master bedroom", "relevant_terms": ["closet", "master", "walk"]},
        {"query": "stainless steel appliances", "relevant_terms": ["stainless", "appliance"]},
        {"query": "fireplace in living room", "relevant_terms": ["fireplace", "living"]},
        {"query": "open floor plan", "relevant_terms": ["open", "floor", "plan"]},
        {"query": "energy efficient home", "relevant_terms": ["energy", "efficient", "solar"]},
        {"query": "close to schools and parks", "relevant_terms": ["school", "park"]},
        {"query": "luxury penthouse with views", "relevant_terms": ["luxury", "penthouse", "view"]},
        {"query": "fixer upper with potential", "relevant_terms": ["fixer", "potential", "opportunity"]},
        {"query": "gated community with security", "relevant_terms": ["gated", "community", "security"]},
        {"query": "home with wine cellar", "relevant_terms": ["wine", "cellar"]},
        {"query": "smart home technology", "relevant_terms": ["smart", "technology", "home"]},
        {"query": "vaulted ceilings", "relevant_terms": ["vaulted", "ceiling"]},
        {"query": "updated HVAC system", "relevant_terms": ["hvac", "heating", "cooling", "air"]},
        {"query": "covered patio", "relevant_terms": ["patio", "covered"]},
        {"query": "double car garage", "relevant_terms": ["garage", "car", "parking"]},
        {"query": "corner lot property", "relevant_terms": ["corner", "lot"]},
        {"query": "mountain view home", "relevant_terms": ["mountain", "view"]},
        {"query": "beachfront condo", "relevant_terms": ["beach", "ocean", "condo"]},
        {"query": "home office space", "relevant_terms": ["office", "work", "space"]},
        {"query": "central air conditioning", "relevant_terms": ["air", "conditioning", "central"]},
        {"query": "tile roof", "relevant_terms": ["tile", "roof"]},
        {"query": "mature trees and landscaping", "relevant_terms": ["tree", "landscape", "mature"]},
        {"query": "quartz countertops", "relevant_terms": ["quartz", "counter"]},
        {"query": "recessed lighting", "relevant_terms": ["recessed", "lighting", "light"]},
        {"query": "in-ground spa", "relevant_terms": ["spa", "jacuzzi", "hot tub"]},
        {"query": "plantation shutters", "relevant_terms": ["plantation", "shutter"]},
        {"query": "crown molding", "relevant_terms": ["crown", "molding"]},
        {"query": "laundry room", "relevant_terms": ["laundry", "washer", "dryer"]},
        {"query": "breakfast nook", "relevant_terms": ["breakfast", "nook"]},
        {"query": "dual pane windows", "relevant_terms": ["dual", "window", "pane"]},
        {"query": "tankless water heater", "relevant_terms": ["tankless", "water", "heater"]},
        {"query": "barn door accents", "relevant_terms": ["barn", "door"]},
        {"query": "solar panels installed", "relevant_terms": ["solar", "panel"]},
        {"query": "private balcony", "relevant_terms": ["balcony", "private"]},
        {"query": "community pool and gym", "relevant_terms": ["pool", "gym", "community"]},
        {"query": "RV parking available", "relevant_terms": ["rv", "parking"]},
        {"query": "cul de sac location", "relevant_terms": ["cul", "sac"]},
        {"query": "home with guest house", "relevant_terms": ["guest", "house", "casita"]},
        {"query": "built-in bookshelves", "relevant_terms": ["bookshelf", "built", "shelv"]},
        {"query": "wrap around porch", "relevant_terms": ["wrap", "porch"]},
        {"query": "freshly painted interior", "relevant_terms": ["paint", "fresh", "interior"]},
    ]
    return pairs


if __name__ == '__main__':
    import pandas as pd

    # Load data
    df = pd.read_csv('data/processed/cleaned_listing.csv')
    remarks = df['cleaned_remarks'].dropna().astype(str).tolist()

    # Filter out non-descriptive remarks (too short or just comma-separated values)
    remarks = [r for r in remarks if len(r) > 50 and ' ' in r]

    print(f"Using {len(remarks)} cleaned listing remarks")

    # Build search engine
    searcher = SemanticSearcher()
    searcher.build_index(remarks)

    # Example search
    print("\n" + "=" * 60)
    print("SEMANTIC SEARCH: 'home with pool and large backyard'")
    print("=" * 60)
    results = searcher.search("home with pool and large backyard", top_k=5)
    for i, (text, score) in enumerate(results, 1):
        print(f"\n  #{i} (score: {score:.4f})")
        print(f"  {text[:150]}...")

    print("\n" + "=" * 60)
    print("BM25 SEARCH: 'home with pool and large backyard'")
    print("=" * 60)
    results = searcher.bm25_search("home with pool and large backyard", top_k=5)
    for i, (text, score) in enumerate(results, 1):
        print(f"\n  #{i} (score: {score:.4f})")
        print(f"  {text[:150]}...")

    # Benchmark latency
    print("\n" + "=" * 60)
    print("LATENCY BENCHMARK")
    print("=" * 60)
    sample_queries = [
        "3 bedroom house with pool",
        "modern condo downtown",
        "spacious kitchen granite countertops",
        "ocean view property",
        "quiet neighborhood family home",
    ]
    latency = searcher.benchmark_latency(sample_queries)
    print(f"  Semantic avg: {latency['semantic_avg_ms']:.1f}ms, p95: {latency['semantic_p95_ms']:.1f}ms")
    print(f"  BM25 avg:     {latency['bm25_avg_ms']:.1f}ms, p95: {latency['bm25_p95_ms']:.1f}ms")

    # Evaluate relevance
    print("\n" + "=" * 60)
    print("RELEVANCE EVALUATION (50 queries)")
    print("=" * 60)
    pairs = generate_evaluation_pairs()
    eval_results = searcher.evaluate_relevance(pairs, top_k=5)
    print(f"  Semantic Precision@5: {eval_results['semantic_precision_at_k']:.3f}")
    print(f"  BM25 Precision@5:     {eval_results['bm25_precision_at_k']:.3f}")
    print(f"  Semantic MRR:         {eval_results['semantic_mrr']:.3f}")
    print(f"  BM25 MRR:             {eval_results['bm25_mrr']:.3f}")
    print(f"  Avg Result Overlap:   {eval_results['avg_overlap']:.3f}")
