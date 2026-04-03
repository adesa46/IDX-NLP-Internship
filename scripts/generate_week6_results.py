"""
Week 6: Generate Signal Extraction Results

Processes the entire sample_listing.csv through SignalExtractor,
saves JSON output, and prints coverage statistics and summaries.
"""

import sys
import os
import json
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from signal_extractor import SignalExtractor


def main():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    taxonomy_path = os.path.join(base, 'data', 'processed', 'taxonomy.json')
    csv_path = os.path.join(base, 'data', 'processed', 'sample_listing.csv')
    output_path = os.path.join(base, 'data', 'processed', 'week6_signals.json')

    print("=" * 60)
    print("Week 6: Listing Signal Extraction")
    print("=" * 60)

    se = SignalExtractor(taxonomy_path=taxonomy_path)
    results = se.process_dataset(csv_path, output_path)

    # ─── Coverage stats ────────────────────────────────────────
    stats = se.compute_coverage_stats(results)
    print("\n=== Coverage Statistics ===")
    for k, v in stats.items():
        label = k.replace('_', ' ').title()
        if isinstance(v, float):
            print(f"  {label}: {v}%")
        else:
            print(f"  {label}: {v}")

    # ─── Top amenities ─────────────────────────────────────────
    all_amenities = []
    for r in results:
        all_amenities.extend(r.get('amenities', []))
    top_amenities = Counter(all_amenities).most_common(15)
    print("\n=== Top 15 Amenities ===")
    for term, count in top_amenities:
        print(f"  {term}: {count}")

    # ─── Top condition keywords ─────────────────────────────────
    all_condition = []
    for r in results:
        all_condition.extend(r.get('condition_keywords', []))
    top_condition = Counter(all_condition).most_common(10)
    print("\n=== Top 10 Condition Keywords ===")
    for term, count in top_condition:
        print(f"  {term}: {count}")

    # ─── Top financing terms ────────────────────────────────────
    all_financing = []
    for r in results:
        all_financing.extend(r.get('financing_terms', []))
    top_financing = Counter(all_financing).most_common(10)
    print("\n=== Top 10 Financing Terms ===")
    for term, count in top_financing:
        print(f"  {term}: {count}")

    # ─── Top location features ──────────────────────────────────
    all_location = []
    for r in results:
        all_location.extend(r.get('location_features', []))
    top_location = Counter(all_location).most_common(10)
    print("\n=== Top 10 Location Features ===")
    for term, count in top_location:
        print(f"  {term}: {count}")

    # ─── Sample output ──────────────────────────────────────────
    print("\n=== Sample Extracted Signals (first 3) ===")
    for r in results[:3]:
        print(json.dumps(r, indent=2))
        print("---")

    # ─── Save summary stats for notebook use ────────────────────
    summary_path = os.path.join(base, 'data', 'processed', 'week6_summary.json')
    summary = {
        'coverage': stats,
        'top_amenities': dict(top_amenities),
        'top_condition': dict(top_condition),
        'top_financing': dict(top_financing),
        'top_location': dict(top_location),
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary stats saved to {summary_path}")
    print("Done!")


if __name__ == '__main__':
    main()
