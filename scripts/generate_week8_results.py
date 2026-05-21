"""
Week 8: Generate Listing Summarization Results

Runs full pipeline:
1. Generates extractive summaries for all listings
2. Builds entity-based reference summaries for ROUGE
3. Computes ROUGE scores
4. Creates 20-listing human evaluation form
5. Demos AnswerabilityChecker
6. Saves week8_summary.json
"""

import sys
import os
import json
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from listing_summarizer import ListingSummarizer
from answerability_checker import AnswerabilityChecker


def main():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    csv_path = os.path.join(base, 'data', 'processed', 'cleaned_listing.csv')
    taxonomy_path = os.path.join(base, 'data', 'processed', 'taxonomy.json')
    schema_path = os.path.join(base, 'data', 'schema.json')
    summaries_path = os.path.join(base, 'data', 'processed', 'week8_summaries.json')
    eval_path = os.path.join(base, 'data', 'processed', 'human_eval_form.json')
    summary_path = os.path.join(base, 'data', 'processed', 'week8_summary.json')

    print("=" * 60)
    print("Week 8: Listing Summarization & Answerability")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1.  Generate extractive summaries
    # ------------------------------------------------------------------
    print("\n-- Generating Summaries --")
    summarizer = ListingSummarizer(taxonomy_path=taxonomy_path)

    import pandas as pd
    df = pd.read_csv(csv_path)

    # Determine text column
    text_col = None
    for col in ('cleaned_remarks', 'remarks', 'L_Remarks'):
        if col in df.columns:
            text_col = col
            break

    results = []
    for idx, row in df.iterrows():
        record = {
            'listing_id': idx,
            'remarks': row[text_col] if pd.notna(row[text_col]) else '',
        }
        result = summarizer.summarize_listing(record)
        result['listing_id'] = idx
        results.append(result)

    # Save summaries
    os.makedirs(os.path.dirname(summaries_path), exist_ok=True)
    with open(summaries_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} summaries to {summaries_path}")

    # Stats
    has_summary = sum(1 for r in results if r['summary'])
    has_beds = sum(1 for r in results if r['beds'] is not None)
    has_baths = sum(1 for r in results if r['baths'] is not None)
    has_price = sum(1 for r in results if r['price'] is not None)
    has_features = sum(1 for r in results if r['top_features'])
    has_location = sum(1 for r in results if r['location'])

    total = len(results)
    stats = {
        'total_listings': total,
        'with_summary_pct': round(has_summary / total * 100, 1),
        'with_beds_pct': round(has_beds / total * 100, 1),
        'with_baths_pct': round(has_baths / total * 100, 1),
        'with_price_pct': round(has_price / total * 100, 1),
        'with_features_pct': round(has_features / total * 100, 1),
        'with_location_pct': round(has_location / total * 100, 1),
    }
    print("\nCoverage:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # ------------------------------------------------------------------
    # 2.  ROUGE evaluation
    # ------------------------------------------------------------------
    print("\n-- ROUGE Evaluation --")

    # Use listings that have rich enough text for meaningful comparison
    eval_indices = [i for i, r in enumerate(results)
                    if r['summary'] and len(r['summary']) > 50]

    # Cap at 200 for efficiency
    if len(eval_indices) > 200:
        random.seed(42)
        eval_indices = random.sample(eval_indices, 200)

    generated = []
    references = []
    for idx in eval_indices:
        generated.append(results[idx]['summary'])
        record = {
            'listing_id': idx,
            'remarks': df.iloc[idx][text_col] if pd.notna(df.iloc[idx][text_col]) else '',
        }
        ref = summarizer.create_reference_summary(record)
        references.append(ref)

    rouge_scores = summarizer.evaluate_rouge(generated, references)
    print(f"  Evaluated on {len(eval_indices)} listings")
    for metric, value in rouge_scores.items():
        print(f"  {metric}: {value:.4f}")

    rouge_target_met = rouge_scores.get('rougeL', 0) > 0.4
    print(f"  ROUGE-L > 0.4: {'PASS' if rouge_target_met else 'FAIL'}")

    # ------------------------------------------------------------------
    # 3.  Human evaluation form (20 samples)
    # ------------------------------------------------------------------
    print("\n-- Human Evaluation Form --")

    # Pick 20 diverse listings with summaries
    eval_candidates = [i for i, r in enumerate(results) if r['summary'] and len(r['summary']) > 30]
    random.seed(42)
    human_eval_indices = random.sample(eval_candidates, min(20, len(eval_candidates)))

    human_eval = []
    for idx in human_eval_indices:
        remarks_text = df.iloc[idx][text_col] if pd.notna(df.iloc[idx][text_col]) else ''
        human_eval.append({
            'listing_id': idx,
            'original_remarks': remarks_text[:500],
            'generated_summary': results[idx]['summary'],
            'beds': results[idx]['beds'],
            'baths': results[idx]['baths'],
            'price': results[idx]['price'],
            'top_features': results[idx]['top_features'],
            'rating': None,          # 1-5 (to be filled by reviewer)
            'completeness': None,    # 1-5 (to be filled)
            'accuracy': None,        # 1-5 (to be filled)
            'feedback': None,        # free text
        })

    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(human_eval, f, indent=2)
    print(f"  Saved {len(human_eval)} entries to {eval_path}")

    # ------------------------------------------------------------------
    # 4.  Answerability demo
    # ------------------------------------------------------------------
    print("\n-- Answerability Checker Demo --")

    checker = AnswerabilityChecker(schema_path=schema_path)

    demo_queries = [
        ("3 bed house in Portland under 500k", True),
        ("homes with pool near good schools", True),
        ("What's the weather like today?", False),
        ("Tell me a joke", False),
        ("condo in Atlantis", False),
        ("2 bath property with garage", True),
    ]

    answerability_results = []
    for query, expected in demo_queries:
        ok, msg = checker.check_pre_query(query)
        status = "Y" if ok else "N"
        match = "MATCH" if ok == expected else "MISMATCH"
        print(f"  {status} [{match}] {msg:50s} | {query}")
        answerability_results.append({
            'query': query,
            'answerable': ok,
            'message': msg,
            'expected': expected,
            'correct': ok == expected,
        })

    ans_accuracy = sum(1 for r in answerability_results if r['correct']) / len(answerability_results)
    print(f"\n  Answerability accuracy: {ans_accuracy:.0%}")

    # ------------------------------------------------------------------
    # 5.  Print sample summaries
    # ------------------------------------------------------------------
    print("\n-- Sample Summaries --")
    for r in results[:5]:
        if r['summary']:
            print(f"\n  ID {r['listing_id']}:")
            print(f"    Beds: {r['beds']}, Baths: {r['baths']}, Price: {r['price']}")
            print(f"    Features: {r['top_features']}")
            print(f"    Location: {r['location']}")
            print(f"    Summary: {r['summary'][:150]}...")

    # ------------------------------------------------------------------
    # 6.  Save summary JSON
    # ------------------------------------------------------------------
    week8_summary = {
        'total_listings': total,
        'coverage': stats,
        'rouge_scores': rouge_scores,
        'rouge_eval_count': len(eval_indices),
        'rouge_l_target_met': rouge_target_met,
        'human_eval_count': len(human_eval),
        'answerability_demo': answerability_results,
        'answerability_accuracy': round(ans_accuracy, 4),
        'sample_summaries': [
            {
                'listing_id': r['listing_id'],
                'summary': r['summary'][:200],
                'beds': r['beds'],
                'baths': r['baths'],
                'price': r['price'],
                'top_features': r['top_features'],
                'location': r['location'],
            }
            for r in results[:10] if r['summary']
        ],
    }

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(week8_summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")
    print("Done!")


if __name__ == '__main__':
    main()
