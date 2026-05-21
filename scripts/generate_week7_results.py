"""
Week 7: Generate Intent Classification Results

Trains IntentClassifier, evaluates on test set, saves model and summary JSON.
"""

import sys
import os
import json
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from intent_classifier import IntentClassifier, IntentQueryParser
from query_parser import QueryParser
from sklearn.model_selection import train_test_split


def main():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dataset_path = os.path.join(base, 'data', 'processed', 'intent_dataset.json')
    model_path = os.path.join(base, 'data', 'models', 'intent_model.pkl')
    summary_path = os.path.join(base, 'data', 'processed', 'week7_summary.json')

    print("=" * 60)
    print("Week 7: Buyer Intent Classification")
    print("=" * 60)

    # Load dataset
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    queries = [d['query'] for d in dataset]
    labels = [d['intent'] for d in dataset]

    print(f"\nDataset: {len(queries)} queries")
    counts = Counter(labels)
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")

    # Train/test split
    train_q, test_q, train_l, test_l = train_test_split(
        queries, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"\nTrain: {len(train_q)}, Test: {len(test_q)}")

    # Train
    clf = IntentClassifier()
    clf.train(train_q, train_l)
    print("Model trained.")

    # Evaluate
    results = clf.evaluate(test_q, test_l)
    print(f"\nAccuracy: {results['accuracy']:.4f}")
    print(f"\n{results['classification_report']}")
    print(f"Confusion Matrix:")
    for row in results['confusion_matrix']:
        print(f"  {row}")

    # Save model
    clf.save_model(model_path)
    print(f"\nModel saved to {model_path}")

    # Sample predictions with confidence
    print("\n=== Sample Predictions ===")
    sample_queries = [
        "What's available in Portland?",
        "Show me what's on the market",
        "How do condos compare to townhouses?",
        "Compare prices in the suburbs vs downtown",
        "3 bed 2 bath under $500k in Portland with pool",
        "I need a condo with garage under $600k",
        "Schedule a showing for condos in downtown",
        "Any nice neighborhoods to explore?",
        "Is Portland a good place to invest?",
        "Pre-approved for $750k, find me 4 bed homes in Irvine",
    ]
    sample_results = []
    for q in sample_queries:
        intent, conf = clf.predict(q)
        probas = clf.predict_proba(q)
        print(f"  [{intent:>12}] ({conf:.3f}) {q}")
        sample_results.append({
            'query': q,
            'predicted_intent': intent,
            'confidence': round(conf, 4),
            'probabilities': {k: round(v, 4) for k, v in probas.items()},
        })

    # Integration demo
    print("\n=== Integration with QueryParser ===")
    parser = QueryParser()
    iqp = IntentQueryParser(clf, parser)
    for q in sample_queries[:5]:
        analysis = iqp.analyze(q)
        print(f"  Query: {analysis['query']}")
        print(f"  Intent: {analysis['intent']} ({analysis['confidence']:.3f})")
        print(f"  Filters: {analysis['filters']}")
        print(f"  SQL: {analysis['sql']}")
        print()

    # Confidence distribution per intent
    all_preds = clf.predict_batch(test_q)
    confidence_by_intent = {}
    for (pred_intent, conf), true_label in zip(all_preds, test_l):
        confidence_by_intent.setdefault(pred_intent, []).append(conf)

    avg_confidence = {}
    for intent, confs in confidence_by_intent.items():
        avg_confidence[intent] = round(sum(confs) / len(confs), 4)

    print("\n=== Average Confidence by Intent ===")
    for intent, avg in sorted(avg_confidence.items()):
        print(f"  {intent}: {avg:.4f}")

    # Per-class metrics for summary
    from sklearn.metrics import precision_recall_fscore_support
    preds_all = [p[0] for p in all_preds]
    prec, rec, f1, sup = precision_recall_fscore_support(
        test_l, preds_all, labels=clf.labels, average=None
    )
    per_class = {}
    for i, label in enumerate(clf.labels):
        per_class[label] = {
            'precision': round(float(prec[i]), 4),
            'recall': round(float(rec[i]), 4),
            'f1': round(float(f1[i]), 4),
            'support': int(sup[i]),
        }

    # Save summary
    summary = {
        'total_queries': len(queries),
        'train_size': len(train_q),
        'test_size': len(test_q),
        'dataset_distribution': dict(sorted(counts.items())),
        'accuracy': round(results['accuracy'], 4),
        'confusion_matrix': results['confusion_matrix'],
        'labels': clf.labels,
        'per_class_metrics': per_class,
        'avg_confidence_by_intent': avg_confidence,
        'sample_predictions': sample_results,
    }
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")
    print("Done!")


if __name__ == '__main__':
    main()
