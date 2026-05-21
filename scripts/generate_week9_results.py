"""
Week 9: Generate Fair Housing Compliance Results

Runs the full compliance-checking pipeline:
1. Scans all listings for Fair Housing violations
2. Generates compliance statistics
3. Runs recall/precision evaluation on test sets
4. Demos listing submission workflow
5. Saves week9_summary.json
"""

import sys
import os
import json
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from compliance_checker import ComplianceChecker


# ── Test Sets ──────────────────────────────────────────────────────────

VIOLATION_TEST_SET = [
    ("No children allowed in this building.", "familial_status"),
    ("Adults only. Quiet mature community.", "familial_status"),
    ("No kids permitted.", "familial_status"),
    ("No families please.", "familial_status"),
    ("No wheelchairs in common areas.", "disability"),
    ("Must be able-bodied to live here.", "disability"),
    ("No disabled persons.", "disability"),
    ("No service animals.", "disability"),
    ("Located in a white neighborhood.", "race"),
    ("Segregated community with great amenities.", "race"),
    ("Great christian community.", "religion"),
    ("Jewish neighborhood living.", "religion"),
    ("English only speakers.", "national_origin"),
    ("No immigrants.", "national_origin"),
    ("Male only building.", "sex"),
    ("Women only floor.", "sex"),
    ("No section 8 accepted.", "exclusionary"),
    ("No vouchers.", "exclusionary"),
    ("Must be physically fit.", "disability"),
    ("Perfect for single women only.", "sex"),
    ("No handicapped individuals allowed.", "disability"),
    ("No teenagers allowed in unit.", "familial_status"),
    ("No blind persons may apply.", "disability"),
    ("Citizenship required to apply.", "national_origin"),
    ("American born tenants preferred.", "national_origin"),
]

CLEAN_TEST_SET = [
    "Beautiful 3 bedroom, 2 bath home with sparkling pool.",
    "Updated kitchen with granite countertops and stainless steel appliances.",
    "Spacious backyard perfect for entertaining.",
    "Walking distance to schools, parks, and shopping.",
    "Hardwood floors and recessed lighting throughout.",
    "New roof, new HVAC, and energy-efficient windows.",
    "Two-car attached garage with storage.",
    "Open floor plan with vaulted ceilings.",
    "Corner lot with mature landscaping.",
    "Victorian home with original character preserved.",
    "Covered patio with ceiling fans.",
    "All bedrooms upstairs with central air.",
    "Close to downtown dining and entertainment.",
    "Turn-key condition, move right in.",
    "Income property with excellent cash flow.",
    "Recently renovated with all permits on file.",
    "Chef's kitchen with center island.",
    "Mountain views from the living room.",
    "Wheelchair accessible first-floor unit with ramp.",
    "HOA-maintained pool and recreation center.",
]


def main():
    import pandas as pd

    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    csv_path = os.path.join(base, 'data', 'processed', 'cleaned_listing.csv')
    summary_path = os.path.join(base, 'data', 'processed', 'week9_summary.json')

    print("=" * 60)
    print("Week 9: Fair Housing Compliance Checker")
    print("=" * 60)

    checker = ComplianceChecker()

    # ------------------------------------------------------------------
    # 1.  Scan all listings
    # ------------------------------------------------------------------
    print("\n-- Scanning Listings --")
    df = pd.read_csv(csv_path)

    text_col = None
    for col in ('cleaned_remarks', 'remarks', 'L_Remarks'):
        if col in df.columns:
            text_col = col
            break

    scan_results = []
    total_violations = 0
    error_count = 0
    warning_count = 0
    info_count = 0
    flagged_listings = 0

    for idx, row in df.iterrows():
        text = row[text_col] if pd.notna(row[text_col]) else ''
        result = checker.check_listing(text)
        n_violations = len(result['violations'])
        total_violations += n_violations
        error_count += result['stats']['error']
        warning_count += result['stats']['warning']
        info_count += result['stats']['info']

        if not result['compliant']:
            flagged_listings += 1

        scan_results.append({
            'listing_id': idx,
            'compliant': result['compliant'],
            'n_violations': n_violations,
            'stats': result['stats'],
            'violations': result['violations'],
        })

    print(f"  Scanned {len(df)} listings")
    print(f"  Flagged: {flagged_listings} ({flagged_listings/len(df)*100:.1f}%)")
    print(f"  Total violations: {total_violations}")
    print(f"    Errors:   {error_count}")
    print(f"    Warnings: {warning_count}")
    print(f"    Info:     {info_count}")

    # Category breakdown
    cat_counts = {}
    for sr in scan_results:
        for v in sr['violations']:
            cat = v['category']
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

    print("\n  By category:")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {count}")

    # ------------------------------------------------------------------
    # 2.  Recall evaluation
    # ------------------------------------------------------------------
    print("\n-- Recall Evaluation (Known Violations) --")
    recall_detected = 0
    recall_total = len(VIOLATION_TEST_SET)

    for text, expected_cat in VIOLATION_TEST_SET:
        result = checker.check_listing(text)
        categories = [v['category'] for v in result['violations']]
        if expected_cat in categories:
            recall_detected += 1
        else:
            print(f"  MISS: {text[:60]} (expected {expected_cat})")

    recall = recall_detected / recall_total
    print(f"  Recall: {recall:.2%} ({recall_detected}/{recall_total})")

    # ------------------------------------------------------------------
    # 3.  Precision evaluation
    # ------------------------------------------------------------------
    print("\n-- Precision Evaluation (Clean Listings) --")
    precision_correct = 0

    for text in CLEAN_TEST_SET:
        result = checker.check_listing(text)
        errors_warnings = [v for v in result['violations']
                           if v['severity'] in ('error', 'warning')]
        if len(errors_warnings) == 0:
            precision_correct += 1
        else:
            print(f"  FP: {text[:60]} -> {[v['pattern'] for v in errors_warnings]}")

    precision = precision_correct / len(CLEAN_TEST_SET)
    print(f"  Precision: {precision:.2%} ({precision_correct}/{len(CLEAN_TEST_SET)})")

    # ------------------------------------------------------------------
    # 4.  Submission workflow demo
    # ------------------------------------------------------------------
    print("\n-- Submission Workflow Demo --")

    demo_listings = [
        {
            'listing_id': 'DEMO-001',
            'remarks': 'Beautiful 4 bed, 3 bath home with pool. Near top schools. $750,000.',
        },
        {
            'listing_id': 'DEMO-002',
            'remarks': 'No children allowed. Adults only building with quiet residents.',
        },
        {
            'listing_id': 'DEMO-003',
            'remarks': 'Lovely home in an exclusive neighborhood for mature residents.',
        },
        {
            'listing_id': 'DEMO-004',
            'remarks': 'Must be able-bodied. Walk-up only, no wheelchairs.',
        },
        {
            'listing_id': 'DEMO-005',
            'remarks': 'Charming 2 bed condo. Wheelchair accessible. Near downtown.',
        },
    ]

    workflow_results = []
    for listing in demo_listings:
        result = checker.check_listing_submission(listing)
        status = "APPROVED" if result['approved'] else ("REVIEW" if result['needs_review'] else "BLOCKED")
        print(f"  [{status}] {listing['listing_id']}: {result['recommendation'][:60]}")
        workflow_results.append({
            'listing_id': result['listing_id'],
            'text': listing['remarks'][:100],
            'approved': result['approved'],
            'needs_review': result['needs_review'],
            'recommendation': result['recommendation'],
            'errors': result['stats']['error'],
            'warnings': result['stats']['warning'],
            'infos': result['stats']['info'],
        })

    # ------------------------------------------------------------------
    # 5.  Sample violations from dataset
    # ------------------------------------------------------------------
    print("\n-- Sample Violations from Dataset --")
    flagged = [sr for sr in scan_results if sr['n_violations'] > 0]
    random.seed(42)
    sample_violations = random.sample(flagged, min(10, len(flagged))) if flagged else []

    for sv in sample_violations[:5]:
        listing_text = df.iloc[sv['listing_id']][text_col]
        if pd.notna(listing_text):
            print(f"\n  Listing {sv['listing_id']}:")
            print(f"    Text: {str(listing_text)[:100]}...")
            for v in sv['violations'][:3]:
                print(f"    [{v['severity'].upper()}] {v['category']}: {v['pattern']}")

    # ------------------------------------------------------------------
    # 6.  Save summary JSON
    # ------------------------------------------------------------------
    week9_summary = {
        'total_listings_scanned': len(df),
        'flagged_listings': flagged_listings,
        'flagged_pct': round(flagged_listings / len(df) * 100, 1),
        'total_violations': total_violations,
        'severity_breakdown': {
            'error': error_count,
            'warning': warning_count,
            'info': info_count,
        },
        'category_breakdown': cat_counts,
        'recall': round(recall, 4),
        'recall_detected': recall_detected,
        'recall_total': recall_total,
        'precision': round(precision, 4),
        'precision_correct': precision_correct,
        'precision_total': len(CLEAN_TEST_SET),
        'workflow_demo': workflow_results,
        'sample_violations': [
            {
                'listing_id': sv['listing_id'],
                'text': str(df.iloc[sv['listing_id']][text_col])[:150] if pd.notna(df.iloc[sv['listing_id']][text_col]) else '',
                'violations': [
                    {
                        'category': v['category'],
                        'pattern': v['pattern'],
                        'severity': v['severity'],
                    }
                    for v in sv['violations'][:5]
                ],
            }
            for sv in sample_violations[:10]
        ],
    }

    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(week9_summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")
    print("Done!")


if __name__ == '__main__':
    main()
