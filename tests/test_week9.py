"""
Week 9 Tests: Fair Housing Compliance Checker

Tests for ComplianceChecker covering:
- Prohibited pattern detection across all protected classes
- Multi-level severity (error, warning, info)
- 100% recall on known violations (no false negatives)
- Precision > 80% (false-positive suppression via allow list)
- Listing submission workflow integration
- Suggested fixes
- Edge cases (empty text, clean text, mixed violations)
"""

import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))
from compliance_checker import ComplianceChecker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def checker():
    return ComplianceChecker()


# ── Known Violation Test Set (must be 100% recall) ─────────────────────

KNOWN_VIOLATIONS = [
    # (text, expected_category, expected_severity)
    # -- Familial Status --
    ("This property has no children allowed on premises.", "familial_status", "error"),
    ("Adults only community, perfect retirement living.", "familial_status", "error"),
    ("No kids permitted in the building.", "familial_status", "error"),
    ("Perfect for singles looking for a quiet place.", "familial_status", "warning"),
    ("No families please.", "familial_status", "error"),
    ("Singles only residence.", "familial_status", "error"),
    ("No teenagers allowed.", "familial_status", "error"),

    # -- Disability --
    ("No wheelchairs in common areas.", "disability", "error"),
    ("Tenants must be able-bodied.", "disability", "error"),
    ("No disabled persons allowed.", "disability", "error"),
    ("No handicapped individuals.", "disability", "error"),
    ("Must be physically fit to navigate stairs.", "disability", "error"),
    ("No service animals permitted.", "disability", "error"),
    ("No emotional support animals.", "disability", "error"),
    ("No blind persons.", "disability", "error"),
    ("No deaf tenants.", "disability", "error"),
    ("Not suitable for wheelchair users.", "disability", "error"),
    ("No mental illness.", "disability", "error"),

    # -- Race --
    ("Located in a white neighborhood.", "race", "error"),
    ("Beautiful home in a black neighborhood.", "race", "error"),
    ("Situated in a white community.", "race", "error"),
    ("Segregated community.", "race", "error"),
    ("Racial composition of area is favorable.", "race", "error"),

    # -- Religion --
    ("Great christian community atmosphere.", "religion", "error"),
    ("Located in a jewish neighborhood.", "religion", "error"),
    ("Wonderful muslim community.", "religion", "error"),
    ("No muslims.", "religion", "error"),
    ("No jews allowed.", "religion", "error"),

    # -- National Origin --
    ("English only speakers.", "national_origin", "error"),
    ("Must speak english.", "national_origin", "error"),
    ("No immigrants allowed.", "national_origin", "error"),
    ("American born residents only.", "national_origin", "error"),
    ("Citizenship required for application.", "national_origin", "error"),

    # -- Sex --
    ("Male only building.", "sex", "error"),
    ("Women only floor.", "sex", "error"),
    ("Men only housing.", "sex", "error"),
    ("Perfect for single women.", "sex", "error"),
    ("Perfect for single men.", "sex", "error"),

    # -- Exclusionary --
    ("No section 8 accepted.", "exclusionary", "error"),
    ("No vouchers accepted.", "exclusionary", "error"),
]


class TestKnownViolationsRecall:
    """Every known violation MUST be detected — 100% recall required."""

    @pytest.mark.parametrize("text,expected_category,expected_severity",
                             KNOWN_VIOLATIONS,
                             ids=[t[0][:50] for t in KNOWN_VIOLATIONS])
    def test_detects_violation(self, checker, text, expected_category, expected_severity):
        result = checker.check_listing(text)
        assert not result['compliant'], f"Should be non-compliant: {text}"
        assert len(result['violations']) > 0, f"No violations detected: {text}"

        # Check that the expected category is among the violations
        categories = [v['category'] for v in result['violations']]
        assert expected_category in categories, (
            f"Expected category '{expected_category}' not found in {categories} for: {text}"
        )

        # Check severity
        relevant = [v for v in result['violations'] if v['category'] == expected_category]
        severities = [v['severity'] for v in relevant]
        assert expected_severity in severities, (
            f"Expected severity '{expected_severity}' not found in {severities} for: {text}"
        )


class TestRecallMetric:
    """Aggregate recall must be exactly 100%."""

    def test_100_percent_recall(self, checker):
        detected = 0
        total = len(KNOWN_VIOLATIONS)

        for text, expected_category, _ in KNOWN_VIOLATIONS:
            result = checker.check_listing(text)
            categories = [v['category'] for v in result['violations']]
            if expected_category in categories:
                detected += 1

        recall = detected / total
        assert recall == 1.0, f"Recall is {recall:.2%} ({detected}/{total}) — must be 100%"


# ── Precision (False Positive Rate) ────────────────────────────────────

CLEAN_LISTINGS = [
    "Beautiful 3 bedroom, 2 bath home with a sparkling pool.",
    "Charming colonial home on a tree-lined street near top-rated schools.",
    "Updated kitchen with granite countertops and stainless steel appliances.",
    "Spacious backyard perfect for entertaining friends and family.",
    "Open floor plan with hardwood floors and recessed lighting throughout.",
    "Close to parks, shopping centers, and major freeways.",
    "Freshly painted with new carpet and modern fixtures.",
    "Two-car garage with ample storage space.",
    "Large master suite with walk-in closet and ensuite bathroom.",
    "Quiet cul-de-sac location with mountain views.",
    "Energy-efficient windows and solar panels installed.",
    "Corner lot with mature landscaping and automatic sprinklers.",
    "Minutes from downtown dining and entertainment.",
    "HOA maintained community pool and recreation area.",
    "Victorian style home with original woodwork preserved.",
    "Recently renovated with permits on file.",
    "Turnkey condition — move right in.",
    "Walking distance to public transit and bike paths.",
    "Gated community with 24/7 security.",
    "Chef's kitchen with center island and breakfast nook.",
    "Covered patio with ceiling fans — great for outdoor living.",
    "All bedrooms upstairs with central air conditioning.",
    "This property checks all the boxes for any homebuyer.",
    "Corner unit with extra natural light and cross ventilation.",
    "Income property with established tenants and positive cash flow.",
]


class TestPrecision:
    """Clean listings should NOT be flagged as violations (error or warning)."""

    @pytest.mark.parametrize("text", CLEAN_LISTINGS,
                             ids=[t[:50] for t in CLEAN_LISTINGS])
    def test_clean_listing_is_compliant(self, checker, text):
        result = checker.check_listing(text)
        errors_warnings = [
            v for v in result['violations']
            if v['severity'] in ('error', 'warning')
        ]
        assert len(errors_warnings) == 0, (
            f"False positive on clean listing: {errors_warnings}"
        )

    def test_precision_above_80_percent(self, checker):
        """Overall precision on clean set must exceed 80%."""
        correct = 0
        for text in CLEAN_LISTINGS:
            result = checker.check_listing(text)
            errors_warnings = [
                v for v in result['violations']
                if v['severity'] in ('error', 'warning')
            ]
            if len(errors_warnings) == 0:
                correct += 1

        precision = correct / len(CLEAN_LISTINGS)
        assert precision > 0.80, (
            f"Precision is {precision:.2%} ({correct}/{len(CLEAN_LISTINGS)}) — must be > 80%"
        )


# ── False Positive Suppression (Allow List) ────────────────────────────

class TestAllowList:
    """Known false-positive patterns should be suppressed."""

    def test_wheelchair_accessible_not_flagged(self, checker):
        text = "This unit is wheelchair accessible with ramp access."
        result = checker.check_listing(text)
        errors = [v for v in result['violations'] if v['severity'] == 'error']
        assert len(errors) == 0, f"wheelchair accessible should not flag errors: {errors}"

    def test_window_blinds_not_flagged(self, checker):
        text = "Living room has beautiful window blinds and curtains."
        result = checker.check_listing(text)
        errors = [v for v in result['violations']
                  if v['severity'] == 'error' and v['category'] == 'disability']
        assert len(errors) == 0, "window blinds should not flag disability error"


# ── Severity Levels ────────────────────────────────────────────────────

class TestSeverityLevels:
    """Verify proper severity classification."""

    def test_error_severity(self, checker):
        result = checker.check_listing("No children allowed.")
        violations = [v for v in result['violations'] if v['severity'] == 'error']
        assert len(violations) > 0

    def test_warning_severity(self, checker):
        result = checker.check_listing("This is an adult community.")
        violations = [v for v in result['violations'] if v['severity'] == 'warning']
        assert len(violations) > 0

    def test_info_severity(self, checker):
        result = checker.check_listing("55+ community with age restricted access.")
        violations = [v for v in result['violations'] if v['severity'] == 'info']
        assert len(violations) > 0

    def test_stats_dict(self, checker):
        result = checker.check_listing("No children. This is an adult community. 55+ housing.")
        stats = result['stats']
        assert 'error' in stats
        assert 'warning' in stats
        assert 'info' in stats
        assert stats['error'] >= 1   # "no children"
        assert stats['warning'] >= 1  # "adult community"
        assert stats['info'] >= 1    # "55+"


# ── Multiple Violations ───────────────────────────────────────────────

class TestMultipleViolations:
    """Test listings with multiple violations."""

    def test_multiple_categories(self, checker):
        text = (
            "No children allowed. Must be able-bodied. "
            "Located in a white neighborhood. No immigrants."
        )
        result = checker.check_listing(text)
        categories = set(v['category'] for v in result['violations'])
        assert 'familial_status' in categories
        assert 'disability' in categories
        assert 'race' in categories
        assert 'national_origin' in categories

    def test_multiple_same_category(self, checker):
        text = "No children, no kids, no families allowed."
        result = checker.check_listing(text)
        fam_violations = [v for v in result['violations']
                          if v['category'] == 'familial_status']
        assert len(fam_violations) >= 3


# ── Edge Cases ─────────────────────────────────────────────────────────

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_text(self, checker):
        result = checker.check_listing("")
        assert result['compliant'] is True
        assert result['violations'] == []

    def test_none_text(self, checker):
        result = checker.check_listing(None)
        assert result['compliant'] is True

    def test_numeric_text(self, checker):
        result = checker.check_listing("12345")
        assert result['compliant'] is True

    def test_case_insensitive(self, checker):
        result_lower = checker.check_listing("no children allowed")
        result_upper = checker.check_listing("NO CHILDREN ALLOWED")
        result_mixed = checker.check_listing("No Children Allowed")
        assert not result_lower['compliant']
        assert not result_upper['compliant']
        assert not result_mixed['compliant']

    def test_compliant_listing(self, checker):
        text = (
            "Beautiful 4 bedroom, 3 bath home with updated kitchen "
            "and spacious backyard. Close to parks and schools. $599,000."
        )
        result = checker.check_listing(text)
        assert result['compliant'] is True

    def test_violation_has_position(self, checker):
        text = "Great house. No children allowed."
        result = checker.check_listing(text)
        assert len(result['violations']) > 0
        for v in result['violations']:
            assert 'position' in v
            assert isinstance(v['position'], int)


# ── Listing Submission Workflow ────────────────────────────────────────

class TestSubmissionWorkflow:
    """Test the check_listing_submission integration method."""

    def test_approved_clean_listing(self, checker):
        record = {
            'listing_id': 1,
            'remarks': 'Beautiful 3 bed, 2 bath home with pool and garage.',
        }
        result = checker.check_listing_submission(record)
        assert result['approved'] is True
        assert result['needs_review'] is False
        assert 'APPROVED' in result['recommendation']

    def test_blocked_violation(self, checker):
        record = {
            'listing_id': 2,
            'remarks': 'No children allowed. Adults only.',
        }
        result = checker.check_listing_submission(record)
        assert result['approved'] is False
        assert 'BLOCKED' in result['recommendation']

    def test_review_warning(self, checker):
        record = {
            'listing_id': 3,
            'remarks': (
                "Lovely home in an exclusive neighborhood. "
                "Very quiet area, great for mature residents."
            ),
        }
        result = checker.check_listing_submission(record)
        # Should have warnings but may or may not have errors
        assert result['stats']['warning'] > 0 or result['stats']['error'] > 0

    def test_returns_listing_id(self, checker):
        record = {'listing_id': 42, 'remarks': 'Nice home.'}
        result = checker.check_listing_submission(record)
        assert result['listing_id'] == 42

    def test_returns_original_text(self, checker):
        text = 'No kids allowed.'
        result = checker.check_listing_submission({'remarks': text})
        assert result['original_text'] == text


# ── Suggested Fixes ────────────────────────────────────────────────────

class TestSuggestedFixes:
    """Test the suggest_fix method."""

    def test_fix_for_no_children(self, checker):
        violation = {'category': 'familial_status', 'pattern': 'no children', 'severity': 'error'}
        fix = checker.suggest_fix(violation)
        assert isinstance(fix, str)
        assert len(fix) > 0

    def test_fix_for_no_wheelchairs(self, checker):
        violation = {'category': 'disability', 'pattern': 'no wheelchairs', 'severity': 'error'}
        fix = checker.suggest_fix(violation)
        assert isinstance(fix, str)
        assert len(fix) > 0

    def test_fix_for_unknown(self, checker):
        violation = {'category': 'unknown', 'pattern': 'xyz', 'severity': 'warning'}
        fix = checker.suggest_fix(violation)
        assert isinstance(fix, str)


# ── Documentation ──────────────────────────────────────────────────────

class TestDocumentation:
    """Verify Fair Housing documentation is available."""

    def test_summary_exists(self, checker):
        summary = checker.get_fair_housing_summary()
        assert isinstance(summary, str)
        assert 'Fair Housing' in summary

    def test_summary_covers_protected_classes(self, checker):
        summary = checker.get_fair_housing_summary()
        for term in ['RACE', 'RELIGION', 'SEX', 'FAMILIAL STATUS',
                      'DISABILITY', 'NATIONAL ORIGIN']:
            assert term in summary, f"Missing protected class: {term}"

    def test_summary_covers_severity(self, checker):
        summary = checker.get_fair_housing_summary()
        for level in ['ERROR', 'WARNING', 'INFO']:
            assert level in summary, f"Missing severity level: {level}"


# ── Result Structure ──────────────────────────────────────────────────

class TestResultStructure:
    """Verify the structure of check_listing results."""

    def test_result_keys(self, checker):
        result = checker.check_listing("Test text with no violations.")
        assert 'compliant' in result
        assert 'violations' in result
        assert 'stats' in result

    def test_violation_keys(self, checker):
        result = checker.check_listing("No children allowed.")
        assert len(result['violations']) > 0
        v = result['violations'][0]
        assert 'category' in v
        assert 'pattern' in v
        assert 'severity' in v
        assert 'message' in v
        assert 'position' in v
