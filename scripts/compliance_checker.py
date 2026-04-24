"""
Week 9: Fair Housing Compliance Checker

Implements Fair Housing Act compliance checking for real estate listings.
Detects prohibited language related to protected classes (race, color, religion,
national origin, sex, familial status, disability) and flags biased descriptions
before publication.

Fair Housing Act (42 U.S.C. §§ 3601-3619) prohibits discrimination in housing
based on seven protected classes. HUD advertising guidelines (24 CFR Part 100)
further specify prohibited language in real estate advertising.

Multi-level severity system:
  - error:   Must fix before publication — clear Fair Housing violation
  - warning: Should review — potentially discriminatory or exclusionary
  - info:    Informational — language that may need context or rewording

Design goals:
  - 100% recall on known violations (no false negatives)
  - Precision > 80% (minimize false positives via contextual checks)
"""

import re
from typing import Any


class ComplianceChecker:
    """Check real estate listings for Fair Housing Act compliance.

    Scans listing text against a curated pattern library organized by
    protected class.  Each pattern has an assigned severity level and
    a human-readable remediation message.

    Usage:
        checker = ComplianceChecker()
        result = checker.check_listing("Beautiful home, no children allowed.")
        # result['compliant'] == False
        # result['violations'][0]['category'] == 'familial_status'
    """

    # ------------------------------------------------------------------
    # Fair Housing protected classes and pattern library
    # ------------------------------------------------------------------

    # Each entry: (pattern_string_or_regex, severity, message)
    # Patterns are matched case-insensitively against listing text.
    # Regex patterns are prefixed with 'regex:'.

    PROHIBITED_PATTERNS: dict[str, list[tuple[str, str, str]]] = {

        # ── Race / Color ─────────────────────────────────────────
        'race': [
            # Direct racial terms
            ('white neighborhood', 'error',
             'Describing a neighborhood by race violates Fair Housing Act.'),
            ('black neighborhood', 'error',
             'Describing a neighborhood by race violates Fair Housing Act.'),
            ('white community', 'error',
             'Describing a community by race violates Fair Housing Act.'),
            ('black community', 'error',
             'Describing a community by race violates Fair Housing Act.'),
            ('caucasian', 'error',
             'Racial terms are prohibited in housing advertisements.'),
            ('african american neighborhood', 'error',
             'Racial references in neighborhood descriptions are prohibited.'),
            ('hispanic area', 'error',
             'Ethnic area descriptions violate Fair Housing Act.'),
            ('latino neighborhood', 'error',
             'Ethnic neighborhood descriptions violate Fair Housing Act.'),
            ('asian community', 'error',
             'Ethnic community descriptions violate Fair Housing Act.'),
            ('colored', 'warning',
             'Term may be racially discriminatory — review context.'),
            ('integrated', 'info',
             'May imply racial composition — consider rewording.'),
            ('exclusive neighborhood', 'warning',
             'May imply racial or socioeconomic exclusion.'),
            ('desirable neighborhood', 'warning',
             'May imply preference for certain demographics.'),
            ('diverse area', 'info',
             'Describing diversity may imply racial steering.'),
            ('ethnic', 'warning',
             'Ethnic references may violate Fair Housing guidelines.'),
            ('segregated', 'error',
             'Segregation language violates Fair Housing Act.'),
            ('racial', 'error',
             'Racial language is prohibited in housing ads.'),
        ],

        # ── National Origin ──────────────────────────────────────
        'national_origin': [
            ('english only', 'error',
             'Language requirements may discriminate by national origin.'),
            ('must speak english', 'error',
             'Language requirements may discriminate by national origin.'),
            ('no immigrants', 'error',
             'Immigration status discrimination violates Fair Housing Act.'),
            ('american born', 'error',
             'National origin restrictions violate Fair Housing Act.'),
            ('foreigners', 'warning',
             'Reference to foreign status may be discriminatory.'),
            ('citizenship required', 'error',
             'Citizenship requirements violate Fair Housing protections.'),
            ('regex:\\b(irish|italian|chinese|japanese|korean|mexican|indian|arab|middle eastern)\\s+(neighborhood|community|area)\\b', 'error',
             'Describing neighborhoods by national origin is prohibited.'),
        ],

        # ── Religion ─────────────────────────────────────────────
        'religion': [
            ('christian community', 'error',
             'Religious community descriptions violate Fair Housing Act.'),
            ('christian neighborhood', 'error',
             'Religious neighborhood descriptions violate Fair Housing Act.'),
            ('jewish neighborhood', 'error',
             'Religious neighborhood descriptions violate Fair Housing Act.'),
            ('jewish community', 'error',
             'Religious community descriptions violate Fair Housing Act.'),
            ('muslim neighborhood', 'error',
             'Religious neighborhood descriptions violate Fair Housing Act.'),
            ('muslim community', 'error',
             'Religious community descriptions violate Fair Housing Act.'),
            ('regex:near\\s+(church|mosque|synagogue|temple)', 'info',
             'Religious landmark proximity references should be used carefully.'),
            ('no muslims', 'error',
             'Religious exclusion violates Fair Housing Act.'),
            ('no jews', 'error',
             'Religious exclusion violates Fair Housing Act.'),
            ('no christians', 'error',
             'Religious exclusion violates Fair Housing Act.'),
            ('catholic community', 'error',
             'Religious community descriptions violate Fair Housing Act.'),
            ('hindu community', 'error',
             'Religious community descriptions violate Fair Housing Act.'),
        ],

        # ── Sex / Gender ─────────────────────────────────────────
        'sex': [
            ('male only', 'error',
             'Gender restrictions violate Fair Housing Act.'),
            ('female only', 'error',
             'Gender restrictions violate Fair Housing Act.'),
            ('men only', 'error',
             'Gender restrictions violate Fair Housing Act.'),
            ('women only', 'error',
             'Gender restrictions violate Fair Housing Act.'),
            ('bachelor pad', 'warning',
             'Gender-suggestive descriptions may discourage applicants.'),
            ('man cave', 'warning',
             'Gender-suggestive descriptions may discourage applicants.'),
            ('perfect for single women', 'error',
             'Gender-targeted marketing violates Fair Housing Act.'),
            ('perfect for single men', 'error',
             'Gender-targeted marketing violates Fair Housing Act.'),
            ('no couples', 'warning',
             'Relationship restrictions may violate Fair Housing Act.'),
        ],

        # ── Familial Status ──────────────────────────────────────
        'familial_status': [
            ('no children', 'error',
             'Excluding children violates Fair Housing Act (familial status).'),
            ('no kids', 'error',
             'Excluding children violates Fair Housing Act (familial status).'),
            ('adults only', 'error',
             'Age-restricted language violates familial status protections '
             '(unless qualifying senior housing under HOPA).'),
            ('adult community', 'warning',
             'May violate familial status protections — verify HOPA exemption.'),
            ('adult living', 'warning',
             'May violate familial status protections — verify HOPA exemption.'),
            ('no families', 'error',
             'Excluding families violates Fair Housing Act.'),
            ('perfect for singles', 'warning',
             'May discourage families — consider neutral language.'),
            ('singles only', 'error',
             'Excluding families violates Fair Housing Act.'),
            ('couple only', 'error',
             'Restricting to couples excludes families with children.'),
            ('mature community', 'warning',
             'May imply exclusion of families — verify HOPA exemption.'),
            ('ideal for professionals', 'info',
             'May indirectly discourage families — consider rewording.'),
            ('quiet community', 'info',
             'May imply children are unwelcome — review context.'),
            ('senior living', 'info',
             'Verify HOPA exemption; otherwise may violate familial status protections.'),
            ('55 and over', 'info',
             'Verify HOPA exemption documentation is on file.'),
            ('55+', 'info',
             'Verify HOPA exemption documentation is on file.'),
            ('age restricted', 'info',
             'Verify HOPA exemption documentation is on file.'),
            ('age-restricted', 'info',
             'Verify HOPA exemption documentation is on file.'),
            ('no pets', 'info',
             'While not a protected class, "no pets" paired with other language '
             'may contribute to discriminatory tone. Review context.'),
            ('no teenagers', 'error',
             'Excluding teenagers violates familial status protections.'),
            ('nursery', 'info',
             'Room description is acceptable — flagged for awareness only.'),
        ],

        # ── Disability ───────────────────────────────────────────
        'disability': [
            ('no wheelchairs', 'error',
             'Excluding wheelchairs violates disability protections.'),
            ('no wheelchair', 'error',
             'Excluding wheelchairs violates disability protections.'),
            ('must be able-bodied', 'error',
             'Ability requirements violate disability protections.'),
            ('able-bodied', 'error',
             'Physical ability requirements violate Fair Housing Act.'),
            ('no disabled', 'error',
             'Excluding disabled persons violates Fair Housing Act.'),
            ('no handicapped', 'error',
             'Excluding handicapped persons violates Fair Housing Act.'),
            ('handicapped', 'warning',
             'Outdated terminology — use "accessible" or "disability-friendly."'),
            ('no mental illness', 'error',
             'Mental health discrimination violates Fair Housing Act.'),
            ('mentally ill', 'error',
             'Mental health references are discriminatory in housing ads.'),
            ('no service animals', 'error',
             'Denying service animals violates disability protections.'),
            ('no emotional support', 'error',
             'Denying emotional support animals violates disability protections.'),
            ('must climb stairs', 'warning',
             'May exclude persons with mobility disabilities.'),
            ('walk-up only', 'warning',
             'May discourage persons with mobility disabilities.'),
            ('physically fit', 'error',
             'Physical fitness requirements violate disability protections.'),
            ('no blind', 'error',
             'Excluding persons by disability violates Fair Housing Act.'),
            ('no deaf', 'error',
             'Excluding persons by disability violates Fair Housing Act.'),
            ('regex:(?:not\\s+suitable|unsuitable)\\s+(?:for\\s+)?(?:disabled|handicapped|wheelchair)', 'error',
             'Suitability statements regarding disability are prohibited.'),
            # Positive accessibility terms (informational, not violations)
            ('wheelchair accessible', 'info',
             'Describing accessibility features is encouraged — no action needed.'),
            ('wheelchair ramp', 'info',
             'Describing accessibility features is encouraged — no action needed.'),
            ('ada compliant', 'info',
             'ADA compliance descriptions are encouraged — no action needed.'),
        ],

        # ── Socioeconomic / Other Exclusionary Language ───────────
        'exclusionary': [
            ('section 8', 'warning',
             'Source-of-income discrimination is prohibited in many jurisdictions.'),
            ('no section 8', 'error',
             'Refusing Section 8 vouchers violates many local/state fair housing laws.'),
            ('no vouchers', 'error',
             'Refusing housing vouchers violates source-of-income protections.'),
            ('credit check required', 'info',
             'Legitimate but ensure applied uniformly without discriminatory intent.'),
            ('background check required', 'info',
             'Legitimate but ensure applied uniformly without discriminatory intent.'),
            ('no felons', 'warning',
             'Blanket criminal history exclusions may have disparate impact.'),
            ('no criminals', 'warning',
             'Blanket criminal history exclusions may have disparate impact.'),
            ('exclusive community', 'warning',
             'May imply discriminatory exclusion — consider rewording.'),
            ('upscale', 'info',
             'May imply economic or racial exclusion — review context.'),
        ],
    }

    # Patterns that indicate false positives (allowable context)
    ALLOW_LIST = [
        # "no children" in "no children's museum" or "no children under X" for pool safety
        re.compile(r"no children(?:'s| under \d+ (?:in|at) (?:the\s+)?pool)", re.IGNORECASE),
        # "wheelchair accessible" should not trigger "wheelchair" as violation
        re.compile(r'wheelchair\s+(?:accessible|ramp|friendly|lift)', re.IGNORECASE),
        # "ada compliant" is positive
        re.compile(r'ada\s+compliant', re.IGNORECASE),
        # "blinds" (window coverings, plural) or "window blind(s)" — NOT "blind" as disability
        re.compile(r'(?:window\s+blinds?\b|(?<!\bno\s)\bblinds\b)', re.IGNORECASE),
    ]

    def __init__(self):
        """Initialize ComplianceChecker with the built-in pattern library."""
        # Pre-compile regex patterns for performance
        self._compiled_patterns: dict[str, list[tuple[Any, str, str, str]]] = {}
        for category, patterns in self.PROHIBITED_PATTERNS.items():
            compiled = []
            for pattern_str, severity, message in patterns:
                if pattern_str.startswith('regex:'):
                    regex = re.compile(pattern_str[6:], re.IGNORECASE)
                    compiled.append((regex, severity, message, pattern_str[6:]))
                else:
                    compiled.append((pattern_str.lower(), severity, message, pattern_str))
            self._compiled_patterns[category] = compiled

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_listing(self, text: str) -> dict:
        """Check a listing description for Fair Housing violations.

        Args:
            text: listing description text to check

        Returns:
            dict with keys:
                compliant (bool): True if no error/warning violations found
                violations (list[dict]): list of violation dicts, each with:
                    category, pattern, severity, message, position
                stats (dict): counts by severity level
        """
        if not text or not isinstance(text, str):
            return {
                'compliant': True,
                'violations': [],
                'stats': {'error': 0, 'warning': 0, 'info': 0},
            }

        text_lower = text.lower()
        violations = []

        for category, patterns in self._compiled_patterns.items():
            for matcher, severity, message, original_pattern in patterns:
                matches = self._find_matches(text_lower, text, matcher)
                for match_text, position in matches:
                    # Check allow list for false-positive suppression
                    if self._is_allowed(text, position, match_text):
                        continue

                    violations.append({
                        'category': category,
                        'pattern': original_pattern,
                        'severity': severity,
                        'message': f'Prohibited language: "{match_text}" — {message}',
                        'position': position,
                        'matched_text': match_text,
                    })

        # Deduplicate overlapping matches (keep highest severity)
        violations = self._deduplicate(violations)

        # Sort by position in text
        violations.sort(key=lambda v: v['position'])

        # Compute stats
        stats = {'error': 0, 'warning': 0, 'info': 0}
        for v in violations:
            stats[v['severity']] = stats.get(v['severity'], 0) + 1

        # Compliant only if zero errors AND zero warnings
        compliant = stats['error'] == 0 and stats['warning'] == 0

        return {
            'compliant': compliant,
            'violations': violations,
            'stats': stats,
        }

    def check_listing_submission(self, listing_record: dict) -> dict:
        """Integration point for listing submission workflow.

        Checks the listing text and returns a structured response suitable
        for embedding in a submission pipeline.

        Args:
            listing_record: dict with text fields (remarks, L_Remarks, etc.)

        Returns:
            dict with:
                approved (bool): True if listing can be published
                needs_review (bool): True if human review recommended
                violations (list): violation details
                recommendation (str): human-readable recommendation
                original_text (str): the checked text
        """
        text = listing_record.get(
            'remarks',
            listing_record.get('L_Remarks',
            listing_record.get('cleaned_remarks', ''))
        )
        if not isinstance(text, str):
            text = ''

        result = self.check_listing(text)

        # Determine approval status
        errors = result['stats']['error']
        warnings = result['stats']['warning']
        infos = result['stats']['info']

        if errors > 0:
            approved = False
            needs_review = False  # Must fix — no point reviewing
            recommendation = (
                f"BLOCKED: {errors} Fair Housing violation(s) found. "
                f"These must be corrected before the listing can be published."
            )
        elif warnings > 0:
            approved = False
            needs_review = True
            recommendation = (
                f"REVIEW REQUIRED: {warnings} potential issue(s) found. "
                f"A compliance officer should review before publication."
            )
        elif infos > 0:
            approved = True
            needs_review = False
            recommendation = (
                f"APPROVED with {infos} informational note(s). "
                f"No action required, but consider reviewing flagged items."
            )
        else:
            approved = True
            needs_review = False
            recommendation = "APPROVED: No Fair Housing concerns detected."

        return {
            'approved': approved,
            'needs_review': needs_review,
            'compliant': result['compliant'],
            'violations': result['violations'],
            'stats': result['stats'],
            'recommendation': recommendation,
            'original_text': text,
            'listing_id': listing_record.get(
                'listing_id',
                listing_record.get('L_ListingID', None)
            ),
        }

    def suggest_fix(self, violation: dict) -> str:
        """Suggest a fix for a specific violation.

        Args:
            violation: a violation dict from check_listing

        Returns:
            str: suggested replacement text or guidance
        """
        fixes = {
            'familial_status': {
                'no children': 'Remove exclusionary language. Describe the property features instead.',
                'no kids': 'Remove exclusionary language. Describe the property features instead.',
                'adults only': 'Remove unless qualifying senior housing (HOPA). Use "all ages welcome."',
                'no families': 'Remove exclusionary language. All family compositions are protected.',
                'singles only': 'Remove restriction. Describe property size and layout instead.',
                'perfect for singles': 'Replace with "great for any lifestyle" or describe the space.',
            },
            'disability': {
                'no wheelchairs': 'Remove. Describe accessibility features (e.g., "second floor unit").',
                'must be able-bodied': 'Remove. Describe the property layout factually.',
                'able-bodied': 'Remove. Describe accessibility features instead.',
                'no disabled': 'Remove discriminatory language entirely.',
                'no handicapped': 'Remove discriminatory language entirely.',
            },
            'race': {
                'white neighborhood': 'Remove racial description. Describe nearby amenities instead.',
                'diverse area': 'Describe specific community features rather than demographics.',
                'ethnic': 'Remove ethnic references. Describe cultural amenities by name.',
            },
            'religion': {
                'christian community': 'Remove religious description. Name nearby landmarks if relevant.',
                'jewish neighborhood': 'Remove religious description. Name nearby landmarks if relevant.',
            },
            'exclusionary': {
                'no section 8': 'Remove. Refusing vouchers is illegal in many jurisdictions.',
                'no vouchers': 'Remove. Source-of-income discrimination is prohibited.',
            },
        }

        category = violation.get('category', '')
        pattern = violation.get('pattern', '').lower()

        category_fixes = fixes.get(category, {})
        for key, fix in category_fixes.items():
            if key in pattern:
                return fix

        # Generic fix based on severity
        severity = violation.get('severity', 'info')
        if severity == 'error':
            return 'Remove this language entirely. It constitutes a Fair Housing violation.'
        elif severity == 'warning':
            return 'Review and consider removing or rewording to neutral, property-focused language.'
        else:
            return 'No action required, but consider whether the language could be misinterpreted.'

    def get_fair_housing_summary(self) -> str:
        """Return a documentation summary of Fair Housing rules for the team.

        Returns:
            str: multi-line documentation string
        """
        return """
═══════════════════════════════════════════════════════════════
           FAIR HOUSING ACT — COMPLIANCE GUIDE
═══════════════════════════════════════════════════════════════

OVERVIEW
--------
The Fair Housing Act (42 U.S.C. §§ 3601-3619) prohibits discrimination
in the sale, rental, and financing of housing based on seven federally
protected classes. HUD advertising guidelines (24 CFR Part 100) further
specify what language is prohibited in real estate advertising.

PROTECTED CLASSES
-----------------
1. RACE / COLOR
   • Do NOT describe neighborhoods, communities, or areas by racial
     composition (e.g., "white neighborhood," "diverse area").
   • Do NOT use racial terms in listings.

2. NATIONAL ORIGIN
   • Do NOT require language proficiency or citizenship.
   • Do NOT describe neighborhoods by national/ethnic origin.

3. RELIGION
   • Do NOT describe communities by religion (e.g., "Christian community").
   • Mentioning proximity to places of worship is acceptable with care.

4. SEX / GENDER
   • Do NOT use gender-restrictive language (e.g., "male only").
   • Avoid gender-implying descriptions (e.g., "bachelor pad").

5. FAMILIAL STATUS
   • Do NOT exclude children or families (e.g., "no children," "adults only").
   • Senior housing (55+/62+) is exempt ONLY under HOPA with proper
     documentation.
   • "Perfect for singles" may discourage families.

6. DISABILITY
   • Do NOT exclude persons with disabilities.
   • Do NOT require physical abilities (e.g., "must be able-bodied").
   • DO describe accessibility features positively.
   • Reasonable accommodations and modifications are legally required.

7. SOURCE OF INCOME (many state/local laws)
   • Do NOT refuse Section 8 vouchers or other housing assistance
     in jurisdictions where source-of-income is protected.

SEVERITY LEVELS
---------------
• ERROR   — Clear violation. MUST be fixed before publication.
• WARNING — Potentially discriminatory. Should be reviewed by compliance.
• INFO    — Informational. No action required, but awareness is important.

BEST PRACTICES
--------------
✓ Describe the PROPERTY, not the desired tenant/buyer.
✓ Focus on features, amenities, location landmarks, and condition.
✓ Use inclusive language ("all welcome," "great for any lifestyle").
✓ When describing accessibility, use positive terms ("wheelchair
  accessible," "ADA compliant").
✗ Never describe the ideal occupant by demographic characteristics.
✗ Never use exclusionary language targeting protected classes.
✗ Never steer buyers/renters based on demographics.

WORKFLOW INTEGRATION
--------------------
All listings should pass through the ComplianceChecker before publication:
1. Agent submits listing text.
2. ComplianceChecker scans for violations.
3. Errors → listing is BLOCKED until corrected.
4. Warnings → listing is HELD for compliance officer review.
5. Info → listing is APPROVED with notes for the agent.
"""

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_matches(self, text_lower, text_original, matcher):
        """Find all occurrences of a pattern in text.

        Returns list of (matched_text, position) tuples.
        """
        results = []
        if isinstance(matcher, str):
            # Plain string matching
            start = 0
            while True:
                idx = text_lower.find(matcher, start)
                if idx == -1:
                    break
                matched = text_original[idx:idx + len(matcher)]
                results.append((matched, idx))
                start = idx + 1
        else:
            # Compiled regex
            for m in matcher.finditer(text_lower):
                results.append((text_original[m.start():m.end()], m.start()))
        return results

    def _is_allowed(self, text, position, match_text):
        """Check if a match is a known false positive via allow list."""
        # Check a window around the match
        window_start = max(0, position - 20)
        window_end = min(len(text), position + len(match_text) + 30)
        window = text[window_start:window_end]

        for allow_pattern in self.ALLOW_LIST:
            if allow_pattern.search(window):
                return True
        return False

    def _deduplicate(self, violations):
        """Remove duplicate/overlapping violations, keeping highest severity."""
        if not violations:
            return violations

        severity_rank = {'error': 3, 'warning': 2, 'info': 1}

        # Sort by position and severity (highest first)
        violations.sort(key=lambda v: (v['position'], -severity_rank.get(v['severity'], 0)))

        deduped = []
        seen_positions = set()

        for v in violations:
            # Create a key from position range and category
            key = (v['position'], v['category'])
            if key not in seen_positions:
                seen_positions.add(key)
                deduped.append(v)

        return deduped
