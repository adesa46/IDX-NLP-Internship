"""
Week 8: Answerability Checker

Determines whether a user query can be answered before and after SQL execution.

Pre-query checks:
  1. Is the query about real estate?
  2. Does it reference valid schema data? (via Week 4 SchemaValidator)

Post-query checks:
  1. Are there any results?
  2. Do results contain meaningful (non-null) data?
"""

import pandas as pd

from query_parser import QueryParser, SchemaValidator


# Real estate vocabulary used to determine if a query is in-domain
REAL_ESTATE_KEYWORDS = [
    'house', 'home', 'bed', 'bath', 'property',
    'listing', 'price', 'sqft', 'pool', 'garage',
    'condo', 'townhouse', 'apartment', 'duplex',
    'bedroom', 'bathroom', 'kitchen', 'yard',
    'square feet', 'acre', 'lot', 'mortgage',
    'rent', 'buy', 'sell', 'showing', 'open house',
    'remodel', 'renovated', 'move-in ready',
    'hoa', 'mls', 'realtor', 'real estate',
]


class AnswerabilityChecker:
    """Check whether a user query can be answered by the system."""

    def __init__(self, taxonomy=None, schema_validator=None,
                 schema_path=None):
        """
        Args:
            taxonomy: optional taxonomy dict (unused placeholder for future)
            schema_validator: optional pre-built SchemaValidator instance
            schema_path: path to schema.json (used if schema_validator is None)
        """
        self.taxonomy = taxonomy or {}
        if schema_validator is not None:
            self.validator = schema_validator
        else:
            self.validator = SchemaValidator(
                schema_path or 'data/schema.json'
            )
        self.parser = QueryParser()
        self.real_estate_keywords = REAL_ESTATE_KEYWORDS

    # ------------------------------------------------------------------
    # Pre-query check (BEFORE generating / executing SQL)
    # ------------------------------------------------------------------

    def check_pre_query(self, query):
        """Validate a query before SQL generation.

        Returns:
            tuple: (is_answerable: bool, message: str)
        """
        query_lower = query.lower()

        # Check 1: Is this a real estate question?
        has_re_terms = any(kw in query_lower for kw in self.real_estate_keywords)
        if not has_re_terms:
            return False, "This doesn't appear to be a real estate question."

        # Check 2: Does the query reference valid schema data?
        filters = self.parser.parse(query)
        valid, errors = self.validator.validate_query(filters)
        if not valid:
            return False, f"Query references invalid data: {'; '.join(errors)}"

        return True, "Query is answerable."

    # ------------------------------------------------------------------
    # Post-query check (AFTER executing SQL)
    # ------------------------------------------------------------------

    def check_post_query(self, query, results_df):
        """Validate query results after SQL execution.

        Args:
            query: original query string (for context in messages)
            results_df: pandas DataFrame of SQL results

        Returns:
            tuple: (has_results: bool, message: str)
        """
        if not isinstance(results_df, pd.DataFrame):
            return False, "Invalid results format."

        if len(results_df) == 0:
            return False, "No listings match your criteria."

        # Check for all-null results
        if results_df.isnull().all().all():
            return False, "Query returned no meaningful data."

        return True, f"Found {len(results_df)} matching listing(s)."

    # ------------------------------------------------------------------
    # Convenience: full pipeline check
    # ------------------------------------------------------------------

    def check_full(self, query, results_df=None):
        """Run pre-query check and optionally post-query check.

        Args:
            query: user query string
            results_df: optional DataFrame of SQL results

        Returns:
            dict with 'answerable', 'message', and 'stage' keys
        """
        ok, msg = self.check_pre_query(query)
        if not ok:
            return {'answerable': False, 'message': msg, 'stage': 'pre_query'}

        if results_df is not None:
            ok, msg = self.check_post_query(query, results_df)
            if not ok:
                return {'answerable': False, 'message': msg, 'stage': 'post_query'}

        return {'answerable': True, 'message': msg, 'stage': 'complete'}


if __name__ == '__main__':
    checker = AnswerabilityChecker()

    # Demo queries
    test_queries = [
        "3 bed house in Portland under 500k",      # valid RE query
        "What's the weather like today?",           # not RE
        "Show me homes in Atlantis",                # invalid city
        "2 bath condo with pool",                   # valid RE query
        "Tell me a joke",                           # not RE
    ]

    for q in test_queries:
        ok, msg = checker.check_pre_query(q)
        status = "Y" if ok else "N"
        print(f"  {status} [{msg}] {q}")

    # Post-query demo
    print("\n--- Post-query checks ---")
    empty_df = pd.DataFrame()
    ok, msg = checker.check_post_query("any query", empty_df)
    print(f"  Empty DF: {ok} — {msg}")

    null_df = pd.DataFrame({'a': [None, None], 'b': [None, None]})
    ok, msg = checker.check_post_query("any query", null_df)
    print(f"  Null DF:  {ok} — {msg}")

    good_df = pd.DataFrame({'price': [500000, 600000], 'beds': [3, 4]})
    ok, msg = checker.check_post_query("any query", good_df)
    print(f"  Good DF:  {ok} — {msg}")
