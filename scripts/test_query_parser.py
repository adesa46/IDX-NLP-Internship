import unittest
import csv
import os
from query_parser import QueryParser, SchemaValidator

class TestQueryParser(unittest.TestCase):
    def setUp(self):
        self.parser = QueryParser()
        self.validator = SchemaValidator()
        self.queries_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'queries.csv')
        self.queries = []
        if os.path.exists(self.queries_file):
            with open(self.queries_file, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.queries.append(row['query'])

    def test_parse_coverage(self):
        if not self.queries:
            self.skipTest("Queries file not found")
            
        parsed_count = 0
        for query in self.queries:
            filters = self.parser.parse(query)
            # If we extracted at least one valid key, count it as a successful parse
            if len(filters) > 0:
                parsed_count += 1
            else:
                print(f"Failed to parse: {query}")
                
        coverage = parsed_count / len(self.queries)
        print(f"\nParse Coverage: {coverage:.2%} ({parsed_count}/{len(self.queries)})")
        self.assertGreaterEqual(coverage, 0.90, "Coverage should be at least 90%")

    def test_sql_injection_protection(self):
        # The parser should create parameterized query and not concatenate strings!
        query = "3 bed in Irvine'; DROP TABLE users;--"
        filters = self.parser.parse(query)
        sql, params = self.parser.to_sql(filters)
        
        # Test that 'DROP TABLE' doesn't appear in the main SQL string
        self.assertNotIn("DROP TABLE", sql, "SQL string should not contain injected commands directly")
        
        # We expect the payload to be passed as a parameter safely
        # Note: actually my parser looks for bounded cities, but let's say a user bypassed it or we passed filters manually.
        filters_malicious = {'city': "Irvine'; DROP TABLE users;--"}
        sql, params = self.parser.to_sql(filters_malicious)
        self.assertNotIn("DROP TABLE", sql)
        self.assertEqual(sql, "SELECT * FROM rets_property WHERE L_City = %s")
        self.assertIn("Irvine'; DROP TABLE users;--", params)

    def test_schema_validator_valid(self):
        filters = {'city': 'Irvine', 'price_max': 500000, 'bedrooms': 3}
        valid, errors = self.validator.validate_query(filters)
        self.assertTrue(valid)
        self.assertEqual(len(errors), 0)

    def test_schema_validator_invalid_city(self):
        filters = {'city': 'Atlantis'}
        valid, errors = self.validator.validate_query(filters)
        self.assertFalse(valid)
        self.assertIn("City 'Atlantis' not found in database", errors[0])

    def test_schema_validator_invalid_range(self):
        filters = {'bedrooms': 99}
        valid, errors = self.validator.validate_query(filters)
        self.assertFalse(valid)
        self.assertIn("Bedroom count 99 seems invalid", errors[0])
        
        filters = {'price_max': 500} # $500 max price is outside typical range
        valid, errors = self.validator.validate_query(filters)
        self.assertFalse(valid)
        self.assertTrue(any("Price 500 outside typical range" in e for e in errors))

    def test_specific_patterns(self):
        # Test a few specific patterns explicitly
        filters = self.parser.parse("Looking for a 4 bed 1 bath near good schools")
        self.assertEqual(filters.get('bedrooms'), 4)
        self.assertEqual(filters.get('bathrooms'), 1)
        self.assertEqual(filters.get('near'), 'good schools')
        
        filters = self.parser.parse("Show me houses under $750k")
        self.assertEqual(filters.get('price_max'), 750000)
        
        filters = self.parser.parse("Exclude homes without a master suite")
        self.assertEqual(filters.get('exclude_amenity'), 'master suite')

if __name__ == '__main__':
    unittest.main()
