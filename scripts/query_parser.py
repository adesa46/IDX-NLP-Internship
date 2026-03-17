import re
import json
import os

class QueryParser:
    def __init__(self):
        pass

    def _parse_number(self, num_str, multiplier_str):
        num = int(num_str)
        if multiplier_str.lower() == 'k':
            num *= 1000
        elif multiplier_str.lower() == 'm':
            num *= 1000000
        return num

    def parse(self, query):
        filters = {}

        # Price patterns
        if match := re.search(r'(?:under|below|less than|max)\s*\$?(\d+)([km]?)', query, re.I):
            filters['price_max'] = self._parse_number(match.group(1), match.group(2))
            
        if match := re.search(r'(?:over|above|more than|min)\s*\$?(\d+)([km]?)', query, re.I):
            filters['price_min'] = self._parse_number(match.group(1), match.group(2))

        # Bedroom patterns
        if match := re.search(r'(\d+)\+?\s*(?:bed|br|beds|bedroom|bedrooms)\b', query, re.I):
            filters['bedrooms_min' if '+' in match.group(0) else 'bedrooms'] = int(match.group(1))

        # Bathroom patterns
        if match := re.search(r'(\d+)\+?\s*(?:bath|ba|baths|bathroom|bathrooms)\b', query, re.I):
            filters['bathrooms_min' if '+' in match.group(0) else 'bathrooms'] = int(match.group(1))

        # Location / City patterns (simple approach: look for 'in <location>' or 'near <location>')
        if match := re.search(r'\bin\s+(the suburbs|the city center|the hills|the valley|the beach|uptown|downtown|Irvine|Portland|Los Angeles|San Diego)\b', query, re.I):
            filters['city'] = match.group(1).lower()

        if match := re.search(r'\bnear\s+(good schools|the park|shopping centers)\b', query, re.I):
            filters['near'] = match.group(1).lower()

        # Property type
        if match := re.search(r'\b(single family|single familys|condo|townhouse|houses)\b', query, re.I):
            filters['property_type'] = match.group(1).lower()

        # Architectural Style
        if match := re.search(r'\b(modern|contemporary|colonial|ranch)\b', query, re.I):
            filters['style'] = match.group(1).lower()

        # Amenities (with)
        if match := re.search(r'\b(?:with|has)\s+(?:a\s+)?(pool|garage|large backyard|updated kitchen|master suite|dining room|walk in closet)\b', query, re.I):
            filters['amenity'] = match.group(1).lower()
            
        if match := re.search(r'\bonly\s+show\s+properties\s+with\s+(?:a\s+)?(pool|garage|large backyard|updated kitchen|master suite|dining room|walk in closet)\b', query, re.I):
            filters['amenity'] = match.group(1).lower()

        # Amenities (without)
        if match := re.search(r'\b(?:without|exclude homes without|no)\s+(?:a\s+)?(pool|garage|large backyard|updated kitchen|master suite|dining room|walk in closet)\b', query, re.I):
            filters['exclude_amenity'] = match.group(1).lower()

        # Sort
        if match := re.search(r'\bsort by\s+(lowest price first|newest listings|largest lot size)\b', query, re.I):
            filters['sort_by'] = match.group(1).lower()

        # Compare intent
        if match := re.search(r'\bCompare\s+(.*?)\s+in\s+(.*?)\s+vs\s+(.*)', query, re.I):
            filters['intent'] = 'Compare'
            filters['compare_type'] = match.group(1).strip().lower()
            filters['compare_location1'] = match.group(2).strip().lower()
            filters['compare_location2'] = match.group(3).strip().lower()
            
        if match := re.search(r'\bprice difference between\s+(.*?)\s+and\s+(.*)', query, re.I):
            filters['intent'] = 'Compare'
            filters['compare_item1'] = match.group(1).strip().lower()
            filters['compare_item2'] = match.group(2).strip().lower()

        return filters

    def to_sql(self, filters):
        conditions = []
        params = []

        if 'price_max' in filters:
            conditions.append('L_SystemPrice <= %s')
            params.append(filters['price_max'])
            
        if 'price_min' in filters:
            conditions.append('L_SystemPrice >= %s')
            params.append(filters['price_min'])

        if 'bedrooms' in filters:
            conditions.append('L_Keyword2 = %s')
            params.append(filters['bedrooms'])
            
        if 'bedrooms_min' in filters:
            conditions.append('L_Keyword2 >= %s')
            params.append(filters['bedrooms_min'])

        if 'bathrooms' in filters:
            conditions.append('L_Keyword3 = %s')
            params.append(filters['bathrooms'])
            
        if 'bathrooms_min' in filters:
            conditions.append('L_Keyword3 >= %s')
            params.append(filters['bathrooms_min'])

        if 'city' in filters:
            conditions.append('L_City = %s')
            params.append(filters['city'])
            
        if 'near' in filters:
            conditions.append('L_Remarks LIKE %s')
            params.append(f"%{filters['near']}%")
            
        if 'property_type' in filters:
            conditions.append('L_Type_ = %s')
            params.append(filters['property_type'])
            
        if 'style' in filters:
            conditions.append('L_Remarks LIKE %s')
            params.append(f"%{filters['style']}%")
            
        if 'amenity' in filters:
            conditions.append('L_Remarks LIKE %s')
            params.append(f"%{filters['amenity']}%")
            
        if 'exclude_amenity' in filters:
            conditions.append('L_Remarks NOT LIKE %s')
            params.append(f"%{filters['exclude_amenity']}%")
            
        v_sort = ""
        if 'sort_by' in filters:
            if filters['sort_by'] == 'lowest price first':
                v_sort = " ORDER BY L_SystemPrice ASC"
            elif filters['sort_by'] == 'newest listings':
                v_sort = " ORDER BY L_ListingDate DESC" 
            elif filters['sort_by'] == 'largest lot size':
                v_sort = " ORDER BY L_LotSizeSqFt DESC" 

        if conditions:
            where_clause = ' AND '.join(conditions)
            sql = f"SELECT * FROM rets_property WHERE {where_clause}{v_sort}"
        else:
            sql = f"SELECT * FROM rets_property{v_sort}"
            
        return sql, params


class SchemaValidator:
    def __init__(self, schema_path='../data/schema.json'):
        if not os.path.exists(schema_path):
            schema_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'schema.json')
            
        with open(schema_path) as f:
            self.schema = json.load(f)
            self.valid_cities = [c.lower() for c in self.schema.get('valid_cities', [])]

    def validate_query(self, filters):
        errors = []

        # Check city exists in database
        if 'city' in filters:
            if filters['city'].lower() not in self.valid_cities:
                errors.append(f"City '{filters['city']}' not found in database")

        # Check price range
        if 'price_max' in filters:
            if filters['price_max'] < 100000 or filters['price_max'] > 10000000:
                errors.append(f"Price {filters['price_max']} outside typical range")
                
        if 'price_min' in filters:
            if filters['price_min'] < 100000 or filters['price_min'] > 10000000:
                errors.append(f"Price {filters['price_min']} outside typical range")

        # Check bedroom count
        if 'bedrooms' in filters:
            if filters['bedrooms'] < 1 or filters['bedrooms'] > 10:
                errors.append(f"Bedroom count {filters['bedrooms']} seems invalid")
                
        if 'bathrooms' in filters:
            if filters['bathrooms'] < 1 or filters['bathrooms'] > 10:
                errors.append(f"Bathroom count {filters['bathrooms']} seems invalid")

        return len(errors) == 0, errors

if __name__ == "__main__":
    parser = QueryParser()
    validator = SchemaValidator()
    filters = parser.parse("3 bed in Portland under 500k")
    valid, errors = validator.validate_query(filters)
    if not valid:
        print(f"Query validation errors: {errors}")
    else:
        sql, params = parser.to_sql(filters)
        print(f"SQL: {sql}\nParams: {params}")
