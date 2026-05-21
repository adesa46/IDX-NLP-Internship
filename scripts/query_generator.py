import csv
import random
import os

def generate_mock_queries():
    intents = {
        "Search": [
            "Find me a {style} home in {city}",
            "Looking for a {beds} bed {baths} bath near {location}",
            "Show me houses under ${price}k",
            "I want a {type} with a {amenity}"
        ],
        "Filter": [
            "Only show properties with a {amenity}",
            "Exclude homes without a {room}",
            "Sort by {sort_criteria}"
        ],
        "Compare": [
            "Compare {type}s in {city} vs {city2}",
            "What's the price difference between {beds} bed and {beds2} bed homes?"
        ]
    }

    variables = {
        "style": ["modern", "ranch", "contemporary", "colonial"],
        "city": ["downtown", "the suburbs", "the valley", "uptown"],
        "city2": ["the hills", "the beach", "the city center"],
        "beds": ["2", "3", "4", "5"],
        "beds2": ["3", "4", "5", "6"],
        "baths": ["1", "2", "3"],
        "price": ["300", "500", "750", "900"],
        "type": ["condo", "single family", "townhouse"],
        "amenity": ["pool", "garage", "large backyard", "updated kitchen"],
        "room": ["master suite", "dining room", "walk in closet"],
        "location": ["good schools", "the park", "shopping centers"],
        "sort_criteria": ["lowest price first", "newest listings", "largest lot size"]
    }

    output_file = 'data/processed/queries.csv'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"Generating 50 mock queries...")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['query', 'intent'])
        
        # Keep generating until we have 60
        generated = set()
        while len(generated) < 60:
            intent_label = random.choices(list(intents.keys()), weights=[0.6, 0.3, 0.1])[0]
            template = random.choice(intents[intent_label])
            
            # fill template
            query = template
            for var_name, var_list in variables.items():
                if f"{{{var_name}}}" in query:
                    query = query.replace(f"{{{var_name}}}", random.choice(var_list))
            
            if query not in generated:
                generated.add(query)
                writer.writerow([query, intent_label])

    print(f"Saved {len(generated)} queries to {output_file}")

if __name__ == "__main__":
    generate_mock_queries()
