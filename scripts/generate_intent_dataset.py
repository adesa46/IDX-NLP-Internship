"""
Week 7: Generate Labeled Intent Dataset

Produces 210+ queries across three intent categories:
  - browsing:     casual, vague interest
  - researching:  comparing, gathering info
  - ready_to_buy: specific criteria, action-oriented

Saves to data/processed/intent_dataset.json
"""

import json
import os
import random

random.seed(42)

# ── Building blocks ──────────────────────────────────────────────────

CITIES = ["Portland", "Irvine", "Los Angeles", "San Diego"]
AREAS = ["the suburbs", "downtown", "the hills", "the valley", "the beach", "uptown"]
PROPERTY_TYPES = ["condo", "townhouse", "single family", "house"]
STYLES = ["modern", "contemporary", "colonial", "ranch"]
AMENITIES = ["pool", "garage", "large backyard", "updated kitchen", "master suite",
             "dining room", "walk in closet", "fireplace"]
PRICES_LOW = ["200k", "250k", "300k", "350k", "400k"]
PRICES_HIGH = ["500k", "600k", "750k", "900k", "1m"]
BEDS = [2, 3, 4, 5]
BATHS = [1, 2, 3]
NEAR = ["good schools", "the park", "shopping centers", "public transit"]


def pick(lst):
    return random.choice(lst)


def generate_browsing_queries():
    """Casual, vague, window-shopping queries."""
    templates = [
        "What's available in {city}?",
        "Show me what's on the market",
        "Any new listings lately?",
        "What does the market look like in {area}?",
        "I'm just looking around",
        "Browsing homes in {city}",
        "What kind of {ptype}s are out there?",
        "Are there any open houses this weekend?",
        "Show me some listings",
        "What's the average price in {city}?",
        "I'm curious about homes in {area}",
        "Any nice neighborhoods to explore?",
        "Show me some {ptype}s",
        "What areas have good value?",
        "Just checking out properties",
        "Anything interesting in {area}?",
        "I want to see what's available",
        "What homes are listed right now?",
        "Any deals in {city}?",
        "Show me the newest listings",
        "What's trending in real estate?",
        "I'm thinking about maybe buying someday",
        "Casually looking at homes in {city}",
        "What can I get for {price}?",
        "Show me the most popular listings",
        "Any {ptype}s on the market?",
        "Give me an overview of {city} real estate",
        "What types of homes are in {area}?",
        "Show me homes with {amenity}",
        "I heard {city} is nice, what's available?",
        "Browse {style} homes",
        "What's for sale near {near}?",
        "General search for homes",
        "Looking around at properties in {area}",
        "Any good neighborhoods in {city}?",
    ]
    queries = []
    for t in templates:
        q = t.format(
            city=pick(CITIES), area=pick(AREAS), ptype=pick(PROPERTY_TYPES),
            style=pick(STYLES), amenity=pick(AMENITIES), near=pick(NEAR),
            price=pick(PRICES_HIGH),
        )
        queries.append(q)

    # Add more with second randomization pass
    for t in random.sample(templates, min(len(templates), 35)):
        q = t.format(
            city=pick(CITIES), area=pick(AREAS), ptype=pick(PROPERTY_TYPES),
            style=pick(STYLES), amenity=pick(AMENITIES), near=pick(NEAR),
            price=pick(PRICES_HIGH),
        )
        queries.append(q)

    return [(q, "browsing") for q in queries]


def generate_researching_queries():
    """Comparison, info-gathering, analytical queries."""
    templates = [
        "How do {ptype}s compare to {ptype2}s in {city}?",
        "What's the price difference between {area} and {area2}?",
        "Compare homes in {city} vs {city2}",
        "Which neighborhoods have the best schools?",
        "What's the average price per sqft in {city}?",
        "Is {city} a good place to invest?",
        "How much do {ptype}s cost in {area}?",
        "Are prices going up or down in {city}?",
        "What are the pros and cons of {ptype}s?",
        "Compare {style} vs {style2} homes",
        "Which areas are appreciating the fastest?",
        "What should I know about buying in {city}?",
        "How long do homes stay on market in {area}?",
        "What's the HOA like for {ptype}s in {city}?",
        "Is it better to buy a {ptype} or a {ptype2}?",
        "What are the property taxes in {city}?",
        "How do homes near {near} compare in price?",
        "What's the average lot size in {area}?",
        "Compare {bed} bed homes vs {bed2} bed homes",
        "What amenities add the most value?",
        "Help me understand the {city} market",
        "Which is a better investment, {city} or {city2}?",
        "What are the school ratings near {area}?",
        "How does {area} rank for families?",
        "What's the inventory like in {city}?",
        "Do {ptype}s appreciate faster than {ptype2}s?",
        "What financing options are available for {ptype}s?",
        "What's the crime rate near {area}?",
        "Tell me about the {city} housing market trends",
        "How much should I budget for a {ptype} in {city}?",
        "What's a fair price for {bed} beds in {city}?",
        "Research {style} homes in {area}",
        "Pros of living in {area} vs {area2}",
        "What are closing costs in {city}?",
        "Show me market analysis for {city}",
    ]
    queries = []
    for t in templates:
        q = t.format(
            city=pick(CITIES), city2=pick(CITIES),
            area=pick(AREAS), area2=pick(AREAS),
            ptype=pick(PROPERTY_TYPES), ptype2=pick(PROPERTY_TYPES),
            style=pick(STYLES), style2=pick(STYLES),
            near=pick(NEAR), bed=pick(BEDS), bed2=pick(BEDS),
        )
        queries.append(q)

    for t in random.sample(templates, min(len(templates), 35)):
        q = t.format(
            city=pick(CITIES), city2=pick(CITIES),
            area=pick(AREAS), area2=pick(AREAS),
            ptype=pick(PROPERTY_TYPES), ptype2=pick(PROPERTY_TYPES),
            style=pick(STYLES), style2=pick(STYLES),
            near=pick(NEAR), bed=pick(BEDS), bed2=pick(BEDS),
        )
        queries.append(q)

    return [(q, "researching") for q in queries]


def generate_ready_to_buy_queries():
    """Specific, action-oriented, criteria-heavy queries."""
    templates = [
        "{bed} bed {bath} bath under ${price} in {city}",
        "I need a {ptype} with {amenity} under ${price}",
        "Find me a {bed} bedroom {ptype} in {city} with {amenity}",
        "Schedule a showing for {ptype}s in {area}",
        "Ready to make an offer on a {ptype} in {city}",
        "Looking for {bed} bed {bath} bath near {near} under ${price}",
        "{bed} bed {style} home in {city} under ${price} with {amenity}",
        "I want to buy a {ptype} in {city} this month",
        "Show me {bed}+ bed homes under ${price} in {area}",
        "Find {ptype}s with {amenity} and {amenity2} in {city}",
        "I'm pre-approved for ${price}, find me {bed} bed homes in {city}",
        "Only show {ptype}s in {city} under ${price} with {amenity}",
        "Serious buyer looking for {bed} bed in {area}",
        "Need a {ptype} with at least {bed} beds and {bath} baths",
        "Want to close on a house in {city} by next month",
        "Submit offer for a {bed} bed {ptype} in {city}",
        "Exclude homes without {amenity}, {bed} bed under ${price}",
        "Move-in ready {bed} bed {bath} bath in {city}",
        "Show me turn-key {ptype}s in {area} under ${price}",
        "I need a {style} {ptype} in {city}, {bed} bed {bath} bath",
        "Looking to close quickly on a {ptype} in {city}",
        "Max budget ${price}, need {bed} beds in {city}",
        "Must have {amenity}, {bed} bed, {city}",
        "Find exactly a {bed} bed {bath} bath {ptype} near {near}",
        "Pre-qualified, looking for a {ptype} under ${price} in {area}",
        "I'm ready to buy a {bed} bed home in {city} with {amenity}",
        "{bed} bedroom {bath} bathroom {ptype} with {amenity} in {city} under ${price}",
        "Sort by lowest price, {bed} bed {ptype} in {city}",
        "Set up alerts for {bed} bed homes under ${price} in {city}",
        "Book a tour for {ptype}s with {amenity} in {area}",
        "Need immediate move-in {bed} bed in {city}",
        "Contact agent about {ptype}s in {city}",
        "Financing ready, need {bed} bed in {area} under ${price}",
        "Filter {ptype}s: {bed}+ beds, {bath}+ baths, under ${price}",
        "Searching for my next home: {bed} bed in {city} with {amenity}",
    ]
    queries = []
    for t in templates:
        q = t.format(
            city=pick(CITIES), area=pick(AREAS),
            ptype=pick(PROPERTY_TYPES), style=pick(STYLES),
            amenity=pick(AMENITIES), amenity2=pick(AMENITIES),
            near=pick(NEAR), bed=pick(BEDS), bath=pick(BATHS),
            price=pick(PRICES_HIGH).replace("k", "000").replace("m", "000000").lstrip("$"),
        )
        queries.append(q)

    for t in random.sample(templates, min(len(templates), 35)):
        q = t.format(
            city=pick(CITIES), area=pick(AREAS),
            ptype=pick(PROPERTY_TYPES), style=pick(STYLES),
            amenity=pick(AMENITIES), amenity2=pick(AMENITIES),
            near=pick(NEAR), bed=pick(BEDS), bath=pick(BATHS),
            price=pick(PRICES_HIGH).replace("k", "000").replace("m", "000000").lstrip("$"),
        )
        queries.append(q)

    return [(q, "ready_to_buy") for q in queries]


def main():
    browsing = generate_browsing_queries()
    researching = generate_researching_queries()
    ready_to_buy = generate_ready_to_buy_queries()

    all_data = browsing + researching + ready_to_buy
    random.shuffle(all_data)

    dataset = [{"query": q, "intent": label} for q, label in all_data]

    out_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'intent_dataset.json')
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2)

    # Stats
    from collections import Counter
    counts = Counter(d['intent'] for d in dataset)
    print(f"Generated {len(dataset)} queries:")
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    main()
