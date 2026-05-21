# Real Estate Taxonomy Project

This project builds a real estate terminology taxonomy with an associated dataset, queries, and analytical tools.

## Structure
- `data/raw/rets_property.sql`: The original 400MB raw dataset (ignored by git).
- `data/processed/sample_listing.csv`: A sampled dataset of property listings.
- `data/processed/taxonomy.json`: A generated taxonomy categorizing prevalent real estate terminology into 8 main categories.
- `data/processed/queries.csv`: Automatically generated mock user searches with intents.
- `notebooks/data_exploration.ipynb`: Jupyter notebook to analyze remark length and taxonomy distributions.
- `scripts/`: Python scripts for data extraction, taxonomy building, and query generation.
- `tests/`: Pytest suite to validate coverage.

## Setup
Install requirements:
```bash
pip install pandas nltk jupyter matplotlib seaborn pytest mysql-connector-python
```

Run notebooks with Jupyter:
```bash
jupyter notebook notebooks/data_exploration.ipynb
```

Run tests with PyTest:
```bash
pytest tests/test_taxonomy_coverage.py -s
pytest tests/test_week2.py -v
```

Run Data Cleaning Pipeline (Week 2):
```bash
python scripts/week2_pipeline.py
```
