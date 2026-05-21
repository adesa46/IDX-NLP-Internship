# 🏠 Real Estate NLP Pipeline

An end-to-end NLP system for real estate listing analysis — from raw MLS data to a production-ready REST API and interactive web UI. Built as an 11-week internship project.

## Features

| Module | Description |
|--------|-------------|
| **Taxonomy Builder** | Generates a categorized terminology taxonomy from 1,000+ listing remarks |
| **Entity Extraction** | Regex-based NER for bedrooms, bathrooms, price, sqft, and amenities |
| **Semantic Search** | FAISS + Sentence Transformers embeddings with BM25 comparison |
| **Signal Extraction** | Structured extraction of amenities, condition, financing, and location signals |
| **Query Parser** | Natural language → structured filters → SQL generation with schema validation |
| **Intent Classification** | TF-IDF + Logistic Regression classifier (browsing / researching / ready_to_buy) |
| **Listing Summarization** | Extractive summarizer with ROUGE evaluation |
| **Compliance Checker** | Fair Housing Act violation detection across 7 protected classes |
| **REST API** | FastAPI backend with caching, rate limiting, and OpenAPI docs |
| **Web UI** | Streamlit dashboard with search, NLP vs keyword comparison, and metrics |

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌────────────────┐
│  Streamlit UI   │────▶│  FastAPI API  │────▶│  NLP Modules   │
│  (Port 8501)    │     │  (Port 8000)  │     │                │
└─────────────────┘     └──────────────┘     │  • Search      │
                              │               │  • Parse       │
                              │               │  • Extract     │
                        ┌─────▼──────┐        │  • Summarize   │
                        │   MySQL    │        │  • Compliance  │
                        │  (3306)    │        │  • Classify    │
                        └────────────┘        └────────────────┘
```

## Project Structure

```
├── scripts/                    # Core NLP modules
│   ├── api_app.py              # FastAPI REST API (8 endpoints)
│   ├── streamlit_app.py        # Streamlit web dashboard
│   ├── semantic_search.py      # FAISS + BM25 search engine
│   ├── query_parser.py         # NL query → structured filters + SQL
│   ├── entity_extractor.py     # Regex NER (beds, baths, price, sqft)
│   ├── signal_extractor.py     # Amenity/condition/financing extraction
│   ├── listing_summarizer.py   # Extractive summarization + ROUGE eval
│   ├── compliance_checker.py   # Fair Housing Act compliance scanning
│   ├── intent_classifier.py    # Buyer intent classification
│   ├── taxonomy_builder.py     # Terminology taxonomy generation
│   ├── text_cleaning.py        # Data preprocessing pipeline
│   └── generate_week*.py       # Result generation scripts
│
├── data/
│   ├── raw/                    # Original MLS dataset (gitignored)
│   ├── processed/              # Cleaned data, taxonomy, signals, summaries
│   ├── schema.json             # Query validation schema
│   └── labeled_dataset.json    # Labeled NER evaluation dataset
│
├── notebooks/
│   └── data_exploration.ipynb  # EDA notebook with visualizations
│
├── tests/                      # Pytest suite (11 test files)
│   ├── test_week1.py – test_week11.py
│   └── test_taxonomy_coverage.py
│
├── Dockerfile                  # Python 3.11 container
├── docker-compose.yml          # Multi-service setup (MySQL, API, UI)
└── requirements.txt            # Python dependencies
```

## Quick Start

### Local Setup

```bash
# Clone the repo
git clone https://github.com/adesa46/IDX-NLP-Internship.git
cd IDX-NLP-Internship

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (required for summarization)
python -c "import nltk; nltk.download('punkt')"
```

### Run App Locally (Without Docker)

You will need two separate terminal windows.

**Terminal 1 (Backend API):**
```bash
python -m uvicorn scripts.api_app:app --reload
```

**Terminal 2 (Web UI):**
```bash
streamlit run scripts/streamlit_app.py
```
Access the Web UI at http://localhost:8501 and API docs at http://localhost:8000/docs.

### Docker Setup

```bash
# Start all services (MySQL + API + Streamlit UI)
docker-compose up -d

# Access:
#   API docs  → http://localhost:8000/docs
#   Web UI    → http://localhost:8501
```

### Run Individual Components

```bash
# Semantic search demo
python scripts/semantic_search.py

# Entity extraction
python scripts/entity_extractor.py

# Listing summarization
python scripts/listing_summarizer.py --input data/processed/cleaned_listing.csv

# Fair Housing compliance check
python -c "from scripts.compliance_checker import ComplianceChecker; c = ComplianceChecker(); print(c.check_listing('Beautiful 3 bed home with pool'))"
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/search` | Semantic search over listings |
| `POST` | `/parse-query` | NL query → structured filters + SQL |
| `POST` | `/extract-entities` | Extract beds/baths/price/sqft/amenities |
| `POST` | `/summarize` | Extractive listing summarization |
| `POST` | `/check-compliance` | Fair Housing violation detection |
| `POST` | `/classify-intent` | Buyer intent classification |
| `POST` | `/extract-signals` | Full signal extraction |
| `GET`  | `/health` | Health check / liveness probe |
| `GET`  | `/api-stats` | Cache & request statistics |

Interactive API docs available at `/docs` (Swagger UI) and `/redoc` when the API is running.

## Tests

```bash
# Run all tests
pytest tests/ -v

# Run a specific week's tests
pytest tests/test_week5.py -v
```

## Tech Stack

- **NLP**: NLTK, Sentence Transformers, scikit-learn, ROUGE
- **Search**: FAISS (vector similarity), BM25Okapi (keyword)
- **API**: FastAPI, Pydantic, uvicorn
- **UI**: Streamlit, Matplotlib
- **Data**: pandas, MySQL
- **Infra**: Docker, Docker Compose
