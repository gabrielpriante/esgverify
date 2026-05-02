# ESGVerify

A local, LLM-powered ESG claim analysis tool for sustainability professionals. ESGVerify analyzes corporate documents — annual reports, sustainability disclosures, marketing materials — and determines whether environmental, social, and governance claims are actually substantiated by evidence in the document.

> **Disclaimer**: ESGVerify is designed to assist with early screening of ESG claims in corporate communications. It does not constitute legal advice, regulatory compliance assessment, or certification of any kind. For formal evaluation of environmental claims, consult qualified experts and verify against recognized standards.

## What is Greenwashing?

Greenwashing occurs when companies make misleading or unsubstantiated environmental claims about their products, services, or operations. Unlike simple keyword scanners, ESGVerify uses a local large language model to reason about whether claims are backed by concrete evidence — distinguishing between a company that says "we are carbon neutral" with third-party certification data and one that says the same thing with nothing to back it up.

## Core Principle

**Everything runs locally. No data leaves your machine.** ESGVerify uses Ollama to run LLaMA 3.1 8B on your hardware. No external APIs, no subscriptions, no data privacy concerns.

## Project Structure

```
esgverify/
│
├── backend/
│   ├── api/                        # FastAPI route handlers
│   ├── core/
│   │   ├── config.py               # Settings (Ollama URL, model, chunk sizes)
│   │   ├── models/
│   │   │   └── report.py           # Pydantic models: ExtractedClaim, ClaimAnalysis, etc.
│   │   └── pipeline/
│   │       ├── orchestrator.py     # Pipeline coordinator (6-stage)
│   │       ├── chunker.py          # Document text → overlapping chunks
│   │       └── claim_extractor.py  # Chunks → structured ESG claims via LLM
│   ├── services/                   # Business logic layer
│   └── utils/                      # Shared utilities
│
├── tests/
│   └── unit/
│       └── test_claim_extractor.py # 18 unit tests, all passing
│
├── requirements.txt
└── README.md
```

## How It Works

ESGVerify processes documents through a multi-stage pipeline:

**Stage 1 — Document Ingestion**: Accepts PDF, DOCX, and TXT files. Extracts raw text using PyMuPDF and python-docx.

**Stage 2 — Claim Extraction** ✅ Complete: Splits document text into overlapping chunks that respect sentence and paragraph boundaries, then sends each chunk to a local LLM (LLaMA 3.1 8B via Ollama). The model identifies ESG claims and returns them as structured JSON — text, context, ESG category (Environmental / Social / Governance), and relevant framework tags (GHG Protocol, TCFD, GRI, UN SDGs, etc.).

**Stages 3–6** — In progress: Claim analysis, evidence cross-referencing, risk scoring, and report generation.

## Requirements

- Python 3.13+
- [Ollama](https://ollama.com/download) installed and running locally
- 8GB+ VRAM recommended (the default model uses ~4.7GB)
- Windows, macOS, or Linux

## Installation

**1. Clone the repository:**
```bash
git clone https://github.com/gabrielpriante/esgverify.git
cd esgverify
```

**2. Install dependencies:**
```bash
pip install -r backend/requirements.txt
```

**3. Install and start Ollama:**

Download from [ollama.com/download](https://ollama.com/download), then pull the recommended model:
```bash
ollama pull llama3.1:8b-instruct-q4_K_M
```

This is a one-time ~4.9GB download. The `q4_K_M` quantization uses ~4.7GB VRAM, leaving comfortable headroom on an 8GB card.

**4. Verify Ollama is running:**
```powershell
Invoke-RestMethod http://localhost:11434/api/tags
```
You should see your model listed in the response.

## Configuration

Settings are managed in `backend/core/config.py` via environment variables or a `.env` file:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b-instruct-q4_K_M
OLLAMA_TIMEOUT_SECONDS=120
CLAIM_EXTRACTION_CHUNK_SIZE=1500
CLAIM_EXTRACTION_CHUNK_OVERLAP=200
```

## Running Tests

```powershell
$env:PYTHONPATH = "C:\path\to\esgverify"; pytest tests/unit/ -v
```

Current test coverage:
- `test_claim_extractor.py` — 18 tests, 18 passing
  - JSON parsing (valid, malformed, markdown fences, missing keys)
  - Claim conversion (valid, missing fields, unknown categories, unique IDs)
  - End-to-end extraction (success, empty input, multi-chunk aggregation, retry/skip behavior)

## Tech Stack

| Component | Technology |
|---|---|
| LLM runtime | Ollama (local) |
| LLM model | LLaMA 3.1 8B (q4_K_M) |
| Backend framework | FastAPI |
| Data validation | Pydantic v2 |
| PDF parsing | PyMuPDF |
| DOCX parsing | python-docx |
| Vector store | ChromaDB |
| HTTP client | httpx |
| Retry logic | tenacity |
| Logging | structlog |
| Frontend | React + Vite (in progress) |

## ESG Categories

Claims are classified into four categories:

- **ENVIRONMENTAL** — emissions, energy, water, waste, biodiversity, climate
- **SOCIAL** — labor practices, supply chain, diversity, community
- **GOVERNANCE** — board composition, executive pay, anti-corruption, transparency
- **UNKNOWN** — claims that don't clearly fit the above

## Supported Frameworks

The LLM is prompted to tag claims against recognized ESG frameworks:
GHG Protocol, TCFD, GRI, UN SDGs, CDP, SASB, EU Taxonomy, ISSB

## Limitations

- Only analyzes English-language documents
- LLM output quality depends on document clarity and structure
- Does not verify claims against external databases or registries
- Does not determine regulatory compliance or legal liability
- Claim extraction accuracy improves with longer, more structured documents

## Intended Use

ESGVerify is most relevant for sustainability professionals, ESG analysts, and researchers analyzing:

- Corporate sustainability reports and annual disclosures
- Marketing materials and product environmental claims
- Press releases and investor communications
- Public filings with environmental statements

It is not suitable for legal proceedings, formal regulatory complaints, or academic research requiring rigorous reproducibility.

## Roadmap

- [x] Stage 1: Document ingestion (PDF, DOCX, TXT)
- [x] Stage 2: LLM-powered claim extraction with structured output
- [ ] Stage 3: Claim analysis — substantiation level and risk scoring
- [ ] Stage 4: Evidence cross-referencing via ChromaDB
- [ ] Stage 5: Report generation (PDF export)
- [ ] Stage 6: React frontend with document upload and results dashboard

## License

MIT License — see [LICENSE](LICENSE) for details.

## Resources

- [EU Green Claims Directive](https://environment.ec.europa.eu/topics/circular-economy/green-claims_en)
- [FTC Green Guides](https://www.ftc.gov/news-events/topics/truth-advertising/green-guides)
- [GRI Standards](https://www.globalreporting.org/standards/)
- [TCFD Recommendations](https://www.fsb-tcfd.org/recommendations/)
- [Ollama Documentation](https://ollama.com/docs)
