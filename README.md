# ESGVerify

A local, LLM-powered ESG claim analysis tool for sustainability professionals. ESGVerify analyzes corporate documents: annual reports, sustainability disclosures, marketing materials — and determines whether environmental, social, and governance claims are actually substantiated by evidence in the document.

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
│   ├── main.py                         # FastAPI app entry point
│   ├── api/
│   │   └── routes/
│   │       ├── analysis.py             # POST /api/v1/analyze
│   │       └── health.py               # GET /api/v1/health
│   ├── core/
│   │   ├── config.py                   # Settings (Ollama URL, model, chunk sizes)
│   │   ├── models/
│   │   │   └── report.py               # Pydantic models: ExtractedClaim, ClaimAnalysis, etc.
│   │   └── pipeline/
│   │       ├── orchestrator.py         # Pipeline coordinator (4 stages complete)
│   │       ├── chunker.py              # Document text → overlapping chunks
│   │       ├── claim_extractor.py      # Chunks → structured ESG claims via LLM
│   │       ├── claim_analyzer.py       # Claims → substantiation level + risk scoring
│   │       └── evidence_retriever.py   # Claims → supporting evidence via ChromaDB
│   ├── services/                       # Business logic layer
│   └── utils/
│       └── document_parser.py          # PDF, DOCX, TXT text extraction
│
├── tests/
│   └── unit/
│       ├── test_claim_extractor.py     # 18 tests, 18 passing
│       ├── test_claim_analyzer.py      # 27 tests, 27 passing
│       └── test_evidence_retriever.py  # 26 tests, 26 passing
│
├── requirements.txt
└── README.md
```

## How It Works

ESGVerify processes documents through a multi-stage pipeline:

**Stage 1 — Document Ingestion** ✅ Complete: Accepts PDF, DOCX, and TXT files. Extracts raw text using PyMuPDF and python-docx.

**Stage 2 — Claim Extraction** ✅ Complete: Splits document text into overlapping chunks that respect sentence and paragraph boundaries, then sends each chunk to a local LLM (LLaMA 3.1 8B via Ollama). The model identifies ESG claims and returns them as structured JSON — text, context, ESG category (Environmental / Social / Governance), and relevant framework tags (GHG Protocol, TCFD, GRI, UN SDGs, etc.).

**Stage 3 — Claim Analysis** ✅ Complete: Each extracted claim is sent back to the LLM for substantiation scoring. Returns a `SubstantiationLevel` (strong / moderate / weak / none), a `RiskLevel` (high / medium / low), a `gap_explanation` describing what evidence is missing, and a `confidence` score.

**Stage 4 — Evidence Retrieval** ✅ Complete: Embeds document chunks into ChromaDB using `all-MiniLM-L6-v2` and performs cosine similarity search for each claim. Returns up to 5 supporting passages per claim with relevance scores and automatically classified evidence types (certification, third_party_reference, data, methodology).

**Stage 5 — React Frontend** — In progress.

**Stage 6 — Report Export (PDF)** — Planned.

## Verified Results

Analyzed the Microsoft 2023 Impact Summary (22MB PDF, 69,896 characters):

| Metric | Value |
|---|---|
| Total claims identified | 141 |
| High greenwashing risk | 86 (61%) |
| Strong substantiation | 25 (18%) |
| Weak or no substantiation | 87 (62%) |
| Overall risk level | HIGH |
| Evidence passages retrieved | 690 |

Most claims related to Environmental (64), followed by Social (60) and Governance (12).

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
pip install -r requirements.txt
```

> **Note:** Python 3.13 requires `torch==2.6.0` and `torchvision==0.21.0` specifically. Other versions will break `sentence-transformers`. If you encounter import errors, run: `pip install torch==2.6.0 torchvision==0.21.0`

**3. Install and start Ollama:**

Download from [ollama.com/download](https://ollama.com/download), then pull the model:
```bash
ollama pull llama3.1:8b
```

**4. Verify Ollama is running:**
```powershell
Invoke-RestMethod http://localhost:11434/api/tags
```

## Running the Server

Open two terminals:

**Terminal 1 — Start Ollama:**
```bash
ollama run llama3.1:8b
```

**Terminal 2 — Start the API server:**
```powershell
$env:PYTHONPATH = "C:\path\to\esgverify"; uvicorn backend.main:app --reload
```

Swagger UI is available at `http://127.0.0.1:8000/docs`.

## Submitting a Document (PowerShell)

```powershell
$env:PYTHONPATH = "C:\path\to\esgverify"
$filePath = "C:\path\to\your\document.pdf"
$fileBytes = [System.IO.File]::ReadAllBytes($filePath)
$boundary = [System.Guid]::NewGuid().ToString()
$body = "--$boundary`r`nContent-Disposition: form-data; name=`"file`"; filename=`"document.pdf`"`r`nContent-Type: application/pdf`r`n`r`n" + [System.Text.Encoding]::GetEncoding("iso-8859-1").GetString($fileBytes) + "`r`n--$boundary--"
$response = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/analyze" -Method POST -ContentType "multipart/form-data; boundary=$boundary" -Body $body -TimeoutSec 3600
$response | ConvertTo-Json -Depth 20 | Out-File "report.json"
```

> **Note:** Processing takes approximately 25 minutes for a 70k character document due to sequential LLM calls (~290 total for extraction + analysis).

## Configuration

Settings are managed in `backend/core/config.py` via environment variables or a `.env` file:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
OLLAMA_TIMEOUT_SECONDS=120
CLAIM_EXTRACTION_CHUNK_SIZE=1500
CLAIM_EXTRACTION_CHUNK_OVERLAP=200
```

## Running Tests

```powershell
$env:PYTHONPATH = "C:\path\to\esgverify"; pytest tests/unit/ -v
```

Current test coverage: **71 tests, 71 passing**

| File | Tests | Status |
|---|---|---|
| `test_claim_extractor.py` | 18 | ✅ Passing |
| `test_claim_analyzer.py` | 27 | ✅ Passing |
| `test_evidence_retriever.py` | 26 | ✅ Passing |

## Tech Stack

| Component | Technology |
|---|---|
| LLM runtime | Ollama (local) |
| LLM model | LLaMA 3.1 8B |
| Backend framework | FastAPI |
| Data validation | Pydantic v2 |
| PDF parsing | PyMuPDF |
| DOCX parsing | python-docx |
| Vector store | ChromaDB |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
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

## Substantiation Levels

| Level | Description |
|---|---|
| **strong** | Claim is backed by specific data, certifications, or third-party verification |
| **moderate** | Claim has partial supporting evidence but gaps remain |
| **weak** | Claim has minimal support; mostly aspirational language |
| **none** | No evidence found in the document to support the claim |

## Supported Frameworks

The LLM is prompted to tag claims against recognized ESG frameworks:
GHG Protocol, TCFD, GRI, UN SDGs, CDP, SASB, EU Taxonomy, ISSB

## Known Limitations

- Only analyzes English-language documents
- Sequential LLM calls result in ~25 minute processing time for typical documents
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
- [x] Stage 3: Claim analysis — substantiation level and risk scoring
- [x] Stage 4: Evidence cross-referencing via ChromaDB
- [ ] Stage 5: React + Vite frontend with document upload and results dashboard
- [ ] Stage 6: Report generation (PDF export)
- [ ] Performance: Async/concurrent LLM calls to reduce processing time

## License

MIT License — see [LICENSE](LICENSE) for details.

## Resources

- [EU Green Claims Directive](https://environment.ec.europa.eu/topics/circular-economy/green-claims_en)
- [FTC Green Guides](https://www.ftc.gov/news-events/topics/truth-advertising/green-guides)
- [GRI Standards](https://www.globalreporting.org/standards/)
- [TCFD Recommendations](https://www.fsb-tcfd.org/recommendations/)
- [Ollama Documentation](https://ollama.com/docs)
