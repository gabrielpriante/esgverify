# ESGVerify

**LLM-powered ESG claim analysis for sustainability professionals.**

ESGVerify analyzes corporate sustainability documents — reports, press releases, filings — and evaluates whether environmental, social, and governance (ESG) claims are substantiated by the evidence presented. It reasons about claims, not just keywords.

> This tool is designed for sustainability professionals, researchers, and journalists who need to evaluate corporate ESG communications at scale. It does not constitute legal advice or regulatory compliance verification.

---

## What makes this different

Most ESG analysis tools flag green-sounding words. ESGVerify reads the document the way an analyst would:

1. **Extract claims** — Identify specific ESG assertions made in the text
2. **Find supporting evidence** — Locate data, certifications, or methodology referenced nearby
3. **Score the gap** — Quantify the distance between what is claimed and what is substantiated
4. **Classify by framework** — Map claims to EU Taxonomy, GRI Standards, TCFD, or SASB categories

Everything runs locally. No API keys. No data leaves your machine.

---

## Tech stack

| Layer | Technology | Why |
|---|---|---|
| LLM inference | [Ollama](https://ollama.com) + LLaMA 3.1 8B | Local, free, strong reasoning |
| Climate NLP | [ClimateBERT](https://huggingface.co/climatebert) | Domain-specific claim classification |
| Backend | FastAPI | Fast async API, excellent docs |
| Frontend | React + Vite + Tailwind | Modern, maintainable UI |
| Document parsing | PyMuPDF + python-docx | Multi-format support |
| Vector search | ChromaDB | Local semantic retrieval |

---

## Project structure

```
esgverify/
│
├── backend/                    # FastAPI application
│   ├── api/
│   │   └── routes/             # Endpoint definitions
│   ├── core/
│   │   ├── pipeline/           # Multi-stage analysis pipeline
│   │   └── models/             # Pydantic data models
│   ├── services/               # External integrations (Ollama, ChromaDB)
│   └── utils/                  # Shared helpers
│
├── frontend/                   # React application
│   └── src/
│       ├── components/         # Reusable UI components
│       ├── pages/              # Route-level page components
│       ├── hooks/              # Custom React hooks
│       └── lib/                # API client, utilities
│
├── config/                     # YAML configuration files
├── data/
│   ├── samples/                # Example ESG documents for testing
│   └── schemas/                # ESG framework schemas (GRI, TCFD, etc.)
├── docs/                       # Architecture and design documentation
├── scripts/                    # Dev tooling and setup scripts
└── tests/
    ├── unit/                   # Unit tests per module
    └── integration/            # End-to-end pipeline tests
```

---

## Quickstart

### Prerequisites

- Python 3.11+
- Node.js 18+
- [Ollama](https://ollama.com/download) installed and running

### 1. Pull the required model

```bash
ollama pull llama3.1:8b
```

### 2. Backend setup

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn api.main:app --reload
```

### 3. Frontend setup

```bash
cd frontend
npm install
npm run dev
```

The app will be available at `http://localhost:5173`. The API runs at `http://localhost:8000`.

---

## How the pipeline works

```
Document input (PDF / DOCX / TXT)
        │
        ▼
  [1] Document parser
      Extracts clean text, preserves structure
        │
        ▼
  [2] Claim extractor  (LLaMA 3.1 8B via Ollama)
      Identifies specific ESG assertions
        │
        ▼
  [3] Evidence retriever  (ChromaDB semantic search)
      Finds supporting data/context for each claim
        │
        ▼
  [4] Claim classifier  (ClimateBERT)
      Maps claims to ESG framework categories
        │
        ▼
  [5] Gap scorer  (LLaMA 3.1 8B via Ollama)
      Reasons about substantiation quality
        │
        ▼
  [6] Report generator
      Structured JSON + human-readable summary
```

---

## Roadmap

- [x] Repository scaffold and architecture design
- [ ] Document parsing (PDF, DOCX, TXT)
- [ ] Claim extraction pipeline
- [ ] Evidence retrieval with ChromaDB
- [ ] ClimateBERT claim classification
- [ ] Gap scoring with Ollama
- [ ] FastAPI backend with full route coverage
- [ ] React frontend — document upload and results view
- [ ] Export to PDF/CSV
- [ ] GRI and TCFD framework mapping
- [ ] Batch processing for multiple documents

---

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

## Disclaimer

ESGVerify is a research and awareness tool. Analysis results should not be used as the sole basis for investment, legal, or compliance decisions. Always consult qualified sustainability and legal professionals for formal assessments.
