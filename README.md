# ESGVerify

**The extraction pipeline behind a research benchmark for environmental commitment extraction and verifiability scoring from corporate sustainability reports.**

ESGVerify reads a sustainability report and answers two questions about every sentence in it: *is this a pledge to a future environmental action or outcome?* and, if so, *what is its structure and how well is it evidenced within the document?*

It began as a greenwashing-screening tool. It is now the instrument for a benchmark paper, and this README describes what it actually is rather than what it was.

**Everything runs locally. No data leaves your machine.** ESGVerify uses Ollama to run an 8B model on your own hardware — no external APIs, no subscriptions. This started as a privacy feature and has become a research argument: the organizations that hold the most interesting sustainability documents are often contractually or legally unable to send them to a commercial API. A benchmark that only works via hosted inference cannot be run by the people with the data.

> **Disclaimer**: ESGVerify assists with the extraction and structuring of environmental commitments. It does not constitute legal advice, regulatory compliance assessment, or certification. Its outputs are model predictions, not findings about any company.

---

## The task

An **environmental commitment** is a stated intention, in writing, to take a future environmental action or reach a future environmental outcome. Future intention is the core test: if a sentence does not point forward in time, it is not a commitment however environmental it sounds.

What is promised must be *specific*. "We are committed to achieving net zero emissions across our value chain" is a commitment even with no date and no number. "We are committed to meeting our own goals" is not — the promised thing is generic.

Four categories are excluded, and each is recorded with its reason rather than silently dropped:

| Rejection | Example |
|---|---|
| `past_action` | "In 2023 we reduced water use by 12%." |
| `values_statement` | "Sustainability is at the heart of everything we do." |
| `factual_disclosure` | A fact carrying no promise. |
| `description` | A product, process or organization as it currently exists. |

Three rulings that recur, locked in the annotation guideline:

- **Conditional commitments count.** "Subject to government policy, we intend to electrify our fleet by 2035" is a commitment; the caveat is recorded in a field, never used to exclude.
- **Restated commitments count**, carrying a `restated` flag so repeat disclosure can be filtered and does not inflate counts.
- **Third-party validation is evidence, not a commitment.** "Our targets were validated by SBTi" describes verification of a pledge made elsewhere.

The annotation unit is the **sentence**. Span-level annotation is future work.

---

## What the pipeline does

Two stages, deliberately split.

**Sentence splitting happens locally, before any model call.** Chunk text is split into sentences, hard-wrapped PDF lines are rejoined, navigation furniture ("Overview", "Learn more") is dropped, and the remaining sentences are numbered.

**Stage 1 — detection.** The numbered sentences go to the model, which must return exactly one verdict per id: `yes` / `no` / `unsure`, plus a rejection reason, a `restated` flag and an `is_evidence` flag.

Numbering exists for two reasons. An earlier version asked the model to find and quote commitment sentences itself, and it silently omitted inconvenient ones — including a past-action sentence that should have been an explicit rejection. Numbering makes an omission *detectable*: any id that comes back missing is recorded as `unsure` with a note rather than vanishing. It also means **record text is taken from the source document by id, never from the model's echo**, so a record can never contain a sentence the model paraphrased or invented.

**Stage 2 — enrichment.** Each sentence judged `yes` goes back on its own, and the model fills seven structural fields:

| Field | Meaning |
|---|---|
| `target` | what is promised |
| `quantity` | how much |
| `deadline` | by when |
| `baseline` | starting point |
| `business_unit` | what part of the business |
| `emissions_scope` | what part of emissions (Scope 1/2/3) |
| `depends_on_outside_factors` | conditional on policy, infrastructure, third parties |

Every field carries a **status**, and a value only when `stated`:

`stated` · `not_stated` (the document is silent) · `not_applicable` (cannot apply here) · `unsure` (cannot tell)

These are kept distinct on purpose. Collapsing them into a null throws away the difference between a company that omitted a deadline and an annotator who could not find one.

Rejected sentences skip stage 2 entirely — their structural fields are `not_applicable` and their verifiability is null, because a sentence that is not a pledge has nothing to verify.

**Verifiability** is a five-level scale, judged **within the document only**:

| Level | Meaning |
|---|---|
| `strong` | specific data, checkable datasets, third-party verification |
| `moderate` | partial supporting evidence, gaps remain |
| `weak` | thin supporting evidence exists |
| `none` | zero corroborating evidence anywhere in the document |
| `unsure` | cannot determine either way |

The `weak` / `none` boundary is about the **presence** of evidence, not the specificity of the commitment. A vague pledge with some supporting context is `weak`, not `none`.

External verification — checking a certification body or regulatory registry — is out of scope for v1 and recorded as a limitation.

The commitment judgement is recorded **independently of ESG category**. A sentence can be environmental without being a commitment, and the two are separate fields.

---

## Provenance and reproducibility

This is a benchmark, so being able to reproduce a result matters more than any individual number.

**Prompts are versioned files, not string literals.** They live in `scripts/prompts/` and are published with the paper:

```
scripts/prompts/detect_v2.txt     stage 1
scripts/prompts/enrich_v1.txt     stage 2
```

**Every record carries its own provenance** — not the run, the record:

```json
{
  "model": "llama3.1:8b-instruct-q4_K_M",
  "detect_prompt_id": "detect_v2.txt@98921feba4f9",
  "enrich_prompt_id": "enrich_v1.txt@5709496bf0d7"
}
```

The identifier is `filename@sha256[:12]` of the prompt file's contents. A hash of an inline string would change silently whenever the module was edited; a hash of a file changes only when the prompt changes. A prompt over the token budget is refused at load time rather than silently degrading output.

`scripts/compare_runs.py` diffs two result files record by record and exits non-zero on any divergence. It also refuses to compare a degenerate run — one where every record is `unsure` because no model response parsed. That failure mode passed a green unit-test suite once; it does not pass this.

---

## The walling-off rule

**`llama3.1:8b-instruct-q4_K_M` is the annotation-assist model. It is excluded from evaluation, permanently.**

The benchmark measures how well language models extract commitments. The same model cannot both help produce the gold labels and be scored against them — its "accuracy" would partly measure agreement with itself. That is circular evaluation, and it invalidates the result.

Models under test are drawn from a disjoint set (Mistral, Qwen, Gemma, larger Llama variants). The annotation-assist model never appears among them.

The human annotator is the final authority on gold labels. The LLM is a second annotator whose disagreements are adjudicated by hand and logged.

---

## Known limitations

**Temporal confusion in the annotation-assist model.** It accepts some past-tense sentences as commitments. On chunk 39 of the development document, 4 sentences were judged commitments and 2 of those were wrong — one past-action sentence and one vague aspiration, both of which appear in the detection prompt as explicit counter-examples.

**This is not being engineered away, and that is a deliberate methodological choice.** Adding machinery to correct errors observed against our own development-set labels would fit the annotator to the gold standard and raise inter-annotator agreement for reasons unrelated to model quality. It would also scaffold away the very behaviour the failure taxonomy exists to measure. Temporal errors are corrected during hand adjudication, logged in the disagreement log, and reported as a finding about model behaviour.

A deterministic past-tense filter was considered and rejected more firmly still: it would hard-code the annotator's bias into the gold labels, and a reviewer could not tell whether the benchmark measures the models or measures the regex.

Other limitations:

- English-language documents only.
- Evidence scope is within-document; no external registry or certification checks.
- Throughput is bounded by local inference. On a single 8 GB card at ~13 tok/s, a 136-page report takes hours, not minutes.
- **Do not set `OLLAMA_NUM_PARALLEL` above 1 on a small card.** It divides the context window between slots, silently truncating the prompt and producing degenerate output that looks like a broken model.
- Model output quality depends on document structure; PDF text extraction from design-heavy reports is imperfect.

---

## Current status

**Section 5 complete** — pipeline built and tagged `v0.1.0-benchmark` as the paper's reference version.

**Corpus assembled** — 155 sustainability reports, 20 companies, 2015–2025, oil & gas and electric utilities. Two further documents (Microsoft 2023 Impact Summary, J&J Health for Humanity 2025) are held separately as guideline-development documents and are **not** part of the corpus frame; the annotation guideline was built on them, so including them would mean reporting results on documents the instrument was fitted to.

**Next** — gold-standard annotation on a sampled subset (~20–30 documents), with the LLM as independent second annotator against hand labels, then model evaluation and a failure taxonomy.

The original greenwashing pipeline (`claim_extractor`, `claim_analyzer`, `evidence_retriever`, ChromaDB retrieval) remains in the codebase and shares the verifiability scale. It supports a separate follow-on paper and is not part of the benchmark.

---

## Requirements

- Python 3.13+
- [Ollama](https://ollama.com/download) installed and running locally
- 8 GB+ VRAM recommended (the model uses ~4.9 GB, plus KV cache at 8192 context)
- Windows, macOS, or Linux

## Installation

```bash
git clone https://github.com/gabrielpriante/esgverify.git
cd esgverify
pip install -r backend/requirements.txt
```

> **Note:** Python 3.13 requires `torch==2.6.0` and `torchvision==0.21.0` specifically. Other versions break `sentence-transformers`, which the evidence-retrieval path depends on.

Pull the annotation-assist model:

```bash
ollama pull llama3.1:8b-instruct-q4_K_M
```

Verify Ollama is up:

```powershell
ollama list
```

## Running the extractor

The entry point is the commitment extractor. Output is auto-named `<document>_<YYYY-MM-DD>.json`:

```powershell
python scripts/run_commitment_extraction.py --pdf "path\to\report.pdf" --model llama3.1:8b-instruct-q4_K_M --timeout 900
```

Useful flags:

| Flag | Purpose |
|---|---|
| `--start N --limit N` | process a chunk range — start small on a new document |
| `--out PATH` | explicit output path instead of auto-naming |
| `--out-dir DIR` | directory for auto-named output (default `data/samples`) |
| `--dump-raw DIR` | write every raw model response before parsing, for debugging |
| `--concurrency N` | requests in flight; **leave at 1** on a single small GPU |
| `--num-ctx N` | context window (default 8192) |

Batch across a corpus, resumable across sessions:

```powershell
python scripts/run_corpus_batch.py --dry-run          # list what would run
python scripts/run_corpus_batch.py --limit-docs 1     # prove it on one document
python scripts/run_corpus_batch.py 2>&1 | Tee-Object -FilePath logs\corpus_batch.log
```

A document whose output already exists is skipped, so a restart never redoes or overwrites completed work. Failures are logged and the batch continues.

Compare two runs:

```powershell
python scripts/compare_runs.py data\samples\before.json data\samples\after.json
```

## Configuration

`backend/core/config.py`, via environment variables or `.env`:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b-instruct-q4_K_M
OLLAMA_TIMEOUT_SECONDS=900
CLAIM_EXTRACTION_CHUNK_SIZE=1500
CLAIM_EXTRACTION_CHUNK_OVERLAP=200
MAX_CONCURRENT_REQUESTS=1
```

## Running tests

```powershell
$env:PYTHONPATH = "C:\path\to\esgverify"; pytest tests/unit/ -v
```

**168 tests.**

| File | Tests | Covers |
|---|---|---|
| `test_commitment_extractor.py` | 89 | rulings, field distinctness, recall, two-stage flow, provenance, prompt guards |
| `test_claim_analyzer.py` | 27 | legacy claim path |
| `test_evidence_retriever.py` | 26 | legacy retrieval path |
| `test_claim_extractor.py` | 18 | legacy claim path |
| `test_orchestrator.py` | 9 | summary generation, UNSURE counting |

These are **offline**; every model call is mocked. They prove the pipeline handles model output correctly. They cannot prove the model behaves correctly on a real document — that requires a live run, and a green suite has masked a broken pipeline at least once here.

## Project structure

```
esgverify/
├── backend/core/
│   ├── models/report.py                  ExtractedCommitment, StructuredField, enums
│   └── pipeline/
│       ├── chunker.py                    text -> overlapping chunks
│       ├── commitment_extractor.py       two-stage extraction  [benchmark]
│       ├── prompts.py                    prompt loading + content hashing
│       ├── claim_extractor.py            legacy ESG claim path
│       ├── claim_analyzer.py             legacy substantiation scoring
│       ├── evidence_retriever.py         legacy ChromaDB retrieval
│       └── orchestrator.py               legacy pipeline coordinator
├── scripts/
│   ├── prompts/                          versioned prompt files (published)
│   ├── run_commitment_extraction.py      single document
│   ├── run_corpus_batch.py               resumable corpus pass
│   ├── compare_runs.py                   record-by-record equivalence gate
│   └── diagnose_ollama_prompt.py         prompt/context bisection tool
└── tests/unit/
```

## Tech stack

| Component | Technology |
|---|---|
| LLM runtime | Ollama (local) |
| Annotation-assist model | llama3.1:8b-instruct-q4_K_M |
| Data validation | Pydantic v2 |
| PDF parsing | PyMuPDF |
| HTTP client | httpx |
| Retry logic | tenacity |
| Logging | structlog |
| Vector store (legacy path) | ChromaDB + all-MiniLM-L6-v2 |
| API (legacy path) | FastAPI |

## License

MIT License — see [LICENSE](LICENSE).

## Resources

- [GRI Standards](https://www.globalreporting.org/standards/)
- [TCFD Recommendations](https://www.fsb-tcfd.org/recommendations/)
- [EU Green Claims Directive](https://environment.ec.europa.eu/topics/circular-economy/green-claims_en)
- [Ollama Documentation](https://ollama.com/docs)
