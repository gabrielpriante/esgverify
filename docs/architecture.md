# Architecture: ESGVerify Analysis Pipeline

## Overview

ESGVerify uses a multi-stage pipeline to analyze ESG documents. This document
explains the design decisions behind that architecture.

---

## Why a staged pipeline?

A single LLM prompt asking "is this document greenwashing?" produces unreliable
results. Large documents exceed context windows, the model can't reason about
evidence it hasn't been shown, and there's no structured output to display.

A staged pipeline solves each problem:

| Problem | Solution |
|---|---|
| Documents exceed LLM context window | Chunking stage splits text into manageable pieces |
| Model hallucinates evidence | Evidence retrieval grounds claims in actual document text |
| Unstructured output | Each stage produces typed Pydantic models |
| Single failure point | Stages can be debugged and improved independently |

---

## Stage design

### Stage 1 — Document chunking
Split the document into overlapping chunks (~1500 chars, 200 char overlap).
Overlap ensures claims that span paragraph boundaries are not lost.

### Stage 2 — Claim extraction (Ollama / LLaMA 3.1 8B)
For each chunk, prompt the LLM to extract ESG assertions. The prompt is
structured to return JSON, which maps directly to `ExtractedClaim` models.
Using a local model means no document data leaves the user's machine.

### Stage 3 — Evidence retrieval (ChromaDB)
Embed the full document text and store in a local vector database.
For each extracted claim, retrieve the top-k most semantically similar
passages. These become the `SupportingEvidence` list for that claim.

### Stage 4 — Claim classification (ClimateBERT)
ClimateBERT is a domain-specific model fine-tuned on climate and ESG text.
It classifies each claim into ESG categories (E/S/G) more reliably than
a general-purpose model would.

### Stage 5 — Gap scoring (Ollama / LLaMA 3.1 8B)
A second LLM call presents the claim and its retrieved evidence side by side,
asking the model to reason about whether the evidence actually substantiates
the claim. This produces the `gap_explanation` and `substantiation_level`.

### Stage 6 — Report assembly
Aggregate all `ClaimAnalysis` results into an `AnalysisReport` with summary
statistics. This is the final object returned to the API and displayed in
the frontend.

---

## Local-first principle

All inference runs locally via Ollama and a local ChromaDB instance.
No document content is sent to external APIs. This is a hard requirement —
sustainability reports often contain sensitive competitive information.

---

## Hardware requirements

The pipeline is designed for the target development machine:
- Ryzen AI 9 + 32 GB RAM + 8 GB VRAM
- LLaMA 3.1 8B fits comfortably in 8 GB VRAM at 4-bit quantization
- ClimateBERT (~110M params) runs on CPU with negligible overhead
- ChromaDB is embedded, no separate server needed
