# Deep Research Agent

This project implements a production-oriented, LangGraph-based deep research and fact-check workflow. The system breaks a large research question into 3-5 independent research tasks, gathers parallel evidence, checks support and quality, and then writes a structured executive research report.

## Architecture overview

The workflow is built around a `StateGraph` with a typed `ResearchState`.

1. Planner creates 3–5 independent research tasks and passes them directly to dispatch.
2. LangGraph's `Send` API starts one parallel worker per task.
3. Each worker runs hybrid retrieval across the curated Chroma corpus and Tavily. Chroma combines cosine-similarity results with BM25-style keyword matches, and each task makes a bounded web search for up to four additional results.
4. Web results are URL-deduplicated and ranked using search relevance plus transparent, heuristic authority and freshness signals; selection prefers distinct registrable domains.
5. Findings are aggregated into shared state using list reducers.
6. The fact checker compares supplied excerpts, and the report writer creates a draft.
7. The writer and critic share a task-by-task evidence matrix. The critic inventories material claims, audits them in bounded claim/source batches, validates every audit reference and quote, and prevents a passing review when a critical claim remains unresolved or citation coverage is incomplete.
8. Failed drafts return to the writer with critic feedback for the configured automatic revision limit.
9. A passing draft pauses at a LangGraph `interrupt()` for final human approval.
10. Approval saves the final report under `outputs/`; revision resumes the writer/critic loop, while rejection ends without approval. After automatic retries, the reviewer can request one final feedback-driven revision; after it, only Approve and Reject are available.

Search-result snippets and local PDF excerpts are identified separately from retrieved page excerpts and cannot be marked independently verified by the fact checker. Local excerpts retain their source title, URL, publication date, and PDF page number. The checker compares supplied text; it does not independently open or validate original source pages. Authority and freshness scores are ranking heuristics, not proof of source quality.

The fact checker operates before report writing and classifies individual retrieved evidence records. The claim auditor operates after drafting and checks each report claim against the gathered source passages. Neither role independently verifies the original source website.

For the final claim audit, report URLs, source aliases, and cited page numbers guide source matching. If the claim inventory omits a cited or numeric sentence, the reviewer retries; if it is still omitted, a warning is logged and the sentence is included verbatim as a critical claim for audit, not treated as verified. Chroma sources are grouped by source ID and all indexed chunks are scanned; cited-page passages are prioritized and missing indexed pages are flagged as incomplete retrieval. Every inventoried material claim receives an audit result; claims that cannot be matched to a returned audit are marked `insufficient_retrieval`. The reviewer receives bounded relevant passages with page/chunk provenance and must quote supplied text exactly for supported or contradicted verdicts. Tavily page text and search snippets are treated as partial retrieval; absence of support there is labeled `insufficient_retrieval`, not `insufficient_evidence`. Critical contradicted or unresolved claims, missing cited excerpts, and uncited numeric claims prevent the critic from passing the report. Claim verdicts, criticality, retrieval completeness, and validated source references are shown with the critic evaluation.

For a failed review, the next draft receives the current cycle's feedback plus validated, source-linked audit passages as separate evidence context. Old feedback is replaced rather than accumulated, and passing reviews clear it. Reviewer notes and audit labels are revision guidance, not report content.

If all three writer attempts reach the output token limit during a revision, the workflow pauses for human review of the previous complete draft and its matching evaluation. The approval message states that the latest revision feedback was not applied. If no previously reviewed draft exists, the workflow fails with an explicit token-limit message rather than returning a truncated report.

The auditor receives nearby report text to resolve references, but treats only retrieved passages as evidence. Proposed supported or contradicted verdicts get a second model check of the claim, context and referenced passages; this is still model-based, not proof of truth. This adds one model request for each claim/source batch containing proposed supported or contradicted verdicts. When a claim includes a URL, a supporting quote must come from a cited source. Citation accuracy is the number of supported material claims divided by all audited material claims, including unverified and contradicted claims. A report with no audited claims cannot pass, and its citation accuracy is shown as unavailable. Completeness and objectivity remain model judgments; the overall score averages the three dimensions. The dashboard shows supported, unverified and contradicted claim counts separately.

The starter knowledge base is scoped to **generative AI in banking** and uses six curated reports from the U.S. Treasury, Federal Reserve, NIST, Financial Stability Board, and European Banking Authority. The source list and official PDF links are maintained in `data/knowledge_sources.json`; this is within the requested 5–10-document range. NIST's document is a cross-sector risk framework; use it for risk-management context, not as banking-specific evidence. Tavily search complements the local corpus with public pages and reports from McKinsey, Deloitte, PwC, IBM, and the World Economic Forum; these publishers receive a moderate ranking signal, not a guarantee of reliability. Their full-text PDFs are not mirrored into Chroma unless their reuse is authorized.

## Project structure

- `app/core` — settings, logging, and exceptions.
- `app/models` — typed domain models.
- `app/graph` — graph state, prompts, routing, and node implementations.
- `app/tools` — search and retrieval interfaces.
- `app/services` — higher-level orchestration for the workflow.
- `app/api` — optional FastAPI layer.
- `data/knowledge_sources.json` — curated PDF source manifest for the banking GenAI corpus.
- `tests` — unit and integration tests.
- `outputs` — generated report output.

## Key design decisions

- The graph state is centralized and typed so that each node receives a known shape.
- Findings accumulate via list reducers so parallel workers do not overwrite each other.
- Human approval is intentionally implemented as a real checkpoint via LangGraph interrupts and a SQLite-backed checkpointer.
- LLMs are selected per role: the planner, writer, and reviewer use Anthropic by default, while the researcher uses Gemini. The fact checker shares the reviewer model. If Gemini fails for the researcher, the workflow retries once and then uses the configured reviewer model, marking the fallback in the evidence notes. Each role's provider and model can be overridden independently through `.env`.

## Running the app

**Source-only archive setup:** The archive includes `data/knowledge_sources.json`,
but not the generated Chroma database or previous run checkpoints. After installing
dependencies, run `uv run python -m app.tools.ingest_knowledge` to download and index
the source documents before starting research. Configure your own API credentials in
`.env`; do not share that file. Live research makes multiple provider API calls and
may incur charges; there is no per-run spending cap.

```bash
uv sync --locked
uv run python -m app.tools.ingest_knowledge
uv run uvicorn app.api.server:app --reload
```

On Windows PowerShell, create and configure the environment file before starting:

```powershell
Copy-Item .env.example .env
notepad .env
uv sync --locked
uv run python -m app.tools.ingest_knowledge
uv run uvicorn app.api.server:app --reload
```

Set provider credentials in `.env` before running ingestion/research. Open `http://127.0.0.1:8000`, submit a focused question, wait for the planner and parallel workers, inspect evidence and the model-assisted evidence cross-check, then review the report and critic evaluation. Approve to save it under `outputs/<thread_id>.md`; reject to end without saving; or request changes. After the automatic revision limit is reached, one final feedback-driven revision remains available; once used, only Approve and Reject are available. `/api/health` checks service availability, and `/docs` exposes the API schema.

The ingestion command downloads the six official PDFs in the manifest, extracts page-aware text chunks, and upserts them into the configured Chroma collection. It does not retain local PDF copies. Re-running it updates the same stable chunk IDs instead of duplicating documents. Chroma's built-in default embedding function is used for both upserts and searches; changing the embedding implementation is not currently a supported `.env` setting.

Open `http://127.0.0.1:8000` for the Research Review dashboard. The API contract is available at `http://127.0.0.1:8000/docs`.

Research runs are checkpointed in SQLite at the configured `CHECKPOINT_DB_PATH` (default: `data/checkpoints.sqlite`) and use a `thread_id` to resume after human decisions. The dashboard starts a run, automatically researches the generated plan, streams worker updates, and pauses for final report approval or revision. Approved reports are saved to `OUTPUT_DIR` (default: `outputs`) as `<thread_id>.md`.

The supported end-to-end workflow is the FastAPI dashboard/API started with Uvicorn. `app.main` is a minimal direct graph-invocation helper, not a supported interactive CLI: it does not manage approval decisions or resume checkpointed runs, so it may return at the final-review interrupt without saving a report.

## Running tests

```bash
uv run pytest
```

## Environment variables

Copy `.env.example` to `.env` and fill in values as needed. `uv.lock` records the exact resolved dependency versions; use `uv sync --locked` to install from that lock file. Update dependencies deliberately with `uv lock --upgrade`, then run the tests before committing the updated lock file.

## Notes

The default planner, writer, and reviewer roles require `ANTHROPIC_API_KEY`. `GEMINI_API_KEY` enables the primary researcher model; if Gemini is unavailable, the researcher retries and then uses the configured reviewer model, with the fallback recorded in evidence notes. `TAVILY_API_KEY` enables bounded web search for every task; if Tavily is unavailable, a worker warning is shown and the workflow continues with local evidence where available. No provider is silently replaced with a test mock. The default researcher model is `gemini-3.5-flash-lite`; set `RESEARCHER_LLM_MODEL` to another model available to your key if needed. Using a different model does not guarantee better factual accuracy; evaluate providers against a fixed set of research questions and compare source-supported claims. Tests use deterministic test-only fakes and do not call provider APIs. The report remains a draft until a reviewer approves it in the dashboard.
