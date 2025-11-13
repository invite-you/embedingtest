# AGENT.md

**Schema Version**: `1.0.0`

## Purpose

This agent analyzes local files, generates summaries/metadata, and proposes folder re-organization plans.  
It runs fully on the user’s machine with local Qwen models.

---

## Runtime & Constraints

- Models
  - LLM: `qwen3-8b` (local)
  - Embedding: `qwen3-embedding-4b` (local)
- Hardware
  - **GPU is mandatory** (minimum 8GB VRAM recommended for qwen3-8b)
  - Fallback: If GPU unavailable, agent must fail fast with clear error message
- Timeouts
  - Per-file end-to-end budget (extract + embed + LLM + indexing): **≤ 50 minutes**
- Text size limit
  - After text extraction, if plain text > **5 MB**, mark file as `TEXT_TOO_LARGE` and skip summarization.

All thresholds (timeouts, limits, chunk sizes, cluster counts, etc.) must be configurable (e.g. `config/agent.yaml`) and must not be hard-coded.

### Configuration

Large production agents usually keep their knobs in declarative configs so they can be reloaded without redeploying binaries. Mirroring that approach, the agent **must** load defaults from `config/agent.yaml` (or an override supplied via CLI/env) and support hot-reload where possible. At a minimum document and validate the following keys:

| Key | Description | Example |
| --- | --- | --- |
| `timeouts.per_file_minutes` | Upper bound for the entire extract→embed→LLM loop. | `50` |
| `limits.max_text_bytes` | Raw text byte ceiling before returning `TEXT_TOO_LARGE`. | `5242880` |
| `clustering.max_clusters` | Hard stop for auto-cluster counts. | `12` |
| `clustering.min_sentences` | Below this, skip clustering and sample directly. | `8` |
| `context.sentences_before` / `context.sentences_after` | Number of neighboring sentences added to each representative block. | `1` |
| `llm.max_tokens` | Token budget handed to `qwen3-8b`. | `2048` |
| `embedding.batch_size` | Sentences batched per embedding call (tune for GPU VRAM). | `64` |

The agent should fail fast with a human-readable error if required config keys are missing, malformed, or outside safe ranges.

---

## Supported Inputs

The agent only attempts analysis for these file types:

- `txt`, `md`, `docx`, `doc`, `pptx`, `ppt`

Other types, or extraction failures (e.g. images in slides), yield **no content** but still produce metadata.

Input to per-file workflows (conceptual schema):

```jsonc
{
  "file_id": "string",          // internal id
  "path": "string",             // local path
  "mime_type": "string",
  "size_bytes": 12345,
  "modified_at": "ISO8601",
  "extracted_text": "string|null", // if null, agent must attempt extraction
  "extra_metadata": {}          // optional
}
```

---

## Per-File Output Contract

Every processed file returns a structured result. This schema should stay stable:

```jsonc
{
  "file_id": "string",
  "status": "OK | NO_CONTENT | TEXT_TOO_LARGE | UNSUPPORTED | ERROR | TIMEOUT",
  "retry_count": 0,  // 재시도 횟수
  "processing_time_ms": 0  // 실제 처리 시간
  "error_message": "string|null",

  "short_summary": "string|null",   // 2–5 sentences
  "long_summary": "string|null",    // optional extended summary
  "tags": ["string"],               // 0–N keywords
  "doc_type": "string|null",        // e.g. 보고서, 계약서, 강의자료, 코드, 메모, 기타
  "topics": ["string"],             // e.g. 재무, LLM, 취업
  "language": "string|null",        // e.g. "ko", "en"

  "embedding": [0.0],               // vector from qwen3-embedding-4b (usually summary-based)
  "raw_features": {                 // optional internal diagnostics
    "sentence_count": 0,
    "chunk_count": 0,
    "text_bytes": 0
  }
}
```

Rules:

* 텍스트 추출 실패, 이미지 전용 파일, 완전히 비어 있는 문서:

* `status = "NO_CONTENT"` or `"UNSUPPORTED"`
  * `short_summary = null`, `tags = []`

### Error handling & retry policy

- Implement a bounded exponential backoff (e.g. max 2 retries) for transient extractor/embedding errors; record the final `retry_count` in the schema.
- Classify failures deterministically so dashboards can alert on spikes (e.g. distinguish extractor crashes from GPU OOMs).
- Emit actionable `error_message` strings that include the failing subsystem plus any upstream error codes/log IDs.
- When timeouts fire, include elapsed time vs. configured budget in telemetry for later tuning.
---

## File Processing Pipeline (High Level)

For files that pass basic checks:

1. **Text extraction**

   * Extract plain text from supported formats.
   * If no text → `NO_CONTENT`.

2. **Guardrails**

   * If `text_bytes > MAX_TEXT_BYTES (default 5 MB)` → `TEXT_TOO_LARGE`.

3. **Sentence segmentation**

   * Split into sentences.
   * If `sentence_count < MIN_SENTENCES_FOR_CLUSTERING`, skip clustering and sample from the full text.

4. **Embedding & clustering**

   * Embed each sentence with `qwen3-embedding-4b`.
   * Cluster using K-means (or HDBSCAN for auto-detection)
   * Number of clusters: `min(MAX_CLUSTERS, ceil(sqrt(sentence_count)))`
   * For each cluster, pick a representative sentence based on:

     * closeness to cluster centroid,
     * minimum length,
     * mild preference for earlier positions.

5. **Context expansion**

   * For each representative sentence, include `±N` neighboring sentences

     * Defaults: `CONTEXT_SENTENCES_BEFORE = 1`, `CONTEXT_SENTENCES_AFTER = 1`.
   * Result: a small set of representative text blocks covering diverse parts of the document.

6. **LLM summarization & tagging (qwen3-8b)**

   * Provide representative blocks + light metadata (filename, etc.).

   * Ask for a **JSON** payload:

     ```jsonc
     {
       "short_summary": "string",
       "long_summary": "string|null",
       "tags": ["string"],
       "doc_type": "string",
       "topics": ["string"],
       "language": "string"
     }
     ```

   * Constraints for the model:

     * Do **not** invent facts, dates, or numbers not implied by the text.
     * If unsure, omit or mark as “정보 부족”.

7. **Final embedding**

   * Compute a final embedding from `short_summary` (or main text) via `qwen3-embedding-4b` and store in `embedding`.

All numeric knobs (cluster counts, context sentences, max tokens, etc.) must be read from configuration and easy to tune.

### Logging, telemetry, and auditability

- Emit structured logs (JSON) with per-file identifiers, durations for each pipeline stage, and GPU utilization snapshots so on-call engineers can debug regressions quickly.
- Export Prometheus (or OpenTelemetry) metrics for throughput, average latency per stage, retry counts, and error categories. Expose health/readiness probes for orchestration platforms (e.g. Kubernetes).
- Persist provenance metadata (config hash, model versions, git SHA) alongside outputs to guarantee reproducibility of summaries and embeddings.

---

## Folder / File Organization Strategies

Re-organization works over an **index** of already-processed files:

```jsonc
{
  "files": [
    {
      "file_id": "string",
      "path": "string",
      "short_summary": "string|null",
      "tags": ["string"],
      "doc_type": "string|null",
      "topics": ["string"],
      "embedding": [0.0]|null,
      "modified_at": "ISO8601"
    }
  ]
}
```

The agent should support multiple strategies, for the UI to present 2–3 options.

### Quality assurance & release readiness

- Provide unit tests for text extraction adapters, schema validators, and clustering heuristics.
- Add golden-file regression tests that feed representative documents (tiny memo, large PPT, empty doc) through the pipeline to ensure summaries remain stable across refactors.
- Include linting/formatting checks plus a smoke-test CLI command (e.g. `agent-cli validate data/sample`) in CI to match the rigor seen in large GitHub-hosted agents.
