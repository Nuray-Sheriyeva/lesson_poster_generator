# genlesson

Turn one lesson file into reviewable, resumable poster-generation runs.

This repository currently implements **Stage 1: ingestion**, **Stage 2:
content classification and language detection**, **Stage 3: poster planning**,
**Stage 4: review gate**, **Stage 5: prompt building**, **Stage 6: image
generation**, **Stage 7: verification and retry**, and **Stage 8: saving**.
The pipeline is complete end to end. Stage 8 also composites a lesson mascot
with a text-fitted question bubble onto every saved poster.







## Status

| Stage | Directory | State |
| --- | --- | --- |
| 1. ingest | `stage-01-ingest/` | Implemented |
| 2. classify & detect language | `stage-02-classify/` | Implemented |
| 3. plan | `stage-03-plan/` | Implemented |
| 4. review gate | `stage-04-review/` |  |
| 5. build prompts | `stage-05-prompts/` | Implemented |
| 6. generate | `stage-06-generate/` | Implemented |
| 7. verify / retry | `stage-07-verify/` | Implemented |
| 8. save | `stage-08-save/` | Implemented |







The locked decisions are reflected in the run manifest where they apply today:
the GPT-6 Luna planner, OpenAI image generation, portrait as the default format,
and the review gate defaulting to off. Whole-poster rendering, lesson-language
output, per-format posters, and poster non-overwrite behavior are implemented.
All eight stages are executable today.

## Milestones

| # | Milestone | Status |
| --- | --- | --- |
| 1 | Ingestion, classification, planner producing a valid `poster_spec`, resumable | Complete |
| 2 | Prompt builder, OpenAI portrait generation, safe saving | Complete |
| 3 | Verification and retry | Complete |
| 4 | Landscape and square formats | Complete |
| 5 | Review gate, switchable | Complete |
| 6 | Anthropic and DeepSeek planner adapters | Complete; live calls need their keys |
| 7 | Test set | Complete: golden case, Russian sample, scanned sample, and unit tests |
| 8 | Mascot and lesson question bubble | Complete: assets, text fit, text-aware placement, preview CLI |

## Quick start

Requires Python 3.11+.

```bash
uv venv
uv pip install -e ".[dev]"
source .venv/bin/activate

# PDF or DOCX
genlesson ingest lessons/photosynthesis.pdf

# Scanned PDFs need a vision model
export OPENAI_API_KEY=...
genlesson ingest lessons/scanned-lesson.pdf

# Machine-readable output
genlesson ingest lessons/lesson.docx --json

# Stage 2: classify content and decide poster language (needs the planner key)
genlesson classify 20261104-154458-photosynthesis

# Stage 3: build the strict poster spec
genlesson plan 20261104-154458-photosynthesis

# Stage 4: optional review gate (off by default)
genlesson review 20261104-154458-photosynthesis

# Stage 5: build one image prompt per format
genlesson prompts 20261104-154458-photosynthesis --formats portrait,landscape

# Stage 6: generate one image per format
genlesson generate 20261104-154458-photosynthesis

# Stage 7: verify text and retry mismatches
genlesson verify 20261104-154458-photosynthesis

# Stage 8: save final posters and sidecars
genlesson save 20261104-154458-photosynthesis

# Preview the mascot and question bubble on a blank canvas (no image credits)
genlesson mascot-preview \
  --size 1024x1536 \
  --question "Бұл суретте не көріп тұрсың?" \
  --out outputs/mascot-preview.png
```

Each invocation prints the generated run id. Use it for status and resume:

```bash
genlesson status 20261104-154458-photosynthesis
genlesson ingest lessons/photosynthesis.pdf \
  --run-id 20261104-154458-photosynthesis \
  --resume
genlesson classify 20261104-154458-photosynthesis
genlesson plan 20261104-154458-photosynthesis
genlesson review 20261104-154458-photosynthesis
genlesson prompts 20261104-154458-photosynthesis
genlesson generate 20261104-154458-photosynthesis
genlesson verify 20261104-154458-photosynthesis
genlesson save 20261104-154458-photosynthesis
```

## Full pipeline, tracing, and experiments

### Setup

Install the optional tracing and graph extras:

```bash
uv venv
uv pip install -e ".[dev,tracing,graph]"
source .venv/bin/activate
```

Put credentials in `.env` (the CLI loads it automatically) or export them:

```dotenv
OPENAI_API_KEY=...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

Tracing is on automatically when both Langfuse keys exist. Use
`--no-tracing` for one command, or `[tracing] enabled = "off"` in
`config.toml`. Add a custom Langfuse model definition for `gpt-6-luna` if
your project's model table does not already include it.

### Graph

`genlesson run` executes the whole pipeline through LangGraph:

```text
START → ingest → classify → plan → review ─┬─ awaiting_review → END
                                           └─ prompts → generate → verify → save → END
```

```bash
genlesson run fixtures/input/relationtype.pdf --run-id demo --format portrait
genlesson run --run-id demo --resume
genlesson run fixtures/input/relationtype.pdf --run-id demo-json --format portrait --json
```

The review gate is unchanged: `run` stops at `awaiting_review`; approve with
`genlesson review <run_id> --resume` (optionally `--spec`), then resume the
graph. `--json` suppresses stage output and prints only the final status.

### Reading a trace

Every run has a deterministic trace id. In Langfuse you will see:

- a root `run:<run_id>` span for `genlesson run`;
- one stage span per stage (`ingest`, `classify`, `plan`, `review`,
  `prompts`, `generate`, `verify`, `save`);
- generation observations named `classify.segments`, `classify.language`,
  `plan.candidates`, `plan.spec`, `ingest.transcribe_page`,
  `verify.transcribe`, and `generate.image`;
- format spans `verify:<format>` with scores `text_match`,
  `exact_mismatch_count`, `extra_strings_count`, `retries_used`, and
  categorical `verify_status`; classify and plan add `language_confidence`,
  `details`, and `conflicts`, and review adds `review_edited` and
  `human_wait_s`.

Filter by score `verify_status` to find passed, needs-attention, or
generation-failed formats. Sidecars also carry `tracing.trace_id` and a
Langfuse URL under `tracing`.

### Prompt management

Planner system prompts and the poster layout templates can live in Langfuse
prompt management. The in-code text remains the offline fallback. Prompt names:

```text
genlesson/planner/classify-system
genlesson/planner/language-system
genlesson/planner/plan-system
genlesson/planner/candidate-system
genlesson/poster/layout-portrait
genlesson/poster/layout-landscape
genlesson/poster/layout-square
```

`python experiments/run.py --sync-prompts` uploads the in-code versions.
Select a label with `GENLESSON_PROMPT_LABEL` (default `production`) or a
version with `GENLESSON_PROMPT_VERSION`.

### Experiments

`experiments/run.py` builds the `genlesson-fixtures` dataset from every
PDF/DOCX in `fixtures/input/`. Only `relationtype.pdf` has golden expectations: language
`kk`, 7 details, slug `biotic-interactions`, and at least one source conflict.

```bash
# Stages 1-3 only; no image generation cost
python experiments/run.py --mode plan

# Relationtype only: review gate + frozen approved spec + stages 5-7
python experiments/run.py --mode image --item relationtype --sync-prompts

# Occasional end-to-end check
python experiments/run.py --mode full --item relationtype
```

Runs use isolated directories under `experiments/runs/<run-name>/`. The
measured three-run baseline protocol is in [`docs/baseline.md`](docs/baseline.md).

## Configuration

Defaults live in [`config.toml`](config.toml), not in pipeline code:

```toml
[planner]
provider = "openai"
model = "gpt-6-luna"
max_context = 1050000

[image]
provider = "openai"
model = "gpt-image-2"

[image.sizes.gpt-image-2]
portrait = "1024x1536"
landscape = "1536x1024"
square = "1024x1024"

[vision]
provider = "openai"
model = "gpt-6-luna"

[verification]
provider = "openai"
model = ""            # empty: reuse planner if it has vision, else [vision]

[defaults]
formats = ["portrait"]
review = "off"
review_on_conflict = false
max_details = 8
max_retries = 2
out_dir = "outputs"

[mascot]
enabled = true
assets_dir = "assets/mascot"
size_ratio = 0.09
fill = "#FFFFFF"
outline = "#6360FF"
text_color = "#3B0764"
outline_width = 4
font_path = "assets/fonts/NotoSans-Regular.ttf"
min_font_ratio = 0.016
max_font_ratio = 0.03
bubble_max_width_ratio = 0.42
edge_margin = 0.025
question_max_chars = 140
avoid_repeat = false
busyness_threshold = 0.08
```

Model IDs are confirmed against OpenAI's docs: [GPT-6
Luna](https://developers.openai.com/api/docs/models/gpt-6-luna) and [GPT Image
2](https://developers.openai.com/api/docs/models/gpt-image-2). Use
`--config PATH` before the subcommand, or set `POSTER_CONFIG`, to choose another
file.

Precedence is: CLI flag, then `POSTER_*` environment variable, then the run
manifest where applicable, then `config.toml`.

### Planner provider layer

The planner contract exposes three capabilities:

```python
PlannerCapabilities(
    structured_output: bool,
    vision: bool,
    max_context: int,
)
```

| Adapter | Structured output | Vision | Context source |
| --- | --- | --- | --- |
| OpenAI | Strict JSON Schema | Yes | `[planner] max_context` |
| Anthropic | Tool-use JSON Schema | Yes | `--planner-context` / config |
| DeepSeek | JSON object + Pydantic validation | No | `--planner-context` / config |

Select an adapter with `--planner-provider openai|anthropic|deepseek` and a model
with `--planner-model`. Non-OpenAI adapters require their own key:
`ANTHROPIC_API_KEY` or `DEEPSEEK_API_KEY`. The model is never guessed: pass the
model ID, or set `POSTER_ANTHROPIC_MODEL` / `POSTER_DEEPSEEK_MODEL`.

Capabilities are applied in the pipeline:

- **No vision:** Stage 2 and Stage 3 send text only; page images are omitted and
  a warning is recorded. Vision-dependent verification resolves to the
  configured vision provider instead of the planner.
- **No strict structured output:** the adapter uses JSON-object mode and
  validates with Pydantic before returning the draft.
- **max_context:** Stage 3 lowers its context/chunk budget when the planner's
  configured context window is smaller.

The image provider is independent. Selecting an Anthropic or DeepSeek planner
does not change Stage 5/6: images still go through the OpenAI image provider
configured in `[image]`.

```bash
# Anthropic planner; OpenAI still renders images.
genlesson classify <run_id> \
  --planner-provider anthropic \
  --planner-model <anthropic-model> \
  --planner-context 200000

genlesson plan <run_id> \
  --planner-provider anthropic \
  --planner-model <anthropic-model> \
  --planner-context 200000
```

`.env` is loaded by the CLI for provider credentials and `POSTER_*` settings
only. Legacy `GENLESSON_*` lines in `.env` are ignored so they cannot silently
override `config.toml`.

### Core flags

```text
--config PATH            Config TOML (before the subcommand).
--format FORMAT          Repeat: portrait, landscape, square. Default: portrait.
--review off|on          Review gate. Default: off; POSTER_REVIEW overrides.
--review-on-conflict     Pause only on planner conflicts. Default: off.
--resume <run_id>        Resume/approve an existing run (run_id is positional).
--planner-provider NAME  openai, anthropic, or deepseek (default: OpenAI).
--planner-model NAME     Default: GPT-6 Luna for OpenAI.
--planner-context N      Planner context window for capability checks.
--image-provider NAME    Default: OpenAI.
--image-model NAME       Default: GPT Image 2.
--max-details N          Default: 8.
--max-retries N          Verification retries. Default: 2.
--out-dir PATH           Root for per-input poster folders. Default: outputs/.
```

### Useful flags

Stage 1 (`ingest`):

```text
--run-id ID             Explicit run id. Required for --resume.
--runs-dir PATH         Run root (default: runs).
--resume                Continue an existing run instead of failing.
--dpi N                 PDF page render DPI, 72-600 (default: 180).
--vision openai|none    Transcription provider (default: openai).
--vision-model NAME     Default: gpt-6-luna, or $GENLESSON_VISION_MODEL.
--vision-timeout SEC    Per-request timeout (default: 120).
--vision-retries N      Retries after the first attempt (default: 3).
--json                  JSON on stdout. Errors are also JSON and exit 2.
```

Stage 2 (`classify`) and Stage 3 (`plan`):

```text
--runs-dir PATH         Run root (default: runs).
--resume                Reuse a completed stage (idempotent either way).
--planner-provider NAME openai, anthropic, or deepseek.
--planner-model NAME    Default: gpt-6-luna for OpenAI.
--planner-context N     Context window used for chunk budgeting.
--planner-timeout SEC   Per-request timeout (default: 120).
--planner-retries N     Retries after the first attempt (default: 3).
--max-page-images N     Page images sent as context, 0-20 (default: 6).
--json                  JSON on stdout. Errors are also JSON and exit 2.
```

Stage 3 only:

```text
--max-details N         Maximum poster details, 1-20 (default: 8).
--context-chars N       Lesson chars before chunking begins (default: 24000).
--chunk-chars N         Candidate-extraction chunk size (default: 12000).
```

Stage 4 (`review`):

```text
--resume                Approve an awaiting review and continue.
--spec PATH             Optional edited poster spec to validate and approve.
--review off|on         Gate mode (default: off).
--review-on-conflict    Pause only when the planner flagged source conflicts.
```

Stage 5 (`prompts`):

```text
--format FORMAT         Repeat for multiple formats (default: portrait).
--formats LIST          Compatibility alias: comma-separated formats.
--image-provider NAME   Default: openai.
--image-model NAME      Default: gpt-image-2, or $POSTER_IMAGE_MODEL.
```

Stage 6 (`generate`):

```text
--image-provider NAME   Must match the provider used to build prompts.
--image-timeout SEC     Per-request timeout (default: 180).
--image-retries N       Retries after the first attempt (default: 3).
--resume                Reuse successful formats and retry failed ones.
```

Stage 7 (`verify`):

```text
--verifier-model NAME   Vision model; defaults to the planner when it has vision.
--similarity FLOAT      Sentence similarity threshold, 0.5-1.0 (default: 0.9).
--max-retries N         Correction retries after the first image (default: 2).
--vision-timeout SEC    Vision request timeout (default: 120).
--vision-retries N      Vision API retries (default: 3).
--image-timeout SEC     Timeout for retry generations (default: 180).
--image-retries N       Retries for retry generations (default: 3).
```

Stage 8 (`save`):

```text
--out-dir PATH          Root for per-input poster folders (default: outputs/).
--resume                Reuse already saved posters (idempotent either way).
```

Mascot preview (`mascot-preview`, no image credits):

```text
--size WxH              Blank-canvas size (default: 1024x1536).
--image PATH            Overlay on an existing poster (no image credits).
--corner CORNER         auto (default), top-left, top-right, bottom-left, bottom-right.
--variant ID            Mascot PNG variant id; default is deterministic from the question.
--question TEXT         Question shown in the bubble (required).
--out PATH              Output PNG (default: outputs/mascot-preview.png).
--background COLOR      Blank canvas color (default: #FFFFFF).
--detector MODE         auto (default), rapidocr, or none.
```

### Environment

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Required for scanned-page OCR, image generation, verification, and the OpenAI planner. Never written to run artifacts. |
| `OPENAI_BASE_URL` | Optional OpenAI-compatible endpoint for the OpenAI planner and image calls. |
| `POSTER_REVIEW` | Review mode: `off`, `on`, or `on-conflict` (default: off). |
| `POSTER_REVIEW_ON_CONFLICT` | Pause only on conflicts (`0`/`1`). |
| `POSTER_PLANNER_PROVIDER` / `POSTER_PLANNER_MODEL` | Planner override. |
| `POSTER_PLANNER_MAX_CONTEXT` | Planner context window for capability checks. |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_BASE_URL` | Anthropic planner credentials. |
| `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` | DeepSeek planner credentials. |
| `POSTER_ANTHROPIC_MODEL` / `POSTER_DEEPSEEK_MODEL` | Provider model IDs when selected. |
| `POSTER_IMAGE_PROVIDER` / `POSTER_IMAGE_MODEL` | Image override. |
| `POSTER_VERIFY_PROVIDER` / `POSTER_VERIFY_MODEL` | Verification vision override. |
| `POSTER_CONFIG` | Alternate config TOML path. |
| `POSTER_FORMATS` | Comma-separated default formats. |
| `POSTER_MAX_DETAILS` | Detail cap (`8`). |
| `POSTER_MAX_RETRIES` | Verification retries (`2`). |
| `POSTER_OUT_DIR` | Final poster directory (`outputs`). |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | Enable Langfuse tracing. |
| `LANGFUSE_HOST` | Langfuse host (default: `https://cloud.langfuse.com`). |
| `GENLESSON_PROMPT_LABEL` | Langfuse prompt label (default: `production`). |
| `GENLESSON_PROMPT_VERSION` | Optional numeric Langfuse prompt version. |
| `GENLESSON_VISION_MODEL` | Default OCR vision model (`gpt-6-luna`). |
| `GENLESSON_PLANNER_MODEL` | Deprecated planner model override. |
| `GENLESSON_RENDER_DPI` | Default page render DPI. |
| `GENLESSON_MAX_PAGE_IMAGES` | Default cap on page images sent to the planner. |
| `GENLESSON_MAX_DETAILS` | Default Stage 3 detail cap (`8`). |
| `GENLESSON_CONTEXT_CHARS` | Lesson chars before Stage 3 chunks (`24000`). |
| `GENLESSON_CHUNK_CHARS` | Stage 3 candidate-extraction chunk size (`12000`). |
| `GENLESSON_IMAGE_MODEL` | Default image model for Stage 5 (`gpt-image-2`). |
| `GENLESSON_FORMATS` | Default comma-separated formats (`portrait`). |
| `GENLESSON_IMAGE_TIMEOUT` | Default image request timeout (`180`). |
| `GENLESSON_IMAGE_RETRIES` | Default image retry count (`3`). |
| `GENLESSON_VERIFY_MODEL` | Optional vision model override for Stage 7. |
| `GENLESSON_VERIFY_SIMILARITY` | Sentence similarity threshold (`0.9`). |
| `GENLESSON_VERIFY_RETRIES` | Correction retries (`2`). |
| `GENLESSON_VERIFY_TIMEOUT` | Vision request timeout (`120`). |
| `GENLESSON_VERIFY_VISION_RETRIES` | Vision API retry count (`3`). |

See `.env.example`. The CLI loads `.env` from the current directory, while
only provider, Langfuse, and `POSTER_*` variables are accepted.

## What ingestion does

### PDF

1. Validates the PDF header, rejects password-protected files, and reports
   damaged files as `corrupt_pdf`.
2. Extracts the text layer with PyMuPDF (`get_text("text", sort=True)`).
3. Detects scanned/image-only documents. A document is treated as scanned when
   it has little total text, no page with a meaningful text layer, and at least
   one page with visual content (including inline images and vector art).
4. Renders **every** page as an RGB PNG at the configured DPI, not just pages
   that need OCR.
5. Transcribes:
   - every visual page of a scanned document, or
   - individual pages in a mixed document that have visual content and fewer
     than `min_page_chars` (20) non-whitespace characters.
6. Writes the transcript to the page artifact and treats it as the lesson text.
   The OCR prompt requires verbatim transcription, preserves the original
   language/script, and forbids translation or commentary.
7. Caches each transcript by page-image hash and model name. A resumed run does
   not call the model again for an unchanged page.

Transcription calls use exponential backoff for rate limits, timeouts,
connection failures, and 5xx responses. Authentication failures and bad
requests fail immediately. One failed page fails the stage; earlier pages stay
cached, so `--resume` continues from the failure.

### DOCX

`python-docx` reads body paragraphs and tables in document order. Table rows
are flattened as `cell | cell | cell`. DOCX has no fixed page images, so it is
stored as one text segment and produces no page PNGs. An image-only DOCX has no
text and fails with `no_text`; export it as PDF if it needs OCR.

## What Stage 2 does

```bash
genlesson classify <run_id> --runs-dir runs
```

Stage 2 reads `stage-01-ingest/lesson.json` and writes exactly one artifact:
`stage-02-classify/classification.json`.

### Segmentation

Ingested page text is split into stable, numbered segments. Paragraph breaks
are the primary boundary. A block longer than 1,800 characters is split at line
and sentence boundaries so one huge PDF paragraph cannot swallow the whole
lesson. Every segment keeps its page number, page label, order, and text source
(`text_layer` or `vision`).

### Content classification (planner call 1)

The planner receives the numbered segments plus up to `--max-page-images` page images
for context. It labels every segment `lesson_content` or `non_content` and
assigns a category:

- `lesson_content`
- `learning_objective`
- `assessment_criteria`
- `exam_instruction`
- `exercise_prompt`
- `other`

Learning objectives, assessment criteria, exam instructions, and exercise
prompts such as `Анықта` or `Мысал келтір` are non-content. Text that appears
only inside a picture, diagram, or decorative image is also non-content. A
scanned page whose main body was transcribed is the exception: that transcribed
body is lesson content. Page images are context only; they are not allowed to
decide the language.

If the planner returns a missing, duplicate, or unknown segment id, the stage fails
with `planner_response_invalid` rather than silently dropping text.

### Language decision (planner call 2)

The classifier's output is filtered to `lesson_content` segments. A second planner
call receives only that retained body text, so non-content instructions cannot
influence the language:

- In a document with a usable text layer, language evidence comes from
  text-layer content only. Vision-transcribed page text is excluded and a
  warning is recorded.
- In a scanned document, the OCR transcript is the only body text, so it is used
  and `language_source` is `vision_transcript`.
- If no text-layer content is available in a non-scanned document, vision
  transcripts are used as a fallback with a warning.

The result is a normalized BCP-47 tag (`kk`, `ru`, `en`, `uz`, `zh-Hans`,
`es-419`, ...), a confidence from 0 to 1, and a short evidence note quoting the
body text. Language names such as `Kazakh` are normalized to tags. An invalid
tag fails with `language_undetermined`. Confidence below 0.6 produces a warning
but does not fail the stage.

### Topics

The planner groups content segments into one or more topics. When several topics are
returned, the topic covering the most content characters becomes
`dominant_topic`, a warning is written into the artifact, and a log warning is
emitted. The poster stages use this dominant topic.

### No content means failure

If every segment is non-content, Stage 2 fails with `no_lesson_content`. It does
not guess a language from objectives, instructions, or prompts.

### Resume behavior

`classification.json` records both the source SHA-256 and the ingest artifact
SHA-256. A completed Stage 2 is reused without calling the planner, even if no API key
is present. If either hash no longer matches, the artifact is rejected instead
of being silently mixed with a different lesson.

## What Stage 3 does

```bash
genlesson plan <run_id> --runs-dir runs
```

Stage 3 reads `stage-02-classify/classification.json` and writes
`stage-03-plan/poster_spec.json`.

### One call or chunked planning

- If the retained lesson text fits the `--context-chars` budget (default
  24,000 characters), The planner makes **one** planning call with the full body text
  and selected source-page images. The response uses a strict JSON schema.
- If it does not fit, the content is split into `--chunk-chars` chunks
  (default 12,000). The planner extracts candidate ideas from each chunk, the
  candidates are deduplicated, and a final planner call merges them into the
  poster spec. Every call uses a strict schema.

Structured output is generated from the Pydantic contract: every object has
`additionalProperties: false`, every property is required, and unsupported JSON
Schema keywords are stripped. Lengths, counts, and semantic rules are validated
after parsing.

### Poster spec

`poster_spec.json` contains:

- `language` — the canonical Stage 2 BCP-47 language, enforced even if the planner
  returns something else.
- `age_band` — `primary` (roughly ages 6-10), `middle` (11-14), or `high`
  (15-18), derived by the planner from the material's vocabulary and concepts.
- `central_idea` — a short title and a one-line definition, for example
  `Байланыс түрлері`.
- `details[]` — the exact label, a short exact explanation, an optional exact
  example, a visual description for the image model, and a source page.
- `composition` — one visual concept such as a hero scene with callouts, a path,
  a radial layout, a cutaway, or a cycle. Plain grids, tables, and mind maps are
  rejected.
- `style` — palette, mood, and illustration style, age-appropriate and
  non-graphic.
- `strings_to_render` — every exact text string that should appear on the
  poster. Canonical strings from the title, definition, labels, explanations,
  and examples are merged with any additional planner strings.
- `source_conflicts[]` — source inconsistencies and how they were resolved.
- `file_slug` — a short lowercase ASCII filename stem.

There is **no fixed 2–3 detail rule**. The number of details follows the
material, up to `--max-details` (default 8, maximum 20). If the planner returns more,
the spec keeps the first `max_details` and records a warning. Duplicate labels
are removed.

Terms stay in their source language and spelling. Titles, definitions, labels,
and explanations are kept short. The planner's output is validated against
length limits, a minimum palette size, a non-empty detail set, and the
composition restriction before it is written.

Page images from Stage 1 are passed to the final planning call as visual
context, capped by `--max-page-images`.

### Resume behavior

`poster_spec.json` records the source SHA-256 and the Stage 2 classification
SHA-256. A completed plan is reused without a planner call. If either hash no
longer matches, the artifact is rejected rather than mixed with a different
lesson or classification.

## What Stage 4 does

```bash
# default: gate off, pass through
genlesson review <run_id>

# pause for human review
genlesson review <run_id> --review on

# pause only when the planner flagged source conflicts
genlesson review <run_id> --review-on-conflict
```

### Gate off (default)

Stage 4 copies the planner spec to `stage-04-review/approved_spec.json`, writes
`review.json` with `status: auto_approved`, and marks the review stage
`skipped`. The run continues. `source_conflicts` are preserved in `review.json`
and printed by the CLI, so they are still visible in the run report.

An auto-approved gate can be reopened by running review again with `--review on`
(or `--review-on-conflict` when conflicts exist); it then pauses before prompts
are built.

### Gate on

Stage 4 writes `review.json`, writes a human-readable
`stage-04-review/review_summary.md`, sets the review stage to
`awaiting_review`, sets the **run** status to `awaiting_review`, and stops.

The summary shows the central idea, details, source pages, composition, style,
source conflicts, and every render string. To continue:

```bash
# approve the current stage-03-plan/poster_spec.json
genlesson review <run_id> --resume

# or approve an edited copy
genlesson review <run_id> --resume --spec path/to/edited.json

# equivalent ordering works too
genlesson review --resume <run_id>
```

An edited spec is revalidated against the `PosterSpecDocument` schema before it
can be approved. The review also checks the source and classification hashes,
that details are non-empty, that render strings are unique, and that
`strings_to_render` contains every canonical title, definition, label,
explanation, and example. Invalid edits fail with `spec_invalid` and leave the
run awaiting review so it can be fixed and resumed again.

Approval writes `approved_spec.json`, records its SHA-256 and whether the spec
was edited, and marks the stage `complete`.

### On-conflict mode

`--review-on-conflict` pauses only if `source_conflicts` is non-empty. With no
conflicts it follows the off path and records `auto_approved`.

## What Stage 5 does

```bash
genlesson prompts <run_id> --formats portrait,landscape,square
```

Stage 5 reads the approved spec from Stage 4 and writes:

- `stage-05-prompts/prompts.json`
- `stage-05-prompts/mascot_layout.json`
- one `prompt-<format>.txt` per requested format

It builds a separate prompt for each format. The default model is OpenAI
`gpt-image-2`, with confirmed sizes:

| Format | Size |
| --- | --- |
| portrait | 1024 × 1536 |
| landscape | 1536 × 1024 |
| square | 1024 × 1024 |

These are the `gpt-image-2` sizes in the [OpenAI image generation API
reference](https://developers.openai.com/api/reference/resources/images/methods/generate).
If you change the image model, add its confirmed size table to
`SUPPORTED_IMAGE_MODELS`; Stage 5 refuses to guess.

The prompt:

- quotes **every** string from the approved spec, verbatim, and says the quotes
  are delimiters rather than rendered text;
- includes typography instructions with the Kazakh letters
  `ә, ғ, қ, ң, ө, ұ, ү, һ, і` and their uppercase forms;
- specifies the composition, palette, mood, illustration style, audience, and
  visual descriptions;
- uses a different layout instruction for portrait, landscape, and square;
- reserves the exact mascot block returned by `layout_mascot`, states its exact
  percentage width/height and anchoring, and forbids any mascot, speech bubble,
  question mark, or thought bubble in the generated image;
- forbids any text not in the spec, watermarks/logos/signatures, wrong-language
  text, placeholder text, and falling back to a grid, table, or mind map;
- requires all text to remain readable and inside the canvas.

The prompt includes a `MASCOT RESERVED AREA — LEAVE THIS AREA EMPTY` section
with the corner, exact pixel and percentage rectangle, and a ban on drawing the
mascot or bubble. The question is not put in the image prompt and is not in
`strings_to_render`; it is serialized in `mascot_layout.json` for Stage 7 and
Stage 8.

Building prompts is deterministic and makes no API call. Re-running with a
different `--formats` list rebuilds the bundle; re-running with the same formats
reuses it.

## What Stage 6 does

```bash
genlesson generate <run_id>
genlesson generate <run_id> --json
```

Stage 6 reads `stage-05-prompts/prompts.json` and makes **one image-provider call
per requested format**. Each format is recorded separately, so a failure in one
format never stops the others.

The provider boundary is:

```python
class ImageProvider:
    name: str
    model: str
    def generate(self, prompt: str, size: str) -> bytes: ...
```

`OpenAIImageProvider` is the default implementation. It uses the OpenAI Images
API and the `gpt-image-2` model selected by Stage 5.

### Failure behavior

- Transient failures, rate limits, connection failures, timeouts, and 5xx
  responses are retried with exponential backoff inside the provider.
- A moderation refusal triggers **one** retry with a toned-down prompt that
  removes explicit safety vocabulary. If the retry is refused too, that format
  is reported failed.
- A failed format is recorded in `generation.json` with its error and warning;
  the run continues.
- If at least one format succeeds, the stage is `complete` and the run continues
  to verification. If every format fails, the stage fails with
  `all_formats_failed`.
- Re-running generation reuses successful images and retries failed or missing
  formats. Images are written exclusively; an existing poster file is never
  overwritten.

Generated image bytes are validated and decoded before being accepted. The
recorded width/height are the actual image dimensions; a mismatch with the
requested size is reported as a warning.

### Per-format records

`generation.json` stores, for each format:

- status (`generated` or `failed`);
- requested size and actual dimensions;
- image path, SHA-256, byte count, and media type;
- prompt hash and whether the original or toned-down prompt was used;
- attempt count and moderation-retry flag;
- error details for failed formats.

Because each format is generated and stored independently, Stage 7 can verify
each poster separately.

## What Stage 7 does

```bash
genlesson verify <run_id>
```

Stage 7 reads each generated image back with a vision model, compares the
transcription to the approved spec, and retries mismatches **per format**.

### Comparison rules

- Both sides are normalized with Unicode NFC, casefold, collapsed whitespace,
  and leading/trailing punctuation trimmed. Curly quotes and dashes are folded
  to their plain equivalents.
- The central idea title and every detail label/term must match exactly.
- The definition, detail explanations, and examples are sentences. They pass
  when their best match meets `--similarity` (default `0.9`).
- Any text in the image that is not present in the spec is reported as an extra
  string.
- A format passes only when there are no exact mismatches, no below-threshold
  sentences, and no extra strings.

### Retry and best attempt

On a mismatch, Stage 7 appends a correction note to the original prompt that
lists the wrong expected strings and the extra strings, then generates another
image. This is repeated up to `--max-retries` (default 2, so up to three total
images including the first).

When retries are exhausted, the best attempt is selected by its verification
score, copied to `stage-07-verify/best/`, and the format is marked
`needs_attention`. The run continues; a needs-attention format does not fail
the whole run.

### Vision provider routing

The verifier defaults to the selected planner's model when the planner provider
is vision-capable. If the planner has no vision support, Stage 7 falls back to
the configured vision model (`GENLESSON_VISION_MODEL`). An explicit
`--verifier-model` overrides either. The chosen path is recorded as
`fallback_used` in `verification.json`.

### Mascot placement

After an attempt's text check passes, Stage 7 resolves the mascot placement. It
tries the Stage 5 preferred corner first, then the others by quietness. A corner
is accepted only when no detected text box intersects the block (with a margin)
and pixel busyness is below `mascot.busyness_threshold`. A non-preferred corner
costs no regeneration.

If no corner passes, the attempt fails and the normal retry loop adds a
correction note naming the corner that must stay empty. Once retries are
exhausted, the format is `needs_attention` with reason
`mascot_no_clear_corner`. Each format result carries an optional
`mascot: {corner, rect, method, text_boxes_hit}` record.

### Per-format outcomes

- `passed` — verified with no retries needed, or a retry passed.
- `needs_attention` — retries exhausted; the best attempt is saved.
- `generation_failed` — Stage 6 did not produce an image for that format.

Verification makes one vision call per attempt and is resumed from the saved
report when completed. The report records source, approval, prompt, and
generation hashes so it cannot silently outlive a changed poster.

## What Stage 8 does

```bash
genlesson save <run_id>
```

Stage 8 saves every verified poster that has a best image. It composes the
mascot at the Stage 7 placement (mascot first, then bubble and text), pixel-checks
that the text stays inside the bubble padding, and never writes the final file
until that check passes. A `mascot_no_clear_corner` format is saved without the
mascot and flagged. Files go into a
folder named after the input file under `--out-dir` (default `outputs/`), while
the run report stays in `stage-08-save/save.json`. For example
`fixtures/input/alashorda.pdf` saves to `outputs/alashorda/`. Formats whose
generation failed are skipped and recorded in `save.json`; formats marked
`needs_attention` are still saved, with that status preserved in the sidecar.

### Filenames

A single-format run uses the input stem for both files:

```text
outputs/alashorda/alashorda.png
outputs/alashorda/alashorda.json
```

Multi-format runs add the format so the files stay distinct:

```text
outputs/alashorda/alashorda-portrait.png
outputs/alashorda/alashorda-landscape.json
```

### No overwrite

Each image and sidecar is opened in exclusive-create mode. If a rerun would
collide with an existing poster, the next number is appended to both files:

```text
outputs/alashorda/alashorda.png
outputs/alashorda/alashorda2.png
outputs/alashorda/alashorda3.png
```

The sidecars follow the same names (`alashorda.json`, `alashorda2.json`, ...).
If a sidecar collision is found after the image was created, that new image is
removed and the next number is tried. Existing posters are never truncated or
replaced.

### Sidecars

A same-stem `.json` sits next to every image and contains:

- the full approved poster `spec`;
- the full image `prompt`;
- the planner, image, and verification `models`, including verifier fallback;
- generation and verification `attempts`;
- the full per-format verification `result`;
- a `mascot` block with the applied flag/reason, variant, preferred and final
  corner, method, text-box hits, question and source, fitted font size, final
  rectangle, raw generated SHA-256/path, final SHA-256, and the pixel self-check
  result;
- a `cost` object with currency, `amount_usd` (null when the provider does not
  return pricing), estimated flag, and image/vision call counts.

The cost note is explicit: provider billing is not queried by this pipeline, so
the sidecar records call counts for reconciliation rather than inventing a
price.

`save.json` is the Stage 8 report and records every saved filename, SHA-256,
dimensions, verification status, skipped format, and the verification hash.

### Resume and completion

A completed save is reused. Re-running `genlesson save` after a partial failure
reuses already-saved, hash-valid posters and retries the rest. Because final
files are exclusive-create, a new run on the same input adds `name2`, `name3`,
and so on rather than overwriting anything. A save record produced before the
mascot feature has no `mascot` metadata; enabling mascots makes that record
non-reusable so the poster is recomposed instead of silently keeping the old
image.

When at least one poster is saved, Stage 8 completes and the run status becomes
`complete`. Needs-attention formats remain visible in both `save.json` and their
sidecars.

### Mascot overlay

Every saved poster gets a mascot asking a lesson-related question. Stage 3
optionally plans the question; otherwise Stage 5 derives one from the central
idea. The question is fitted into a white, `#6360FF`-outlined speech bubble and
is rendered by code, never by the image model.

![Mascot preview](docs/mascot-preview-crop.png)

The five icon-style assets live in `assets/mascot/` and keep their original
aspect ratio. The loader requires exactly five alpha PNGs; the short side scales
with `size_ratio` (0.09 by default), and the long side stays proportional:

| Canvas short side | Mascot short side |
| --- | --- |
| 1024 px | 92 px |
| 2000 px | 180 px |

`layout_mascot(width, height, question, corner, config)` is the single source of
geometry. It returns the mascot rectangle, the tightest bubble rectangle, the
largest fitting font (trimmed by 2 px), the wrapped lines, and the combined
block. It measures real rendered pixels, including Kazakh accents.

Stage 5 writes `stage-05-prompts/mascot_layout.json` and tells the image model
to keep the exact percentage-sized block empty. The question is not in
`strings_to_render`. If the question cannot fit even at the minimum font, Stage 5
raises a `mascot_text_overflow` error with a hint to shorten it in review.

Stage 7 resolves the placement after the text check passes. Candidate corners
are tried in this order: the Stage 5 preferred corner, then the rest by
quietness. A corner is accepted only when no detected text box intersects the
block (with a margin) and pixel busyness is below `mascot.busyness_threshold`.
A non-preferred corner costs no regeneration. If no corner passes, the attempt
is retried with a correction note naming the corner that must stay empty. After
retries, the format is `needs_attention` with reason
`mascot_no_clear_corner`; Stage 8 then saves the poster without a mascot and
flags it. Text is never covered.

Stage 8 composes at the recorded placement in this order: mascot, then bubble
and text. This happens automatically at the end of the same `run`; no separate
preview or compose command is needed. The text block is optically centered in
the bubble, and the detector ignores low-confidence boxes that are usually illustration
detail. A pixel self-check verifies that text pixels sit inside the
bubble padding; failure raises instead of silently saving. The raw generated image
stays in the run folder, and the sidecar records variant, preferred/final
corner, question and source, font size, rectangle, and raw/final SHA-256.

```bash
# Inspect the fit without spending image credits.
genlesson mascot-preview \
  --size 1024x1536 \
  --corner bottom-right \
  --variant "Vector (3)" \
  --question "«Байланыс түрлері» туралы не білесің?" \
  --out outputs/mascot-preview.png
```

`--corner auto` (the default) runs the text detector on each candidate block.
`--detector none` skips OCR. The preview prints the chosen corner, font size,
bubble rectangle, and wrapped lines; `--json` returns the same data.

To test the overlay on an already generated poster without spending credits, pass
`--image`:

```bash
genlesson mascot-preview \
  --image outputs/alashorda/alashorda.png \
  --question "«Алаш қозғалысы» туралы не білесің?" \
  --out /tmp/alashorda-mascot.png
```

With `--image`, `--corner auto` uses the same Stage 7 acceptance rules: if no
corner is free of detected text and quiet enough, it exits with
`mascot_no_clear_corner` instead of covering text. Use `--corner ...` only when
you deliberately want to inspect a forced placement.

The bundled open-licensed Noto Sans font covers `ә ғ қ ң ө ұ ү һ і` and their
uppercase forms. The detector method and benchmark are in
[`docs/mascot-text-detector.md`](docs/mascot-text-detector.md).

### Run artifacts

```text
runs/<run_id>/
├── manifest.json
├── stage-01-ingest/
│   ├── lesson.json
│   ├── pages/
│   │   ├── page-0001.png
│   │   └── ...
│   ├── transcripts/              # only when vision was used
│   │   ├── page-0001.txt
│   │   └── page-0001.json
│   └── vision_calls.jsonl        # one line per real model call
├── stage-02-classify/
│   └── classification.json
├── stage-03-plan/
│   └── poster_spec.json
├── stage-04-review/
│   ├── review.json
│   ├── review_summary.md   # when the gate pauses or is approved manually
│   └── approved_spec.json  # when auto-approved or approved
├── stage-05-prompts/
│   ├── prompts.json
│   ├── mascot_layout.json
│   ├── prompt-portrait.txt
│   └── ...
├── stage-06-generate/
│   ├── generation.json
│   └── images/
│       ├── communication-types-portrait.png
│       └── ...
├── stage-07-verify/
│   ├── verification.json
│   ├── best/
│   │   ├── communication-types-portrait-attempt-2.png
│   │   └── ...
│   └── attempts/
│       └── portrait/
│           └── ...
└── stage-08-save/
    └── save.json

outputs/
└── lesson/
    ├── lesson-portrait.png
    └── lesson-portrait.json
```

`lesson.json` is the Stage 1 contract:

```json
{
  "schema_version": 1,
  "source": {
    "path": "/abs/path/lesson.pdf",
    "filename": "lesson.pdf",
    "media_type": "pdf",
    "sha256": "...",
    "size_bytes": 12345,
    "modified_at": "2026-11-04T10:00:00Z"
  },
  "title": "Photosynthesis",
  "pages": [
    {
      "index": 1,
      "label": "Page 1",
      "text": "...",
      "text_source": "text_layer",
      "char_count": 512,
      "image_path": "stage-01-ingest/pages/page-0001.png",
      "image_sha256": "...",
      "width": 1240,
      "height": 1754,
      "transcription_model": null
    }
  ],
  "text": "...",
  "char_count": 512,
  "warnings": [],
  "stats": {
    "page_count": 1,
    "text_layer_pages": 1,
    "vision_pages": 0,
    "blank_pages": 0,
    "char_count": 512,
    "scanned_document": false
  },
  "config": {
    "render_dpi": 180,
    "vision_mode": "openai",
    "vision_model": "gpt-6-luna",
    "min_page_chars": 20,
    "min_document_chars": 40
  },
  "extracted_at": "2026-11-04T10:00:00Z"
}
```

Image paths are relative to the run root, so a run directory can be moved
without breaking its references. `text_source` is `text_layer`, `vision`, or
`blank`. `transcription_model` is set only for pages that actually used a
vision model.

`classification.json` is the Stage 2 contract:

```json
{
  "schema_version": 1,
  "source_sha256": "...",
  "lesson_sha256": "...",
  "poster_language": "kk",
  "confidence": 0.94,
  "evidence": "Photosynthesis is the process...",
  "language_source": "text_layer",
  "dominant_topic": "Light reactions",
  "topics": [
    {"title": "Light reactions", "segment_ids": ["s0005", "s0006"], "char_count": 742}
  ],
  "warnings": [],
  "segments": [
    {
      "id": "s0001",
      "page": 1,
      "page_label": "Page 1",
      "order": 1,
      "text": "Learning objective: ...",
      "char_count": 48,
      "text_source": "text_layer",
      "label": "non_content",
      "category": "learning_objective",
      "reason": "learning objective"
    }
  ],
  "lesson_text": "Photosynthesis is the process...",
  "stats": {
    "segment_count": 12,
    "content_segments": 6,
    "non_content_segments": 6,
    "content_char_count": 1840,
    "language_char_count": 1840,
    "page_images_used": 3
  },
  "config": {
    "planner_provider": "openai",
    "planner_model": "gpt-6-luna",
    "max_page_images": 6
  },
  "classified_at": "2026-11-04T10:05:00Z"
}
```

`lesson_text` is the concatenation of retained lesson content only. Stage 3
consumes it directly, or uses the individual segments for chunked planning.

`poster_spec.json` is the Stage 3 contract:

```json
{
  "schema_version": 1,
  "source_sha256": "...",
  "classification_sha256": "...",
  "language": "kk",
  "age_band": "middle",
  "central_idea": {
    "title": "Байланыс түрлері",
    "definition": "Байланыс — ақпарат алмасудың түрлері."
  },
  "details": [
    {
      "label": "Ауызша байланыс",
      "explanation": "Сөйлеу арқылы ақпарат беру.",
      "example": "Әңгімелесу.",
      "visual": "Two people talking with speech bubbles.",
      "source_page": 2
    }
  ],
  "composition": {
    "concept": "hero scene with callouts",
    "description": "A welcoming hero scene with one callout per communication type."
  },
  "style": {
    "palette": ["deep teal", "warm ochre", "cream"],
    "mood": "bright and encouraging",
    "illustration_style": "friendly flat vector illustration, age-appropriate"
  },
  "strings_to_render": ["Байланыс түрлері", "Ауызша байланыс", "Сөйлеу арқылы ақпарат беру."],
  "source_conflicts": [],
  "file_slug": "communication-types",
  "warnings": [],
  "stats": {
    "content_chars": 1840,
    "chunks": 1,
    "candidate_details": 0,
    "details": 1,
    "strings": 3,
    "page_images_used": 2
  },
  "config": {
    "planner_provider": "openai",
    "planner_model": "gpt-6-luna",
    "max_details": 8,
    "context_char_limit": 24000,
    "chunk_char_limit": 12000,
    "max_page_images": 6
  },
  "planned_at": "2026-11-04T10:10:00Z"
}
```

## Failure behavior

All expected failures exit with status 2 and print a stable code, a message,
and usually a hint. `--json` returns:

```json
{"ok": false, "error": {"code": "...", "message": "...", "hint": "..."}}
```

| Code | Meaning |
| --- | --- |
| `unsupported_file_type` | Not a `.pdf` or `.docx` |
| `input_not_found` / `input_not_file` | Missing or not a regular file |
| `encrypted_pdf` | Password-protected PDF |
| `corrupt_pdf` / `empty_pdf` | Damaged or zero-page PDF |
| `encrypted_docx` | Encrypted or legacy `.doc` renamed to `.docx` |
| `corrupt_docx` | Invalid DOCX container |
| `no_text` | No text layer and no usable transcript |
| `vision_required` | A page needs OCR but vision is disabled or no key is set |
| `vision_auth_failed` | OpenAI rejected the API key |
| `vision_failed` / `vision_rate_limited` | Vision call failed after retries |
| `no_lesson_content` | The planner marked every segment non-content |
| `language_undetermined` | The planner returned no valid BCP-47 language tag |
| `no_plan_details` | The planner produced no usable poster details |
| `plan_invalid` | The spec failed post-parse validation (lengths, forbidden composition, palette, etc.) |
| `planner_required` / `planner_unavailable` | Stage 2/3 has no API key, or the planner SDK/provider is unavailable |
| `planner_auth_failed` | OpenAI rejected the planner API key |
| `planner_failed` / `planner_rate_limited` | Planner call failed after retries |
| `planner_response_invalid` | Planner returned invalid JSON or omitted, duplicated, or invented segment ids |
| `review_pending` | The run is paused for review or has not been approved |
| `review_not_pending` | `--spec` was supplied when no review was awaiting approval |
| `spec_invalid` | The edited spec failed schema or consistency validation |
| `unsupported_image_model` | No confirmed size table exists for the chosen image model |
| `image_provider_required` / `image_provider_unavailable` | Stage 6 has no API key or provider |
| `all_formats_failed` | Every requested format failed to generate |
| `image_failed` | A format failed after provider retries |
| `moderation_refused` | A format was refused by moderation twice |
| `invalid_image` | The provider returned undecodable or unsupported bytes |
| `image_save_failed` | The generated image could not be safely written |
| `vision_provider_required` | Stage 7 has no vision API key |
| `vision_failed` | A vision transcription attempt failed |
| `save_failed` | A final image or sidecar could not be written |
| `no_posters_saved` | Every format was skipped or failed to save |
| `stage_incomplete` | A stage ran before its prerequisite completed |
| `run_exists` | Run id already exists; use `--resume` |
| `input_changed` | Source bytes differ from the run's recorded hash |
| `config_changed` | Resume options differ from the run's recorded Stage 1 config |
| `path_outside_run` | Artifact path attempted to escape the run root |
| `output_exists` | Exclusive write refused to overwrite an existing file |

A failed stage is recorded in `manifest.json` with its error. The run is not
deleted; inspect it or resume it.

## Resume and safety guarantees

- `runs/<run_id>/` is created exclusively. A new run never reuses an existing
  directory.
- A completed ingest or classification is reused as-is. Missing artifacts are
  reported rather than silently regenerated.
- The source SHA-256 is recorded at creation and rechecked on resume. Changed
  input is refused with `input_changed`.
- Stage 1 options are recorded in the manifest. While ingest is incomplete, a
  resume with different options is refused with `config_changed`.
- `classification.json` records both the source and ingest artifact hashes, so a
  completed Stage 2 cannot be reused against a different lesson.
- `poster_spec.json` records the source and classification hashes, so a completed
  Stage 3 cannot be reused against a changed classification.
- A paused review sets the review stage and the whole run to `awaiting_review`.
  Resuming with `--resume` approves the current or supplied spec and returns the
  run to `in_progress`.
- `approved_spec.json` and `prompts.json` record hashes of the artifacts they
  derive from, so Stage 5 cannot silently outlive a changed approval.
- `generation.json` records the prompt bundle hash and per-format image hashes.
  Successful formats are reused; failed formats are retried without overwriting
  any existing poster image.
- `verification.json` records the generation, prompt, approval, and source
  hashes. A completed verification is reused without another vision call.
- Final posters and sidecars are exclusive-create. A completed save is reused;
  a retry adds a new counter/time-stamped file instead of overwriting.
- `classify` and `plan` are idempotent: re-running after a failure retries the
  stage; a completed stage is returned without a model call.
- Stage artifacts are written through same-directory temporary files and atomic
  renames. A crash does not leave a half-written `lesson.json` or manifest.
- Completed stage output is not overwritten during a normal resume.
- `storage.write_bytes_exclusive()` is the primitive for final deliverables:
  it opens with `xb` and raises `output_exists` instead of replacing a file.
  The poster stages will use it so posters are never overwritten.
- API keys are never serialized. Only provider/model names are stored.

## Real lesson fixtures

The repo has a dedicated place for real lesson material and a reference poster:

```text
fixtures/
├── input/       # real lessons + small Russian/scanned samples
└── expected/    # example poster, frozen spec, golden result
```

The first real golden case is present:

```text
fixtures/input/relationtype.pdf
fixtures/expected/relationtype_reference.png
fixtures/expected/relationtype_approved_spec.json
fixtures/expected/relationtype_golden_result.md
```

It is a 33-page Kazakh lesson with mixed Russian/English source pages and a
1536×1024 landscape reference poster. The generated golden poster and sidecar
are under `outputs/relationtype-golden/`; `tests/test_golden_case.py` validates
the real PDF, reference dimensions, frozen spec, prompt guards, and Kazakh
letters without network calls. `fixtures/input/russian_sample.docx` and
`fixtures/input/scanned_sample.pdf` are small deterministic ingest cases.

Put additional lessons at `fixtures/input/<lesson>.pdf` (or `.docx`) and
matching reference at `fixtures/expected/<lesson>__reference.png`. The pytest
suite still uses synthetic temp fixtures for most unit tests, so large files do
not slow normal tests. The reference poster is for visual/LLM comparison, not
image conditioning or pixel matching; Stage 7 verifies text and mascot-safe placement.

## Tests

```bash
.venv/bin/pytest
```

The suite covers text PDFs, scanned PDFs, mixed text/image PDFs, DOCX
paragraph/table ordering, Unicode lesson text, encrypted and corrupt files,
no-text documents, transcription caching across a failed resume, run-id
collisions, path traversal, and CLI JSON/error behavior. Stage 2 tests cover
segment filtering, no-content failure, topic dominance warnings, BCP-47
normalization, text-layer versus vision language evidence, invalid planner
responses, and resume reuse. Stage 3 tests cover one-call planning, chunked
candidate extraction and merging, `max_details`, strict-schema generation,
forbidden compositions, language enforcement, slug normalization, page-image
context, and resume reuse. Stage 4 tests cover off/on/on-conflict gates,
`awaiting_review`, summary writing, edited-spec validation, and resume.
Stage 5 tests cover per-format prompts and sizes, quoted strings, Kazakh
typography, layout differences, forbidden content, format rebuilds, and reuse.
Stage 6 tests cover separate per-format generation, partial failure, moderation
retry with a toned-down prompt, invalid bytes, backoff, all-failed behavior, and
resume reuse. Stage 7 tests cover normalization, exact and fuzzy matching, extra
strings, correction-note retries, exhausted retries with best-attempt saving,
generation-failed formats, planner-to-vision fallback, and resume reuse.
Stage 8 tests cover filenames, sidecar contents, needs-attention saving,
generation-failed skipping, collision counters, no-overwrite behavior, and
resume reuse. Mascot tests cover non-square icon assets, 1024/2000 geometry,
inward bubbles, real-pixel font fitting with Kazakh accents, font coverage for
all nine letters, long-question self-checks, seeded variant coverage, synthetic
text-box overlap and busyness rejection, question language/length/source,
golden-spec back-compat, Stage 7 no-clear-corner retries, Stage 8 recorded
placement and sidecar hashes, and the credit-free preview CLI. Provider tests cover the capability contract, Anthropic tool-use
and image content, DeepSeek JSON mode without images, provider-specific keys,
text-only planner routing, and `.env` loading. Golden tests validate the real
`relationtype` PDF, reference dimensions, frozen spec, prompt guards, and Kazakh
letter distinctions. Vision, planner, and image unit tests use fake clients; the
suite makes no network calls.

## Deliberate current limits

- All eight stages are implemented.
- OCR and the planner require OpenAI. There is no local OCR or local language
  model fallback.
- Only the OpenAI planner adapter exists. The planner boundary is a protocol so
  Anthropic and DeepSeek adapters can be added.
- PDF passwords are rejected, not stored or guessed.
- DOCX pages are not rendered or OCR'd.
- Pages are transcribed sequentially. Stage 2 makes two sequential planner calls
  (classification, then language), and Stage 3 makes one planning call or one
  candidate call per chunk plus a merge call. Parallelism can be added when cost
  and rate-limit policy are settled.
- The planner receives at most `--max-page-images` page images. Stage 2 still
  sends a long lesson in one classification request; it does not chunk.
- Stage 3 chunks on a character budget, not an exact model-tokenizer count. The
  defaults are deliberately conservative; tune `--context-chars` and
  `--chunk-chars` for the deployed model.
- Stage 4 has no interactive editor. It writes the spec and summary, pauses, and
  expects the reviewer to edit the file (or pass `--spec`) before resuming.
- Stage 5's confirmed size tables live in `config.toml` (`gpt-image-2` default,
  with `gpt-image-1` kept for compatibility). Other image models must be added
  explicitly rather than guessed.
- Stage 6 retries moderation only once. A second refusal is recorded and the
  format is left failed; no prompt is rewritten beyond the single toned-down
  variant.
- Stage 7 verifies text and mascot-safe corner placement; it does not judge
  overall aesthetics or illustration quality. Text verification uses a
  similarity heuristic for sentences and exact normalization for
  titles/labels/terms.
- Stage 7 retries generation with a correction note, but the underlying image
  model can still fail to render a string correctly. Such formats remain
  `needs_attention` with the best attempt preserved.
- Sidecar `cost.amount_usd` is null unless the provider reports pricing. The
  pipeline records image/vision call counts but does not invent a price.
- Mascot placement is text-aware and never covers detected text: if no corner
  passes, the format is retried with a blank-corner note, then marked
  `needs_attention` with reason `mascot_no_clear_corner`. Stage 8 saves that
  poster without a mascot and flags it rather than cropping or erasing content.
- Stage 3 may plan the question; when it does not, Stage 5 derives one from the
  central idea. `mascot.question_max_chars`, the bubble width cap, and the
  2 px font trim keep it inside the overlay.
- There is no interactive review editor or gallery UI; artifacts are files under
  the run directory.
- Anthropic and DeepSeek adapters are unit-tested with fake HTTP responses. A live
  OpenAI planner smoke test was run; live Anthropic/DeepSeek calls require their
  provider keys.
- The default model names come from the locked plan (`gpt-6-luna`). Point
  `--vision-model` or `--planner-model` at the exact model available to your
  account.
