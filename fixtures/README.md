# Real lesson/poster fixtures

This directory is for real, non-synthetic test material. The pytest suite still
uses generated temp files, so adding large lesson files here does not slow down
normal tests.

## Layout

```text
fixtures/
├── input/
│   ├── relationtype.pdf                  # real Kazakh lesson
│   ├── russian_sample.docx               # small Russian lesson
│   └── scanned_sample.pdf                # scanned Kazakh lesson
└── expected/
    ├── relationtype_reference.png        # example poster
    ├── relationtype_approved_spec.json   # frozen approved spec
    ├── relationtype_golden.json          # golden metadata
    └── relationtype_golden_result.md     # run/visual notes
```

The `relationtype` case is the first real golden case. See
`relationtype_golden_result.md` for the generated output path and known
differences. The Russian DOCX and scanned PDF are small deterministic ingest
samples; `tests/test_sample_fixtures.py` exercises them offline.

Place exactly one primary lesson file in `input/` for a run. The reference image
is for human and LLM review; it is not fed into Stage 7, which verifies text only.

## Run a lesson

```bash
RUN=photosynthesis

genlesson ingest "fixtures/input/$RUN.pdf" --run-id "$RUN"
genlesson classify "$RUN"
genlesson plan "$RUN"
# If the review gate is on:
genlesson review "$RUN" --resume
genlesson prompts "$RUN" --format portrait
genlesson generate "$RUN"
genlesson verify "$RUN"
genlesson save "$RUN" --out-dir "outputs/$RUN"
```

The final images and sidecars land in `outputs/<run_id>/`. Put the example
poster at `fixtures/expected/<run_id>__reference.png` and compare the generated
portrait against it visually, or provide both to an LLM for a structured review.

## Notes

- Do not commit copyrighted lesson material or student data unless you have
  permission.
- Large PDFs belong here, not in `tests/`.
- Generation is nondeterministic, so “close to the example” is a review judgment,
  not a pixel-equality test. Stage 7 only verifies rendered text.
- Reference posters are not used as image-to-image input by the current pipeline.
