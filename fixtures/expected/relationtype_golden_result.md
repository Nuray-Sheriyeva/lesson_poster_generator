# relationtype golden result

Run: `relationtype-golden`

Input: `fixtures/input/relationtype.pdf` (33 pages, Kazakh, mixed Russian/English source pages)
Reference: `fixtures/expected/relationtype_reference.png` (1536×1024 landscape)
Approved spec: `fixtures/expected/relationtype_approved_spec.json`
Format: landscape, `gpt-image-2`

## Result

- Language: `kk`, confidence `0.98`.
- Planning: 7 details, 1 recorded source conflict about conflicting `+/-` signs.
- Final output: `outputs/relationtype-golden/biotic-interactions__kk__landscape__20261105-065019__c407e2.png`
- Verification: passed on attempt 3 after the verifier was fixed to handle wrapped lines,
  list-marker stripping, and fuzzy-matched line fragments. Best score: `0.998`.
- Sidecar: same-stem `.json` next to the saved poster.

## Visual comparison

The generated poster covers the same seven relationship types and uses the same
green/nature/educational direction as the reference. The reference uses an
8-card 4×2 grid; the generated poster uses a 7-card two-column layout with a
unified ecosystem illustration. Those are structural differences, not text
failures.

The first generation included `1., 2., 3.` prompt-ordering numbers and a
somewhat graphic carcass. The prompt was tightened to forbid list markers and
blood/gore/carcasses; the final attempt removed both.

## Known gaps

- Reference has 8 interaction cards; the approved spec has 7 because the source
  table on page 2 lists seven types. Neutralism/amensalism are represented in
  the reference but not in that table.
- The generated layout is not pixel-identical to the reference, and image
  generation is nondeterministic. Treat this as a visual review baseline, not a
  pixel-diff test.
- The reference poster is not passed to the model as image conditioning.
