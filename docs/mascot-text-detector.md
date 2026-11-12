# Mascot text detector spike

The mascot and question bubble must never cover rendered poster text. The
verifier only returns a transcript, so placement needs coordinates. This spike
compares box detectors on the candidate mascot rectangle, not the full poster.

## Candidate

**Chosen: PP-OCR DB through RapidOCR (`rapidocr>=3.4`).** DB is trained to
produce text boxes, not transcripts, so it does not depend on the lesson
language. It is bundled in the Python wheel and runs on CPU through ONNX
Runtime. `src/genlesson/mascot/zones.py` uses it only when a corner is being
scored; `--detector none` and the null detector keep tests and offline previews
dependency-light.

Alternatives considered:

| Detector | Script coverage | Setup | Result |
| --- | --- | --- | --- |
| PP-OCR DB (RapidOCR) | language-agnostic boxes | `pip` dependency, bundled model | chosen |
| Tesseract | needs per-language traineddata | system binary + language packs | not available in this environment |
| Morphological edges | none (pixels only) | OpenCV only | noisier, used as baseline |
| Vision model | model-dependent | paid API per crop | rejected for placement cost and nondeterminism |

## Method

`experiments/mascot_text_detector.py` renders a 1024 x 1536 synthetic poster
with known text lines from 8 px to 30 px, including Kazakh letters and numbers.
It runs each detector **only on the four candidate mascot rectangles** returned
by `layout_mascot`, then counts a ground-truth line as detected when a predicted
box has IoU >= 0.3 with it or contains its centre.

The same script scans every existing `outputs/**/*.png` poster and reports the
number of detected boxes and runtime for the bottom-right candidate crop.

## Result (2026-11-11)

```text
Synthetic small-text recall by detector
---------------------------------------
rapidocr-db    recall=1.000 (16/16) boxes= 28 extra= 12 time=1.090s
opencv-edges   recall=1.000 (16/16) boxes= 39 extra= 23 time=0.001s
```

Recall was tied at 100% on this clean synthetic set. DB was chosen on the
recall tie-break: it produced 12 extra boxes versus 23 for the pixel baseline,
and it is text-specific rather than a general edge response. The edge detector
would be cheaper but is more likely to mistake illustration detail for text,
which can move the mascot away from a perfectly good corner.

Existing-posters scan (bottom-right crop, DB detector):

```text
outputs/Grade12_Term2_Week2_Lesson1/Grade12_Term2_Week2_Lesson1.png    boxes=  3 time=0.282s
outputs/alashorda/alash-kozgalysy-tarihy__kk__portrait__20261105-072558__24ad5b.png boxes=  5 time=0.184s
outputs/alashorda/alashorda.png                                        boxes=  3 time=0.254s
outputs/government/tortinshi-bilik__kk__portrait__20261105-071551__2a3520.png boxes=  1 time=0.171s
outputs/kazakhstan-inflation-currency-shocks__en__portrait__20261110-220810__bd43e9.png boxes=  4 time=0.160s
outputs/relationtype/organism-relationships__kk__portrait__20261105-063533__7c9236.png boxes=  6 time=0.164s
outputs/relationtype-golden/biotic-interactions__kk__landscape__20261105-065019__c407e2.png boxes=  3 time=0.221s
```

Run it yourself:

```bash
python experiments/mascot_text_detector.py
python experiments/mascot_text_detector.py --json
```

## Placement behavior

1. Stage 5 computes the reserved mascot block with `layout_mascot` and asks the
   image model to leave that exact percentage rectangle free of text, labels,
   numbers, and key illustrations.
2. Stage 7 tries the preferred corner first, then the rest by quietness. It runs
   DB only on each candidate crop and accepts a corner only when no detected text
   box intersects the block (with a margin) and pixel busyness is below the
   configured threshold. Detector boxes below `0.5` confidence are ignored; on
   the trace poster, illustration false positives scored `0.25–0.28` while real
   text scored `0.65–0.83`.
3. If no corner passes, the attempt is retried with a correction note naming the
   corner that must stay empty. Exhausted retries mark the format
   `needs_attention` with reason `mascot_no_clear_corner`; Stage 8 saves it
   without a mascot and flags it. No text is covered.
