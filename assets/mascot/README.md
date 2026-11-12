# Mascot assets

These are the five icon-style mascot states used by the poster overlay. They are
the assets that were originally supplied in the mascot folder; the loader uses
them in filename order:

| Variant | State |
| --- | --- |
| `Vector.png` | smiling / closed-eye state |
| `Vector (1).png` | open-eye state |
| `Vector (2).png` | open-eye state (alternate) |
| `Vector (3).png` | code / thinking state |
| `Vector (4).png` | error / surprised state |

The loader (`src/genlesson/mascot/assets.py`) requires exactly five PNG files,
each with an alpha channel and a short side larger than the 92 px rendered size
at 1024 px. Assets keep their original aspect ratio: the short side is scaled to
`mascot.size_ratio * poster_short_side` and the long side stays proportional.
Variant ids are the file stems.

Do not add a sixth PNG. The bundled Noto Sans font is licensed under the SIL Open
Font License; see `assets/fonts/OFL.txt`.
