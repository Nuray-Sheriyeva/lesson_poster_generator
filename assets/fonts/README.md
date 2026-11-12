# Fonts

`NotoSans-Regular.ttf` and `NotoSans-Bold.ttf` are Noto Sans, licensed under the
SIL Open Font License 1.1. The license text is in `OFL.txt`.

Noto Sans covers the Kazakh Cyrillic letters used by the poster pipeline:
`ә ғ қ ң ө ұ ү һ і` and `Ә Ғ Қ Ң Ө Ұ Ү Һ І`. The mascot text fitter measures
real rendered pixels with this font, including accents and descenders.

`mascot.font_path` in `config.toml` points at the regular face. Swap in another
Open Font License TTF with the same coverage if needed.
