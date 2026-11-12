"""Load and validate the five bundled mascot PNGs.

Variant ids are the file names, without the ``.png`` suffix.  The loader is
strict on purpose: a poster overlay must not silently use an opaque, or
under-resolution asset.  Assets keep their original aspect ratio; the short
side is scaled to the configured mascot size.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ..errors import GenLessonError

MIN_ASSET_PX = 93  # at 1024 px the rendered short side is 92 px
MIN_ASSET_SIZE = MIN_ASSET_PX
EXPECTED_ASSETS = 5
EXPECTED_MASCOT_COUNT = EXPECTED_ASSETS
_REPO_ROOT = Path(__file__).resolve().parents[3]


def repository_root() -> Path:
    return _REPO_ROOT


def resolve_asset_path(path: Path | str, *, base: Path | None = None) -> Path:
    """Resolve a configured asset path against the CWD and repository root.

    Absolute paths are returned unchanged.  Relative paths prefer the explicit
    ``base`` (normally the config file's directory), then the current working
    directory, then the repository root.  The first existing candidate wins; if
    none exists, the repository-root candidate is returned so the error message
    points at the bundled location.
    """
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    roots: list[Path] = []
    if base is not None:
        roots.append(Path(base).expanduser())
    roots.append(Path.cwd())
    roots.append(_REPO_ROOT)
    for root in roots:
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved
    return (_REPO_ROOT / candidate).resolve()


@dataclass(frozen=True)
class MascotAsset:
    """A validated mascot PNG."""

    variant: str
    path: Path
    width: int
    height: int

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

    def target_size(self, short_side: int) -> tuple[int, int]:
        """Return the rendered size for a given short side, keeping the ratio."""
        short_side = max(1, int(short_side))
        if self.width <= self.height:
            width = short_side
            height = max(short_side, round(short_side * self.height / self.width))
        else:
            height = short_side
            width = max(short_side, round(short_side * self.width / self.height))
        return width, height

    def load(self, short_side: int) -> Image.Image:
        """Return the asset as RGBA, scaled so its short side is ``short_side``."""
        width, height = self.target_size(short_side)
        return self.load_for_box(width, height)

    def load_for_box(self, width: int, height: int) -> Image.Image:
        """Return the asset as RGBA, resized to the exact box with LANCZOS."""
        if width < 1 or height < 1:
            raise GenLessonError(
                "mascot_canvas_too_small",
                f"Mascot render size must be positive, got {width}x{height}.",
            )
        try:
            with Image.open(self.path) as image:
                converted = image.convert("RGBA")
                if converted.size != (width, height):
                    converted = converted.resize((width, height), Image.Resampling.LANCZOS)
                return converted
        except OSError as exc:
            raise GenLessonError(
                "mascot_assets_invalid",
                f"Could not read mascot asset {self.path.name!r}: {exc}",
            ) from exc


def _png_files(directory: Path) -> list[Path]:
    try:
        entries = list(directory.iterdir())
    except FileNotFoundError as exc:
        raise GenLessonError(
            "mascot_assets_missing",
            f"Mascot assets directory does not exist: {directory}",
            hint="Create assets/mascot with exactly five alpha PNGs.",
        ) from exc
    except OSError as exc:
        raise GenLessonError(
            "mascot_assets_missing",
            f"Could not read mascot assets directory: {directory}",
            details={"reason": str(exc)},
        ) from exc
    return sorted(
        (entry for entry in entries if entry.is_file() and entry.suffix.casefold() == ".png"),
        key=lambda entry: entry.name.casefold(),
    )


def _validate_one(path: Path) -> MascotAsset:
    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            has_alpha = "A" in image.getbands() or image.mode in {"LA", "RGBA"}
            if min(width, height) < MIN_ASSET_PX:
                raise GenLessonError(
                    "mascot_assets_invalid",
                    f"Mascot asset {path.name!r} is {width}x{height}; the short side must be "
                    f"at least {MIN_ASSET_PX}px.",
                )
            if not has_alpha:
                raise GenLessonError(
                    "mascot_assets_invalid",
                    f"Mascot asset {path.name!r} has no alpha channel.",
                )
    except GenLessonError:
        raise
    except OSError as exc:
        raise GenLessonError(
            "mascot_assets_invalid",
            f"Mascot asset {path.name!r} is not a readable PNG: {exc}",
        ) from exc
    return MascotAsset(
        variant=path.stem,
        path=path.resolve(),
        width=width,
        height=height,
    )


def load_mascot_assets(assets_dir: Path | str) -> dict[str, MascotAsset]:
    """Load exactly five alpha PNG assets keyed by file stem."""
    directory = resolve_asset_path(Path(assets_dir))
    pngs = _png_files(directory)
    if len(pngs) != EXPECTED_ASSETS:
        names = ", ".join(path.name for path in pngs) or "(none)"
        raise GenLessonError(
            "mascot_assets_invalid",
            f"Mascot assets directory must contain exactly {EXPECTED_ASSETS} PNGs; "
            f"found {len(pngs)}: {names}.",
        )
    assets: dict[str, MascotAsset] = {}
    for path in pngs:
        asset = _validate_one(path)
        if asset.variant in assets:
            raise GenLessonError(
                "mascot_assets_invalid",
                f"Duplicate mascot variant id: {asset.variant!r}.",
            )
        assets[asset.variant] = asset
    return assets
