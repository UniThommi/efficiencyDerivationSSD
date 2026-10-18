#!/usr/bin/env python3
"""Scale the SSD/Tyvek optical-property matrices inside a GDML file.

This standalone utility reads a GDML geometry produced by the geometry-generation
workflow and writes a *modified copy* in which the property values of specific
optical-property matrices are multiplied by user-supplied factors. It is used to
adapt the SSD reflectivity (and detection efficiency) without rebuilding the
geometry: the scaled copies are simulated and compared with the PMT+Tyvek
simulation by ``dataAnalysis/analyzeEfficiency.py``. The matrix set is chosen by
``--geometry``:

    sphere   (default):
        my_tyvek_foil_surface_REFLECTIVITY            <- scaled by --reflectivity-factor
        my_tyvek_foil_surface_EFFICIENCY              <- scaled by --efficiency-factor

    detector (LEGEND-1000 cylinder; each foil surface is scaled independently):
        my_tyvek_wall_foil_surface_REFLECTIVITY       <- scaled by --r-wall
        my_tyvek_wall_foil_surface_EFFICIENCY         <- scaled by --e-wall
        my_tyvek_bot_foil_surface_REFLECTIVITY        <- scaled by --r-bot
        my_tyvek_bot_foil_surface_EFFICIENCY          <- scaled by --e-bot
        my_tyvek_top_foil_surface_REFLECTIVITY        <- scaled by --r-top
        my_tyvek_top_foil_surface_EFFICIENCY          <- scaled by --e-top
        my_tyvek_pit_foil_surface_REFLECTIVITY        <- scaled by --r-pit
        my_tyvek_pit_foil_surface_EFFICIENCY          <- scaled by --e-pit

    (The global --reflectivity-factor/--efficiency-factor apply to the sphere
    geometry only; for the detector each of the four foils — wall, bot, top, pit
    — carries its own REFLECTIVITY and EFFICIENCY matrix and its own factor.)

Each matrix stores alternating ``energy value`` pairs, e.g.::

    <matrix name="my_tyvek_foil_surface_REFLECTIVITY" coldim="2"
            values="1.5498024804150035*eV 0.913 2.4796839686640055*eV 0.913 ..."/>

Only the property entries (every second token) are scaled. The energy entries
(tokens carrying a ``*eV`` unit) are left untouched, as is every other byte of the
file: the modification is a targeted in-place edit of the matrix ``values="..."``
attributes, so the multi-MB GDML structure is preserved exactly.

Examples
--------
    python scale_tyvek_properties.py \\
        --input-gdml geometry.gdml \\
        --reflectivity-factor 0.95 \\
        --efficiency-factor 1.10 \\
        --output-dir ./modified_geometries

writes ``geometry_R0p95_E1p10.gdml`` (sphere) into ``./modified_geometries``.

    python scale_tyvek_properties.py \\
        --input-gdml geometry.gdml --geometry detector \\
        --r-pit 0.9 --e-pit 1.2 \\
        --output-dir ./modified_geometries

writes ``geometry_pit_R0p9_E1p2.gdml`` (only the pit foil deviates from 1.0;
wall/bot/top stay at 1.0 and are omitted from the filename).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Exact ``name`` attribute values of the matrices to scale. Matching is exact (not
# substring) because some geometries also define a standard ``tyvek_*`` set of
# matrices that must NOT be touched.
#
#   sphere   — the single-surface SSD sphere/wall geometry (one matrix each),
#              scaled by the global --reflectivity-factor / --efficiency-factor.
#   detector — the LEGEND-1000 cylinder geometry. Each of the four foils
#              (wall/bot/top/pit) carries its own REFLECTIVITY/EFFICIENCY matrix
#              and is scaled by its own --r-<surface> / --e-<surface> factor.
GEOMETRIES: tuple[str, ...] = ("detector", "sphere")

SPHERE_REFLECTIVITY_MATRIX: str = "my_tyvek_foil_surface_REFLECTIVITY"
SPHERE_EFFICIENCY_MATRIX: str = "my_tyvek_foil_surface_EFFICIENCY"

# Detector foil surfaces, each independently scalable.
DETECTOR_SURFACES: tuple[str, ...] = ("wall", "bot", "top", "pit")


def detector_matrix_name(surface: str, prop: str) -> str:
    """GDML ``name`` of a detector foil matrix (``prop`` = REFLECTIVITY|EFFICIENCY)."""
    return f"my_tyvek_{surface}_foil_surface_{prop}"

# A self-closing ``<matrix .../>`` element. GDML matrix tags never contain a literal
# '>' inside their attributes, so a non-greedy ``[^>]*`` is a safe, fast tokenizer.
_MATRIX_TAG_RE = re.compile(r"<matrix\b[^>]*/>")
_VALUES_ATTR_RE = re.compile(r'values="([^"]*)"')


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define and parse the command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Create a copy of a GDML file with the REFLECTIVITY and EFFICIENCY "
            "property values of the SSD (my_tyvek_*) foil surfaces scaled by the "
            "given factors."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input-gdml",
        type=Path,
        required=True,
        help="Path to the original GDML file (left unmodified).",
    )
    parser.add_argument(
        "-r",
        "--reflectivity-factor",
        type=str,
        default="1.0",
        help=(
            "Sphere geometry only: factor for my_tyvek_foil_surface_REFLECTIVITY. "
            "Defaults to 1.0 (no change). Ignored for --geometry detector."
        ),
    )
    parser.add_argument(
        "-e",
        "--efficiency-factor",
        type=str,
        default="1.0",
        help=(
            "Sphere geometry only: factor for my_tyvek_foil_surface_EFFICIENCY. "
            "Defaults to 1.0 (no change). Ignored for --geometry detector."
        ),
    )

    # Per-surface factors for the detector geometry (wall/bot/top/pit).
    detector_group = parser.add_argument_group(
        "detector per-surface factors",
        "Used only for --geometry detector. Each foil surface is scaled "
        "independently; every factor defaults to 1.0 (no change).",
    )
    for surface in DETECTOR_SURFACES:
        detector_group.add_argument(
            f"--r-{surface}",
            type=str,
            default="1.0",
            help=f"Factor for {detector_matrix_name(surface, 'REFLECTIVITY')}.",
        )
        detector_group.add_argument(
            f"--e-{surface}",
            type=str,
            default="1.0",
            help=f"Factor for {detector_matrix_name(surface, 'EFFICIENCY')}.",
        )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Existing directory where the modified GDML file is written.",
    )
    parser.add_argument(
        "-g",
        "--geometry",
        choices=GEOMETRIES,
        default="sphere",
        help=(
            "Which set of matrices to scale. 'sphere' (default) scales the single "
            "my_tyvek_foil_surface_* matrices with --reflectivity-factor/"
            "--efficiency-factor; 'detector' scales each LEGEND-1000 foil "
            "(wall/bot/top/pit) independently with its --r-<surface>/--e-<surface> "
            "factors."
        ),
    )
    return parser.parse_args(argv)


def parse_factor(raw: str, label: str) -> float:
    """Parse a scaling-factor string into a float, raising a clear error otherwise.

    The raw string is kept separately (by the caller) for the output filename so that
    trailing zeros typed on the command line (e.g. ``1.10``) are preserved.
    """
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{label} must be a number, got {raw!r}.") from exc


def filesystem_safe_factor(raw: str) -> str:
    """Render a factor string for use inside a filename.

    Preserves the exact characters typed on the command line (including trailing
    zeros), turning ``.`` into ``p`` and a leading ``-`` into ``m`` so the result is
    safe across filesystems. E.g. ``0.95`` -> ``0p95``, ``1.10`` -> ``1p10``,
    ``-0.5`` -> ``m0p5``.
    """
    return raw.replace("-", "m").replace("+", "").replace(".", "p")


def scale_values_string(values: str, factor: float) -> tuple[str, int]:
    """Scale the property tokens of a ``values`` attribute string.

    The string is a whitespace-separated sequence of alternating ``energy value``
    pairs. Energy tokens carry a unit (``*eV``) and are returned unchanged; value
    tokens are plain floats and are multiplied by ``factor``. Scaled values are
    formatted with :func:`repr` (shortest decimal that round-trips exactly).

    Returns the rebuilt values string and the number of scaled property values.

    Raises:
        ValueError: if the token layout does not match the expected
            ``energy value energy value ...`` pattern.
        RuntimeError: if any scaled property value falls outside the physical
            range [0, 1].
    """
    tokens = values.split()
    if not tokens:
        raise ValueError("matrix has an empty 'values' attribute.")
    if len(tokens) % 2 != 0:
        raise ValueError(
            f"expected an even number of tokens (energy/value pairs), "
            f"got {len(tokens)}."
        )

    # The largest property value is the binding constraint for the upper bound:
    # the highest factor keeping every scaled value <= 1 is 1 / max(value).
    max_value = max(
        (float(tok) for i, tok in enumerate(tokens) if i % 2 == 1),
        default=0.0,
    )

    out_tokens: list[str] = []
    n_scaled = 0
    for i, tok in enumerate(tokens):
        if i % 2 == 0:
            # Energy token: must carry a unit; never modified.
            if "*" not in tok:
                raise ValueError(
                    f"expected an energy token with a unit (e.g. '*eV') at "
                    f"position {i}, got {tok!r}."
                )
            out_tokens.append(tok)
        else:
            # Property token: must be a plain float; scaled.
            try:
                value = float(tok)
            except ValueError as exc:
                raise ValueError(
                    f"expected a numeric property value at position {i}, "
                    f"got {tok!r}."
                ) from exc
            scaled = value * factor
            if not 0.0 <= scaled <= 1.0:
                msg = (
                    f"scaled property value out of the physical range [0, 1] at "
                    f"token {i}: {value!r} * {factor!r} = {scaled!r}."
                )
                if scaled > 1.0 and max_value > 0.0:
                    msg += (
                        f" The highest factor that keeps all values <= 1 is "
                        f"{1.0 / max_value!r} (= 1 / {max_value!r}, the largest "
                        f"property value in this matrix)."
                    )
                raise RuntimeError(msg)
            out_tokens.append(repr(scaled))
            n_scaled += 1

    return " ".join(out_tokens), n_scaled


def scale_matrix_in_text(text: str, matrix_name: str, factor: float) -> tuple[str, int]:
    """Scale one named matrix's property values within the raw GDML text.

    Locates the unique ``<matrix .../>`` element whose ``name`` attribute equals
    ``matrix_name`` exactly, scales its property values, and substitutes only that
    element's ``values="..."`` attribute. All other bytes are preserved verbatim.

    Returns the new text and the number of scaled property values.

    Raises:
        ValueError: if the matrix is not found exactly once, or its format is
            unexpected.
    """
    name_attr = f'name="{matrix_name}"'
    matches = [m for m in _MATRIX_TAG_RE.finditer(text) if name_attr in m.group(0)]

    if len(matches) == 0:
        raise ValueError(f"matrix {matrix_name!r} not found in the GDML file.")
    if len(matches) > 1:
        raise ValueError(
            f"matrix {matrix_name!r} found {len(matches)} times; expected exactly one."
        )

    tag = matches[0].group(0)
    values_match = _VALUES_ATTR_RE.search(tag)
    if values_match is None:
        raise ValueError(
            f"matrix {matrix_name!r} has no 'values' attribute (unexpected format)."
        )

    try:
        new_values, n_scaled = scale_values_string(values_match.group(1), factor)
    except ValueError as exc:
        raise ValueError(f"matrix {matrix_name!r}: {exc}") from exc
    except RuntimeError as exc:
        raise RuntimeError(f"matrix {matrix_name!r}: {exc}") from exc

    new_values_attr = f'values="{new_values}"'
    new_tag = tag[: values_match.start()] + new_values_attr + tag[values_match.end() :]
    # The full original tag is unique in the document, so a single-count replace is safe.
    return text.replace(tag, new_tag, 1), n_scaled


def build_jobs(args: argparse.Namespace) -> list[tuple[str, float, str, str, str | None]]:
    """Return the list of scaling jobs for the selected geometry.

    Each job is ``(matrix_name, factor, raw_factor, prop, surface)`` where
    ``prop`` is ``"R"`` or ``"E"`` and ``surface`` is the foil name for the
    detector geometry (``None`` for the sphere). Parsing each factor here means
    a malformed factor string raises before any file is written.

    Raises:
        ValueError: if any factor string is not a valid number.
    """
    if args.geometry == "sphere":
        return [
            (SPHERE_REFLECTIVITY_MATRIX,
             parse_factor(args.reflectivity_factor, "reflectivity-factor"),
             args.reflectivity_factor, "R", None),
            (SPHERE_EFFICIENCY_MATRIX,
             parse_factor(args.efficiency_factor, "efficiency-factor"),
             args.efficiency_factor, "E", None),
        ]

    jobs: list[tuple[str, float, str, str, str | None]] = []
    for surface in DETECTOR_SURFACES:
        r_raw = getattr(args, f"r_{surface}")
        e_raw = getattr(args, f"e_{surface}")
        jobs.append((detector_matrix_name(surface, "REFLECTIVITY"),
                     parse_factor(r_raw, f"r-{surface}"), r_raw, "R", surface))
        jobs.append((detector_matrix_name(surface, "EFFICIENCY"),
                     parse_factor(e_raw, f"e-{surface}"), e_raw, "E", surface))
    return jobs


def build_filename_tag(args: argparse.Namespace) -> str:
    """Filename tag encoding the scaling factors.

    Sphere: ``_R<r>_E<e>`` (always). Detector: ``_<surface>_R<r>_E<e>`` for each
    foil whose factor differs from 1.0, including only the property that actually
    deviates; if every foil is left at 1.0 the tag is ``_R1p0_E1p0``.
    """
    if args.geometry == "sphere":
        return (f"_R{filesystem_safe_factor(args.reflectivity_factor)}"
                f"_E{filesystem_safe_factor(args.efficiency_factor)}")

    parts: list[str] = []
    for surface in DETECTOR_SURFACES:
        r_raw = getattr(args, f"r_{surface}")
        e_raw = getattr(args, f"e_{surface}")
        seg = ""
        if parse_factor(r_raw, f"r-{surface}") != 1.0:
            seg += f"_R{filesystem_safe_factor(r_raw)}"
        if parse_factor(e_raw, f"e-{surface}") != 1.0:
            seg += f"_E{filesystem_safe_factor(e_raw)}"
        if seg:
            parts.append(f"_{surface}{seg}")
    return "".join(parts) if parts else "_R1p0_E1p0"


def main(argv: list[str] | None = None) -> int:
    """Entry point: validate inputs, scale the matrices, write the modified copy."""
    args = parse_args(argv)

    input_path: Path = args.input_gdml
    output_dir: Path = args.output_dir

    # --- Input validation -----------------------------------------------------
    if not input_path.is_file():
        print(f"ERROR: input GDML file does not exist: {input_path}", file=sys.stderr)
        return 1
    if not output_dir.is_dir():
        print(f"ERROR: output directory does not exist: {output_dir}", file=sys.stderr)
        return 1

    try:
        jobs = build_jobs(args)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Warn about flags that have no effect for the chosen geometry.
    if args.geometry == "detector" and (
        parse_factor(args.reflectivity_factor, "reflectivity-factor") != 1.0
        or parse_factor(args.efficiency_factor, "efficiency-factor") != 1.0
    ):
        print("WARNING: --reflectivity-factor/--efficiency-factor are ignored for "
              "--geometry detector; use the per-surface --r-<surface>/--e-<surface> "
              "flags instead.", file=sys.stderr)
    if args.geometry == "sphere" and any(
        parse_factor(getattr(args, f"r_{s}"), f"r-{s}") != 1.0
        or parse_factor(getattr(args, f"e_{s}"), f"e-{s}") != 1.0
        for s in DETECTOR_SURFACES
    ):
        print("WARNING: the per-surface --r-<surface>/--e-<surface> flags are "
              "ignored for --geometry sphere; use --reflectivity-factor/"
              "--efficiency-factor instead.", file=sys.stderr)

    # --- Read, scale, write ---------------------------------------------------
    text = input_path.read_text(encoding="utf-8")

    n_reflectivity = 0
    n_efficiency = 0
    try:
        for name, factor, _raw, prop, _surface in jobs:
            text, n = scale_matrix_in_text(text, name, factor)
            if prop == "R":
                n_reflectivity += n
            else:
                n_efficiency += n
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    out_name = f"{input_path.stem}{build_filename_tag(args)}{input_path.suffix}"
    output_path = output_dir / out_name
    output_path.write_text(text, encoding="utf-8")

    # --- Summary --------------------------------------------------------------
    print("Tyvek property scaling complete.")
    print(f"  Geometry                    : {args.geometry}")
    print(f"  Input file                  : {input_path}")
    print(f"  Output file                 : {output_path}")
    print("  Scaled matrices             :")
    for name, factor, _raw, prop, surface in jobs:
        tag = f"{surface} " if surface else ""
        prop_name = "REFLECTIVITY" if prop == "R" else "EFFICIENCY"
        print(f"      {tag}{prop_name:<12} x{factor:<8} : {name}")
    print(f"  Modified reflectivity values: {n_reflectivity}")
    print(f"  Modified efficiency values  : {n_efficiency}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
