#!/usr/bin/env python3
"""Scale the SSD/Tyvek optical-property matrices inside a GDML file.

This standalone utility reads a GDML geometry produced by the geometry-generation
workflow and writes a *modified copy* in which the property values of two specific
optical-property matrices are multiplied by user-supplied factors:

    my_tyvek_foil_surface_REFLECTIVITY   <- scaled by --reflectivity-factor
    my_tyvek_foil_surface_EFFICIENCY     <- scaled by --efficiency-factor

Each matrix stores alternating ``energy value`` pairs, e.g.::

    <matrix name="my_tyvek_foil_surface_REFLECTIVITY" coldim="2"
            values="1.5498024804150035*eV 0.913 2.4796839686640055*eV 0.913 ..."/>

Only the property entries (every second token) are scaled. The energy entries
(tokens carrying a ``*eV`` unit) are left untouched, as is every other byte of the
file: the modification is a targeted in-place edit of the two ``values="..."``
attributes, so the multi-MB GDML structure is preserved exactly.

Example
-------
    python scale_tyvek_properties.py \\
        --input-gdml geometry.gdml \\
        --reflectivity-factor 0.95 \\
        --efficiency-factor 1.10 \\
        --output-dir ./modified_geometries

writes ``geometry_R0p95_E1p10.gdml`` into ``./modified_geometries``.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Exact ``name`` attribute values of the two matrices to scale. Matching is exact
# (not substring) because some geometries also define a standard ``tyvek_*`` set of
# matrices that must NOT be touched.
REFLECTIVITY_MATRIX_NAME = "my_tyvek_foil_surface_REFLECTIVITY"
EFFICIENCY_MATRIX_NAME = "my_tyvek_foil_surface_EFFICIENCY"

# A self-closing ``<matrix .../>`` element. GDML matrix tags never contain a literal
# '>' inside their attributes, so a non-greedy ``[^>]*`` is a safe, fast tokenizer.
_MATRIX_TAG_RE = re.compile(r"<matrix\b[^>]*/>")
_VALUES_ATTR_RE = re.compile(r'values="([^"]*)"')


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define and parse the command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Create a copy of a GDML file with the my_tyvek_foil_surface "
            "REFLECTIVITY and EFFICIENCY property values scaled by the given factors."
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
            "Multiplicative factor for my_tyvek_foil_surface_REFLECTIVITY values. "
            "Defaults to 1.0 (no change)."
        ),
    )
    parser.add_argument(
        "-e",
        "--efficiency-factor",
        type=str,
        default="1.0",
        help=(
            "Multiplicative factor for my_tyvek_foil_surface_EFFICIENCY values. "
            "Defaults to 1.0 (no change)."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Existing directory where the modified GDML file is written.",
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
    """
    tokens = values.split()
    if not tokens:
        raise ValueError("matrix has an empty 'values' attribute.")
    if len(tokens) % 2 != 0:
        raise ValueError(
            f"expected an even number of tokens (energy/value pairs), "
            f"got {len(tokens)}."
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
            out_tokens.append(repr(value * factor))
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

    new_values_attr = f'values="{new_values}"'
    new_tag = tag[: values_match.start()] + new_values_attr + tag[values_match.end() :]
    # The full original tag is unique in the document, so a single-count replace is safe.
    return text.replace(tag, new_tag, 1), n_scaled


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
        reflectivity_factor = parse_factor(args.reflectivity_factor, "reflectivity-factor")
        efficiency_factor = parse_factor(args.efficiency_factor, "efficiency-factor")
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # --- Read, scale, write ---------------------------------------------------
    text = input_path.read_text(encoding="utf-8")

    try:
        text, n_reflectivity = scale_matrix_in_text(
            text, REFLECTIVITY_MATRIX_NAME, reflectivity_factor
        )
        text, n_efficiency = scale_matrix_in_text(
            text, EFFICIENCY_MATRIX_NAME, efficiency_factor
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    out_name = (
        f"{input_path.stem}"
        f"_R{filesystem_safe_factor(args.reflectivity_factor)}"
        f"_E{filesystem_safe_factor(args.efficiency_factor)}"
        f"{input_path.suffix}"
    )
    output_path = output_dir / out_name
    output_path.write_text(text, encoding="utf-8")

    # --- Summary --------------------------------------------------------------
    print("Tyvek property scaling complete.")
    print(f"  Input file                  : {input_path}")
    print(f"  Output file                 : {output_path}")
    print(f"  Reflectivity factor         : {reflectivity_factor}")
    print(f"  Efficiency factor           : {efficiency_factor}")
    print(f"  Modified reflectivity values: {n_reflectivity}")
    print(f"  Modified efficiency values  : {n_efficiency}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
