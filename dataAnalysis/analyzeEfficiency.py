#!/usr/bin/env python3
"""
SSD vs. PMT+Tyvek efficiency analysis script.

Analyses performed (in order):
  2.1  Photon validity filter (vertex position + momentum direction)
  2.2  GDML reflectivity verification (SSD surface vs. analytic formula)
  2.3  Total hit comparison (SSD vs. PMT+Tyvek, global efficiency correction)
  2.4  Per-voxel efficiency analysis
  2.5  3D voxel visualisation

Usage:
  python analyzeEfficiency.py \\
      --gdml  path/to/ssd.gdml \\
      --data-ssd  dir/ssd_hdf5/ \\
      --data-pmt  dir/pmt_hdf5/ \\
      --voxel-json  voxels.json \\
      --geometry  sphere \\
      --output-dir  results/

Optional:
  --selected-json  voxels_selected.json   (default: voxels_selected.json next to --voxel-json)
  --data-pmt-no-tyvek  dir/notyvek_hdf5/  (wall geometry only)
  --n-pmts-from-gdml                       infer N_PMT from GDML rawids instead of JSON
"""

import argparse
import json
import sys
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

# ══════════════════════════════════ Constants ══════════════════════════════════

HC_EV_NM: float = 1239.84            # hc in eV·nm for energy↔wavelength conversion
M_TO_MM: float = 1000.0              # conversion factor

TYVEK_DETECTOR_ID: int = 1000        # SSD / Tyvek wall rawid
INNER_SPHERE_DETECTOR_ID: int = 9000
PMT_DETECTOR_ID_BASE: int = 10000    # PMT rawids start here

# Reference Tyvek reflectivity table (wavelength in nm, reflectivity dimensionless)
TYVEK_REF_TABLE: np.ndarray = np.array([
    [250, 0.80],
    [300, 0.90],
    [355, 0.945],
    [400, 0.96],
    [440, 0.96],
    [500, 0.96],
    [800, 0.96],
])

PMT_CATHODE_R: float = 0.028725181851905604   # wavelength-independent
PMT_CATHODE_RADIUS_MM: float = 110.0           # effective cathode radius

SPHERE_INNER_RADIUS_MM: float = 1739.52
INNER_SPHERE_RADIUS_MM: float = 1000.0
WALL_X_MM: float = 1950.0
WALL_Y_MM: float = 1950.0
WALL_Z_MM: float = 1.0


# ══════════════════════════════ HDF5 Loading ══════════════════════════════════

def _load_one_hdf5(path: Path) -> dict[str, np.ndarray]:
    """Load optical hits and emission vertices from one HDF5 file."""
    with h5py.File(path, "r") as f:
        opt = f["hit"]["optical"]
        vert = f["hit"]["vertices"]

        data = {
            # Optical hits (positions in mm, momenta dimensionless unit vector)
            "x":          np.asarray(opt["x_position_in_m"]["pages"]) * M_TO_MM,
            "y":          np.asarray(opt["y_position_in_m"]["pages"]) * M_TO_MM,
            "z":          np.asarray(opt["z_position_in_m"]["pages"]) * M_TO_MM,
            "px":         np.asarray(opt["x_momentum_direction"]["pages"]),
            "py":         np.asarray(opt["y_momentum_direction"]["pages"]),
            "pz":         np.asarray(opt["z_momentum_direction"]["pages"]),
            "evtid":      np.asarray(opt["evtid"]["pages"]),
            "det_uid":    np.asarray(opt["det_uid"]["pages"]),
            "wavelength": np.asarray(opt["wavelength_in_nm"]["pages"]),
            "time":       np.asarray(opt["time_in_ns"]["pages"]),
            # Emission vertices (positions in mm)
            "v_evtid": np.asarray(vert["evtid"]["pages"]),
            "vx":      np.asarray(vert["xloc_in_m"]["pages"]) * M_TO_MM,
            "vy":      np.asarray(vert["yloc_in_m"]["pages"]) * M_TO_MM,
            "vz":      np.asarray(vert["zloc_in_m"]["pages"]) * M_TO_MM,
        }
    return data


def load_hdf5_dir(directory: Path) -> dict[str, np.ndarray]:
    """Concatenate all HDF5 files in a directory into one dataset."""
    files = sorted(directory.glob("*.hdf5")) + sorted(directory.glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No HDF5 files found in {directory}")

    print(f"  Loading {len(files)} HDF5 file(s) from {directory} …")
    parts: list[dict] = []
    for f in files:
        parts.append(_load_one_hdf5(f))
        print(f"    {f.name}: {len(parts[-1]['x'])} hits, "
              f"{len(parts[-1]['v_evtid'])} vertices")

    # Concatenate arrays; evtid values may repeat across files — disambiguate
    # by adding a per-file offset. Cast to int64 to avoid int32 overflow.
    offset = np.int64(0)
    for i, p in enumerate(parts):
        p["evtid"]   = p["evtid"].astype(np.int64)
        p["v_evtid"] = p["v_evtid"].astype(np.int64)
        max_id = np.int64(p["v_evtid"].max()) + 1 if len(p["v_evtid"]) else np.int64(0)
        if i > 0:
            p["evtid"]   += offset
            p["v_evtid"] += offset
        offset += max_id

    combined: dict[str, np.ndarray] = {}
    for key in parts[0]:
        combined[key] = np.concatenate([p[key] for p in parts])

    print(f"    Total: {len(combined['x'])} optical hits, "
          f"{len(combined['v_evtid'])} vertices")
    return combined


def load_particles_wavelength(directory: Path) -> np.ndarray:
    """
    Load emitted-photon wavelengths (nm) from the particles table.

    Reads ekin_in_MeV from //hit/particles for every HDF5 file in *directory*,
    converts to wavelength via λ = hc/E, and returns only entries that fall
    in the 100–1500 nm range (i.e. genuine optical photons).
    """
    files = sorted(directory.glob("*.hdf5")) + sorted(directory.glob("*.h5"))
    parts: list[np.ndarray] = []
    for f in files:
        with h5py.File(f, "r") as hf:
            ekin_mev = np.asarray(hf["hit"]["particles"]["ekin_in_MeV"]["pages"],
                                  dtype=np.float64)
        wl = HC_EV_NM / (ekin_mev * 1e6)
        parts.append(wl)
    return np.concatenate(parts) if parts else np.array([], dtype=float)


def load_and_filter_hdf5_dir(
    directory: Path,
    geometry: str,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict]:
    """
    Load and validity-filter HDF5 files one at a time to minimise peak RAM.

    Each file's vertex arrays (~800 MB for 25 M vertices) are freed as soon as
    the validity mask is computed.  Only the four columns needed downstream
    (x, y, z, det_uid) are kept for valid hits.

    Returns
    -------
    data       : dict with keys x, y, z, det_uid — valid hits only
    valid_mask : all-True boolean array (hits are already filtered)
    stats      : aggregated validity stats across all files
    """
    files = sorted(directory.glob("*.hdf5")) + sorted(directory.glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No HDF5 files found in {directory}")

    print(f"  Loading {len(files)} HDF5 file(s) from {directory} …")

    kept: list[dict] = []
    stats: dict = {
        "total": 0, "valid": 0, "invalid_vertex": 0,
        "invalid_momentum": 0, "no_vertex_match": 0,
        "outside_per_surface": {},
    }

    offset = np.int64(0)
    for f in files:
        fdata = _load_one_hdf5(f)
        n_hits  = len(fdata["x"])
        n_verts = len(fdata["v_evtid"])
        print(f"    {f.name}: {n_hits} hits, {n_verts} vertices")

        fdata["evtid"]   = fdata["evtid"].astype(np.int64)   + offset
        fdata["v_evtid"] = fdata["v_evtid"].astype(np.int64) + offset
        max_id = np.int64(fdata["v_evtid"].max()) + 1 if n_verts else np.int64(0)
        offset += max_id

        vmask, fstats = apply_validity_filter(fdata, geometry)

        stats["total"]            += fstats["total"]
        stats["valid"]            += fstats["valid"]
        stats["invalid_vertex"]   += fstats["invalid_vertex"]
        stats["invalid_momentum"] += fstats["invalid_momentum"]
        stats["no_vertex_match"]  += fstats["no_vertex_match"]
        for uid, cnt in fstats["outside_per_surface"].items():
            stats["outside_per_surface"][uid] = (
                stats["outside_per_surface"].get(uid, 0) + cnt
            )

        kept.append({
            "x":          fdata["x"][vmask],
            "y":          fdata["y"][vmask],
            "z":          fdata["z"][vmask],
            "det_uid":    fdata["det_uid"][vmask],
            "wavelength": fdata["wavelength"][vmask],
            "time":       fdata["time"][vmask],
        })
        # fdata (vertices + full hit arrays) freed here

    n_valid = sum(len(p["x"]) for p in kept)
    print(f"    Total: {stats['total']} optical hits, {n_valid} valid")

    combined: dict[str, np.ndarray] = {
        k: np.concatenate([p[k] for p in kept]) for k in kept[0]
    }
    return combined, np.ones(len(combined["x"]), dtype=bool), stats


# ══════════════════════════════ Validity Filter ═══════════════════════════════

def _vertex_position(hit_evtid: np.ndarray, data: dict) -> np.ndarray:
    """
    Return emission vertex (x, y, z) in mm for each optical hit.

    Uses searchsorted for O(N log N) performance instead of a dict loop.
    Rows with no matching vertex are filled with NaN.
    """
    v_evtid = data["v_evtid"]
    vx = data["vx"]
    vy = data["vy"]
    vz = data["vz"]

    sort_idx = np.argsort(v_evtid)
    sorted_evtid = v_evtid[sort_idx]

    # For each hit evtid, find insertion point in sorted vertex evtids
    ins = np.searchsorted(sorted_evtid, hit_evtid, side="left")
    within = ins < len(sorted_evtid)
    matched = np.zeros(len(hit_evtid), dtype=bool)
    matched[within] = sorted_evtid[ins[within]] == hit_evtid[within]

    result = np.full((len(hit_evtid), 3), np.nan)
    vi = sort_idx[ins[matched]]
    result[matched, 0] = vx[vi]
    result[matched, 1] = vy[vi]
    result[matched, 2] = vz[vi]
    return result


def _outward_normal_sphere(
    hit_pos: np.ndarray,
    inner_sphere_hits: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Normal vectors pointing toward the active volume (sphere interior).

    For outer sphere hits: n_active = -r_hat (toward centre).
    For inner sphere hits: n_active = +r_hat (toward annular active volume).
    inner_sphere_hits: boolean mask, True if hit is on the inner sphere.
    """
    r = np.linalg.norm(hit_pos, axis=1, keepdims=True)
    r = np.maximum(r, 1e-9)
    r_hat = hit_pos / r                     # outward unit vector from centre
    n_active = -r_hat                       # default: toward sphere interior

    if inner_sphere_hits is not None and np.any(inner_sphere_hits):
        n_active[inner_sphere_hits] = r_hat[inner_sphere_hits]  # toward annular region

    return n_active


def apply_validity_filter(
    data: dict,
    geometry: str,
) -> tuple[np.ndarray, dict]:
    """
    Flag invalid photons.

    Invalid if:
      a) emission vertex is outside the active volume, OR
      b) momentum direction at the hit implies arrival from outside
         (dot-product with active-volume-pointing normal > 0)

    Returns
    -------
    valid_mask : boolean array (True = keep)
    stats      : dict with counts
    """
    n = len(data["x"])
    valid = np.ones(n, dtype=bool)

    hit_pos = np.column_stack([data["x"], data["y"], data["z"]])
    momentum = np.column_stack([data["px"], data["py"], data["pz"]])

    # ── Look up emission vertex for each optical hit ───────────────────────
    vertex_pos = _vertex_position(data["evtid"], data)
    vertex_found = ~np.any(np.isnan(vertex_pos), axis=1)
    n_no_vertex = int((~vertex_found).sum())
    if n_no_vertex:
        print(f"  Warning: {n_no_vertex} optical hits have no matching emission vertex.")

    n_invalid_vertex = 0
    momentum_invalid = np.zeros(n, dtype=bool)

    if geometry == "sphere":
        r_vert = np.linalg.norm(vertex_pos, axis=1)
        vertex_outside = vertex_found & (r_vert > SPHERE_INNER_RADIUS_MM)
        valid[vertex_outside] = False
        n_invalid_vertex = int(vertex_outside.sum())

        n_active = _outward_normal_sphere(hit_pos)
        dot = np.sum(momentum * n_active, axis=1)
        momentum_invalid = valid & (dot > 0)
        # counted for diagnostics only — not removed from valid

    elif geometry == "sphere_in_sphere":
        r_vert = np.linalg.norm(vertex_pos, axis=1)
        vertex_outside = vertex_found & (
            (r_vert < INNER_SPHERE_RADIUS_MM) | (r_vert > SPHERE_INNER_RADIUS_MM)
        )
        valid[vertex_outside] = False
        n_invalid_vertex = int(vertex_outside.sum())

        inner_mask = data["det_uid"] == INNER_SPHERE_DETECTOR_ID
        n_active = _outward_normal_sphere(hit_pos, inner_sphere_hits=inner_mask)
        dot = np.sum(momentum * n_active, axis=1)
        momentum_invalid = valid & (dot > 0)
        # counted for diagnostics only — not removed from valid

    elif geometry == "wall":
        # Source is on the -z side of the wall; photons travel toward +z.
        # (Matches createGeometry.py: wall PMTs face -z, so the active volume
        #  — and hence the photon source — lies on the -z side of the wall.)
        vert_z = vertex_pos[:, 2]
        vertex_outside = vertex_found & (vert_z > -WALL_Z_MM / 2.0)
        valid[vertex_outside] = False
        n_invalid_vertex = int(vertex_outside.sum())

        # Outside-in = arriving from the +z side, i.e. travelling toward -z.
        dot = -data["pz"]
        momentum_invalid = valid & (dot > 0)
        # counted for diagnostics only — not removed from valid

    else:
        raise ValueError(f"Unknown geometry: {geometry!r}")

    # Per-surface breakdown of outside-in hits
    outside_per_surface: dict[int, int] = {}
    if momentum_invalid.any():
        for uid in np.unique(data["det_uid"][momentum_invalid]):
            outside_per_surface[int(uid)] = int(
                (momentum_invalid & (data["det_uid"] == uid)).sum()
            )

    stats = {
        "total": n,
        "valid": int(valid.sum()),
        "invalid_vertex": n_invalid_vertex,
        "invalid_momentum": int(momentum_invalid.sum()),
        "no_vertex_match": n_no_vertex,
        "outside_per_surface": outside_per_surface,
    }
    return valid, stats


# ═════════════════════════ GDML Reflectivity Parsing ══════════════════════════

def parse_gdml_matrices(gdml_path: Path) -> dict[str, np.ndarray]:
    """
    Parse all matrix entries from the GDML <define> section.

    Returns dict: matrix_name → Nx2 array of (energy_eV, value).
    Energy is the raw eV value from the file (not yet converted to wavelength).
    """
    tree = ET.parse(gdml_path)
    root = tree.getroot()
    define = root.find("define")
    if define is None:
        raise ValueError(f"No <define> section found in {gdml_path}")

    matrices: dict[str, np.ndarray] = {}
    for mat in define.findall("matrix"):
        name = mat.get("name", "")
        values_str = mat.get("values", "").strip()
        if not values_str:
            continue

        nums: list[float] = []
        for tok in values_str.split():
            if "*eV" in tok:
                nums.append(float(tok.replace("*eV", "")))
            else:
                try:
                    nums.append(float(tok))
                except ValueError:
                    pass

        if len(nums) % 2 != 0:
            print(f"  Warning: odd token count in matrix '{name}', skipping.")
            continue

        matrices[name] = np.array(nums).reshape(-1, 2)

    return matrices


def extract_reflectivity(
    matrices: dict[str, np.ndarray],
    surface_keyword: str = "REFLECTIVITY",
    exclude_keywords: tuple[str, ...] = ("pmt", "cathode", "borosilicate"),
) -> tuple[str, np.ndarray]:
    """
    Find the Tyvek/SSD surface REFLECTIVITY matrix, convert eV → nm.

    Returns (matrix_name, Nx2 array of (wavelength_nm, reflectivity)).
    """
    candidates = {
        name: data
        for name, data in matrices.items()
        if surface_keyword in name.upper()
        and not any(kw in name.lower() for kw in exclude_keywords)
    }

    if not candidates:
        raise ValueError(
            f"No REFLECTIVITY matrix found (excluding {exclude_keywords}). "
            f"Available: {list(matrices)}"
        )
    if len(candidates) > 1:
        print(f"  Multiple REFLECTIVITY matrices found: {list(candidates)}")
        print(f"  Using first: {next(iter(candidates))}")

    name, data = next(iter(candidates.items()))
    energy_ev = data[:, 0]
    reflectivity = data[:, 1]
    wavelength_nm = HC_EV_NM / energy_ev

    # Sort ascending wavelength
    idx = np.argsort(wavelength_nm)
    return name, np.column_stack([wavelength_nm[idx], reflectivity[idx]])


def compute_expected_ssd_reflectivity(
    wavelengths_nm: np.ndarray,
    geometry: str,
    n_pmts: int,
) -> np.ndarray:
    """
    Compute R_SSD(λ) = (A_PMT/A_total)·R_cathode + (1 - A_PMT/A_total)·R_Tyvek(λ).

    R_Tyvek(λ) is linearly interpolated from TYVEK_REF_TABLE.
    """
    a_pmt = n_pmts * np.pi * PMT_CATHODE_RADIUS_MM**2  # mm²
    a_total = _geometry_area_mm2(geometry, n_pmts)
    fraction_pmt = a_pmt / a_total

    # Interpolate reference Tyvek values at requested wavelengths
    r_tyvek = np.interp(
        wavelengths_nm,
        TYVEK_REF_TABLE[:, 0],
        TYVEK_REF_TABLE[:, 1],
        left=TYVEK_REF_TABLE[0, 1],
        right=TYVEK_REF_TABLE[-1, 1],
    )
    r_ssd = fraction_pmt * PMT_CATHODE_R + (1.0 - fraction_pmt) * r_tyvek
    return r_ssd


def _geometry_area_mm2(geometry: str, n_pmts: int = 0) -> float:
    """Total surface area of the detector in mm²."""
    if geometry in ("sphere", "sphere_in_sphere"):
        return 4.0 * np.pi * SPHERE_INNER_RADIUS_MM**2
    if geometry == "wall":
        return WALL_X_MM * WALL_Y_MM
    raise ValueError(f"Unknown geometry: {geometry!r}")


def analyze_gdml_reflectivity(
    gdml_path: Path,
    geometry: str,
    n_pmts: int,
    output_dir: Path,
) -> None:
    """Section 2.2 — Parse GDML and verify SSD reflectivity against formula."""
    print("\n── 2.2  GDML Reflectivity Verification ──────────────────────────────")

    matrices = parse_gdml_matrices(gdml_path)

    # Locate SSD (Tyvek-like) reflectivity
    try:
        ssd_name, ssd_refl = extract_reflectivity(matrices)
        print(f"  Found SSD reflectivity matrix: '{ssd_name}'")
    except ValueError as e:
        print(f"  ERROR: {e}")
        print("  Available matrices: " + ", ".join(matrices))
        return

    # Also show Tyvek reference if present
    ref_name, ref_refl = None, None
    for name, data in matrices.items():
        if "tyvek" in name.lower() and "REFLECTIVITY" in name.upper() and name != ssd_name:
            wl = HC_EV_NM / data[:, 0]
            idx = np.argsort(wl)
            ref_name, ref_refl = name, np.column_stack([wl[idx], data[idx, 1]])
            break

    # Verify Tyvek reference values (first check: is this a standard Tyvek GDML?)
    if ref_refl is not None:
        print(f"  Found Tyvek reference matrix: '{ref_name}'")
        print(f"  {'Wavelength (nm)':<18} {'GDML R':<12} {'Reference':<12} {'Match?'}")
        for wl_ref, r_ref in TYVEK_REF_TABLE:
            r_gdml = float(np.interp(wl_ref, ref_refl[:, 0], ref_refl[:, 1]))
            ok = "✓" if abs(r_gdml - r_ref) < 0.001 else "✗"
            print(f"  {wl_ref:<18.0f} {r_gdml:<12.4f} {r_ref:<12.4f} {ok}")

    # Compute expected R_SSD at the GDML wavelength points
    wl_pts = ssd_refl[:, 0]
    r_gdml = ssd_refl[:, 1]
    r_expected = compute_expected_ssd_reflectivity(wl_pts, geometry, n_pmts)

    a_pmt = n_pmts * np.pi * PMT_CATHODE_RADIUS_MM**2
    a_total = _geometry_area_mm2(geometry)
    frac = a_pmt / a_total

    print(f"\n  A_PMT / A_total = {frac:.6f}  (N_PMT={n_pmts}, R_cathode={PMT_CATHODE_R:.6f})")
    print(f"\n  {'λ (nm)':<12} {'R_SSD GDML':<15} {'R_SSD expected':<18} {'Δ':<10}")
    for wl, r_g, r_e in zip(wl_pts, r_gdml, r_expected):
        print(f"  {wl:<12.1f} {r_g:<15.6f} {r_e:<18.6f} {r_g - r_e:<+10.6f}")

    max_dev = float(np.max(np.abs(r_gdml - r_expected)))
    print(f"\n  Max deviation: {max_dev:.2e}  {'OK' if max_dev < 0.01 else 'WARNING: deviation > 0.01'}")

    # Plot
    wl_fine = np.linspace(240, 820, 300)
    r_fine_expected = compute_expected_ssd_reflectivity(wl_fine, geometry, n_pmts)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(wl_fine, r_fine_expected, "b-", label="$R_{SSD}$ (formula)", linewidth=1.5)
    ax.scatter(wl_pts, r_gdml, c="red", s=60, zorder=5, label="$R_{SSD}$ (GDML)")
    if ref_refl is not None:
        ax.plot(ref_refl[:, 0], ref_refl[:, 1], "k--", alpha=0.5,
                label="$R_{Tyvek}$ (GDML)")
    ax.axhline(PMT_CATHODE_R, color="green", linestyle=":", alpha=0.7,
               label=f"$R_{{cathode}}={PMT_CATHODE_R:.3f}$")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectivity")
    ax.set_title(f"SSD Reflectivity Verification — {geometry}, $N_{{PMT}}={n_pmts}$")
    ax.legend(fontsize=9)
    ax.set_xlim(240, 820)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = output_dir / "reflectivity_verification.png"
    plt.savefig(out, dpi=200)
    plt.close(fig)
    print(f"\n  Saved: {out}")


# ═══════════════════════ Total Hit Comparison ════════════════════════════════

def analyze_total_hits(
    ssd_data: dict,
    ssd_valid: np.ndarray,
    pmt_data: dict,
    pmt_valid: np.ndarray,
    geometry: str,
    output_dir: Path,
    pmt_notyvek_data: Optional[dict] = None,
    pmt_notyvek_valid: Optional[np.ndarray] = None,
) -> dict:
    """Section 2.3 — Compare total hits SSD vs. PMT+Tyvek."""
    print("\n── 2.3  Total Hit Comparison ─────────────────────────────────────────")

    # SSD: all hits on SSD surface
    ssd_hits = ssd_valid & (ssd_data["det_uid"] == TYVEK_DETECTOR_ID)
    n_ssd = int(ssd_hits.sum())

    # PMT+Tyvek: all hits on Tyvek wall + PMT cathodes
    pmt_all_mask = pmt_valid & (
        (pmt_data["det_uid"] == TYVEK_DETECTOR_ID) |
        (pmt_data["det_uid"] >= PMT_DETECTOR_ID_BASE)
    )
    n_pmt_all = int(pmt_all_mask.sum())

    # PMT+Tyvek: PMT cathode hits only
    pmt_cathode_mask = pmt_valid & (pmt_data["det_uid"] >= PMT_DETECTOR_ID_BASE)
    n_pmt_cathode = int(pmt_cathode_mask.sum())

    eps_global = n_pmt_cathode / n_ssd if n_ssd > 0 else float("nan")

    print(f"  SSD total hits:             {n_ssd}")
    print(f"  PMT+Tyvek total hits:       {n_pmt_all}")
    print(f"  PMT cathode hits:           {n_pmt_cathode}")
    print(f"  ε_global = N_PMT / N_SSD = {eps_global:.6f}")

    results = {"n_ssd": n_ssd, "n_pmt_all": n_pmt_all, "n_pmt_cathode": n_pmt_cathode, "eps_global": eps_global}

    # Inner sphere check (sphere_in_sphere)
    if geometry == "sphere_in_sphere":
        inner_ssd = ssd_valid & (ssd_data["det_uid"] == INNER_SPHERE_DETECTOR_ID)
        inner_pmt = pmt_valid & (pmt_data["det_uid"] == INNER_SPHERE_DETECTOR_ID)
        n_inner_ssd = int(inner_ssd.sum())
        n_inner_pmt = int(inner_pmt.sum())
        ratio = n_inner_pmt / n_inner_ssd if n_inner_ssd > 0 else float("nan")
        print(f"\n  Inner sphere hits (SSD):       {n_inner_ssd}")
        print(f"  Inner sphere hits (PMT+Tyvek): {n_inner_pmt}")
        print(f"  Ratio (should be ≈ 1):         {ratio:.4f}")
        results.update({
            "n_inner_ssd": n_inner_ssd,
            "n_inner_pmt": n_inner_pmt,
            "inner_sphere_ratio": ratio,
        })

        # Inner sphere comparison plot
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.bar(["SSD", "PMT+Tyvek"], [n_inner_ssd, n_inner_pmt],
               color=["steelblue", "orange"], alpha=0.8)
        ax.set_ylabel("Inner sphere hits")
        ax.set_title(f"Inner sphere hits (UID {INNER_SPHERE_DETECTOR_ID}): SSD vs. PMT+Tyvek\n(should be identical)")
        ax.text(0.5, 0.98, f"ratio = {ratio:.4f}",
                ha="center", va="top", transform=ax.transAxes)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        out = output_dir / "inner_sphere_hits_comparison.png"
        plt.savefig(out, dpi=200)
        plt.close(fig)
        print(f"  Saved: {out}")

    # Optionally include pmt_no_tyvek
    datasets = {
        f"SSD\n(UID {TYVEK_DETECTOR_ID})": n_ssd,
        "PMT+Tyvek\n(total)": n_pmt_all,
        "PMT cathode\nonly": n_pmt_cathode,
    }
    n_notyvek: Optional[int] = None
    if pmt_notyvek_data is not None and pmt_notyvek_valid is not None:
        nt_hits = pmt_notyvek_valid & (pmt_notyvek_data["det_uid"] >= PMT_DETECTOR_ID_BASE)
        n_notyvek = int(nt_hits.sum())
        datasets["PMT only\n(no Tyvek)"] = n_notyvek
        eps_notyvek = n_notyvek / n_ssd if n_ssd > 0 else float("nan")
        print(f"  PMT-only total hits:        {n_notyvek}  (ε = {eps_notyvek:.6f})")
        results.update({"n_pmt_notyvek": n_notyvek, "eps_pmt_notyvek": eps_notyvek})

    # Bar chart
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["steelblue", "darkorange", "orange", "green"][: len(datasets)]
    ax.bar(list(datasets.keys()), list(datasets.values()), color=colors, alpha=0.8)
    for i, (label, n) in enumerate(datasets.items()):
        ax.text(i, n + max(datasets.values()) * 0.01, str(n),
                ha="center", fontsize=10)
    ax.set_ylabel("Valid optical hits")
    ax.set_title(f"Total hit comparison — {geometry}")
    ax.grid(axis="y", alpha=0.3)
    ax.text(0.98, 0.97,
            f"$\\epsilon_{{global}}$ = {eps_global:.4f}",
            ha="right", va="top", transform=ax.transAxes, fontsize=11)
    plt.tight_layout()
    out = output_dir / "total_hits_comparison.png"
    plt.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  Saved: {out}")

    return results


# ═══════════════════════ Per-Voxel Efficiency ════════════════════════════════

def _map_hits_to_voxels_wall(
    x: np.ndarray,
    y: np.ndarray,
    all_voxels: list[dict],
) -> np.ndarray:
    """Return voxel list-index for each hit based on grid position."""
    if not all_voxels:
        return np.full(len(x), -1, dtype=int)

    # Read grid parameters from first voxel
    v0 = all_voxels[0]
    n_x: int = v0.get("n_x", 10)
    n_y: int = v0.get("n_y", 10)
    vox_x: float = v0["size"][0] if "size" in v0 else WALL_X_MM / n_x
    vox_y: float = v0["size"][2] if "size" in v0 else WALL_Y_MM / n_y
    x_min = -WALL_X_MM / 2.0
    y_min = -WALL_Y_MM / 2.0

    # Build index lookup: (ix, iy) → list position
    lookup: dict[tuple[int, int], int] = {}
    for pos, v in enumerate(all_voxels):
        ix = v.get("grid_ix", int(v["index"][:2]))
        iy = v.get("grid_iy", int(v["index"][2:]))
        lookup[(ix, iy)] = pos

    ix_arr = np.clip(np.floor((x - x_min) / vox_x).astype(int), 0, n_x - 1)
    iy_arr = np.clip(np.floor((y - y_min) / vox_y).astype(int), 0, n_y - 1)

    result = np.full(len(x), -1, dtype=int)
    for i in range(len(x)):
        result[i] = lookup.get((ix_arr[i], iy_arr[i]), -1)
    return result


def _map_hits_to_voxels_sphere(
    xyz: np.ndarray,
    all_voxels: list[dict],
) -> np.ndarray:
    """Return voxel list-index for each hit using KD-tree on voxel centres."""
    centers = np.array([v["center"] for v in all_voxels])
    tree = cKDTree(centers)
    _, indices = tree.query(xyz)
    return indices.astype(int)


def analyze_per_voxel_efficiency(
    ssd_data: dict,
    ssd_valid: np.ndarray,
    pmt_data: dict,
    pmt_valid: np.ndarray,
    all_voxels: list[dict],
    selected_voxels: list[dict],
    geometry: str,
    output_dir: Path,
) -> dict:
    """Section 2.4 — Per-voxel hit counts and efficiency correction."""
    print("\n── 2.4  Per-Voxel Efficiency Analysis ───────────────────────────────")

    if not selected_voxels:
        print("  No PMT voxels loaded — skipping per-voxel analysis.")
        return {}

    # ── Map SSD hits to voxels ─────────────────────────────────────────────
    ssd_surface_mask = ssd_valid & (ssd_data["det_uid"] == TYVEK_DETECTOR_ID)
    ssd_xyz = np.column_stack([ssd_data["x"], ssd_data["y"], ssd_data["z"]])
    ssd_hit_pos = ssd_xyz[ssd_surface_mask]

    if geometry == "wall":
        ssd_voxel_idx = _map_hits_to_voxels_wall(
            ssd_hit_pos[:, 0], ssd_hit_pos[:, 1], all_voxels
        )
    else:
        ssd_voxel_idx = _map_hits_to_voxels_sphere(ssd_hit_pos, all_voxels)

    # Hit count per voxel (all voxels)
    ssd_voxel_counts = np.zeros(len(all_voxels), dtype=int)
    for idx in ssd_voxel_idx:
        if 0 <= idx < len(all_voxels):
            ssd_voxel_counts[idx] += 1

    # ── Map PMT hits by det_uid to selected voxels ─────────────────────────
    pmt_cathode_mask = pmt_valid & (pmt_data["det_uid"] >= PMT_DETECTOR_ID_BASE)
    pmt_uid = pmt_data["det_uid"][pmt_cathode_mask]
    pmt_rawids = sorted(set(int(u) for u in pmt_uid))
    pmt_counts_by_rawid: dict[int, int] = {
        uid: int((pmt_uid == uid).sum()) for uid in pmt_rawids
    }

    # Match PMT rawid = PMT_DETECTOR_ID_BASE + i to selected_voxels[i]
    n_pmt_vox = len(selected_voxels)
    pmt_counts: np.ndarray = np.zeros(n_pmt_vox, dtype=int)
    for i, sv in enumerate(selected_voxels):
        expected_rawid = PMT_DETECTOR_ID_BASE + int(sv["index"])
        pmt_counts[i] = pmt_counts_by_rawid.get(expected_rawid, 0)

    # SSD counts at PMT voxel positions
    # Find list-index of each selected voxel in all_voxels
    all_indices_by_index = {v["index"]: pos for pos, v in enumerate(all_voxels)}
    ssd_at_pmt: np.ndarray = np.zeros(n_pmt_vox, dtype=int)
    for i, sv in enumerate(selected_voxels):
        pos = all_indices_by_index.get(sv["index"], -1)
        if pos >= 0:
            ssd_at_pmt[i] = ssd_voxel_counts[pos]

    eps_voxel = np.where(
        ssd_at_pmt > 0, pmt_counts.astype(float) / ssd_at_pmt, np.nan
    )
    eps_voxel_global = (
        float(pmt_counts.sum() / ssd_at_pmt.sum()) if ssd_at_pmt.sum() > 0 else float("nan")
    )

    print(f"  Number of PMT voxels:       {n_pmt_vox}")
    print(f"  SSD hits at PMT voxels:     {ssd_at_pmt.sum()}")
    print(f"  PMT hits total:             {pmt_counts.sum()}")
    print(f"  ε_voxel (global average):   {eps_voxel_global:.6f}")
    eps_finite = eps_voxel[np.isfinite(eps_voxel)]
    if len(eps_finite):
        print(f"  ε_voxel range:              [{eps_finite.min():.4f}, {eps_finite.max():.4f}]")
        print(f"  ε_voxel std:                {eps_finite.std():.4f}")

    # ── Per-voxel bar plot ─────────────────────────────────────────────────
    labels = [sv["index"] for sv in selected_voxels]
    x_pos = np.arange(n_pmt_vox)

    fig, axes = plt.subplots(2, 1, figsize=(max(10, n_pmt_vox * 0.4), 10))

    ax = axes[0]
    bar_w = 0.4
    ax.bar(x_pos - bar_w / 2, ssd_at_pmt, bar_w, label=f"SSD, UID {TYVEK_DETECTOR_ID} (at PMT voxel)", color="steelblue", alpha=0.8)
    ax.bar(x_pos + bar_w / 2, pmt_counts,   bar_w, label="PMT cathode",                               color="orange",   alpha=0.8)
    # Average hit counts over the PMT voxels (unweighted means) — shows whether
    # the SSD surface and the PMT cathodes record the same number of hits.
    avg_ssd_at_pmt = float(ssd_at_pmt.mean()) if n_pmt_vox else 0.0
    avg_pmt = float(pmt_counts.mean()) if n_pmt_vox else 0.0
    ax.axhline(avg_ssd_at_pmt, color="steelblue", linestyle="--", linewidth=1.5,
               label=f"mean SSD / voxel = {avg_ssd_at_pmt:.1f}")
    ax.axhline(avg_pmt, color="darkorange", linestyle="--", linewidth=1.5,
               label=f"mean PMT / PMT = {avg_pmt:.1f}")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Hit count")
    ax.set_title(f"Per-voxel hits — {geometry}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    ax2.scatter(x_pos, eps_voxel, color="purple", alpha=0.7, s=30, zorder=3)
    ax2.axhline(eps_voxel_global, color="red", linestyle="--",
                label=f"$\\epsilon_{{voxel}}$ = {eps_voxel_global:.4f}")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax2.set_ylabel("$\\epsilon_{voxel}$")
    ax2.set_title("Per-voxel efficiency correction")
    # Zoom y-axis to the relevant range of (finite) per-voxel efficiencies
    if len(eps_finite):
        lo, hi = float(eps_finite.min()), float(eps_finite.max())
        pad = (hi - lo) * 0.1 if hi > lo else max(abs(hi) * 0.1, 1e-3)
        ax2.set_ylim(lo - pad, hi + pad)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = output_dir / "per_voxel_efficiency.png"
    plt.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  Saved: {out}")

    return {
        "eps_voxel_global": eps_voxel_global,
        "eps_voxel_values": eps_voxel.tolist(),
        "pmt_counts": pmt_counts.tolist(),
        "ssd_at_pmt_counts": ssd_at_pmt.tolist(),
        "voxel_indices": labels,
    }


# ═══════════════════════════ Validity Report Plot ════════════════════════════

def plot_validity_report(
    ssd_stats: dict,
    pmt_stats: dict,
    geometry: str,
    output_dir: Path,
    pmt_notyvek_stats: Optional[dict] = None,
) -> None:
    """Print validity filter summary and save plots."""
    print("\n── 2.1  Photon Validity Filter Summary ───────────────────────────────")
    all_stats = [("SSD", ssd_stats), ("PMT+Tyvek", pmt_stats)]
    if pmt_notyvek_stats is not None:
        all_stats.append(("PMT only", pmt_notyvek_stats))

    for label, stats in all_stats:
        n = stats["total"]
        v = stats["valid"]
        pct = v / n * 100 if n else 0.0
        print(f"  {label}: {v}/{n} valid ({pct:.1f}%),  "
              f"invalid vertex={stats['invalid_vertex']},  "
              f"invalid momentum={stats['invalid_momentum']}")
        if stats["invalid_momentum"] > 0:
            print(f"  Note: {label}: {stats['invalid_momentum']} photons hit surface "
                  f"from outside (not filtered, see outside_in_hits.png)")
            per_surf = stats.get("outside_per_surface", {})
            for uid, cnt in sorted(per_surf.items()):
                print(f"      det_uid={uid}: {cnt} outside-in hits")

    labels = [s[0] for s in all_stats]
    valids = [s[1]["valid"] for s in all_stats]
    totals = [s[1]["total"] for s in all_stats]

    # Main bar chart
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(labels))
    ax.bar(x, totals, 0.4, label="Total simulated", color="gray", alpha=0.5)
    ax.bar(x, valids, 0.4, label="Valid", color="steelblue", alpha=0.9)
    # Annotate each bar with its hit count
    ymax = max(totals) if totals else 0
    for xi, tot, val in zip(x, totals, valids):
        ax.text(xi, tot + ymax * 0.01, f"{tot:,}", ha="center", va="bottom",
                fontsize=8, color="dimgray")
        ax.text(xi, val + ymax * 0.01, f"{val:,}", ha="center", va="bottom",
                fontsize=8, color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Photon count")
    ax.set_title(f"Photon validity filter — {geometry}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    uid_note = f"SSD/Tyvek surface: UID {TYVEK_DETECTOR_ID}"
    if geometry == "sphere_in_sphere":
        uid_note += f"  |  inner sphere: UID {INNER_SPHERE_DETECTOR_ID}"
    ax.text(0.02, 0.98, uid_note, transform=ax.transAxes,
            va="top", fontsize=7, color="dimgray")
    plt.tight_layout()
    out = output_dir / "validity_filter.png"
    plt.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════ Outside-in Hits Plot ════════════════════════════

def plot_outside_in_hits(
    ssd_stats: dict,
    pmt_stats: dict,
    geometry: str,
    output_dir: Path,
    pmt_notyvek_stats: Optional[dict] = None,
) -> None:
    """Bar chart of outside-in photon counts per surface and scenario (not filtered)."""
    print("\n── Outside-in Hits ───────────────────────────────────────────────────")

    all_stats = [("SSD", ssd_stats), ("PMT+Tyvek", pmt_stats)]
    if pmt_notyvek_stats is not None:
        all_stats.append(("PMT only", pmt_notyvek_stats))

    all_uids: set[int] = set()
    for _, stats in all_stats:
        all_uids.update(stats.get("outside_per_surface", {}).keys())

    if not all_uids:
        print("  No outside-in hits found in any scenario.")
        return

    all_uids_sorted = sorted(all_uids)

    fig, ax = plt.subplots(figsize=(max(8, len(all_uids_sorted) * 0.8), 5))
    bar_w = 0.8 / max(len(all_stats), 1)
    for j, (label, stats) in enumerate(all_stats):
        per_surf = stats.get("outside_per_surface", {})
        counts = [per_surf.get(uid, 0) for uid in all_uids_sorted]
        x = np.arange(len(all_uids_sorted)) + (j - len(all_stats) / 2) * bar_w
        ax.bar(x, counts, bar_w, label=label, alpha=0.8)

    ax.set_xticks(np.arange(len(all_uids_sorted)))
    ax.set_xticklabels([str(u) for u in all_uids_sorted], rotation=45)
    ax.set_xlabel("det_uid")
    ax.set_ylabel("Outside-in hits (not filtered)")
    ax.set_title(f"Photons hitting surface from outside — {geometry}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = output_dir / "outside_in_hits.png"
    plt.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  Saved: {out}")


# ════════════════════════════ 3D Voxel Plot ══════════════════════════════════

def plot_3d_voxels(
    all_voxels: list[dict],
    selected_voxels: list[dict],
    geometry: str,
    output_dir: Path,
) -> None:
    """Section 2.5 — 3D scatter plot with PMT voxels highlighted."""
    print("\n── 2.5  3D Voxel Plot ────────────────────────────────────────────────")

    sel_indices_set = {sv["index"] for sv in selected_voxels}

    all_centers = np.array([v["center"] for v in all_voxels])
    sel_mask = np.array([v["index"] in sel_indices_set for v in all_voxels])

    fig = plt.figure(figsize=(13, 11))
    ax = fig.add_subplot(111, projection="3d")

    # Transparent reference surface
    if geometry != "wall":
        r = SPHERE_INNER_RADIUS_MM
        u = np.linspace(0, 2 * np.pi, 40)
        v = np.linspace(0, np.pi, 40)
        ax.plot_surface(
            r * np.outer(np.cos(u), np.sin(v)),
            r * np.outer(np.sin(u), np.sin(v)),
            r * np.outer(np.ones(40), np.cos(v)),
            alpha=0.05, color="lightblue", edgecolor="none",
        )

    other = all_centers[~sel_mask]
    if len(other):
        ax.scatter(other[:, 0], other[:, 1], other[:, 2],
                   c="cornflowerblue", s=20, alpha=0.55,
                   label=f"All voxels, UID {TYVEK_DETECTOR_ID} ({(~sel_mask).sum()})")

    sel = all_centers[sel_mask]
    if len(sel):
        ax.scatter(sel[:, 0], sel[:, 1], sel[:, 2],
                   c="red", s=50, alpha=0.9,
                   edgecolors="darkred", linewidths=0.8,
                   label=f"PMT voxels ({sel_mask.sum()})")

    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    ax.set_title(
        f"Voxel grid — {geometry}\n"
        f"{len(all_voxels)} voxels, {len(selected_voxels)} PMTs"
    )
    ax.legend(fontsize=9)
    ax.set_box_aspect([1, 1, 1])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = output_dir / "voxels_3d.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════ Wavelength Histograms ═══════════════════════════

def plot_wavelength_histograms(
    ssd_data: dict,
    pmt_data: dict,
    ssd_dir: Path,
    geometry: str,
    output_dir: Path,
) -> None:
    """Side-by-side histograms: emitted photon spectrum vs detected spectrum."""
    print("\n── Wavelength Histograms ─────────────────────────────────────────────")

    bins = np.linspace(200, 800, 121)

    # ── Emitted photons (from particles table, SSD dataset) ──────────────────
    print("  Loading emitted-photon wavelengths from particles table …")
    emit_wl = load_particles_wavelength(ssd_dir)
    print(f"    {len(emit_wl)} emitted photons in 200–800 nm range")

    # ── Detected photons ─────────────────────────────────────────────────────
    ssd_surface = ssd_data["det_uid"] == TYVEK_DETECTOR_ID
    ssd_wl = ssd_data["wavelength"][ssd_surface]

    pmt_cathode = pmt_data["det_uid"] >= PMT_DETECTOR_ID_BASE
    pmt_wl = pmt_data["wavelength"][pmt_cathode]

    print(f"    SSD surface detected:   {len(ssd_wl)}")
    print(f"    PMT cathode detected:   {len(pmt_wl)}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: emitted spectrum
    ax1.hist(emit_wl, bins=bins, color="steelblue", alpha=0.8)
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Counts")
    ax1.set_title("Emitted photon spectrum\n(vertex photons, SSD simulation)")
    ax1.set_xlim(200, 800)
    ax1.grid(True, alpha=0.3)

    # Right: detected spectra (SSD + PMT cathode overlaid)
    ax2.hist(ssd_wl, bins=bins, color="steelblue", alpha=0.7,
             label=f"SSD surface, UID {TYVEK_DETECTOR_ID} ({len(ssd_wl):,})")
    ax2.hist(pmt_wl, bins=bins, color="orange", alpha=0.7,
             label=f"PMT cathode ({len(pmt_wl):,})")
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Counts")
    ax2.set_title("Detected photon spectrum")
    ax2.set_xlim(200, 800)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Photon wavelength spectra — {geometry}", fontsize=13)
    plt.tight_layout()
    out = output_dir / "wavelength_histograms.png"
    plt.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════ Time Distributions ══════════════════════════════

def plot_time_distributions(
    ssd_data: dict,
    pmt_data: dict,
    geometry: str,
    output_dir: Path,
) -> None:
    """Hit-time histograms for outer and (if sphere_in_sphere) inner sphere."""
    print("\n── Time Distributions ────────────────────────────────────────────────")

    def _hist_pair(
        t_ssd: np.ndarray,
        t_pmt: np.ndarray,
        title_ssd: str,
        title_pmt: str,
        suptitle: str,
        filename: Path,
    ) -> None:
        t_min = min(
            t_ssd.min() if len(t_ssd) else 0.0,
            t_pmt.min() if len(t_pmt) else 0.0,
        )
        t_max = max(
            t_ssd.max() if len(t_ssd) else 1.0,
            t_pmt.max() if len(t_pmt) else 1.0,
        )
        bins = np.linspace(t_min, t_max, 101)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

        ax1.hist(t_ssd, bins=bins, color="steelblue", alpha=0.8)
        ax1.set_xlabel("Time (ns)")
        ax1.set_ylabel("Hit count")
        ax1.set_title(f"{title_ssd}\n{len(t_ssd):,} hits")
        ax1.grid(True, alpha=0.3)

        ax2.hist(t_pmt, bins=bins, color="orange", alpha=0.8)
        ax2.set_xlabel("Time (ns)")
        ax2.set_ylabel("Hit count")
        ax2.set_title(f"{title_pmt}\n{len(t_pmt):,} hits")
        ax2.grid(True, alpha=0.3)

        fig.suptitle(suptitle, fontsize=13)
        plt.tight_layout()
        plt.savefig(filename, dpi=200)
        plt.close(fig)
        print(f"  Saved: {filename}")

    # ── Outer sphere ──────────────────────────────────────────────────────────
    # SSD simulation: hits on SSD/Tyvek surface (UID 1000)
    ssd_outer_t = ssd_data["time"][ssd_data["det_uid"] == TYVEK_DETECTOR_ID]
    # PMT+Tyvek simulation: PMT cathode hits (UID ≥ 10000) together with
    # Tyvek wall hits (UID 1000).
    pmt_outer_t = pmt_data["time"][
        (pmt_data["det_uid"] >= PMT_DETECTOR_ID_BASE)
        | (pmt_data["det_uid"] == TYVEK_DETECTOR_ID)
    ]

    _hist_pair(
        ssd_outer_t, pmt_outer_t,
        title_ssd=f"SSD (UID {TYVEK_DETECTOR_ID})",
        title_pmt=f"PMT+Tyvek — cathode (UID ≥ {PMT_DETECTOR_ID_BASE}) + Tyvek (UID {TYVEK_DETECTOR_ID})",
        suptitle=f"Outer sphere hit time distribution — {geometry}",
        filename=output_dir / "time_distribution_outer.png",
    )

    # ── Inner sphere (sphere_in_sphere only) ──────────────────────────────────
    if geometry == "sphere_in_sphere":
        ssd_inner_t = ssd_data["time"][ssd_data["det_uid"] == INNER_SPHERE_DETECTOR_ID]
        pmt_inner_t = pmt_data["time"][pmt_data["det_uid"] == INNER_SPHERE_DETECTOR_ID]

        _hist_pair(
            ssd_inner_t, pmt_inner_t,
            title_ssd=f"SSD — inner sphere (UID {INNER_SPHERE_DETECTOR_ID})",
            title_pmt=f"PMT+Tyvek — inner sphere (UID {INNER_SPHERE_DETECTOR_ID})",
            suptitle=f"Inner sphere hit time distribution — {geometry}",
            filename=output_dir / "time_distribution_inner.png",
        )


# ═══════════════════════════════════ Main ════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SSD vs PMT+Tyvek efficiency analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--gdml",          type=Path, required=True,
                        help="Path to SSD GDML file")
    parser.add_argument("--data-ssd",      type=Path, required=True,
                        help="Directory with HDF5 files from SSD simulation")
    parser.add_argument("--data-pmt",      type=Path, required=True,
                        help="Directory with HDF5 files from PMT+Tyvek simulation")
    parser.add_argument("--voxel-json",    type=Path, required=True,
                        help="Path to all-voxels JSON (generated by createGeometry.py --variant ssd)")
    parser.add_argument("--geometry",
                        choices=["wall", "sphere", "sphere_in_sphere"], required=True,
                        help="Detector geometry type")
    parser.add_argument("--output-dir",    type=Path, required=True,
                        help="Directory for output plots and results JSON")
    parser.add_argument("--selected-json", type=Path, default=None,
                        help="Path to selected-voxels JSON (default: <voxel-json>_selected.json)")
    parser.add_argument("--data-pmt-no-tyvek", type=Path, default=None,
                        help="Optional: directory with PMT-only HDF5 files (wall geometry)")
    args = parser.parse_args()

    # ── Validate inputs ──────────────────────────────────────────────────────
    for path, name in [
        (args.gdml, "--gdml"),
        (args.data_ssd, "--data-ssd"),
        (args.data_pmt, "--data-pmt"),
        (args.voxel_json, "--voxel-json"),
    ]:
        if not path.exists():
            print(f"ERROR: {name} path not found: {path}", file=sys.stderr)
            sys.exit(1)

    selected_json = args.selected_json or Path(
        str(args.voxel_json).replace(".json", "_selected.json")
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("═" * 65)
    print("  SSD Efficiency Analysis")
    print("═" * 65)
    print(f"  Geometry:  {args.geometry}")
    print(f"  GDML:      {args.gdml}")
    print(f"  SSD data:  {args.data_ssd}")
    print(f"  PMT data:  {args.data_pmt}")
    print(f"  Voxels:    {args.voxel_json}")
    print(f"  Selected:  {selected_json}")
    print(f"  Output:    {args.output_dir}")

    # ── Load voxel grids ────────────────────────────────────────────────────
    print("\n── Loading voxel definitions ─────────────────────────────────────────")
    with open(args.voxel_json) as f:
        all_voxels: list[dict] = json.load(f)
    print(f"  All voxels:      {len(all_voxels)}")

    selected_voxels: list[dict] = []
    if selected_json.exists():
        with open(selected_json) as f:
            selected_voxels = json.load(f)
        print(f"  Selected voxels: {len(selected_voxels)}")
    else:
        print(f"  Warning: selected voxels file not found: {selected_json}")
        print("  Per-voxel analysis will be skipped.")

    n_pmts = len(selected_voxels)

    # ── Load HDF5 data + validity filter (one file at a time to cap peak RAM) ─
    print("\n── Loading HDF5 data and applying validity filters ───────────────────")
    print("  SSD scenario:")
    ssd_data, ssd_valid, ssd_stats = load_and_filter_hdf5_dir(
        args.data_ssd, args.geometry
    )

    print("  PMT+Tyvek scenario:")
    pmt_data, pmt_valid, pmt_stats = load_and_filter_hdf5_dir(
        args.data_pmt, args.geometry
    )

    pmt_notyvek_data: Optional[dict] = None
    pmt_notyvek_valid: Optional[np.ndarray] = None
    pmt_notyvek_stats: Optional[dict] = None
    if args.data_pmt_no_tyvek is not None and args.data_pmt_no_tyvek.exists():
        print("  PMT-only scenario:")
        pmt_notyvek_data, pmt_notyvek_valid, pmt_notyvek_stats = load_and_filter_hdf5_dir(
            args.data_pmt_no_tyvek, args.geometry
        )

    plot_validity_report(
        ssd_stats, pmt_stats, args.geometry, args.output_dir, pmt_notyvek_stats
    )
    plot_outside_in_hits(
        ssd_stats, pmt_stats, args.geometry, args.output_dir, pmt_notyvek_stats
    )

    # ── 2.2 GDML reflectivity verification ─────────────────────────────────
    analyze_gdml_reflectivity(args.gdml, args.geometry, n_pmts, args.output_dir)

    # ── 2.3 Total hit comparison ─────────────────────────────────────────────
    total_results = analyze_total_hits(
        ssd_data, ssd_valid, pmt_data, pmt_valid,
        args.geometry, args.output_dir,
        pmt_notyvek_data, pmt_notyvek_valid,
    )

    # ── 2.4 Per-voxel efficiency ─────────────────────────────────────────────
    voxel_results: dict = {}
    if selected_voxels:
        voxel_results = analyze_per_voxel_efficiency(
            ssd_data, ssd_valid, pmt_data, pmt_valid,
            all_voxels, selected_voxels, args.geometry, args.output_dir,
        )

    # ── 2.5 3D voxel plot ───────────────────────────────────────────────────
    plot_3d_voxels(all_voxels, selected_voxels, args.geometry, args.output_dir)

    # ── Wavelength histograms ────────────────────────────────────────────────
    plot_wavelength_histograms(
        ssd_data, pmt_data, args.data_ssd, args.geometry, args.output_dir
    )

    # ── Time distributions ───────────────────────────────────────────────────
    plot_time_distributions(
        ssd_data, pmt_data, args.geometry, args.output_dir
    )

    # ── Save summary JSON ───────────────────────────────────────────────────
    summary = {
        "geometry": args.geometry,
        "n_pmts": n_pmts,
        "validity_filter": {
            "ssd": ssd_stats,
            "pmt": pmt_stats,
        },
        "total_hits": total_results,
        "per_voxel": voxel_results,
    }
    if pmt_notyvek_stats:
        summary["validity_filter"]["pmt_notyvek"] = pmt_notyvek_stats

    summary_path = args.output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "═" * 65)
    print(f"  Analysis complete. Results saved to: {args.output_dir}")
    print(f"  Summary JSON: {summary_path}")
    print("═" * 65)


if __name__ == "__main__":
    main()
