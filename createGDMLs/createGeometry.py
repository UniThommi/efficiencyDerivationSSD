#!/usr/bin/env python3
"""
Unified GDML geometry creation for SSD efficiency validation.

Geometries: wall | sphere | sphere_in_sphere
Variants:
  ssd           - SSD surface (area-weighted reflectivity); generates voxel JSON
  pmt           - Tyvek + PMTs; reads voxel JSON, selects N homogeneous PMTs
  pmt_no_tyvek  - PMTs only, no Tyvek wall (wall geometry only)

Usage examples:
  # 1. Generate SSD sphere + voxel grid
  python createGeometry.py --geometry sphere --variant ssd --out sphere_ssd.gdml --voxels sphere_voxels.json

  # 2. Place PMTs on sphere (reads voxel grid, saves selection)
  python createGeometry.py --geometry sphere --variant pmt --n-pmts 50 \
      --out sphere_pmt.gdml --voxels sphere_voxels.json

  # 3. Wall without Tyvek (PMTs only)
  python createGeometry.py --geometry wall --variant pmt_no_tyvek --n-pmts 5 \
      --out wall_pmt_notyvek.gdml --voxels wall_voxels.json
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.append(os.path.abspath("./src"))
import pyg4ometry.geant4 as geant4
import pyg4ometry.gdml as gdml
from l1000geom import materials
from pygeomtools import RemageDetectorInfo, detectors, geometry, visualization

# ══════════════════════════════════ Constants ══════════════════════════════════

PMT_EFF_RADIUS: float = 131.0         # mm — PMT ellipsoid semi-axis
PMT_CATHODE_RADIUS: float = 110.0     # mm — effective cathode radius at z-cut
PMT_VACUUM_RADIUS: float = 128.0      # mm — vacuum inner radius (glass ~2-3 mm)
PMT_Z_BOTTOM_CUT: float = 41.0       # mm — lower z-cut of window ellipsoid
PMT_CATHODE_Z_CUT: float = 65.0      # mm — lower z-cut of cathode ellipsoid
PMT_Z_TOP_CUT: float = 200.0          # mm — upper z-cut

SPHERE_INNER_RADIUS: float = 1739.52  # mm
SPHERE_THICKNESS: float = 1.0         # mm
INNER_SPHERE_RADIUS: float = 1000.0   # mm — steel sphere (sphere_in_sphere only)

WALL_X: float = 1950.0                # mm
WALL_Y: float = 1950.0                # mm
WALL_Z: float = 1.0                   # mm — wall thickness

PMT_DETECTOR_ID_BASE: int = 10000     # rawid = PMT_DETECTOR_ID_BASE + i
TYVEK_DETECTOR_ID: int = 1000         # rawid for Tyvek / SSD surface
INNER_SPHERE_DETECTOR_ID: int = 9000  # rawid for inner steel sphere


# ══════════════════════════════ PMT Geometry ═══════════════════════════════════

def construct_pmt_front(
    vac_mat: geant4.Material,
    surfaces,
    reg: geant4.Registry,
) -> list:
    """Construct shared PMT solid geometry (window, vacuum, cathode)."""
    pmt_window = geant4.solid.Ellipsoid(
        "PMT_window",
        PMT_EFF_RADIUS, PMT_EFF_RADIUS, PMT_EFF_RADIUS,
        PMT_Z_BOTTOM_CUT, PMT_Z_TOP_CUT,
        reg, "mm",
    )
    pmt_vacuum = geant4.solid.Ellipsoid(
        "PMT_vacuum",
        PMT_VACUUM_RADIUS, PMT_VACUUM_RADIUS, PMT_EFF_RADIUS - 2.0,
        PMT_Z_BOTTOM_CUT, PMT_Z_TOP_CUT,
        reg, "mm",
    )
    pmt_cathode = geant4.solid.Ellipsoid(
        "PMT_cathode",
        PMT_VACUUM_RADIUS, PMT_VACUUM_RADIUS, PMT_EFF_RADIUS - 2.0,
        PMT_CATHODE_Z_CUT, PMT_Z_TOP_CUT,
        reg, "mm",
    )
    pmt_cathode_lv = geant4.LogicalVolume(pmt_cathode, vac_mat, "PMT_cathode", reg)
    pmt_cathode_lv.pygeom_color_rgba = [0.545, 0.271, 0.074, 1.0]
    geant4.SkinSurface("pmt_cathode_surface", pmt_cathode_lv, surfaces.to_photocathode, reg)
    return [pmt_window, pmt_vacuum, pmt_cathode_lv]


def get_euler_angles(target_direction: np.ndarray) -> list:
    """
    Compute XYZ Euler angles (radians) that rotate [0,0,1] → target_direction.

    Uses cross(from, to) rotation axis convention, consistent across all geometries.
    """
    default = np.array([0.0, 0.0, 1.0])
    target = np.asarray(target_direction, dtype=float)
    target = target / np.linalg.norm(target)

    cos_angle = float(np.clip(np.dot(default, target), -1.0, 1.0))
    angle = np.arccos(cos_angle)

    if angle < 1e-6:
        return [0.0, 0.0, 0.0]
    if abs(angle - np.pi) < 1e-6:
        return [float(np.pi), 0.0, 0.0]  # 180° around x-axis

    axis = np.cross(default, target)  # correct: cross(from, to)
    axis /= np.linalg.norm(axis)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        euler = Rotation.from_rotvec(axis * angle).as_euler("xyz")
    return [float(e) for e in euler]


def place_pmt_front(
    rotation: list,
    translation: list,
    pmt_volumes: list,
    name: str,
    mother_lv: geant4.LogicalVolume,
    reg: geant4.Registry,
    mat_registry,
    surface_registry,
    rawid: int,
) -> geant4.PhysicalVolume:
    """Instantiate unique PMT logical volumes and place into mother volume."""
    window_lv = geant4.LogicalVolume(
        pmt_volumes[0], mat_registry.borosilicate, name + "_window", reg
    )
    window_lv.pygeom_color_rgba = [0.9, 0.8, 0.5, 0.5]
    vacuum_lv = geant4.LogicalVolume(
        pmt_volumes[1], mat_registry.vacuum, name + "_vacuum", reg
    )
    geant4.PhysicalVolume([0, 0, 0], [0, 0, 0], vacuum_lv, name + "_vacuum", window_lv, reg)
    cathode_pv = geant4.PhysicalVolume(
        [0, 0, 0], [0, 0, 0], pmt_volumes[2], name, vacuum_lv, reg
    )
    cathode_pv.pygeom_active_detector = RemageDetectorInfo("optical", rawid)
    return geant4.PhysicalVolume(rotation, translation, window_lv, name + "_window", mother_lv, reg)


def _sphere_pmt_placement(voxel: dict) -> tuple[list, list]:
    """Compute rotation and translation for a sphere PMT at given voxel center."""
    cx, cy, cz = voxel["center"]
    r = float(np.linalg.norm([cx, cy, cz]))
    nx, ny, nz = cx / r, cy / r, cz / r

    # PMT should face inward: cathode top (local +z) points toward sphere center
    target_direction = np.array([-nx, -ny, -nz])

    # Shift PMT outward so window bottom clears the Tyvek wall (approximate)
    cutoff_r = np.sqrt(PMT_EFF_RADIUS**2 - PMT_Z_BOTTOM_CUT**2)
    radial_shift = SPHERE_INNER_RADIUS - np.sqrt(SPHERE_INNER_RADIUS**2 - cutoff_r**2)
    offset = PMT_Z_BOTTOM_CUT - radial_shift
    x = cx + nx * offset
    y = cy + ny * offset
    z = cz + nz * offset

    return get_euler_angles(target_direction), [x, y, z]


def _wall_pmt_placement(voxel: dict) -> tuple[list, list]:
    """Compute rotation and translation for a wall PMT at given voxel center."""
    cx, cy, cz = voxel["center"]
    # PMT faces -z (toward wall from above), center shifts so window bottom is at wall face
    target_direction = np.array([0.0, 0.0, -1.0])
    z = cz - (PMT_Z_BOTTOM_CUT - 0.5)  # place window bottom at z = cz - 0.5 (wall surface)
    return get_euler_angles(target_direction), [cx, cy, z]


# ══════════════════════════════ Voxel Generation ══════════════════════════════

def generate_sphere_voxels(
    radius: float = SPHERE_INNER_RADIUS,
    pmt_radius: float = PMT_CATHODE_RADIUS,
) -> list[dict]:
    """
    Generate approximately equal-area voxels on a sphere surface.

    Target voxel area equals one PMT cathode area (pi * pmt_radius^2).
    Uses equal solid-angle latitude bands with uniform longitude spacing.
    """
    pmt_area = np.pi * pmt_radius**2
    sphere_area = 4.0 * np.pi * radius**2
    target_count = int(round(sphere_area / pmt_area))

    # Equal-area latitude bands: each band covers 1/n_lat of solid angle
    n_lat = int(round(np.sqrt(target_count / 2.0)))
    n_lon = int(round(target_count / n_lat))
    total = n_lat * n_lon

    print(f"Sphere voxel grid: {n_lat} lat × {n_lon} lon = {total} voxels "
          f"(target {target_count}, radius {radius:.1f} mm, PMT cathode R={pmt_radius:.0f} mm)")

    i_arr = np.arange(n_lat)
    theta_arr = np.arccos(1.0 - 2.0 * (i_arr + 0.5) / n_lat)
    phi_arr = np.linspace(0.0, 2.0 * np.pi, n_lon, endpoint=False)

    voxels: list[dict] = []
    idx = 0
    for t in theta_arr:
        for p in phi_arr:
            voxels.append({
                "index": f"{idx:06d}",
                "center": [
                    float(radius * np.sin(t) * np.cos(p)),
                    float(radius * np.sin(t) * np.sin(p)),
                    float(radius * np.cos(t)),
                ],
                "layer": "sphere",
            })
            idx += 1
    return voxels


def generate_wall_voxels(
    wall_x: float = WALL_X,
    wall_y: float = WALL_Y,
    pmt_radius: float = PMT_CATHODE_RADIUS,
) -> list[dict]:
    """
    Generate a uniform rectangular grid of voxels on the wall surface.

    Each voxel's area approximates one PMT cathode area (pi * pmt_radius^2).
    Voxels are centered in the xy-plane at z = 0 (wall center in GDML coordinates).
    """
    pmt_area = np.pi * pmt_radius**2
    n_total = wall_x * wall_y / pmt_area
    n_x = max(1, int(round(np.sqrt(n_total * wall_x / wall_y))))
    n_y = max(1, int(round(np.sqrt(n_total * wall_y / wall_x))))

    vox_x = wall_x / n_x
    vox_y = wall_y / n_y
    actual_area = vox_x * vox_y

    print(f"Wall voxel grid: {n_x} × {n_y} = {n_x * n_y} voxels "
          f"(size {vox_x:.1f}×{vox_y:.1f} mm, area {actual_area:.0f} mm² "
          f"vs PMT {pmt_area:.0f} mm²)")

    voxels: list[dict] = []
    x_min = -wall_x / 2.0
    y_min = -wall_y / 2.0
    for ix in range(n_x):
        for iy in range(n_y):
            cx = x_min + (ix + 0.5) * vox_x
            cy = y_min + (iy + 0.5) * vox_y
            voxels.append({
                "index": f"{ix:02d}{iy:02d}",
                "center": [float(cx), float(cy), 0.0],
                "size": [float(vox_x), float(WALL_Z), float(vox_y)],
                "rotation_matrix": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                "layer": "wall",
                "grid_ix": ix,
                "grid_iy": iy,
                "n_x": n_x,
                "n_y": n_y,
            })
    return voxels


# ══════════════════════════════ PMT Selection ════════════════════════════════

def select_sphere_pmts(voxels: list[dict], n_select: int) -> list[int]:
    """
    Select n_select voxels as homogeneously as possible on a sphere.

    Uses Fibonacci spiral to generate n_select ideal positions, then assigns
    each to the nearest voxel center via KD-tree.
    """
    if n_select >= len(voxels):
        return list(range(len(voxels)))
    if n_select == 1:
        return [0]

    centers = np.array([v["center"] for v in voxels])
    radius = float(np.mean(np.linalg.norm(centers, axis=1)))

    # Fibonacci spiral on unit sphere
    k = np.arange(n_select)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - (k / (n_select - 1)) * 2.0
    r = np.sqrt(np.maximum(1.0 - y**2, 0.0))
    fib_pts = np.column_stack([
        np.cos(golden_angle * k) * r,
        y,
        np.sin(golden_angle * k) * r,
    ]) * radius

    tree = cKDTree(centers)
    _, indices = tree.query(fib_pts, k=1)
    return sorted(set(int(i) for i in indices))


def select_wall_pmts(voxels: list[dict], n_select: int) -> list[int]:
    """
    Select n_select voxels as homogeneously as possible on a flat wall.

    Uses a 2D Fibonacci lattice to generate ideal positions, then assigns
    each to the nearest voxel center via KD-tree.
    """
    if n_select >= len(voxels):
        return list(range(len(voxels)))
    if n_select == 1:
        return [len(voxels) // 2]

    centers_2d = np.array([[v["center"][0], v["center"][1]] for v in voxels])

    # 2D Fibonacci lattice (quasi-random low-discrepancy sequence)
    x_min = centers_2d[:, 0].min()
    x_max = centers_2d[:, 0].max()
    y_min = centers_2d[:, 1].min()
    y_max = centers_2d[:, 1].max()

    phi = (1.0 + np.sqrt(5.0)) / 2.0  # golden ratio
    k = np.arange(n_select)
    # Van der Corput sequence in base 2 and golden ratio
    fib_x = x_min + (x_max - x_min) * ((k * phi) % 1.0)
    fib_y = y_min + (y_max - y_min) * (k / n_select)
    fib_pts = np.column_stack([fib_x, fib_y])

    tree = cKDTree(centers_2d)
    _, indices = tree.query(fib_pts, k=1)
    return sorted(set(int(i) for i in indices))


def compute_homogeneity(selected_centers: np.ndarray) -> dict:
    """
    Compute nearest-neighbour distance statistics as a homogeneity metric.

    Lower coefficient of variation (std/mean) = more homogeneous distribution.
    """
    if len(selected_centers) < 2:
        return {"n": len(selected_centers), "mean_nn_mm": 0.0, "std_nn_mm": 0.0, "cv": 0.0}

    tree = cKDTree(selected_centers)
    # k=2: first neighbour is the point itself (distance 0), second is nearest other
    distances, _ = tree.query(selected_centers, k=2)
    nn_distances = distances[:, 1]

    mean_d = float(np.mean(nn_distances))
    std_d = float(np.std(nn_distances))
    cv = std_d / mean_d if mean_d > 0 else 0.0
    return {
        "n": len(selected_centers),
        "mean_nn_mm": mean_d,
        "std_nn_mm": std_d,
        "cv": cv,
        "min_nn_mm": float(np.min(nn_distances)),
        "max_nn_mm": float(np.max(nn_distances)),
    }


# ══════════════════════════════ Visualization ═════════════════════════════════

def plot_voxels_3d(
    all_voxels: list[dict],
    selected_indices: list[int],
    geometry_type: str,
    filename: str,
    sphere_radius: Optional[float] = None,
) -> None:
    """3D scatter plot with PMT-selected voxels highlighted in green."""
    centers = np.array([v["center"] for v in all_voxels])
    sel_set = set(selected_indices)
    sel_mask = np.array([i in sel_set for i in range(len(all_voxels))])

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Transparent reference sphere (for sphere geometries)
    if sphere_radius is not None and geometry_type != "wall":
        u = np.linspace(0, 2 * np.pi, 40)
        v = np.linspace(0, np.pi, 40)
        xs = sphere_radius * np.outer(np.cos(u), np.sin(v))
        ys = sphere_radius * np.outer(np.sin(u), np.sin(v))
        zs = sphere_radius * np.outer(np.ones(40), np.cos(v))
        ax.plot_surface(xs, ys, zs, alpha=0.06, color="lightblue", edgecolor="none")

    # Non-selected voxels
    other = centers[~sel_mask]
    if len(other):
        ax.scatter(other[:, 0], other[:, 1], other[:, 2],
                   c="gray", s=4, alpha=0.25, label=f"All voxels ({(~sel_mask).sum()})")

    # Selected (PMT) voxels
    sel = centers[sel_mask]
    if len(sel):
        ax.scatter(sel[:, 0], sel[:, 1], sel[:, 2],
                   c="green", s=40, alpha=0.9, edgecolors="darkgreen", linewidths=0.8,
                   label=f"PMT voxels ({sel_mask.sum()})")

    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    ax.set_title(f"Voxel grid — {geometry_type} ({len(all_voxels)} voxels, "
                 f"{sel_mask.sum()} PMTs)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_box_aspect([1, 1, 1])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filename}")


# ═══════════════════════════ World Builder ════════════════════════════════════

def _build_world(
    reg: geant4.Registry,
    mat_registry,
    world_is_vacuum: bool,
    size_m: float = 20.0,
) -> geant4.LogicalVolume:
    world_mat = mat_registry.vacuum if world_is_vacuum else mat_registry.water
    world_solid = geant4.solid.Box("world", size_m, size_m, size_m, reg, "m")
    world_lv = geant4.LogicalVolume(world_solid, world_mat, "world", reg)
    reg.setWorld(world_lv)
    return world_lv


def _macro_filename(gdml_path: str) -> str:
    p = Path(gdml_path)
    stem = p.stem
    return str(p.parent / ("macro" + stem[0].upper() + stem[1:] + ".mac"))


def _write_gdml(reg: geant4.Registry, out_filename: str) -> None:
    detectors.write_detector_auxvals(reg)
    visualization.write_color_auxvals(reg)
    geometry.check_registry_sanity(reg, reg)
    writer = gdml.Writer()
    writer.addDetector(reg)
    writer.write(out_filename)
    print(f"  GDML written: {out_filename}")
    macro_file = _macro_filename(out_filename)
    detectors.generate_detector_macro(reg, macro_file)
    print(f"  Macro written: {macro_file}")


# ══════════════════════════ Sphere Geometry ═══════════════════════════════════

def _build_tyvek_sphere_solid(reg: geant4.Registry) -> geant4.solid.Sphere:
    return geant4.solid.Sphere(
        "tyvek_sphere",
        SPHERE_INNER_RADIUS, SPHERE_INNER_RADIUS + SPHERE_THICKNESS,
        0.0, 2.0 * np.pi, 0.0, np.pi,
        reg, "mm",
    )


def build_sphere_geometry(
    variant: str,
    out_filename: str,
    voxels_file: str,
    n_pmts: int,
    plot_filename: str,
    world_is_vacuum: bool,
    with_inner_sphere: bool = False,
) -> None:
    """Build sphere (or sphere_in_sphere) geometry in the requested variant."""
    reg = geant4.Registry()
    mat_registry = materials.OpticalMaterialRegistry(reg)
    surface_registry = materials.surfaces.OpticalSurfaceRegistry(reg)
    world_lv = _build_world(reg, mat_registry, world_is_vacuum)

    # ── Tyvek / SSD shell ──────────────────────────────────────────────────
    sphere_solid = _build_tyvek_sphere_solid(reg)

    if variant == "ssd":
        shell_mat = mat_registry.my_tyvek
        shell_surface = surface_registry.to_my_tyvek
        shell_surface_name = "tyvek_sphere_surface"
    else:
        shell_mat = mat_registry.tyvek
        shell_surface = surface_registry.to_tyvek
        shell_surface_name = "tyvek_sphere_surface"

    shell_lv = geant4.LogicalVolume(sphere_solid, shell_mat, "tyvek_sphere_log", reg)
    shell_lv.pygeom_color_rgba = [1.0, 1.0, 1.0, 0.15]
    geant4.SkinSurface(shell_surface_name, shell_lv, shell_surface, reg)

    shell_pv = geant4.PhysicalVolume(
        [0, 0, 0], [0, 0, 0], shell_lv, "TyvekSphere_phys", world_lv, reg
    )
    shell_pv.pygeom_active_detector = RemageDetectorInfo("optical", TYVEK_DETECTOR_ID)

    # ── Inner steel sphere (sphere_in_sphere only) ─────────────────────────
    if with_inner_sphere:
        inner_solid = geant4.solid.Sphere(
            "inner_sphere", 0.0, INNER_SPHERE_RADIUS,
            0.0, 2.0 * np.pi, 0.0, np.pi, reg, "mm",
        )
        inner_lv = geant4.LogicalVolume(
            inner_solid, mat_registry.metal_steel, "inner_sphere_log", reg
        )
        inner_lv.pygeom_color_rgba = [0.8, 0.8, 0.8, 0.3]
        geant4.SkinSurface("inner_sphere_surface", inner_lv, surface_registry.to_tyvek, reg)
        inner_pv = geant4.PhysicalVolume(
            [0, 0, 0], [0, 0, 0], inner_lv, "InnerSphere_phys", world_lv, reg
        )
        inner_pv.pygeom_active_detector = RemageDetectorInfo(
            "optical", INNER_SPHERE_DETECTOR_ID
        )

    # ── Voxels (SSD: generate + save; PMT: load + select) ─────────────────
    if variant == "ssd":
        print("\n=== Generating sphere voxel grid ===")
        all_voxels = generate_sphere_voxels(SPHERE_INNER_RADIUS)
        with open(voxels_file, "w") as f:
            json.dump(all_voxels, f, indent=2)
        print(f"  Saved: {voxels_file} ({len(all_voxels)} voxels)")

        selected_indices: list[int] = []  # SSD has no PMTs
        print(f"  Homogeneity (all voxels): {compute_homogeneity(np.array([v['center'] for v in all_voxels]))}")

    else:  # pmt
        print(f"\n=== Loading voxel grid from {voxels_file} ===")
        with open(voxels_file) as f:
            all_voxels = json.load(f)
        print(f"  {len(all_voxels)} voxels loaded")

        print(f"\n=== Selecting {n_pmts} PMT positions (Fibonacci) ===")
        selected_indices = select_sphere_pmts(all_voxels, n_pmts)
        # Handle duplicates when Fibonacci maps to fewer unique voxels
        actual_n = len(selected_indices)
        if actual_n < n_pmts:
            print(f"  Warning: only {actual_n} unique voxels selected (requested {n_pmts})")

        sel_voxels = [all_voxels[i] for i in selected_indices]
        selected_file = voxels_file.replace(".json", "_selected.json")
        with open(selected_file, "w") as f:
            json.dump(sel_voxels, f, indent=2)
        print(f"  Saved: {selected_file} ({actual_n} PMT voxels)")

        sel_centers = np.array([v["center"] for v in sel_voxels])
        hom = compute_homogeneity(sel_centers)
        print(f"  Homogeneity: mean NN = {hom['mean_nn_mm']:.1f} mm, "
              f"CV = {hom['cv']:.3f} (lower = more uniform)")

        # ── Place PMTs ─────────────────────────────────────────────────────
        print(f"\n=== Placing {actual_n} PMTs ===")
        pmt_volumes = construct_pmt_front(mat_registry.vacuum, surface_registry, reg)
        for i, voxel in enumerate(sel_voxels):
            rotation, translation = _sphere_pmt_placement(voxel)
            rawid = PMT_DETECTOR_ID_BASE + i
            place_pmt_front(
                rotation, translation, pmt_volumes,
                f"PMT{voxel['index']}", world_lv, reg,
                mat_registry, surface_registry, rawid,
            )

    # ── Export ──────────────────────────────────────────────────────────────
    print(f"\n=== Writing GDML ===")
    _write_gdml(reg, out_filename)

    print(f"\n=== Plotting voxels ===")
    plot_voxels_3d(all_voxels, selected_indices, "sphere", plot_filename,
                   sphere_radius=SPHERE_INNER_RADIUS)

    _print_sphere_summary(all_voxels, selected_indices, voxels_file, out_filename, plot_filename)


# ══════════════════════════ Wall Geometry ═════════════════════════════════════

def build_wall_geometry(
    variant: str,
    out_filename: str,
    voxels_file: str,
    n_pmts: int,
    plot_filename: str,
    world_is_vacuum: bool,
) -> None:
    """Build wall geometry in the requested variant (ssd / pmt / pmt_no_tyvek)."""
    reg = geant4.Registry()
    mat_registry = materials.OpticalMaterialRegistry(reg)
    surface_registry = materials.surfaces.OpticalSurfaceRegistry(reg)
    world_lv = _build_world(reg, mat_registry, world_is_vacuum, size_m=6.0)

    has_tyvek_wall = variant in ("ssd", "pmt")

    # ── Tyvek / SSD wall ───────────────────────────────────────────────────
    if has_tyvek_wall:
        wall_solid = geant4.solid.Box(
            "wall_solid", WALL_X, WALL_Y, WALL_Z, registry=reg
        )
        wall_mat = mat_registry.my_tyvek if variant == "ssd" else mat_registry.tyvek
        wall_surface = (
            surface_registry.to_my_tyvek if variant == "ssd"
            else surface_registry.to_tyvek
        )
        wall_lv = geant4.LogicalVolume(wall_solid, wall_mat, "wall_log", registry=reg)
        wall_lv.pygeom_color_rgba = [1.0, 1.0, 1.0, 0.20]
        geant4.SkinSurface("wall_surface", wall_lv, wall_surface, reg)
        wall_pv = geant4.PhysicalVolume(
            [0, 0, 0], [0, 0, 0], wall_lv, "Wall_phys", world_lv, registry=reg
        )
        wall_pv.pygeom_active_detector = RemageDetectorInfo("optical", TYVEK_DETECTOR_ID)

    # ── Voxels ─────────────────────────────────────────────────────────────
    if variant == "ssd":
        print("\n=== Generating wall voxel grid ===")
        all_voxels = generate_wall_voxels()
        with open(voxels_file, "w") as f:
            json.dump(all_voxels, f, indent=2)
        print(f"  Saved: {voxels_file} ({len(all_voxels)} voxels)")
        selected_indices: list[int] = []

    else:  # pmt or pmt_no_tyvek
        print(f"\n=== Loading voxel grid from {voxels_file} ===")
        with open(voxels_file) as f:
            all_voxels = json.load(f)
        print(f"  {len(all_voxels)} voxels loaded")

        print(f"\n=== Selecting {n_pmts} PMT positions (2D Fibonacci) ===")
        selected_indices = select_wall_pmts(all_voxels, n_pmts)
        actual_n = len(selected_indices)
        if actual_n < n_pmts:
            print(f"  Warning: only {actual_n} unique voxels selected (requested {n_pmts})")

        sel_voxels = [all_voxels[i] for i in selected_indices]
        selected_file = voxels_file.replace(".json", "_selected.json")
        with open(selected_file, "w") as f:
            json.dump(sel_voxels, f, indent=2)
        print(f"  Saved: {selected_file} ({actual_n} PMT voxels)")

        sel_centers_2d = np.array([[v["center"][0], v["center"][1]] for v in sel_voxels])
        # 2D homogeneity (use x/y only)
        hom = compute_homogeneity(sel_centers_2d)
        print(f"  Homogeneity (2D): mean NN = {hom['mean_nn_mm']:.1f} mm, CV = {hom['cv']:.3f}")

        # ── Place PMTs ─────────────────────────────────────────────────────
        print(f"\n=== Placing {actual_n} PMTs ===")
        pmt_volumes = construct_pmt_front(mat_registry.vacuum, surface_registry, reg)
        for i, voxel in enumerate(sel_voxels):
            rotation, translation = _wall_pmt_placement(voxel)
            rawid = PMT_DETECTOR_ID_BASE + i  # sequential: 10000, 10001, ...
            place_pmt_front(
                rotation, translation, pmt_volumes,
                f"PMT{voxel['index']}", world_lv, reg,
                mat_registry, surface_registry, rawid,
            )

    # ── Export ──────────────────────────────────────────────────────────────
    print(f"\n=== Writing GDML ===")
    _write_gdml(reg, out_filename)

    print(f"\n=== Plotting voxels ===")
    plot_voxels_3d(all_voxels, selected_indices, "wall", plot_filename)

    _print_wall_summary(all_voxels, selected_indices, voxels_file, out_filename, variant)


# ══════════════════════════ Summary Printers ══════════════════════════════════

def _print_sphere_summary(
    all_voxels: list[dict],
    selected_indices: list[int],
    voxels_file: str,
    gdml_file: str,
    plot_file: str,
) -> None:
    n_vox = len(all_voxels)
    n_pmt = len(selected_indices)
    r = SPHERE_INNER_RADIUS
    sphere_area_m2 = 4.0 * np.pi * r**2 / 1e6
    pmt_area_mm2 = n_pmt * np.pi * PMT_CATHODE_RADIUS**2
    coverage_pct = pmt_area_mm2 / (4.0 * np.pi * r**2) * 100.0

    print("\n" + "=" * 60)
    print("SPHERE GEOMETRY SUMMARY")
    print("=" * 60)
    print(f"  Inner radius:      {r:.2f} mm")
    print(f"  Shell area:        {sphere_area_m2:.3f} m²")
    print(f"  Total voxels:      {n_vox}")
    print(f"  PMT voxels:        {n_pmt}")
    print(f"  PMT coverage:      {coverage_pct:.2f}%  (A_PMT / A_total)")
    print(f"  GDML:              {gdml_file}")
    print(f"  Voxels JSON:       {voxels_file}")
    print(f"  Plot:              {plot_file}")
    print("=" * 60)


def _print_wall_summary(
    all_voxels: list[dict],
    selected_indices: list[int],
    voxels_file: str,
    gdml_file: str,
    variant: str,
) -> None:
    n_vox = len(all_voxels)
    n_pmt = len(selected_indices)
    wall_area_mm2 = WALL_X * WALL_Y
    pmt_area_mm2 = n_pmt * np.pi * PMT_CATHODE_RADIUS**2
    coverage_pct = pmt_area_mm2 / wall_area_mm2 * 100.0

    print("\n" + "=" * 60)
    print(f"WALL GEOMETRY SUMMARY ({variant})")
    print("=" * 60)
    print(f"  Wall size:         {WALL_X:.0f} × {WALL_Y:.0f} mm")
    print(f"  Wall area:         {wall_area_mm2 / 1e6:.4f} m²")
    print(f"  Total voxels:      {n_vox}")
    print(f"  PMT voxels:        {n_pmt}")
    print(f"  PMT coverage:      {coverage_pct:.2f}%")
    print(f"  GDML:              {gdml_file}")
    print(f"  Voxels JSON:       {voxels_file}")
    print("=" * 60)


# ═══════════════════════════════════ Main ════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate GDML geometry for SSD efficiency validation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--geometry", choices=["wall", "sphere", "sphere_in_sphere"], required=True,
        help="Detector geometry type",
    )
    parser.add_argument(
        "--variant", choices=["ssd", "pmt", "pmt_no_tyvek"], required=True,
        help=(
            "ssd: SSD surface + voxel generation; "
            "pmt: Tyvek + PMTs (reads voxels file); "
            "pmt_no_tyvek: PMTs only, no Tyvek wall (wall geometry only)"
        ),
    )
    parser.add_argument(
        "--n-pmts", type=int, default=50,
        help="Number of PMTs to place (pmt/pmt_no_tyvek variants only, default: 50)",
    )
    parser.add_argument(
        "--out", type=str, required=True,
        help="Output GDML filename",
    )
    parser.add_argument(
        "--voxels", type=str, required=True,
        help=(
            "ssd variant: output path for voxel JSON; "
            "pmt* variants: input path to existing voxel JSON"
        ),
    )
    parser.add_argument(
        "--plot", type=str, default=None,
        help="Output filename for 3D voxel plot (default: derived from --voxels)",
    )
    parser.add_argument(
        "--vacuum", action="store_true",
        help="Set world material to vacuum instead of water",
    )
    args = parser.parse_args()

    # Validate combinations
    if args.variant == "pmt_no_tyvek" and args.geometry != "wall":
        parser.error("--variant pmt_no_tyvek is only valid for --geometry wall")

    if args.variant in ("pmt", "pmt_no_tyvek") and not Path(args.voxels).exists():
        parser.error(
            f"Voxel JSON '{args.voxels}' not found. "
            "Run with --variant ssd first to generate the voxel grid."
        )

    plot_file = args.plot or args.voxels.replace(".json", "_voxel_plot.png")

    print(f"\n=== createGeometry.py ===")
    print(f"  geometry: {args.geometry}")
    print(f"  variant:  {args.variant}")
    if args.variant != "ssd":
        print(f"  n_pmts:   {args.n_pmts}")
    print(f"  out:      {args.out}")
    print(f"  voxels:   {args.voxels}")
    print(f"  vacuum:   {args.vacuum}")
    print()

    if args.geometry == "sphere":
        build_sphere_geometry(
            variant=args.variant,
            out_filename=args.out,
            voxels_file=args.voxels,
            n_pmts=args.n_pmts,
            plot_filename=plot_file,
            world_is_vacuum=args.vacuum,
            with_inner_sphere=False,
        )
    elif args.geometry == "sphere_in_sphere":
        build_sphere_geometry(
            variant=args.variant,
            out_filename=args.out,
            voxels_file=args.voxels,
            n_pmts=args.n_pmts,
            plot_filename=plot_file,
            world_is_vacuum=args.vacuum,
            with_inner_sphere=True,
        )
    elif args.geometry == "wall":
        build_wall_geometry(
            variant=args.variant,
            out_filename=args.out,
            voxels_file=args.voxels,
            n_pmts=args.n_pmts,
            plot_filename=plot_file,
            world_is_vacuum=args.vacuum,
        )


if __name__ == "__main__":
    main()
