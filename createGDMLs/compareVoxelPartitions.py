#!/usr/bin/env python3
"""
Compare the existing sphere voxelisation against a recursive zonal equal-area
(Leopardi) partition, for the exact sphere/voxel parameters used in
``createGeometry.py``.

This is a standalone diagnostic — it does NOT modify or import
``createGeometry.py`` (the sphere constants and the existing
``generate_sphere_voxels`` algorithm are reproduced locally so the script runs
without pyg4ometry / l1000geom).

Two partitions are built for the same sphere:

  1. "existing"  — equal-solid-angle latitude bands with a *fixed* number of
                   longitude cells per band (n_lat x n_lon grid), exactly as in
                   createGeometry.generate_sphere_voxels.
  2. "leopardi"  — recursive zonal equal-area partition (Leopardi 2006): two
                   polar caps + collars whose cell count scales with sin(theta).
                   Hits an *exact* target N and keeps cells near-isotropic.

Reported / plotted:
  * voxel count N of each partition,
  * per-voxel area derivation  A_i = R^2 * dOmega_i,  dOmega_i = dphi_i * dcos(theta)_i,
    with spread statistics (mean, std, min, max, CV) and deviation from the
    PMT-cathode target area,
  * 3D scatter of voxel centres on the sphere (coloured by area),
  * 2D Lambert cylindrical equal-area projection (phi, cos theta) of the cells
    (coloured by area), where equal-area cells appear as equal-area rectangles.
  * 2D raw angular (phi, theta) view (NOT area-preserving), where equal-area
    cells appear square near the equator and as tall thin slivers near the poles
    — making the shape (aspect-ratio) distortion visible.

Usage:
  python compareVoxelPartitions.py [--output-dir .]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ── Constants copied verbatim from createGeometry.py ──────────────────────────
SPHERE_INNER_RADIUS: float = 1739.52  # mm
PMT_CATHODE_RADIUS: float = 110.0     # mm


# ══════════════════════════ Existing lat-lon grid ════════════════════════════

def build_latlon_grid(
    n_lat: int,
    n_lon: int,
    label: str,
    short: str,
    radius: float = SPHERE_INNER_RADIUS,
    target_count: int = 0,
) -> dict:
    """Build an equal-solid-angle latitude-band grid (uniform in cos theta) with
    a *fixed* ``n_lon`` longitude cells per band — the algorithm used by
    createGeometry.generate_sphere_voxels — for an explicit factorisation
    ``n_lat x n_lon``.  Returns per-cell centres, angular bounds and exact areas.
    """
    total = n_lat * n_lon
    d_phi = 2.0 * np.pi / n_lon
    # Band edges are uniform in u = (1 - cos theta) / 2.
    centers, areas = [], []
    th_top, th_bot, ph_min, ph_max, th_c = [], [], [], [], []
    for i in range(n_lat):
        cos_top = 1.0 - 2.0 * (i / n_lat)          # nearer north pole
        cos_bot = 1.0 - 2.0 * ((i + 1) / n_lat)
        theta_i = np.arccos(1.0 - 2.0 * (i + 0.5) / n_lat)
        d_cos = cos_top - cos_bot                  # = 2 / n_lat
        cell_area = radius**2 * d_phi * d_cos
        for j in range(n_lon):
            phi_j = 2.0 * np.pi * j / n_lon        # cell centre longitude
            centers.append([
                radius * np.sin(theta_i) * np.cos(phi_j),
                radius * np.sin(theta_i) * np.sin(phi_j),
                radius * np.cos(theta_i),
            ])
            areas.append(cell_area)
            th_top.append(np.arccos(cos_top))
            th_bot.append(np.arccos(cos_bot))
            th_c.append(theta_i)
            ph_min.append(phi_j - d_phi / 2.0)
            ph_max.append(phi_j + d_phi / 2.0)

    return {
        "label": label,
        "short": short,
        "n": total,
        "n_lat": n_lat,
        "n_lon": n_lon,
        "centers": np.array(centers),
        "areas": np.array(areas),
        "theta_top": np.array(th_top),
        "theta_bot": np.array(th_bot),
        "theta_c": np.array(th_c),
        "phi_min": np.array(ph_min),
        "phi_max": np.array(ph_max),
        "is_cap": np.zeros(total, dtype=bool),
        "target_count": target_count or total,
    }


def generate_existing_grid(
    radius: float = SPHERE_INNER_RADIUS,
    pmt_radius: float = PMT_CATHODE_RADIUS,
) -> dict:
    """Reproduce createGeometry.generate_sphere_voxels exactly: derive
    (n_lat, n_lon) from the PMT-equivalence target via the same heuristic
    (n_lat = round(sqrt(target/2)), n_lon = round(target/n_lat))."""
    pmt_area = np.pi * pmt_radius**2
    sphere_area = 4.0 * np.pi * radius**2
    target_count = int(round(sphere_area / pmt_area))

    n_lat = int(round(np.sqrt(target_count / 2.0)))
    n_lon = int(round(target_count / n_lat))
    return build_latlon_grid(
        n_lat, n_lon,
        f"Existing lat-lon grid ({n_lat}x{n_lon})", "existing",
        radius, target_count,
    )


# ═══════════════════ Recursive zonal equal-area (Leopardi) ════════════════════

def generate_leopardi_grid(
    n_regions: int,
    radius: float = SPHERE_INNER_RADIUS,
) -> dict:
    """Recursive zonal equal-area partition of the sphere into exactly
    ``n_regions`` cells of equal area (Leopardi 2006).

    North/south polar caps + collars; the number of cells per collar scales
    with the collar circumference so every cell has solid angle 4*pi/N and the
    cells stay near-isotropic at all latitudes.
    """
    N = int(n_regions)
    v_solid = 4.0 * np.pi / N                       # solid angle per cell
    cell_area = radius**2 * v_solid                 # equal physical area

    # 1. Polar cap colatitude (cap area = v_solid).
    theta_c = float(np.arccos(1.0 - v_solid / (2.0 * np.pi)))  # = arccos(1 - 2/N)

    # 2-3. Ideal collar width and collar count.
    delta_i = np.sqrt(v_solid)
    n_collars = max(1, int(round((np.pi - 2.0 * theta_c) / delta_i)))
    delta_f = (np.pi - 2.0 * theta_c) / n_collars

    # 4. Ideal cell count per collar (uniform-width boundaries), then round with
    #    cumulative remainder so the totals sum to exactly N - 2.
    y = np.empty(n_collars)
    for i in range(n_collars):
        a = theta_c + i * delta_f
        b = theta_c + (i + 1) * delta_f
        y[i] = 2.0 * np.pi * (np.cos(a) - np.cos(b)) / v_solid
    m = np.empty(n_collars, dtype=int)
    rem = 0.0
    for i in range(n_collars):
        m[i] = int(round(y[i] + rem))
        rem += y[i] - m[i]
    # Numerical safety: force the exact total.
    diff = (N - 2) - int(m.sum())
    if diff != 0:
        m[int(np.argmax(m))] += diff

    # 5. Recompute collar boundaries so each cell has area exactly v_solid.
    cos_edges = [np.cos(theta_c)]
    for mi in m:
        cos_edges.append(cos_edges[-1] - mi * v_solid / (2.0 * np.pi))
    cos_edges = np.clip(np.array(cos_edges), -1.0, 1.0)
    theta_edges = np.arccos(cos_edges)              # length n_collars + 1

    centers, areas = [], []
    th_top, th_bot, ph_min, ph_max, th_c, is_cap = [], [], [], [], [], []

    def _add(c_top, c_bot, p0, p1, x, y_, z_, cap):
        centers.append([x, y_, z_])
        areas.append(cell_area)
        th_top.append(float(np.arccos(np.clip(c_top, -1.0, 1.0))))
        th_bot.append(float(np.arccos(np.clip(c_bot, -1.0, 1.0))))
        th_c.append(float(np.arccos(np.clip(0.5 * (c_top + c_bot), -1.0, 1.0))))
        ph_min.append(p0)
        ph_max.append(p1)
        is_cap.append(cap)

    # North cap (centre at north pole).
    _add(1.0, np.cos(theta_c), 0.0, 2.0 * np.pi, 0.0, 0.0, radius, True)

    # Collars.
    for i in range(n_collars):
        c_top = cos_edges[i]
        c_bot = cos_edges[i + 1]
        cos_mid = 0.5 * (c_top + c_bot)
        sin_mid = np.sqrt(max(1.0 - cos_mid**2, 0.0))
        mi = m[i]
        for k in range(mi):
            p0 = 2.0 * np.pi * k / mi
            p1 = 2.0 * np.pi * (k + 1) / mi
            phi_mid = 0.5 * (p0 + p1)
            _add(
                c_top, c_bot, p0, p1,
                radius * sin_mid * np.cos(phi_mid),
                radius * sin_mid * np.sin(phi_mid),
                radius * cos_mid,
                False,
            )

    # South cap (centre at south pole).
    _add(cos_edges[-1], -1.0, 0.0, 2.0 * np.pi, 0.0, 0.0, -radius, True)

    return {
        "label": f"Leopardi equal-area (N={N}, {n_collars} collars + 2 caps)",
        "short": "leopardi",
        "n": len(centers),
        "centers": np.array(centers),
        "areas": np.array(areas),
        "theta_top": np.array(th_top),
        "theta_bot": np.array(th_bot),
        "theta_c": np.array(th_c),
        "phi_min": np.array(ph_min),
        "phi_max": np.array(ph_max),
        "is_cap": np.array(is_cap, dtype=bool),
        "target_count": N,
    }


# ════════════════════════════════ Metrics ════════════════════════════════════

def compute_metrics(grid: dict, radius: float, pmt_area: float) -> dict:
    """Area-spread and shape (aspect-ratio) statistics for one partition."""
    areas = grid["areas"]
    mean_a = float(areas.mean())
    std_a = float(areas.std())

    # Cell aspect ratio (>= 1): latitudinal vs longitudinal arc length at the
    # cell centre.  Polar caps are round -> aspect 1.
    d_theta = grid["theta_bot"] - grid["theta_top"]
    d_phi = grid["phi_max"] - grid["phi_min"]
    s_theta = radius * d_theta
    s_phi = radius * np.sin(grid["theta_c"]) * d_phi
    with np.errstate(divide="ignore", invalid="ignore"):
        aspect = np.maximum(s_theta, s_phi) / np.minimum(s_theta, s_phi)
    aspect[grid["is_cap"]] = 1.0
    aspect = aspect[np.isfinite(aspect)]

    return {
        "n": grid["n"],
        "mean_area": mean_a,
        "std_area": std_a,
        "min_area": float(areas.min()),
        "max_area": float(areas.max()),
        "cv_area": std_a / mean_a if mean_a else 0.0,
        "dev_from_target_pct": (mean_a - pmt_area) / pmt_area * 100.0,
        "total_area": float(areas.sum()),
        "sphere_area": 4.0 * np.pi * radius**2,
        "aspect_mean": float(aspect.mean()),
        "aspect_median": float(np.median(aspect)),
        "aspect_max": float(aspect.max()),
    }


def print_comparison(grids: list[dict], radius: float, pmt_area: float) -> None:
    print("\n" + "=" * 74)
    print("VOXEL PARTITION COMPARISON")
    print("=" * 74)
    print(f"  Sphere radius:        {radius:.2f} mm")
    print(f"  Sphere area:          {4.0 * np.pi * radius**2:,.0f} mm^2")
    print(f"  PMT cathode radius:   {PMT_CATHODE_RADIUS:.0f} mm")
    print(f"  PMT target area:      {pmt_area:,.1f} mm^2  (= pi * r_PMT^2)")
    print("\n  Area derivation per cell:")
    print("    A_i = R^2 * dOmega_i,   dOmega_i = dphi_i * (cos theta_top - cos theta_bot)_i")

    mets = [compute_metrics(g, radius, pmt_area) for g in grids]
    rows = [
        ("voxel count N",            "{n:d}"),
        ("mean area [mm^2]",         "{mean_area:,.1f}"),
        ("std area [mm^2]",          "{std_area:,.3f}"),
        ("min area [mm^2]",          "{min_area:,.1f}"),
        ("max area [mm^2]",          "{max_area:,.1f}"),
        ("area CV (std/mean)",       "{cv_area:.2e}"),
        ("mean dev. from target [%]","{dev_from_target_pct:+.3f}"),
        ("area closure (sum/4piR^2)", None),
        ("aspect ratio mean",        "{aspect_mean:.2f}"),
        ("aspect ratio median",      "{aspect_median:.2f}"),
        ("aspect ratio max",         "{aspect_max:.2f}"),
    ]

    w = 30
    header = " " * w + "".join(f"{g['short']:>20}" for g in grids)
    print("\n" + header)
    print("  " + "-" * (w + 20 * len(grids) - 2))
    for label, fmt in rows:
        line = f"  {label:<{w-2}}"
        for met in mets:
            if label.startswith("area closure"):
                val = f"{met['total_area'] / met['sphere_area']:.6f}"
            else:
                val = fmt.format(**met)
            line += f"{val:>20}"
        print(line)
    print("=" * 74)


# ════════════════════════════════ Plotting ═══════════════════════════════════

def plot_3d(grids: list[dict], radius: float, out: Path) -> None:
    """Side-by-side 3D scatter of voxel centres, coloured by cell area."""
    vmin = min(g["areas"].min() for g in grids)
    vmax = max(g["areas"].max() for g in grids)
    # Guard against a degenerate (all-equal) colour range.
    if vmax - vmin < 1e-6 * vmax:
        vmin, vmax = 0.99 * vmin, 1.01 * vmax

    fig = plt.figure(figsize=(7.0 * len(grids), 7))
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 40)
    xs = radius * np.outer(np.cos(u), np.sin(v))
    ys = radius * np.outer(np.sin(u), np.sin(v))
    zs = radius * np.outer(np.ones_like(u), np.cos(v))

    sc = None
    for col, g in enumerate(grids):
        ax = fig.add_subplot(1, len(grids), col + 1, projection="3d")
        ax.plot_surface(xs, ys, zs, alpha=0.05, color="lightblue", edgecolor="none")
        c = g["centers"]
        sc = ax.scatter(c[:, 0], c[:, 1], c[:, 2], c=g["areas"],
                        cmap="viridis", vmin=vmin, vmax=vmax, s=12, alpha=0.9)
        ax.set_title(f"{g['label']}\nN = {g['n']}", fontsize=10)
        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y [mm]")
        ax.set_zlabel("Z [mm]")
        ax.set_box_aspect([1, 1, 1])

    cbar = fig.colorbar(sc, ax=fig.axes, shrink=0.6, pad=0.02)
    cbar.set_label("Cell area [mm$^2$]")
    fig.suptitle("Sphere voxelisation — 3D voxel centres", fontsize=13)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_2d(grids: list[dict], out: Path) -> None:
    """Lambert cylindrical equal-area projection (phi, cos theta) of the cells.

    In this projection true area is preserved, so equal-area cells appear as
    equal-area rectangles.
    """
    vmin = min(g["areas"].min() for g in grids)
    vmax = max(g["areas"].max() for g in grids)
    if vmax - vmin < 1e-6 * vmax:
        vmin, vmax = 0.99 * vmin, 1.01 * vmax

    fig, axes = plt.subplots(1, len(grids), figsize=(6.5 * len(grids), 6), sharey=True)
    if len(grids) == 1:
        axes = [axes]

    pc = None
    for ax, g in zip(axes, grids):
        rects = []
        cos_top = np.cos(g["theta_top"])
        cos_bot = np.cos(g["theta_bot"])
        for k in range(g["n"]):
            x0 = g["phi_min"][k]
            width = g["phi_max"][k] - g["phi_min"][k]
            y0 = min(cos_top[k], cos_bot[k])
            height = abs(cos_top[k] - cos_bot[k])
            rects.append(Rectangle((x0, y0), width, height))
        pc = PatchCollection(rects, cmap="viridis", edgecolor="white", linewidth=0.3)
        pc.set_array(g["areas"])
        pc.set_clim(vmin, vmax)
        ax.add_collection(pc)
        ax.set_xlim(-0.3, 2.0 * np.pi + 0.3)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel(r"Longitude $\varphi$ [rad]")
        ax.set_title(f"{g['label']}\nN = {g['n']}", fontsize=10)
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel(r"$\cos\theta$  (Lambert equal-area)")

    cbar = fig.colorbar(pc, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label("Cell area [mm$^2$]")
    fig.suptitle("Sphere voxelisation — 2D equal-area projection of cells", fontsize=13)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_2d_angular(grids: list[dict], out: Path) -> None:
    """Cells in raw angular coordinates (phi, theta) — NOT area-preserving.

    Unlike the Lambert projection, this stretches the polar regions, so cells
    of equal area appear with very different shapes: square-ish near the equator
    and tall thin slivers near the poles.  Same colour (= equal area) but
    different rectangle proportions directly illustrates the "equal area,
    unequal shape" point.
    """
    vmin = min(g["areas"].min() for g in grids)
    vmax = max(g["areas"].max() for g in grids)
    if vmax - vmin < 1e-6 * vmax:
        vmin, vmax = 0.99 * vmin, 1.01 * vmax

    fig, axes = plt.subplots(1, len(grids), figsize=(6.5 * len(grids), 6), sharey=True)
    if len(grids) == 1:
        axes = [axes]

    pc = None
    for ax, g in zip(axes, grids):
        rects = []
        for k in range(g["n"]):
            x0 = g["phi_min"][k]
            width = g["phi_max"][k] - g["phi_min"][k]
            y0 = min(g["theta_top"][k], g["theta_bot"][k])
            height = abs(g["theta_bot"][k] - g["theta_top"][k])
            rects.append(Rectangle((x0, y0), width, height))
        pc = PatchCollection(rects, cmap="viridis", edgecolor="white", linewidth=0.3)
        pc.set_array(g["areas"])
        pc.set_clim(vmin, vmax)
        ax.add_collection(pc)
        ax.set_xlim(-0.3, 2.0 * np.pi + 0.3)
        ax.set_ylim(np.pi, 0.0)          # north pole (theta=0) at top
        ax.set_xlabel(r"Longitude $\varphi$ [rad]")
        ax.set_title(f"{g['label']}\nN = {g['n']}", fontsize=10)
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel(r"Colatitude $\theta$ [rad]  (not area-preserving)")

    cbar = fig.colorbar(pc, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label("Cell area [mm$^2$]")
    fig.suptitle(
        "Sphere voxelisation — cells in raw angular coordinates "
        "(equal area, unequal shape)", fontsize=13)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════════════ Main ════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare existing vs Leopardi equal-area sphere voxelisation.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."),
                        help="Directory for output PNGs (default: current dir)")
    parser.add_argument("--n-lat", type=int, default=25,
                        help="Latitude bands for the exact-N lat-lon grid (default: 25)")
    parser.add_argument("--n-lon", type=int, default=40,
                        help="Longitude cells for the exact-N lat-lon grid (default: 40)")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    radius = SPHERE_INNER_RADIUS
    pmt_area = np.pi * PMT_CATHODE_RADIUS**2

    existing = generate_existing_grid(radius)

    # Exact lat-lon grid: same equal-solid-angle algorithm, but with a
    # factorisation chosen so n_lat * n_lon hits the target exactly.
    exact = build_latlon_grid(
        args.n_lat, args.n_lon,
        f"Lat-lon exact ({args.n_lat}x{args.n_lon})", "latlon_exact",
        radius,
    )

    # Match N to the same PMT-equivalence target the existing grid aims for.
    leopardi = generate_leopardi_grid(existing["target_count"], radius)
    grids = [existing, exact, leopardi]

    print_comparison(grids, radius, pmt_area)
    print("\n-- Plotting " + "-" * 58)
    plot_3d(grids, radius, args.output_dir / "voxel_partition_comparison_3d.png")
    plot_2d(grids, args.output_dir / "voxel_partition_comparison_2d.png")
    plot_2d_angular(grids, args.output_dir / "voxel_partition_comparison_angular.png")


if __name__ == "__main__":
    main()
