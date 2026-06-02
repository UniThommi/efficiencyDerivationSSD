# wand_to_gdml.py
# Benötigt: pygeant4ometry installiert (pip install pygeant4ometry)
# Aufruf: python wand_to_gdml.py

import pyg4ometry.geant4 as geant4
import pyg4ometry.gdml as gdml
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
import sys
import os
import warnings
import json
import matplotlib.pyplot as plt
import argparse
from mpl_toolkits.mplot3d import Axes3D

# src zum Python-Suchpfad hinzufügen
sys.path.append(os.path.abspath("./src"))
from l1000geom import materials
from pygeomtools import RemageDetectorInfo, detectors, geometry, visualization
import legendoptics.water

pmt_eff_radius = 131
cutoff = 41  # Cutoff to take the top part of the ellipsoid resulting in 250mm diameter.
cathode_cutoff = 65  # cutoff such that the effective cathode radius is 220mm.
# pmt_base_height = 145

# --- Voxel Parameter ---
voxel_size = 195.0  # mm
inner_radius = 1739.52  # mm
outer_radius = inner_radius + 1.0

inner_sphere_outer_radius = 1000

def construct_PMT_front(
    vac_mat: geant4.Material,
    surfaces,
    reg: geant4.Registry,
) -> list:
    """Construct the solids for the front part of the PMT."""
    # Borosilcate glass window of the PMT
    pmt_window = geant4.solid.Ellipsoid(
        "PMT_window",       # Name
         pmt_eff_radius,    # pxSemiAxis
         pmt_eff_radius,    # pySemiAxis
         pmt_eff_radius,    # pySemiAxis
         cutoff,            # pzBottomCut
         200,               # pzTopCut
         reg,               # Registry
         "mm"               # lUnit
    )

    vacuum_radius = 128  # Results in a glass window thickness of ~2-3mm
    vacuum_height = pmt_eff_radius - 2
    # The vacuum inside of the PMT window
    pmt_vacuum = geant4.solid.Ellipsoid(
        "PMT_vacuum", vacuum_radius, vacuum_radius, vacuum_height, cutoff, 200, reg, "mm"
    )
    # The actual sensitive part of the PMT. Optical hits will be registered once they hit this volume
    pmt_cathode = geant4.solid.Ellipsoid(
        "PMT_cathode", vacuum_radius, vacuum_radius, vacuum_height, cathode_cutoff, 200, reg, "mm"
    )

    pmt_cathode_lv = geant4.LogicalVolume(pmt_cathode, vac_mat, "PMT_cathode", reg)
    pmt_cathode_lv.pygeom_color_rgba = [0.545, 0.271, 0.074, 1]
    geant4.SkinSurface("pmt_cathode_surface", pmt_cathode_lv, surfaces.to_photocathode, reg)

    return [pmt_window, pmt_vacuum, pmt_cathode_lv]

def generate_sphere_voxels(radius, voxel_size, target_count=1012):
    """
    Generiert gleichflächige Voxel auf einer Kugeloberfläche.
    Die Mittelpunkte liegen exakt auf der Kugeloberfläche.
    """
    n_lat = int(np.sqrt(target_count / 2))
    n_lon = int(np.ceil(target_count / n_lat))
    N_real = n_lat * n_lon

    print(f"Generiere gleichflächige Voxel-Unterteilung:")
    print(f"  n_lat = {n_lat}, n_lon = {n_lon}")
    print(f"  Tatsächliche Anzahl Voxel: {N_real}")
    print(f"  Voxel-Größe: {voxel_size}x{voxel_size} mm")
    print(f"  Radius: {radius:.2f} mm")

    # Gleichflächige Aufteilung in theta (polar angle)
    i = np.arange(0, n_lat)
    theta = np.arccos(1 - 2 * (i + 0.5) / n_lat)  # Mittelpunkte zwischen den Rändern
    phi = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)  # gleichmäßige Verteilung

    voxels = []
    idx = 0

    for t in theta:
        for p in phi:
            x = radius * np.sin(t) * np.cos(p)
            y = radius * np.sin(t) * np.sin(p)
            z = radius * np.cos(t)
            voxel = {
                "index": f"{idx:06d}",
                "center": [float(x), float(y), float(z)],
                "layer": "sphere"
            }
            voxels.append(voxel)
            idx += 1

    print(f"✅ {len(voxels)} gleichflächige Voxel generiert")
    return voxels

def select_homogeneous_voxels(voxels, n_select):
    """
    Wählt n_select Voxel aus, die möglichst homogen über die Kugeloberfläche verteilt sind.
    Verwendet Fibonacci-Sampling auf der Einheitssphäre und wählt die nächsten Voxel.
    """

    if n_select >= len(voxels):
        return list(range(len(voxels)))

    if n_select == 1:
        return [0]

    # Alle Voxel-Mittelpunkte
    centers = np.array([v["center"] for v in voxels])
    radius = np.mean(np.linalg.norm(centers, axis=1))

    # Fibonacci-Sphere Punkte generieren
    indices = np.arange(0, n_select)
    phi = np.pi * (3. - np.sqrt(5.))  # Goldener Winkel
    y = 1 - (indices / float(n_select - 1)) * 2  # y ∈ [1, -1]
    r = np.sqrt(1 - y * y)
    x = np.cos(phi * indices) * r
    z = np.sin(phi * indices) * r
    sphere_points = np.stack((x, y, z), axis=1) * radius

    # Nächste Voxel zu jedem idealen Punkt finden
    tree = cKDTree(centers)
    _, voxel_indices = tree.query(sphere_points, k=1)

    return sorted(set(voxel_indices))

def place_PMT_front(
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
    """Place PMT with unique logical volumes."""
    # In order to have unique PMT physical volumes, we need to re-create the mother logical volumes.
    pmt_window_lv = geant4.LogicalVolume(pmt_volumes[0], mat_registry.borosilicate, name + "_window", reg)
    pmt_window_lv.pygeom_color_rgba = [0.9, 0.8, 0.5, 0.5]
    pmt_vacuum_lv = geant4.LogicalVolume(pmt_volumes[1], mat_registry.vacuum, name + "_vacuum", reg)

    # We have to place the new logical volumes for every single PMT
    geant4.PhysicalVolume([0, 0, 0], [0, 0, 0], pmt_vacuum_lv, name + "_vacuum", pmt_window_lv, reg)
    pmt_pv = geant4.PhysicalVolume([0, 0, 0], [0, 0, 0], pmt_volumes[2], name, pmt_vacuum_lv, reg)
    pmt_pv.pygeom_active_detector = RemageDetectorInfo("optical", rawid)

    return geant4.PhysicalVolume(rotation, translation, pmt_window_lv, name + "_window", mother_lv, reg)

def plot_sphere_voxels(voxels, selected_indices, filename="sphere_voxels_overview.png"):
    """
    Plotte nur die Voxel-Mittelpunkte.
    Ausgewählte Voxel grün, alle anderen rot.
    """
    print(f"\n=== Erstelle Visualisierung der Voxel-Mittelpunkte ===")
    
    centers = np.array([v["center"] for v in voxels])
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 🔹 Ausgewählte Voxel grün
    if selected_indices:
        selected_centers = np.array([voxels[i]["center"] for i in selected_indices])
        ax.scatter(selected_centers[:, 0], selected_centers[:, 1], selected_centers[:, 2],
                   c='green', s=20, alpha=0.8, label='Ausgewählte Voxel')
    
    # 🔸 Alle anderen Voxel rot (außer die, die bereits ausgewählt wurden)
    all_indices = set(range(len(voxels)))
    other_indices = list(all_indices - set(selected_indices))
    if other_indices:
        other_centers = np.array([voxels[i]["center"] for i in other_indices])
        ax.scatter(other_centers[:, 0], other_centers[:, 1], other_centers[:, 2],
                   c='red', s=5, alpha=0.6, label='Alle anderen Voxel')
    
    # Achsen & Layout
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Z [mm]')
    ax.set_title(f"Voxel-Mittelpunkte")
    ax.set_box_aspect([1, 1, 1])
    
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualisierung gespeichert: {filename}")


def print_summary(radius, voxels, selected_indices, voxel_json, selected_json, gdml_file, plot_file):
    """
    Gibt eine Zusammenfassung aller wichtigen Parameter und Pfade aus.
    """
    print("\n" + "="*70)
    print(" "*20 + "ZUSAMMENFASSUNG")
    print("="*70)
    print(f"\n📐 GEOMETRIE:")
    print(f"   Innerer Kugelradius:        {radius:.2f} mm ({radius/1000:.4f} m)")
    print(f"   Äußerer Kugelradius:        {outer_radius:.2f} mm")
    print(f"   Tyvek-Dicke:                {outer_radius - radius:.2f} mm")
    print(f"   Kugeloberfläche:            {4 * np.pi * radius**2 / 1e6:.2f} m²")
    
    print(f"\n📦 VOXELISIERUNG:")
    print(f"   Anzahl aller Voxel:         {len(voxels)}")
    print(f"   Ziel-Voxelgröße:            {voxel_size:.1f} x {voxel_size:.1f} mm")
    print(f"   Durchschn. Voxelfläche:     {4 * np.pi * radius**2 / len(voxels) / 1e2:.2f} cm²")
    
    print(f"\n🔴 PMT-PLATZIERUNG:")
    print(f"   Anzahl platzierte PMTs:     {len(selected_indices)}")
    print(f"   Abdeckung:                  {len(selected_indices)/len(voxels)*100:.1f}% der Voxel")
    print(f"   PMT-Durchmesser:            {2*pmt_eff_radius:.0f} mm")
    print(f"   Kathoden-Durchmesser:       ~440 mm (effektiv)")
    
    # Berechne durchschnittlichen Abstand zwischen PMTs
    if len(selected_indices) > 1:
        selected_centers = np.array([voxels[i]["center"] for i in selected_indices])
        distances = []
        for i in range(len(selected_centers)):
            for j in range(i+1, len(selected_centers)):
                dist = np.linalg.norm(selected_centers[i] - selected_centers[j])
                distances.append(dist)
        avg_dist = np.mean(distances)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        print(f"   Durchschn. PMT-Abstand:     {avg_dist:.1f} mm")
        print(f"   Min/Max PMT-Abstand:        {min_dist:.1f} / {max_dist:.1f} mm")
    
    print(f"\n💾 AUSGABEDATEIEN:")
    print(f"   GDML-Geometrie:             {gdml_file}")
    print(f"   Alle Voxel (JSON):          {voxel_json}")
    print(f"   Ausgewählte Voxel (JSON):   {selected_json}")
    print(f"   Visualisierung (PNG):       {plot_file}")
    
    print(f"\n📊 DATEIGRÖSSEN:")
    if os.path.exists(gdml_file):
        print(f"   {gdml_file:30s} {os.path.getsize(gdml_file)/1024:.1f} KB")
    if os.path.exists(voxel_json):
        print(f"   {voxel_json:30s} {os.path.getsize(voxel_json)/1024:.1f} KB")
    if os.path.exists(selected_json):
        print(f"   {selected_json:30s} {os.path.getsize(selected_json)/1024:.1f} KB")
    if os.path.exists(plot_file):
        print(f"   {plot_file:30s} {os.path.getsize(plot_file)/1024:.1f} KB")
    
    print("\n" + "="*70)
    print("✅ ERFOLGREICH ABGESCHLOSSEN!")
    print("="*70 + "\n")

# Euler Angles
def get_euler_angles(target_direction: np.array):
    """
    Calculate the Euler angles to rotate the default direction to the target direction.
    The default direction is [0, 0, 1]
    """
    default_direction = np.array([0, 0, 1])

    rotation_axis = np.cross(target_direction, default_direction)

    # Calculate the angle using the dot product
    cos_angle = np.dot(default_direction, target_direction)
    angle = np.arccos(cos_angle)

    # If angle is 0 or 180, no rotation is needed or full rotation is needed
    if np.abs(angle) < 1e-6:
        return [0, 0, 0]
    if np.abs(angle - np.pi) < 1e-6:
        # Special case for 180 degree rotation, any axis is valid, pick (1, 0, 0)
        return [np.pi, 0, 0]

    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

    # Create a rotation object
    rotation = R.from_rotvec(rotation_axis * angle)

    # Get the rotation matrix or Euler angles
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        euler_angles = rotation.as_euler("xyz")
    return [euler_angles[0], euler_angles[1], euler_angles[2]]


def place_pmts(pmt_volumes: list, world_lv: geant4.LogicalVolume, reg: geant4.Registry, 
               mat_registry, surface_registry, selected_voxels):
    """Place PMTs at specified voxel positions."""
    print(f"\nPlatziere {len(selected_voxels)} PMTs an ausgewählten Voxel-Positionen:")
    print("Voxel IDs mit PMTs:")
    
    for i, voxel in enumerate(selected_voxels):
        cx, cy, cz = voxel["center"]
        voxel_id = voxel["index"]
        
        # Berechne radiale Normalenrichtung (nach außen zeigend)
        r = np.sqrt(cx**2 + cy**2 + cz**2)
        nx = cx / r
        ny = cy / r
        nz = cz / r

        print("R: ", r, "; NX: ", nx, "; NY: ", ny, "; NZ: ", nz)
        
        # damit es keinen Überlapp mit der Tyvek-Wand gibt
        # berechne zusätzlichen radialen Abstand
        cutoff_radius = np.sqrt(pmt_eff_radius**2 - cutoff**2)
        radial_shift = inner_radius - np.sqrt(inner_radius**2 - cutoff_radius**2) #FIX, stimmt nicht
        offset = cutoff - radial_shift 
        #offset = 0
        x = cx + nx * offset
        y = cy + ny * offset
        z = cz + nz * offset
        
        # Target direction: PMT schaut nach innen (negative Normalenrichtung)
        target_direction = np.array([-nx, -ny, -nz])
        rotation = get_euler_angles(target_direction)
        
        rawid = 10000 + i
        namePMT = f"PMT{voxel_id}"
        
        place_PMT_front(rotation, [x, y, z], pmt_volumes, namePMT,
                        world_lv, reg, mat_registry, surface_registry, rawid)
        
        if i < 10 or i % 10 == 0:  # Zeige erste 10 und dann jeden 10.
            print(f"  {voxel_id} (Nummer {i})")

def make_tyvek_sphere_gdml(out_filename, voxel_json="voxelsTyvekSphere.json",  plot_filename="sphere_voxels_overview.png", n_pmts = 50, world_is_vacuum=False):
    # Registry für Geometrie-Objekte
    reg = geant4.Registry()
    mat_registry = materials.OpticalMaterialRegistry(reg)
    surface_registry = materials.surfaces.OpticalSurfaceRegistry(reg)

    # --- World (groß genug, um die Kugel zu enthalten) ---
    # World: 20 m Kantenlänge
    if world_is_vacuum:
        world_material = mat_registry.vacuum  # Geant4 vacuum
    else:
        world_material = mat_registry.water

    world = geant4.solid.Box("world", 20, 20, 20, reg, "m")
    world_lv = geant4.LogicalVolume(world, world_material, "world", reg)
    reg.setWorld(world_lv)

    # Kugel: volle 4π Kugelschale (0° bis 180° polar, 0° bis 360° azimut)
    sphere_solid = geant4.solid.Sphere(
        "tyvek_sphere",
        inner_radius,  # Innenradius
        outer_radius,  # Außenradius
        0.0,           # Start phi
        2*np.pi,         # Delta phi
        0.0,           # Start theta
        np.pi,         # Delta theta
        reg,            
        "mm"
    )

    sphere_logical = geant4.LogicalVolume(sphere_solid, mat_registry.tyvek, "tyvek_sphere_log", reg)
    sphere_logical.pygeom_color_rgba = [1, 1, 1, 0.20]
    geant4.SkinSurface("tyvek_sphere_surface", sphere_logical, surface_registry.to_tyvek, reg)

    sphere_physical = geant4.PhysicalVolume([0, 0, 0], [0, 0, 0],
                                          sphere_logical, "TyvekSphere_phys",
                                          world_lv, reg)
    sphere_physical.pygeom_active_detector = RemageDetectorInfo("optical", 1000)
    
    # Innere Kugel aus Stahl
    inner_sphere_solid = geant4.solid.Sphere(
        "inner_sphere",
        0,  # Innenradius
        inner_sphere_outer_radius,  # Außenradius
        0.0,           # Start phi
        2*np.pi,         # Delta phi
        0.0,           # Start theta
        np.pi,         # Delta theta
        reg,            
        "mm"
    )

    inner_sphere_logical = geant4.LogicalVolume(inner_sphere_solid, mat_registry.metal_steel, "inner_sphere_log", reg)
    inner_sphere_logical.pygeom_color_rgba = [1, 0, 1, 0.20]
    geant4.SkinSurface("inner_sphere_surface", inner_sphere_logical, surface_registry.to_tyvek, reg)

    inner_sphere_physical = geant4.PhysicalVolume([0, 0, 0], [0, 0, 0],
                                          inner_sphere_logical, "InnerSphere_phys",
                                          world_lv, reg)
    inner_sphere_physical.pygeom_active_detector = RemageDetectorInfo("optical", 9000)
    
     # --- Voxelisierung generieren ---
    print(f"\n=== VOXELISIERUNG ===")
    voxels = generate_sphere_voxels(inner_radius, voxel_size, target_count=1012)
    
    # Speichere alle Voxel in JSON
    with open(voxel_json, "w") as f:
        json.dump(voxels, f, indent=2)
    print(f"✅ Alle Voxel gespeichert: {voxel_json}")
    
    # --- Auswahl von 50 homogen verteilten Voxeln ---
    print(f"\n=== PMT-AUSWAHL ===")
    selected_indices = select_homogeneous_voxels(voxels, n_select=n_pmts)
    
    # Speichere ausgewählte Voxel separat
    selected_voxels = [voxels[i] for i in selected_indices]
    selected_json = voxel_json.replace(".json", "_selected.json")
    with open(selected_json, "w") as f:
        json.dump(selected_voxels, f, indent=2)
    print(f"✅ Ausgewählte Voxel gespeichert: {selected_json}")

    # --- Place PMTs ---
    print(f"\n=== PMT-PLATZIERUNG ===")
    pmt_volumes = construct_PMT_front(mat_registry.vacuum, surface_registry, reg)
    place_pmts(pmt_volumes, world_lv, reg, mat_registry, surface_registry, selected_voxels)

    # --- Writer: Registry -> GDML-Datei ---
    print(f"\n=== GDML-EXPORT ===")
    detectors.write_detector_auxvals(reg)
    visualization.write_color_auxvals(reg)
    geometry.check_registry_sanity(reg, reg)
    writer = gdml.Writer()
    writer.addDetector(reg)
    writer.write(out_filename)
    print(f"✅ GDML geschrieben: {out_filename}")
    
    # --- Visualisierung erstellen ---
    plot_sphere_voxels(voxels, selected_indices, plot_filename)
    
    # --- Zusammenfassung ausgeben ---
    print_summary(inner_radius, voxels, selected_indices, voxel_json, 
                  selected_json, out_filename, plot_filename)


# --- Command-line interface ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Tyvek Sphere with PMTs and Inner Sphere and export as GDML")
    parser.add_argument("--n", type=int, default=50, help="Number of PMTs to place (default: 50)")
    parser.add_argument("--out", type=str, default="tyvekSphereWithInnerSphere50PMTs.gdml", help="Output GDML filename")
    parser.add_argument("--voxels", type=str, default="voxelsTyvekSphereWithInnerSphere.json", help="Voxel JSON filename")
    parser.add_argument("--plot", type=str, default="sphere_with_inner_sphere_voxels_overview.png", help="Plot filename")
    parser.add_argument("--vacuum", action="store_true", help="Set world material to vacuum instead of water")
    args = parser.parse_args()

    make_tyvek_sphere_gdml(
        out_filename=args.out,
        voxel_json=args.voxels,
        plot_filename=args.plot,
        n_pmts=args.n,
        world_is_vacuum=args.vacuum
    )
