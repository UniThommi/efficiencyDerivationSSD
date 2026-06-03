# sphere_to_gdml.py
# Benötigt: pygeant4ometry installiert (pip install pyg4ometry)
# Aufruf: python sphere_to_gdml.py

import pyg4ometry.geant4 as geant4
import pyg4ometry.gdml as gdml
import sys
import os
import numpy as np
import argparse

# src zum Python-Suchpfad hinzufügen
sys.path.append(os.path.abspath("./src"))
from l1000geom import materials
from pygeomtools import RemageDetectorInfo, detectors, geometry, visualization
import legendoptics.water


def create_ssd_sphere(out_filename, world_is_vacuum=False):
    # Registry für Geometrie-Objekte
    reg = geant4.Registry()
    mat_registry = materials.OpticalMaterialRegistry(reg)
    surface_registry = materials.surfaces.OpticalSurfaceRegistry(reg)

    # --- World (muss die Kugel komplett enthalten) ---
    # Wir machen es großzügig, z. B. 20 m Kantenlänge
    if world_is_vacuum:
        world_material = mat_registry.vacuum  # Geant4 vacuum
    else:
        world_material = mat_registry.water

    world = geant4.solid.Box("world", 10000, 10000, 10000, reg, "mm")
    world_lv = geant4.LogicalVolume(world, world_material, "world", reg)
    reg.setWorld(world_lv)

    # --- Tyvek-Kugel ---
    inner_radius = 1739.52  # mm
    outer_radius = inner_radius + 1.0  # 1 mm Wanddicke
    inner_sphere_outer_radius = 1000   
    sphere_solid = geant4.solid.Sphere(
        "my_tyvek_sphere_solid",
        inner_radius,
        outer_radius,
        0.0, 2 * np.pi,     # phi 0-360°
        0.0, np.pi,     # theta 0-180° (volle Kugel)
        registry=reg
    )

    # --- Logisches Volume mit Material ---
    sphere_logical = geant4.LogicalVolume(
        sphere_solid, mat_registry.my_tyvek, "my_tyvek_sphere_log", registry=reg
    )
    sphere_logical.pygeom_color_rgba = [1, 1, 1, 0.20]  # halbtransparent weiß
    geant4.SkinSurface("my_tyvek_sphere_surface", sphere_logical, surface_registry.to_my_tyvek, reg)

    # --- Physikalisches Volume ins World platzieren ---
    sphere_physical = geant4.PhysicalVolume(
        [0, 0, 0], [0, 0, 0], sphere_logical, "TyvekSphere_phys", world_lv, registry=reg
    )

    # Photon-Detektor aktivieren
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

    # --- Writer: Registry -> GDML-Datei ---
    detectors.write_detector_auxvals(reg)
    visualization.write_color_auxvals(reg)
    geometry.check_registry_sanity(reg, reg)

    writer = gdml.Writer()
    writer.addDetector(reg)
    writer.write(out_filename)
    print(f"GDML geschrieben: {out_filename}")


# --- Command-line interface ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Tyvek SSD Sphere with Inner Sphere and export as GDML")
    parser.add_argument("--out", type=str, default="tyvekSphereWithInnerSphereSSD.gdml", help="Output GDML filename")
    parser.add_argument("--vacuum", action="store_true", help="Set world material to vacuum instead of water")
    args = parser.parse_args()

    create_ssd_sphere(
        out_filename=args.out,
        world_is_vacuum=args.vacuum
    )
