# wand_to_gdml.py
# Benötigt: pygeant4ometry installiert (pip install pygeant4ometry)
# Aufruf: python wand_to_gdml.py

import pyg4ometry.geant4 as geant4
import pyg4ometry.gdml as gdml
import sys
import os
import argparse

# src zum Python-Suchpfad hinzufügen
sys.path.append(os.path.abspath("./src"))
from l1000geom import materials
from pygeomtools import RemageDetectorInfo, detectors, geometry, visualization
import legendoptics.water


def make_tyvek_wall_gdml(out_filename, world_is_vacuum=False):
    # Registry für Geometrie-Objekte
    reg = geant4.Registry()
    mat_registry = materials.OpticalMaterialRegistry(reg)
    surface_registry = materials.surfaces.OpticalSurfaceRegistry(reg)

    # --- World (muss die Wand komplett enthalten) ---
    # World: 6 m Kantenlänge (sicher größer als Wand)
    if world_is_vacuum:
        world_material = mat_registry.vacuum  # Geant4 vacuum
    else:
        world_material = mat_registry.water

    world = geant4.solid.Box("world", 6, 6, 6, reg, "m")
    world_lv = geant4.LogicalVolume(world, world_material, "world", reg)
    reg.setWorld(world_lv)

    # --- Wand-Solid (Box) ---
    # Symmetrisch im Ursprung
    wall_x = 1950.0  # mm
    wall_y = 1950.0  # mm
    wall_z = 1.0     # mm thickness
    wall_solid = geant4.solid.Box("wall_solid", wall_x, wall_y, wall_z, registry=reg)

    # --- Logisches Volume mit Material ---
    # Verwende den NIST-Name "G4_WATER" — Geant4 kennt diesen Namen
    wall_logical = geant4.LogicalVolume(wall_solid, mat_registry.my_tyvek, "wall_log", registry=reg)
    wall_logical.pygeom_color_rgba = [1, 1, 1, 0.20]
    geant4.SkinSurface("wall_surface", wall_logical, surface_registry.to_my_tyvek, reg)
    wall_physical = geant4.PhysicalVolume([0, 0, 0], [0, 0, 0], wall_logical, "Wall_phys", world_lv, registry=reg)
    wall_physical.pygeom_active_detector = RemageDetectorInfo("optical", 1000)

    # --- Writer: Registry -> GDML-Datei ---
    detectors.write_detector_auxvals(reg)
    visualization.write_color_auxvals(reg)
    geometry.check_registry_sanity(reg, reg)
    writer = gdml.Writer()
    writer.addDetector(reg)
    writer.write(out_filename)
    print(f"GDML geschrieben: {out_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Tyvek SSD Wall and export as GDML")
    parser.add_argument("--out", type=str, default="tyvekWallSSD.gdml", help="Output GDML filename")
    parser.add_argument("--vacuum", action="store_true", help="Set world material to vacuum instead of water")
    args = parser.parse_args()

    make_tyvek_wall_gdml(
        out_filename=args.out,
        world_is_vacuum=args.vacuum
    )
