# wand_to_gdml.py
# Benötigt: pygeant4ometry installiert (pip install pygeant4ometry)
# Aufruf: python wand_to_gdml.py

import pyg4ometry.geant4 as geant4
import pyg4ometry.gdml as gdml
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
import os
import warnings
import json
import argparse

# src zum Python-Suchpfad hinzufügen
sys.path.append(os.path.abspath("./src"))
from l1000geom import materials
from pygeomtools import RemageDetectorInfo, detectors, geometry, visualization
import legendoptics.water

pmt_eff_radius = 131
cutoff = 41  # Cutoff to take the top part of the ellipsoid resulting in 250mm diameter.
cathode_cutoff = 65  # cutoff such that the effective cathode radius is 220mm.
# pmt_base_height = 145

def readVoxelFile(json_file_path: str): 
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    voxel_list = []
    for entry in data:
        voxel = {
            'index': entry.get('index'),
            'center': entry.get('center'),
            'corners': entry.get('corners', None),
            'layer': entry.get('layer')
        }
        voxel_list.append(voxel)

    return voxel_list

def construct_PMT_front(
    vac_mat: geant4.Material,
    surfaces,
    reg: geant4.Registry,
) -> list:
    """Construct the solids for the front part of the PMT."""
    # Borosilcate glass window of the PMT
    pmt_window = geant4.solid.Ellipsoid(
        "PMT_window", pmt_eff_radius, pmt_eff_radius, pmt_eff_radius, cutoff, 200, reg, "mm"
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

# Euler Angles
def get_euler_angles(target_direction: np.array):
    """
    Calculate the Euler angles to rotate the default direction to the target direction.
    The default direction is [0, 0, 1]
    """
    default_direction = np.array([0, 0, 1])

    rotation_axis = np.cross(default_direction, target_direction)

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

def place_pmts(pmt_volumes: list, world_lv: geant4.LogicalVolume, reg: geant4.Registry, mat_registry, surface_registry):
    """Place PMTs at specified voxel positions."""
    voxel_file_path = "/home/tbueger/code/weiteres/inspectPhotonsSSD/voxelsTyvekWall.json"
    voxels = readVoxelFile(voxel_file_path)
    voxel_list = ["0202", "0208", "0802", "0808", "0505"]
    filtered_voxels = [v for v in voxels if v['index'] in voxel_list]

    for voxel in filtered_voxels:
        cx = voxel["center"][0] 
        cy = voxel["center"][1] 
        cz = voxel["center"][2]

        # if voxel["layer"] == "wall":
        #     r = (np.sqrt(cx ** 2 + cy ** 2))
        #     nx =  - cx / r
        #     ny =  - cy / r
        #     nz = 0
        #     x = cx + nx * (0.5 - cutoff)
        #     y = cy + ny * (0.5 - cutoff) 
        #     z = cz        
        #     target_direction = np.array([-nx, -ny, nz])

        # Ausgerichtet in -z Richtung
        nx = 0
        ny = 0
        nz = -1
        x = cx 
        y = cy
        z = cz + nz * (0.5 - cutoff)   
        target_direction = np.array([nx, ny, nz])


        rawid = 10000 + int(voxel["index"])
        namePMT = "PMT" + voxel["index"]
        place_PMT_front(
            get_euler_angles(target_direction),
            [x, y, z],
            pmt_volumes,
            namePMT,
            world_lv,
            reg,
            mat_registry,
            surface_registry,
            rawid,
        )

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

    # Place PMTs:
    pmt_volumes = construct_PMT_front(mat_registry.vacuum, surface_registry, reg)
    place_pmts(pmt_volumes, world_lv, reg, mat_registry, surface_registry)


    # --- Writer: Registry -> GDML-Datei ---
    detectors.write_detector_auxvals(reg)
    visualization.write_color_auxvals(reg)
    geometry.check_registry_sanity(reg, reg)
    writer = gdml.Writer()
    writer.addDetector(reg)
    writer.write(out_filename)
    print(f"GDML geschrieben: {out_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Tyvek PMT Wall and export as GDML")
    parser.add_argument("--out", type=str, default="wallPMTsNoTyvek.gdml", help="Output GDML filename")
    parser.add_argument("--vacuum", action="store_true", help="Set world material to vacuum instead of water")
    args = parser.parse_args()

    make_tyvek_wall_gdml(
        out_filename=args.out,
        world_is_vacuum=args.vacuum
    )
