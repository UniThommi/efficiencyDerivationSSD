import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LogNorm
from pathlib import Path
import json
from collections import Counter

# ----------------------
# Konfiguration
# ----------------------
files = sorted(Path("/home/tbueger/code/weiteres/inspectPhotonsSSD/SSDWallLowR").glob("*.hdf5"))  # Pfad zu deinen HDF5-Dateien
wall_size_x = 1950.0  # mm
wall_size_y = 1950.0  # mm
voxel_nx = 10
voxel_ny = 10
voxel_size_x = wall_size_x / voxel_nx
voxel_size_y = wall_size_y / voxel_ny
voxel_size_z = 1.0  # mm
wall_layer = "wall"
selected_voxels = ["0202", "0208", "0802", "0808", "0505"]

# Ursprung unten links
x_min = -wall_size_x / 2
y_min = -wall_size_y / 2
center_z = 0.5  # Mitte der 1 mm Wand

# ----------------------
# Voxeldaten vorbereiten
# ----------------------
voxels = {}
for ix in range(voxel_nx):
    for iy in range(voxel_ny):
        index = f"{ix:02d}{iy:02d}"  # z.B. "0000", "0001", ... "0909"
        center_x = x_min + (ix + 0.5) * voxel_size_x
        center_y = y_min + (iy + 0.5) * voxel_size_y
        voxels[index] = {
            "index": index,
            "center": [center_x, center_y, center_z],
            "rotation_matrix": [1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0],
            "size": [voxel_size_x, voxel_size_z, voxel_size_y],
            "layer": "wall",
            "hits": []
        }


# ----------------------
# HDF5-Dateien lesen
# ----------------------
total_simulated = 0
total_detected = 0
backward_photons = 0  # pz < 0

for i, file_path in enumerate(files):
    with h5py.File(file_path, "r") as f:
        # Simulierte Photonen pro Datei (falls gespeichert, sonst auf Länge der Arrays)
        if 'hit/vertices/evtid/entries' in f:
            total_simulated += np.sum(f['hit/vertices/entries'])
        else:
            # Fallback, falls nicht vorhanden
            total_simulated += f['hit']['optical']['x_position_in_m']['pages'].shape[0]
        
        # Alle Hits auslesen
        x = np.array(f['hit']['optical']['x_position_in_m']['pages'], dtype=np.float32) * 1000
        y = np.array(f['hit']['optical']['y_position_in_m']['pages'], dtype=np.float32) * 1000
        z = np.array(f['hit']['optical']['z_position_in_m']['pages'], dtype=np.float32) * 1000
        px = np.array(f['hit']['optical']['x_momentum_direction']['pages'], dtype=np.float32)
        py = np.array(f['hit']['optical']['y_momentum_direction']['pages'], dtype=np.float32)
        pz = np.array(f['hit']['optical']['z_momentum_direction']['pages'], dtype=np.float32)

        total_detected += len(x)

        backward_photons += np.sum(pz < 0)

        # ----------------------
        # Photonen zu Voxeln zuordnen
        # ----------------------
        ix = np.floor((x - x_min) / voxel_size_x).astype(int)
        iy = np.floor((y - y_min) / voxel_size_y).astype(int)
        ix = np.clip(ix, 0, voxel_nx-1)
        iy = np.clip(iy, 0, voxel_ny-1)

        for j in range(len(x)):
            voxel_index = f"{ix[j]:02d}{iy[j]:02d}"
            voxels[voxel_index]["hits"].append([x[j], y[j], z[j]])

hit_counts_selected = [len(voxels[idx]["hits"]) for idx in selected_voxels]
total_selected_hits = sum(hit_counts_selected)

# ----------------------
# Allgemeine Statistik
# ----------------------
print(f"Simulierte Photonen: {total_simulated}")
print(f"Detektierte Photonen: {total_detected}")
print(f"Photon Hits in den ausgewählten Voxeln ({', '.join(selected_voxels)}): {total_selected_hits}")
print(f"Photonen von hinten (pz < 0): {backward_photons}")

# ----------------------
# JSON exportieren (ohne Hits, nur voxel info)
# ----------------------
json_data = []
for voxel in voxels.values():
    # optional: nur voxels mit Hits speichern, sonst alle
    json_data.append({
        "index": voxel["index"],
        "center": voxel["center"],
        "rotation_matrix": voxel["rotation_matrix"],
        "size": voxel["size"],
        "layer": voxel["layer"]
    })

with open("voxelsTyvekWall.json", "w") as f:
    json.dump(json_data, f, indent=2)

# ----------------------
# Plot: Hitverteilung
# ----------------------
hit_counts = np.zeros((voxel_nx, voxel_ny), dtype=int)

for voxel in voxels.values():
    ix_plot = int(voxel["index"][:2])
    iy_plot = int(voxel["index"][2:])
    hit_counts[ix_plot, iy_plot] = len(voxel["hits"])


plt.figure(figsize=(10,10))
plt.imshow(hit_counts.T, origin="lower", cmap="viridis", interpolation="nearest", norm=LogNorm(vmin=hit_counts[hit_counts > 0].min(), vmax=hit_counts.max()))
plt.colorbar(label="Photon Hits")
plt.title("Photon Hitverteilung auf der Wand")

# ausgewählte Voxel markieren
for idx in selected_voxels:
    ix = int(idx[:2])
    iy = int(idx[2:])
    plt.scatter(ix, iy, color="red", s=50, marker="x", label=f"Voxel {idx}")

plt.xlabel("Voxel X")
plt.ylabel("Voxel Y")
plt.show()


plt.figure(figsize=(12,6))

hit_counts_selected = [len(voxels[idx]["hits"]) for idx in selected_voxels]
counts_ssd = Counter({idx: len(voxels[idx]["hits"]) for idx in selected_voxels})

plt.figure(figsize=(8,6))
plt.bar(selected_voxels, hit_counts_selected, color="orange", alpha=0.4)
plt.xlabel("Voxel Index")
plt.ylabel("Photon Hits")
plt.title("Photon Hits in ausgewählten Voxeln")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# ----------------------
# Konfiguration
# ----------------------
files = sorted(Path("/home/tbueger/code/weiteres/inspectPhotonsSSD/PMTWall").glob("*.hdf5"))  # Pfad zu deinen HDF5-Dateien

total_simulated = 0
total_detected = 0
all_uids = []

for i, file_path in enumerate(files):
    with h5py.File(file_path, "r") as f:
        # Simulierte Photonen pro Datei (falls gespeichert, sonst auf Länge der Arrays)
        if 'hit/vertices/evtid/entries' in f:
            total_simulated += np.sum(f['hit/vertices/entries'])
        else:
            # Fallback, falls nicht vorhanden
            total_simulated += f['hit']['optical']['x_position_in_m']['pages'].shape[0]
        
        # Alle Hits auslesen
        uid = np.array(f['hit']['optical']['det_uid']['pages'], dtype="U")

        total_detected += len(uid)

        # In Strings umwandeln und erstes Zeichen löschen
        uid_str = [str(u)[1:] for u in uid]
        all_uids.extend(uid_str)

# ----------------------
# Allgemeine Statistik
# ----------------------
print(f"Simulierte Photonen: {total_simulated}")
print(f"Detektierte Photonen: {total_detected}")

counts_pmt_tyvek = Counter(all_uids)
labels = list(counts_pmt_tyvek.keys())
values = list(counts_pmt_tyvek.values())

plt.figure(figsize=(12,6))
plt.bar(labels, values, color="steelblue", alpha=0.8)
plt.xlabel("Detector UID")
plt.ylabel("Photon Hits")
plt.title("Photon Hits pro PMT")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# ----------------------
# Konfiguration
# ----------------------
files = sorted(Path("/home/tbueger/code/weiteres/inspectPhotonsSSD/PMTsOnly").glob("*.hdf5"))  # Pfad zu deinen HDF5-Dateien

total_simulated = 0
total_detected = 0
all_uids = []

for i, file_path in enumerate(files):
    with h5py.File(file_path, "r") as f:
        # Simulierte Photonen pro Datei (falls gespeichert, sonst auf Länge der Arrays)
        if 'hit/vertices/evtid/entries' in f:
            total_simulated += np.sum(f['hit/vertices/entries'])
        else:
            # Fallback, falls nicht vorhanden
            total_simulated += f['hit']['optical']['x_position_in_m']['pages'].shape[0]
        
        # Alle Hits auslesen
        uid = np.array(f['hit']['optical']['det_uid']['pages'], dtype="U")

        total_detected += len(uid)

        # In Strings umwandeln und erstes Zeichen löschen
        uid_str = [str(u)[1:] for u in uid]
        all_uids.extend(uid_str)

# ----------------------
# Allgemeine Statistik
# ----------------------
print(f"Simulierte Photonen: {total_simulated}")
print(f"Detektierte Photonen: {total_detected}")

counts_pmt_only = Counter(all_uids)
labels = list(counts_pmt_only.keys())
values = list(counts_pmt_only.values())

plt.figure(figsize=(12,6))
plt.bar(labels, values, color="steelblue", alpha=0.8)
plt.xlabel("Detector UID")
plt.ylabel("Photon Hits")
plt.title("Photon Hits pro PMT")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# Alle Detector UIDs sammeln
all_uids = sorted(set(counts_ssd.keys()) | set(counts_pmt_tyvek.keys()) | set(counts_pmt_only.keys()))

# Werte vorbereiten (falls UID in einer Konfiguration fehlt -> 0 setzen)
ssd_values = [counts_ssd.get(uid, 0) for uid in all_uids]
tyvek_values = [counts_pmt_tyvek.get(uid, 0) for uid in all_uids]
pmt_only_values = [counts_pmt_only.get(uid, 0) for uid in all_uids]

# Plot
x = np.arange(len(all_uids))  # Positionen der Gruppen
width = 0.25  # Breite der Balken

plt.figure(figsize=(14,7))
plt.bar(x - width, ssd_values, width, label="SSD", color="orange", alpha=0.7)
plt.bar(x, tyvek_values, width, label="PMT mit Folie", color="steelblue", alpha=0.7)
plt.bar(x + width, pmt_only_values, width, label="nur PMT", color="green", alpha=0.7)

plt.xticks(x, all_uids, rotation=45, ha="right")
plt.xlabel("Detector UID")
plt.ylabel("Photon Hits")
plt.title("Photon Hits pro Detector UID – Vergleich SSD vs PMT")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()