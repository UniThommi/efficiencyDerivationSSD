#!/usr/bin/env python3
"""
3D Visualisierung der Voxel auf der Tyvek-Kugel
Zeigt alle Voxel und hebt die 50 ausgewählten PMT-Positionen rot hervor
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def load_json(filepath):
    """Lade JSON-Datei mit Voxel-Daten"""
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_sphere(ax, radius, alpha=0.1, color='cyan'):
    """Plotte eine transparente Kugel"""
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, alpha=alpha, color=color, edgecolor='none')

def plot_voxels_3d(all_voxels_file, selected_voxels_file, sphere_radius=1739.52):
    """
    Hauptfunktion für 3D-Plot
    
    Args:
        all_voxels_file: JSON mit allen Voxeln
        selected_voxels_file: JSON mit ausgewählten Voxeln
        sphere_radius: Radius der Kugel in mm
    """
    # Lade Voxel-Daten
    print(f"Lade alle Voxel aus: {all_voxels_file}")
    all_voxels = load_json(all_voxels_file)
    
    print(f"Lade ausgewählte Voxel aus: {selected_voxels_file}")
    selected_voxels = load_json(selected_voxels_file)
    
    # Extrahiere IDs der ausgewählten Voxel
    selected_ids = set([v['index'] for v in selected_voxels])
    
    print(f"Anzahl aller Voxel: {len(all_voxels)}")
    print(f"Anzahl ausgewählter Voxel: {len(selected_voxels)}")
    
    # Erstelle Figure und 3D Axes
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plotte Kugel
    print("Plotte Tyvek-Kugel...")
    plot_sphere(ax, sphere_radius, alpha=0.15, color='lightblue')
    
    # Separiere Voxel in ausgewählt und nicht-ausgewählt
    all_positions = []
    selected_positions = []
    
    for voxel in all_voxels:
        pos = voxel['center']
        if voxel['index'] in selected_ids:
            selected_positions.append(pos)
        else:
            all_positions.append(pos)
    
    all_positions = np.array(all_positions)
    selected_positions = np.array(selected_positions)
    
    # Plotte alle Voxel (grau/klein)
    print("Plotte alle Voxel...")
    if len(all_positions) > 0:
        ax.scatter(all_positions[:, 0], all_positions[:, 1], all_positions[:, 2],
                  c='gray', s=5, alpha=0.3, label=f'Alle Voxel ({len(all_positions)})')
    
    # Plotte ausgewählte Voxel (rot/groß)
    print("Plotte ausgewählte PMT-Positionen...")
    if len(selected_positions) > 0:
        ax.scatter(selected_positions[:, 0], selected_positions[:, 1], selected_positions[:, 2],
                  c='red', s=50, alpha=0.9, label=f'PMT-Positionen ({len(selected_positions)})',
                  edgecolors='darkred', linewidths=1)
    
    # Achsenbeschriftung und Titel
    ax.set_xlabel('X [mm]', fontsize=12)
    ax.set_ylabel('Y [mm]', fontsize=12)
    ax.set_zlabel('Z [mm]', fontsize=12)
    ax.set_title('Voxelisierung der Tyvek-Kugel mit PMT-Positionen', fontsize=14, fontweight='bold')
    
    # Legende
    ax.legend(loc='upper right', fontsize=10)
    
    # Setze gleiche Skalierung für alle Achsen
    max_range = sphere_radius * 1.1
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    # Gleiche Aspekt-Ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Speichern und anzeigen
    output_file = 'voxel_visualization_3d.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot gespeichert als: {output_file}")
    
    plt.show()

def plot_voxels_multiple_views(all_voxels_file, selected_voxels_file, sphere_radius=1739.52
):
    """
    Erstelle Plots aus verschiedenen Blickwinkeln
    """
    # Lade Voxel-Daten
    all_voxels = load_json(all_voxels_file)
    selected_voxels = load_json(selected_voxels_file)
    selected_ids = set([v['index'] for v in selected_voxels])
    
    # Separiere Positionen
    all_positions = []
    selected_positions = []
    
    for voxel in all_voxels:
        pos = voxel['center']
        if voxel['index'] in selected_ids:
            selected_positions.append(pos)
        else:
            all_positions.append(pos)
    
    all_positions = np.array(all_positions)
    selected_positions = np.array(selected_positions)
    
    # Erstelle Figure mit 2x2 Subplots
    fig = plt.figure(figsize=(16, 14))
    
    views = [
        (30, 45, 'Standard View'),
        (0, 0, 'Front View (XY)'),
        (0, 90, 'Side View (XZ)'),
        (90, 0, 'Top View (YZ)')
    ]
    
    for idx, (elev, azim, title) in enumerate(views, 1):
        ax = fig.add_subplot(2, 2, idx, projection='3d')
        
        # Plotte Kugel
        plot_sphere(ax, sphere_radius, alpha=0.1, color='lightblue')
        
        # Plotte Voxel
        if len(all_positions) > 0:
            ax.scatter(all_positions[:, 0], all_positions[:, 1], all_positions[:, 2],
                      c='gray', s=3, alpha=0.2)
        
        if len(selected_positions) > 0:
            ax.scatter(selected_positions[:, 0], selected_positions[:, 1], selected_positions[:, 2],
                      c='red', s=40, alpha=0.9, edgecolors='darkred', linewidths=0.5)
        
        # Setze Blickwinkel
        ax.view_init(elev=elev, azim=azim)
        
        # Achsen
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        max_range = sphere_radius * 1.1
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        ax.set_box_aspect([1, 1, 1])
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Voxelisierung: {len(all_voxels)} Voxel, {len(selected_positions)} PMTs',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_file = 'voxel_visualization_multiview.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Multi-View Plot gespeichert als: {output_file}")
    
    plt.show()

if __name__ == "__main__":
    # Pfade zu den JSON-Dateien
    path = "/home/tbueger/code/sim/legend-pygeom-l1000"
    all_voxels_file = os.path.join(path, "voxelsTyvekSphere.json")
    selected_voxels_file = os.path.join(path, "voxelsTyvekSphere_selected.json")
    sphere_radius = 1739.52  # mm
    
    print("=== 3D Voxel Visualisierung ===\n")
    
    # Hauptplot
    plot_voxels_3d(all_voxels_file, selected_voxels_file, sphere_radius)
    
    # Multi-View Plot
    print("\n=== Erstelle Multi-View Plot ===\n")
    plot_voxels_multiple_views(all_voxels_file, selected_voxels_file, sphere_radius)
    
    print("\n=== Fertig! ===")