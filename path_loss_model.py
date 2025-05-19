#!/usr/bin/env python3
# coding: utf-8
"""
path_loss_model.py
==================
Ray-Tracing Path-Loss Optimisation Tool (Enhanced)

Adds:
 - Path-loss exponent n
 - Calibration offset C0
 - Weighted, robust (Huber) least-squares
"""

import argparse
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import trimesh
from trimesh.ray.ray_triangle import RayMeshIntersector
from scipy.optimize import least_squares
import pyproj

# Speed of light [m/s]
C_LIGHT = 299_792_458.0

def parse_args():
    p = argparse.ArgumentParser(
        description="Optimise L_b, L_v, n, and C0 so simulated path-loss matches measurements."
    )
    p.add_argument("--meas-xlsx",  default="5g_dl.xlsx",
                   help="Drive-test measurements Excel file")
    p.add_argument("--cells-xlsx", default="hucre_bilgileri.xlsx",
                   help="Cell (eNB/gNB) info Excel file")
    p.add_argument("--obj-dir",    default=".",
                   help="Directory containing .obj files")
    p.add_argument("--freq-ghz",   type=float, default=3.5,
                   help="Carrier frequency in GHz")
    p.add_argument("--output",     default="results.csv",
                   help="Output CSV for per-sample results")
    p.add_argument("--use-ground-reflection", action="store_true",
                   help="Enable simple ground reflection model")
    args, _ = p.parse_known_args()
    return args

def build_coordinate_transform(ref_ll, ref_xyz):
    """
    Create lon/lat → scene (x,z) transform using fixed UTM-35N (EPSG:32635).
    ref_ll: (lon, lat) of reference
    ref_xyz: corresponding scene (x, y, z)
    """
    transformer = pyproj.Transformer.from_crs(4326, 32635, always_xy=True)
    utm_x0, utm_y0 = transformer.transform(*ref_ll)
    t_x = utm_x0 - ref_xyz[0]
    t_y = utm_y0 - ref_xyz[2]

    def ll_to_scene(lon, lat):
        ux, uy = transformer.transform(lon, lat)
        return np.array([ux - t_x, 0.0, uy - t_y], dtype=np.float64)

    return ll_to_scene

def segment_length(mesh_intersector, origin, direction, max_dist):
    locs, _, _ = mesh_intersector.intersects_location(
        ray_origins=[origin],
        ray_directions=[direction],
        multiple_hits=True
    )
    if len(locs) < 2:
        return 0.0
    t_vals = np.dot(locs - origin, direction)
    t_vals = np.sort(t_vals[(t_vals >= 0.0) & (t_vals <= max_dist)])
    if len(t_vals) % 2 == 1:
        t_vals = t_vals[:-1]
    segs = t_vals.reshape(-1, 2)
    return float(np.sum(segs[:,1] - segs[:,0]))

def main():
    args = parse_args()
    freq_hz = args.freq_ghz * 1e9
    const_fs = 20.0 * math.log10(4.0 * math.pi * freq_hz / C_LIGHT)

    # 1) Load measurements and cells
    meas  = pd.read_excel(args.meas_xlsx).dropna(subset=["NR_UE_Pathloss_DL_0_mean"])
    cells = pd.read_excel(args.cells_xlsx)
    meas.rename(columns={"NR_UE_PCI_0_mode":"PCI"}, inplace=True)

    # 2) Load geometry
    obj_dir    = Path(args.obj_dir)
    buildings  = trimesh.load(obj_dir/"buildings.obj",  force='mesh')
    vegetation = trimesh.load(obj_dir/"vegetation.obj", force='mesh')
    ground     = trimesh.load(obj_dir/"ground.obj",     force='mesh')
    bs_scene   = trimesh.load(obj_dir/"base_stations.obj",
                              split_object=True, group_material=False)

    inter_b = RayMeshIntersector(buildings)
    inter_v = RayMeshIntersector(vegetation)

    # 3) Coordinate transform
    bs_keys = list(bs_scene.geometry.keys())
    ref_xyz = np.array(bs_scene.geometry[bs_keys[0]].centroid)
    REF_LL  = (29.0233668, 41.1073413)
    ll2sc   = build_coordinate_transform(REF_LL, ref_xyz)

    # 4) PCI → index map
    pci_map = {}
    for idx, pci in enumerate(cells["PCI"]):
        try:
            pci_map[int(pci)] = idx
        except:
            continue

    # 5) Build feature arrays
    D, LB, LV, PL = [], [], [], []
    for _, row in tqdm(meas.iterrows(), total=len(meas), desc="Building features"):
        try:
            pci = int(row.PCI)
            if pci not in pci_map: continue
            bs_idx = pci_map[pci]
            bs_pos = np.array(bs_scene.geometry[bs_keys[bs_idx]].centroid)
            ue_pos = ll2sc(row.Longitude_mean, row.Latitude_mean)
            ue_pos[1] = 1.5
            d_vec = ue_pos - bs_pos
            dist = np.linalg.norm(d_vec)
            if dist < 1.0: continue
            dir_ = d_vec / dist
            lb = segment_length(inter_b, bs_pos, dir_, dist)
            lv = segment_length(inter_v, bs_pos, dir_, dist)
            D .append(dist)
            LB.append(lb)
            LV.append(lv)
            PL.append(row.NR_UE_Pathloss_DL_0_mean)
        except:
            continue

    D  = np.array(D)
    LB = np.array(LB)
    LV = np.array(LV)
    PL = np.array(PL)

    # 6) Optimisation: [L_b, L_v, n, C0]
    W = 1.0 / np.maximum(D, 1.0)

    def residuals(params):
        L_b, L_v, n, C0 = params
        pred = (
            C0
            + const_fs
            + 10.0 * n * np.log10(D)
            + L_b * LB
            + L_v * LV
        )
        return (pred - PL) * W

    x0     = np.array([0.5, 0.2, 2.0, 0.0])
    bounds = (
        [0.0, 0.0, 1.0, -10.0],
        [10.0,10.0,6.0,  10.0]
    )

    res = least_squares(
        residuals,
        x0,
        bounds=bounds,
        loss='huber',
        f_scale=1.0,
        verbose=1
    )

    L_b_opt, L_v_opt, n_opt, C0_opt = res.x
    rmse = math.sqrt(np.mean(((res.fun) / W)**2))

    print("\nOptimisation complete")
    print("----------------------")
    print(f"  L_b  = {L_b_opt:.3f} dB/m")
    print(f"  L_v  = {L_v_opt:.3f} dB/m")
    print(f"  n    = {n_opt:.3f}")
    print(f"  C0   = {C0_opt:.3f} dB")
    print(f"  RMSE = {rmse:.3f} dB  (N = {len(PL)})")

    # 7) Build output DataFrame
    preds = (
        C0_opt
        + const_fs
        + 10.0 * n_opt * np.log10(D)
        + L_b_opt * LB
        + L_v_opt * LV
    )
    df_out = pd.DataFrame({
        "distance_m"    : D,
        "len_building"  : LB,
        "len_veg"       : LV,
        "pathloss_meas" : PL,
        "pathloss_pred" : preds,
        "residual_db"   : preds - PL
    })
    df_out.to_csv(args.output, index=False)
    print(f"Per-sample results written → {args.output}")

    # 8) Optional: plot measured vs predicted
    idx = np.arange(len(PL))
    plt.figure(figsize=(10,4))
    plt.plot(idx, PL,    label="Measured",    color="gold",      linewidth=1)
    plt.plot(idx, preds, label="Predicted",   color="orangered", linewidth=1)
    plt.xlabel("Sample index")
    plt.ylabel("Path-loss (dB)")
    plt.title("Measured vs Predicted Path-loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

if _name_ == "_main_":
    warnings.filterwarnings("ignore")
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
