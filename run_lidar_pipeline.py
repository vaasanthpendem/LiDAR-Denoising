# run_lidar_pipeline.py
'''This program applies the denoising algorithms and process the data'''
import os
import sys
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import pywt
from numpy.linalg import svd
from vmdpy import VMD
import open3d as o3d

# ------------------------------
# Configuration
# ------------------------------
BASE = r"C:\Users\Vaasanth\Documents\Project\LiDAR_Proj\Level-1"  
RAW_TRAIN = os.path.join(BASE, "training", "velodyne")
RAW_TEST  = os.path.join(BASE, "testing",  "velodyne")

OUT_BASE      = os.path.join(BASE, "results")
PRE_TRAIN_OUT = os.path.join(OUT_BASE, "training", "preprocessed")
PRE_TEST_OUT  = os.path.join(OUT_BASE, "testing",  "preprocessed")
FIG_OUT       = os.path.join(OUT_BASE, "figures")

os.makedirs(PRE_TRAIN_OUT, exist_ok=True)
os.makedirs(PRE_TEST_OUT,  exist_ok=True)
os.makedirs(FIG_OUT,       exist_ok=True)

# Preprocess params
DIST_LIMIT = 100.0
MIN_Z, MAX_Z = -2.0, 2.5
VOXEL_SIZE = 0.05
NB_NEIGHBORS = 24
STD_RATIO = 1.5
USE_RADIUS_FILTER = True
RADIUS_NB_POINTS = 8
RADIUS_VALUE = 0.30

# Denoising params
NUM_AZ_BINS = 1024
WAVELET_NAME = "db4"     # better than db1 for spiky noise
WAVELET_MODE = "soft"
WAVELET_LEVEL = None      # auto max
SVD_RANK = 10
VMD_PARAMS = dict(alpha=2000, tau=0., K=3, DC=0, init=1, tol=1e-7)

# ------------------------------
# IO helpers
# ------------------------------
def list_bins(dir_path):
    return sorted(glob.glob(os.path.join(dir_path, "*.bin")))

def load_kitti_bin(bin_path):
    """Return Nx4: x,y,z,intensity"""
    arr = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return arr

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

# ------------------------------
# Preprocessing utilities
# ------------------------------
def clean_points(points, distance_lim=DIST_LIMIT, min_z=MIN_Z, max_z=MAX_Z):
    """Remove NaNs, clip by distance and z band."""
    mask = np.all(np.isfinite(points), axis=1)
    if points.shape[1] >= 3:
        dist = np.linalg.norm(points[:, :3], axis=1)
        mask &= (dist < distance_lim)
        mask &= (points[:, 2] > min_z) & (points[:, 2] < max_z)
    return points[mask]

def voxel_downsample(points_xyz, intensity, voxel_size=VOXEL_SIZE):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz)
    if intensity is not None and len(intensity) == len(points_xyz):
        cols = np.repeat(intensity.reshape(-1,1), 3, axis=1)
        pcd.colors = o3d.utility.Vector3dVector(cols)
    ds = pcd.voxel_down_sample(voxel_size=voxel_size)
    pts = np.asarray(ds.points)
    cols = np.asarray(ds.colors) if len(ds.colors) == len(ds.points) else None
    inten = cols[:,0].astype(np.float32) if cols is not None else np.zeros(len(pts), dtype=np.float32)
    return pts, inten, ds

def remove_statistical_outliers(pcd, nb_neighbors=NB_NEIGHBORS, std_ratio=STD_RATIO):
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd.select_by_index(ind)

def remove_radius_outliers(pcd, nb_points=RADIUS_NB_POINTS, radius=RADIUS_VALUE):
    cl, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    return pcd.select_by_index(ind)


# 1D signal constructions
def to_azimuth_bins(points_xyz, intensity, num_bins=NUM_AZ_BINS):
    """Max intensity per azimuth bin."""
    if intensity is None or len(intensity) != len(points_xyz):
        intensity = np.zeros(len(points_xyz), dtype=np.float32)
    x, y = points_xyz[:,0], points_xyz[:,1]
    az = np.arctan2(y, x)  # [-pi, pi]
    bins = ((az + np.pi) / (2*np.pi) * num_bins).astype(np.int32)
    bins = np.clip(bins, 0, num_bins-1)
    out = np.zeros(num_bins, dtype=np.float32)
    for i in range(num_bins):
        vals = intensity[bins == i]
        out[i] = vals.max() if vals.size else 0.0
    return out

# ------------------------------
# Denoising methods
# ------------------------------
def wavelet_universal_denoise(sig, wavelet=WAVELET_NAME, level=WAVELET_LEVEL, mode=WAVELET_MODE):
    coeffs = pywt.wavedec(sig, wavelet=wavelet, level=level, mode='periodization')
    sigma = np.median(np.abs(coeffs[-1]))/0.6745 if coeffs[-1].size else 0.0
    thr = sigma * np.sqrt(2*np.log(len(sig))) if sigma > 0 else 0.0
    den = [coeffs[0]]  # keep approximation untouched for stronger structure retention
    for c in coeffs[1:]:
        den.append(pywt.threshold(c, thr, mode=mode))
    rec = pywt.waverec(den, wavelet=wavelet, mode='periodization')
    return rec[:len(sig)]

def svd_lowrank(points_xyz, rank=SVD_RANK):
    """Low-rank geometric smoothing on an Nx3 slice (centered)."""
    if len(points_xyz) == 0:
        return points_xyz
    X = points_xyz.astype(np.float64)
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    r = max(1, min(rank, min(Xc.shape)))
    U, s, Vt = svd(Xc, full_matrices=False)
    s[r:] = 0.0
    Xs = (U * s) @ Vt + mu
    return Xs.astype(np.float32)

def vmd_denoise(sig, params=VMD_PARAMS):
    sig = sig.astype(np.float64)
    u, u_hat, omega = VMD(sig, **params)
    # Sum IMFs as denoised signal (alternative: select subset of modes)
    return np.sum(u, axis=0).astype(np.float32)

# ------------------------------
# Plots and saving
# ------------------------------
def plot_signal_pair(a, b, title, out_png):
    plt.figure(figsize=(11,4))
    plt.plot(a, label="Original", alpha=0.65)
    plt.plot(b, label="Denoised", linewidth=2)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def save_pcd(points_xyz, intensity, out_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz)
    cols = np.repeat(intensity.reshape(-1,1), 3, axis=1) if intensity is not None else np.zeros((len(points_xyz),3), dtype=np.float32)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    o3d.io.write_point_cloud(out_path, pcd)

# ------------------------------
# Pipeline
# ------------------------------
def preprocess_split(in_folder, out_folder, split_name):
    params = dict(distance_lim=DIST_LIMIT, min_z=MIN_Z, max_z=MAX_Z,
                  voxel_size=VOXEL_SIZE, nb_neighbors=NB_NEIGHBORS, std_ratio=STD_RATIO,
                  use_radius=USE_RADIUS_FILTER, radius_pts=RADIUS_NB_POINTS, radius=RADIUS_VALUE)
    save_json(params, os.path.join(out_folder, f"{split_name}_preprocess_params.json"))

    bins = list_bins(in_folder)
    for idx, bp in enumerate(bins):
        print(f"[{split_name}] {idx+1}/{len(bins)}: {os.path.basename(bp)}")
        raw = load_kitti_bin(bp)            # Nx4
        raw = clean_points(raw, DIST_LIMIT, MIN_Z, MAX_Z)
        if raw.size == 0:
            continue
        xyz, inten = raw[:, :3], raw[:, 3]

        xyz_ds, inten_ds, pcd_ds = voxel_downsample(xyz, inten, VOXEL_SIZE)
        pcd_clean = remove_statistical_outliers(pcd_ds, NB_NEIGHBORS, STD_RATIO)
        if USE_RADIUS_FILTER:
            pcd_clean = remove_radius_outliers(pcd_clean, RADIUS_NB_POINTS, RADIUS_VALUE)

        xyz_c = np.asarray(pcd_clean.points)
        cols  = np.asarray(pcd_clean.colors) if len(pcd_clean.colors) == len(pcd_clean.points) else None
        inten_c = cols[:,0].astype(np.float32) if cols is not None else np.zeros(len(xyz_c), dtype=np.float32)

        np.save(os.path.join(out_folder, f"pre_{idx:06d}.npy"),
                np.hstack([xyz_c, inten_c.reshape(-1,1)]))
        save_pcd(xyz_c, inten_c, os.path.join(out_folder, f"pre_{idx:06d}.pcd"))

    print(f"[{split_name}] Preprocessing done -> {out_folder}")

def demo_on_first(pre_folder, split_tag):
    npys = sorted(glob.glob(os.path.join(pre_folder, "*.npy")))
    if not npys:
        print(f"[WARN] No preprocessed .npy in {pre_folder}")
        return
    npy = npys[0]
    arr = np.load(npy)  # Nx4
    xyz, inten = arr[:, :3], arr[:, 3]

    # 1) Wavelet on azimuth intensity
    az = to_azimuth_bins(xyz, inten, NUM_AZ_BINS)
    az_den = wavelet_universal_denoise(az, WAVELET_NAME, WAVELET_LEVEL, WAVELET_MODE)
    np.save(os.path.join(pre_folder, f"az_wavelet_{split_tag}.npy"), az_den)
    plot_signal_pair(az, az_den, f"{split_tag} Azimuth Intensity Wavelet ({WAVELET_NAME})",
                     os.path.join(FIG_OUT, f"{split_tag}_az_wavelet.png"))

    # 2) Wavelet on X-signal
    xsig = xyz[:,0].astype(np.float32)
    x_den = wavelet_universal_denoise(xsig, WAVELET_NAME, WAVELET_LEVEL, WAVELET_MODE)
    plot_signal_pair(xsig, x_den, f"{split_tag} X-signal Wavelet ({WAVELET_NAME})",
                     os.path.join(FIG_OUT, f"{split_tag}_x_wavelet.png"))

    # 3) VMD on short X-signal
    short = xsig[:min(2000, len(xsig))].astype(np.float64)
    vmd_den = vmd_denoise(short, VMD_PARAMS)
    np.save(os.path.join(pre_folder, f"x_vmd_{split_tag}.npy"), vmd_den)
    plot_signal_pair(short, vmd_den, f"{split_tag} X-signal VMD (K={VMD_PARAMS['K']}, alpha={VMD_PARAMS['alpha']})",
                     os.path.join(FIG_OUT, f"{split_tag}_x_vmd.png"))

    # 4) SVD low-rank smoothing on slice
    blk = xyz[:min(800, len(xyz)), :3]
    svd_sm = svd_lowrank(blk, SVD_RANK)
    # quick 2D projection comparison
    plt.figure(figsize=(6,6))
    plt.scatter(blk[:,0], blk[:,1], s=3, alpha=0.35, label="Original")
    plt.scatter(svd_sm[:,0], svd_sm[:,1], s=3, alpha=0.8,  label=f"SVD rank {SVD_RANK}")
    plt.title(f"{split_tag} SVD Low-rank Geometric Smoothing")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_OUT, f"{split_tag}_svd_scatter.png"), dpi=150)
    plt.close()

    # Save a “cleaned cloud” PCD again for this demo item
    save_pcd(xyz, inten, os.path.join(pre_folder, f"{split_tag}_clean_preview.pcd"))
    print(f"[{split_tag}] Denoising demos saved in {FIG_OUT} and arrays saved in {pre_folder}")

def main():
    # 1) Preprocess both splits if present
    if os.path.isdir(RAW_TRAIN) and len(list_bins(RAW_TRAIN)):
        preprocess_split(RAW_TRAIN, PRE_TRAIN_OUT, "train")
    if os.path.isdir(RAW_TEST) and len(list_bins(RAW_TEST)):
        preprocess_split(RAW_TEST, PRE_TEST_OUT, "test")

    # 2) Demo denoising on first preprocessed frame of each split
    if os.path.isdir(PRE_TRAIN_OUT):
        demo_on_first(PRE_TRAIN_OUT, "train")
    if os.path.isdir(PRE_TEST_OUT):
        demo_on_first(PRE_TEST_OUT, "test")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
