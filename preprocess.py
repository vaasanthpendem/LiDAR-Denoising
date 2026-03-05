# preprocess.py

'''This program is for preprocessing the .bin files and get the minimal 
denoising of the point cloud data'''

import os
import numpy as np
import open3d as o3d

def load_bin(bin_path):
    """Load KITTI-format .bin LiDAR file"""
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points

def clean_points(points, distance_lim=100.0):
    """Remove NaNs and far-away points"""
    mask = np.all(np.isfinite(points), axis=1) & (np.linalg.norm(points[:, :3], axis=1) < distance_lim)
    return points[mask]

def voxel_downsample(points, voxel_size=0.05):
    """Voxel downsample point cloud"""
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[:, :3]))
    pcd.colors = o3d.utility.Vector3dVector(np.tile(points[:, 3:4], (1, 3)))
    ds_pcd = pcd.voxel_down_sample(voxel_size)
    ds_points = np.asarray(ds_pcd.points)
    ds_intensity = np.asarray(ds_pcd.colors)[:, 0]
    return np.hstack([ds_points, ds_intensity.reshape(-1, 1)]), ds_pcd

def remove_outliers(pcd, nb_neighbors=20, std_ratio=2.0):
    """Statistical outlier removal"""
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)
    cleaned = pcd.select_by_index(ind)
    cpoints = np.asarray(cleaned.points)
    cintensity = np.asarray(cleaned.colors)[:, 0]
    return np.hstack([cpoints, cintensity.reshape(-1, 1)]), cleaned

def preprocess_folder(in_folder, out_folder):
    os.makedirs(out_folder, exist_ok=True)
    bin_files = sorted([f for f in os.listdir(in_folder) if f.endswith(".bin")])
    for i, fname in enumerate(bin_files):
        print(f"Preprocessing {fname} ({i+1}/{len(bin_files)})...")
        raw_points = load_bin(os.path.join(in_folder, fname))
        cleaned_points = clean_points(raw_points, distance_lim=100.0)
        ds_points, ds_pcd = voxel_downsample(cleaned_points, voxel_size=0.05)
        so_points, so_pcd = remove_outliers(ds_pcd, nb_neighbors=20, std_ratio=2.0)
        np.save(os.path.join(out_folder, f"pre_{i:06d}.npy"), so_points)
        o3d.io.write_point_cloud(os.path.join(out_folder, f"pre_{i:06d}.pcd"), so_pcd)

if __name__ == "__main__":
    base = r"C:\Users\Vaasanth\Documents\Project\LiDAR_Proj\Level-1"
    preprocess_folder(os.path.join(base, "training", "velodyne"),
                      os.path.join(base, "results", "training", "preprocessed"))
    preprocess_folder(os.path.join(base, "testing", "velodyne"),
                      os.path.join(base, "results", "testing", "preprocessed"))
