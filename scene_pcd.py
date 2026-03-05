# view_many_kitti.py
'''This program plots the preprocessed data and we can see the plot per scene'''
import os, glob, numpy as np, matplotlib.pyplot as plt, open3d as o3d

BASE = r"C:\Users\Vaasanth\Documents\Project\LiDAR_Proj\Level-1"
RAW_DIR = os.path.join(BASE, "training", "velodyne")  # or "testing/velodyne"

bin_list = sorted(glob.glob(os.path.join(RAW_DIR, "*.bin")))
assert bin_list, f"No .bin files in {RAW_DIR}"
print(f"Found {len(bin_list)} frames")

def load_and_color(bin_path, color_by="z"):
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1,4)
    xyz, inten = pts[:,:3], pts[:,3]
    scalar = xyz[:,2] if color_by=="z" else inten
    smin, smax = np.percentile(scalar, 1), np.percentile(scalar, 99)
    sn = np.clip((scalar - smin) / (smax - smin + 1e-9), 0, 1)
    colors = plt.get_cmap("turbo")(sn)[:,:3]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

# Choose an index to change frames
idx = 10
pcd = load_and_color(bin_list[idx], color_by="z")

# Pretty settings
pcd = pcd.voxel_down_sample(0.05)
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=24, std_ratio=1.5)

vis = o3d.visualization.Visualizer()
vis.create_window(window_name=f"Frame {idx}: {os.path.basename(bin_list[idx])}", width=1600, height=900)
vis.add_geometry(pcd)
opt = vis.get_render_option()
opt.background_color = np.array([0,0,0])
opt.point_size = 2.0
vis.run()
vis.destroy_window()
