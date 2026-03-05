# plot_snr_rmse.py
# Reads *_snr_rmse.csv from your metrics folder and plots SNR/RMSE per file and means.

import os, glob, csv, numpy as np
import matplotlib.pyplot as plt

BASE = r"C:\Users\Vaasanth\Documents\Project\LiDAR_Proj\Level-1"
MET_DIR = os.path.join(BASE, "results", "metrics_pre")   # adjust if you used a different folder
OUT_DIR = os.path.join(BASE, "results", "figures_pre")
os.makedirs(OUT_DIR, exist_ok=True)

# Pick which CSV(s) to plot: last run for 'test' split with any K,alpha
csv_paths = sorted(glob.glob(os.path.join(MET_DIR, "test*_snr_rmse.csv")))
if not csv_paths:
    raise FileNotFoundError(f"No CSVs like test*_snr_rmse.csv found in {MET_DIR}")
csv_path = csv_paths[-1]  # latest

# Load rows
with open(csv_path, "r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Extract series
files = [r["file"] for r in rows]
idx = np.arange(len(files))

# after computing idx = np.arange(len(files))
step = max(1, len(files)//20)  # ~20 labels max
tick_idx = idx[::step]
tick_lbl = [os.path.splitext(f)[0] for f in files][::step]
plt.xticks(tick_idx, tick_lbl, rotation=45, ha="right")


def to_float_list(key):
    vals = []
    for r in rows:
        try:
            vals.append(float(r[key]))
        except:
            vals.append(np.nan)
    return np.array(vals, dtype=float)

az_wav_snr  = to_float_list("az_wav_SNR_dB")
az_vmd_snr  = to_float_list("az_vmd_SNR_dB")
x_wav_snr   = to_float_list("x_wav_SNR_dB")
x_vmd_snr   = to_float_list("x_vmd_SNR_dB")

az_wav_rmse = to_float_list("az_wav_RMSE")
az_vmd_rmse = to_float_list("az_vmd_RMSE")
x_wav_rmse  = to_float_list("x_wav_RMSE")
x_vmd_rmse  = to_float_list("x_vmd_RMSE")

# 1) SNR per file
plt.figure(figsize=(12,5))
plt.plot(idx, az_wav_snr,  "-o", label="Azimuth-Wavelet")
plt.plot(idx, az_vmd_snr,  "-o", label="Azimuth-VMD")
plt.plot(idx, x_wav_snr,   "-o", label="X-Wavelet")
plt.plot(idx, x_vmd_snr,   "-o", label="X-VMD")
plt.xticks(idx, [os.path.splitext(f)[0] for f in files], rotation=45, ha="right")
plt.ylabel("SNR (dB)")
plt.title("SNR per file")
plt.grid(True, alpha=0.3); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "snr_per_file.png"), dpi=180); plt.close()

# 2) RMSE per file

plt.figure(figsize=(12,5))
plt.plot(idx, az_wav_snr, "-o", label="Azimuth-Wavelet", markersize=3)
plt.plot(idx, az_vmd_snr, "-o", label="Azimuth-VMD",   markersize=3)
plt.plot(idx, x_wav_snr,  "-o", label="X-Wavelet",     markersize=3)
plt.plot(idx, x_vmd_snr,  "-o", label="X-VMD",         markersize=3)
# subsample ticks: ~20 labels max
step = max(1, len(files)//20)
tick_idx = idx[::step]
tick_lbl = [os.path.splitext(f)[0] for f in files][::step]
ax = plt.gca()
ax.set_xticks(tick_idx)
ax.set_xticklabels(tick_lbl, rotation=45, ha="right")
plt.ylabel("SNR (dB)")
plt.title("SNR per file")
plt.grid(True, alpha=0.3); plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "snr_per_file.png"), dpi=180)
plt.close()

plt.figure(figsize=(12,5))
plt.plot(idx, az_wav_rmse, "-o", label="Azimuth-Wavelet", markersize=3)
plt.plot(idx, az_vmd_rmse, "-o", label="Azimuth-VMD",   markersize=3)
plt.plot(idx, x_wav_rmse,  "-o", label="X-Wavelet",     markersize=3)
plt.plot(idx, x_vmd_rmse,  "-o", label="X-VMD",         markersize=3)
step = max(1, len(files)//20)
tick_idx = idx[::step]
tick_lbl = [os.path.splitext(f)[0] for f in files][::step]
ax = plt.gca()
ax.set_xticks(tick_idx)
ax.set_xticklabels(tick_lbl, rotation=45, ha="right")
plt.ylabel("RMSE")
plt.title("RMSE per file")
plt.grid(True, alpha=0.3); plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "rmse_per_file.png"), dpi=180)
plt.close()


# 3) Means bar chart
labels = ["Azimuth-Wavelet","Azimuth-VMD","X-Wavelet","X-VMD"]
snr_means  = [np.nanmean(az_wav_snr), np.nanmean(az_vmd_snr),
              np.nanmean(x_wav_snr),  np.nanmean(x_vmd_snr)]
rmse_means = [np.nanmean(az_wav_rmse), np.nanmean(az_vmd_rmse),
              np.nanmean(x_wav_rmse),  np.nanmean(x_vmd_rmse)]

fig, axes = plt.subplots(1,2, figsize=(12,5))
# SNR means
ax = axes[0]
ax.bar(labels, snr_means, color=["#4c72b0","#55a868","#c44e52","#8172b3"])
ax.set_ylabel("SNR (dB)"); ax.set_title("Mean SNR")
for i,v in enumerate(snr_means):
    ax.text(i, v + (0.02*max(1,abs(v))), f"{v:.2f}", ha="center", va="bottom")
ax.tick_params(axis="x", rotation=25)

# RMSE means
ax = axes[1]
ax.bar(labels, rmse_means, color=["#4c72b0","#55a868","#c44e52","#8172b3"])
ax.set_ylabel("RMSE"); ax.set_title("Mean RMSE")
for i,v in enumerate(rmse_means):
    ax.text(i, v + (0.02*max(1,abs(v))), f"{v:.3f}", ha="center", va="bottom")
ax.tick_params(axis="x", rotation=25)

plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "means_bar.png"), dpi=180); plt.close()

print("Saved plots to:", OUT_DIR)
