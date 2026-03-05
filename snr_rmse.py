# run_lidar_pipeline_preprocessed.py
# Assumes dataset has already been preprocessed and saved as pre_*.npy/.pcd/.bin
# Performs: 1D signal construction -> Wavelet & VMD denoising -> SNR/RMSE -> figures/metrics.

import os, glob, json, math, numpy as np, matplotlib.pyplot as plt

# Optional (for .pcd)
try:
    import open3d as o3d
except Exception:
    o3d = None

# Optional (for VMD)
try:
    from vmdpy import VMD
    HAVE_VMD = True
except Exception:
    HAVE_VMD = False

import pywt

# --------------------------- Config ---------------------------
BASE = r"C:\Users\Vaasanth\Documents\Project\LiDAR_Proj\Level-1"
PRE_TRAIN = os.path.join(BASE, "results", "training", "preprocessed")
PRE_TEST  = os.path.join(BASE, "results", "testing",  "preprocessed")

OUT_BASE = os.path.join(BASE, "results")
FIG_OUT  = os.path.join(OUT_BASE, "figures_pre")
MET_OUT  = os.path.join(OUT_BASE, "metrics_pre")
os.makedirs(FIG_OUT, exist_ok=True)
os.makedirs(MET_OUT, exist_ok=True)

NUM_AZ_BINS   = 1024
WAVELET_NAME  = "db4"
WAVELET_MODE  = "soft"
WAVELET_LEVEL = None
VMD_PARAMS    = dict(alpha=1500, tau=0., K=4, DC=0, init=1, tol=1e-7)
MAX_FILES     = 18  # set None to run all

# ----------------------- Loaders ------------------------------
def load_frame(path: str):
    """
    Returns:
      xyz (N,3) float32, intensity (N,) float32 (0..1 if available else zeros)
    Supports: .npy (Nx4), .pcd (needs open3d), .bin (KITTI xyzI)
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path).astype(np.float32)
        if arr.shape[1] >= 4:
            return arr[:, :3], arr[:, 3]
        xyz = arr[:, :3]; inten = np.zeros(len(xyz), dtype=np.float32)
        return xyz, inten
    elif ext == ".pcd":
        if o3d is None:
            raise RuntimeError("open3d is required to read .pcd files")
        pcd = o3d.io.read_point_cloud(path)
        xyz = np.asarray(pcd.points, dtype=np.float32)
        if len(pcd.colors) == len(pcd.points) and len(pcd.points) > 0:
            cols = np.asarray(pcd.colors, dtype=np.float32)
            inten = cols.mean(axis=1).astype(np.float32)  # RGB->intensity proxy
        else:
            inten = np.zeros(len(xyz), dtype=np.float32)
        return xyz, inten
    elif ext == ".bin":
        arr = np.fromfile(path, dtype=np.float32).reshape(-1,4)
        return arr[:, :3].astype(np.float32), arr[:, 3].astype(np.float32)
    else:
        raise ValueError(f"Unsupported file: {path}")

# ------------------- 1D signal builders -----------------------
def to_azimuth_bins(points_xyz, intensity, num_bins=NUM_AZ_BINS):
    x, y = points_xyz[:,0], points_xyz[:,1]
    az = np.arctan2(y, x)  # [-pi, pi]
    bins = ((az + np.pi)/(2*np.pi) * num_bins).astype(np.int32).clip(0, num_bins-1)
    out = np.zeros(num_bins, dtype=np.float32)
    if np.var(intensity) > 1e-12:
        for i in range(num_bins):
            v = intensity[bins==i]
            out[i] = v.max() if v.size else 0.0
    else:
        # fallback: normalized counts
        for i in range(num_bins):
            out[i] = np.count_nonzero(bins==i)
        if out.max() > 0:
            out /= out.max()
    return out

# ----------------------- Denoisers -----------------------------
def wavelet_universal_denoise(sig,
                              wavelet=WAVELET_NAME,
                              level=WAVELET_LEVEL,
                              mode=WAVELET_MODE):
    coeffs = pywt.wavedec(sig, wavelet=wavelet, level=level, mode="periodization")
    sigma = np.median(np.abs(coeffs[-1]))/0.6745 if coeffs[-1].size else 0.0
    thr = sigma * np.sqrt(2*np.log(len(sig))) if sigma > 0 else 0.0
    den = [coeffs[0]]
    for c in coeffs[1:]:
        den.append(pywt.threshold(c, thr, mode=mode))
    rec = pywt.waverec(den, wavelet=wavelet, mode="periodization")
    return rec[:len(sig)].astype(np.float32)

def vmd_denoise(sig, params=VMD_PARAMS):
    if not HAVE_VMD:
        raise RuntimeError("vmdpy not installed. pip install vmdpy")
    z = sig.astype(np.float64)
    mu, sd = np.mean(z), np.std(z) + 1e-12
    z = (z - mu)/sd
    u, _, _ = VMD(z, **params)
    energies = np.sum(u**2, axis=1)
    frac = energies/(np.sum(energies)+1e-12)
    keep = frac >= 0.05
    z_rec = np.sum(u[keep], axis=0)
    x_rec = z_rec*sd + mu
    return x_rec.astype(np.float32)

# ---------------------- Metrics/Plots --------------------------
def rmse_snr(original, denoised):
    o = np.asarray(original, dtype=np.float64)
    d = np.asarray(denoised, dtype=np.float64)
    N = min(len(o), len(d))
    if N == 0:
        return {"RMSE": np.nan, "SNR_dB": np.nan}
    o, d = o[:N], d[:N]
    e = o - d
    rmse = float(np.sqrt(np.mean(e**2)))
    snr  = float(10*np.log10((np.var(d)+1e-12)/(np.var(e)+1e-12)))
    return {"RMSE": rmse, "SNR_dB": snr}

def plot_pair(a, b, title, out_png):
    plt.figure(figsize=(11,4))
    plt.plot(a, label="Original", alpha=0.6)
    plt.plot(b, label="Denoised", linewidth=2)
    plt.title(title); plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=150); plt.close()

# ------------------------- Runner ------------------------------
def run_on_folder(pre_dir, tag, max_files=MAX_FILES):
    files = sorted(glob.glob(os.path.join(pre_dir, "pre_*.*")))
    files = [p for p in files if os.path.splitext(p)[1].lower() in [".npy",".pcd",".bin"]]
    if not files:
        print(f"[WARN] No pre_* files in {pre_dir}"); return
    if max_files is not None:
        files = files[:max_files]

    rows = []
    for i, path in enumerate(files):
        print(f"[{tag}] {i+1}/{len(files)} {os.path.basename(path)}")
        xyz, inten = load_frame(path)

        # Build 1D signals
        az   = to_azimuth_bins(xyz, inten, NUM_AZ_BINS)
        xsig = xyz[:,0].astype(np.float32)

        # Wavelet metrics
        az_wav = wavelet_universal_denoise(az)
        x_wav  = wavelet_universal_denoise(xsig)
        m_az_w = rmse_snr(az, az_wav)
        m_x_w  = rmse_snr(xsig, x_wav)

        # VMD metrics (if available)
        if HAVE_VMD:
            az_vmd = vmd_denoise(az)
            x_vmd  = vmd_denoise(xsig[:min(4000,len(xsig))])
            m_az_v = rmse_snr(az, az_vmd)
            m_x_v  = rmse_snr(xsig[:len(x_vmd)], x_vmd)
        else:
            az_vmd, x_vmd = None, None
            m_az_v = {"RMSE": np.nan, "SNR_dB": np.nan}
            m_x_v  = {"RMSE": np.nan, "SNR_dB": np.nan}

        # Save example plots for first file
        if i == 0:
            plot_pair(az, az_wav, f"{tag} Azimuth Wavelet (db4)", os.path.join(FIG_OUT, f"{tag}_az_wavelet.png"))
            if HAVE_VMD:
                plot_pair(az, az_vmd, f"{tag} Azimuth VMD", os.path.join(FIG_OUT, f"{tag}_az_vmd.png"))
            plot_pair(xsig, x_wav, f"{tag} X Wavelet (db4)", os.path.join(FIG_OUT, f"{tag}_x_wavelet.png"))
            if HAVE_VMD:
                plot_pair(xsig[:len(x_vmd)], x_vmd, f"{tag} X VMD", os.path.join(FIG_OUT, f"{tag}_x_vmd.png"))

        rows.append({
            "file": os.path.basename(path),
            "az_wav_RMSE": m_az_w["RMSE"], "az_wav_SNR_dB": m_az_w["SNR_dB"],
            "az_vmd_RMSE": m_az_v["RMSE"], "az_vmd_SNR_dB": m_az_v["SNR_dB"],
            "x_wav_RMSE":  m_x_w["RMSE"],  "x_wav_SNR_dB":  m_x_w["SNR_dB"],
            "x_vmd_RMSE":  m_x_v["RMSE"],  "x_vmd_SNR_dB":  m_x_v["SNR_dB"],
        })

    # Save JSON and CSV summaries
    jpath = os.path.join(MET_OUT, f"{tag}_snr_rmse.json")
    with open(jpath, "w") as f:
        json.dump(rows, f, indent=2)

    cpath = os.path.join(MET_OUT, f"{tag}_snr_rmse.csv")
    with open(cpath, "w") as f:
        cols = list(rows[0].keys())
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")

    # Averages for PPT
    def avg(key):
        vals = [r[key] for r in rows if (r[key] is not None and not np.isnan(r[key]))]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "files": len(rows),
        "az_wav_RMSE_mean": avg("az_wav_RMSE"),
        "az_wav_SNR_dB_mean": avg("az_wav_SNR_dB"),
        "az_vmd_RMSE_mean": avg("az_vmd_RMSE"),
        "az_vmd_SNR_dB_mean": avg("az_vmd_SNR_dB"),
        "x_wav_RMSE_mean": avg("x_wav_RMSE"),
        "x_wav_SNR_dB_mean": avg("x_wav_SNR_dB"),
        "x_vmd_RMSE_mean": avg("x_vmd_RMSE"),
        "x_vmd_SNR_dB_mean": avg("x_vmd_SNR_dB"),
    }
    with open(os.path.join(MET_OUT, f"{tag}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    for i,j in summary.items():
        print(f"{i }: {j}")

def main():
    if os.path.isdir(PRE_TRAIN):
        run_on_folder(PRE_TRAIN, "train", MAX_FILES)
    if os.path.isdir(PRE_TEST):
        run_on_folder(PRE_TEST,  "test",  MAX_FILES)

if __name__ == "__main__":
    main()
