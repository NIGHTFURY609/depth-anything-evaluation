"""Full-resolution crops of projected points on sharp objects, to judge alignment by eye."""
import sys
sys.path.insert(0, ".")
import cv2
import numpy as np
from src.config import load_config
from src.gt import load_calibration
from src.projection import project_points
from src.rellis import list_frames, load_scan

cfg = load_config()
W, H = cfg["gt"]["source_size"]
OUT = "results/overlays/"
for seq, wanted, (x0, y0, x1, y1), name in [
    ("00001", 1545, (800, 450, 1700, 800), "barrier"),
    ("00002", 1382, (640, 480, 1560, 1000), "tree"),
    ("00003", 1455, (380, 240, 1100, 720), "trees"),
]:
    frames = {f.frame: f for f in list_frames(cfg["paths"]["data_root"], seq)}
    f = frames[wanted]
    calib = load_calibration(cfg, seq)
    xyz, labels = load_scan(f.scan, f.point_labels)
    u, v, z, kept = project_points(xyz, calib.T_cam_lidar, calib.K, calib.dist, (W, H), (W, H))
    im = cv2.imread(str(f.image))
    log_depth = np.log(np.clip(z, 1.0, 80.0)) / np.log(80.0)
    colours = cv2.applyColorMap((log_depth * 255).astype(np.uint8), cv2.COLORMAP_TURBO).reshape(-1, 3)
    for a, b, c in zip(u, v, colours):
        cv2.circle(im, (int(round(a)), int(round(b))), 2, c.tolist(), -1)
    cv2.imwrite(f"{OUT}crop_{seq}_{wanted:06d}_{name}.jpg", im[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("wrote", seq, wanted)
