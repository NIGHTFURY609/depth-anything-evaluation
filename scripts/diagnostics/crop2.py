"""Same barrier crop, official on top and corrected below."""
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
seq, wanted = "00001", 1545
f = {x.frame: x for x in list_frames(cfg["paths"]["data_root"], seq)}[wanted]
xyz, _ = load_scan(f.scan, f.point_labels)
panels = []
for corrections in (None, cfg["gt"]["sensitivity_calibration"]):
    calib = load_calibration(cfg, seq, corrections)
    u, v, z, _ = project_points(xyz, calib.T_cam_lidar, calib.K, calib.dist, (W, H), (W, H))
    im = cv2.imread(str(f.image))
    for a, b in zip(u, v):
        cv2.circle(im, (int(round(a)), int(round(b))), 2, (0, 255, 255), -1)
    panels.append(im[450:640, 880:1680])
cv2.imwrite("results/overlays/crop_barrier_official_vs_corrected.jpg", np.vstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 92])
