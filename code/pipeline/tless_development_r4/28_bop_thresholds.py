"""Shared BOP19 thresholds and explicit image-width normalization."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


CORRECTNESS = np.arange(0.05, 0.51, 0.05)
MSPD_CORRECTNESS = np.arange(5.0, 51.0, 5.0)
VSD_TAUS = np.arange(0.05, 0.51, 0.05)


def normalized_mspd(data: pd.DataFrame) -> np.ndarray:
    if "image_width" not in data:
        raise ValueError("MSPD requires measured image_width; raw-pixel thresholds are not BOP19 normalized thresholds")
    widths = data.image_width.to_numpy(dtype=float)
    errors = data.MSPD_px.to_numpy(dtype=float)
    if not np.isfinite(widths).all() or (widths <= 0).any():
        raise ValueError("Invalid image_width")
    if not np.isfinite(errors).all() or (errors < 0).any():
        raise ValueError("Invalid MSPD_px")
    values = errors * (640.0 / widths)
    if "MSPD_normalized" in data and not np.allclose(
        values, data.MSPD_normalized.to_numpy(dtype=float), rtol=1e-9, atol=1e-9
    ):
        raise ValueError("MSPD_normalized disagrees with raw error and measured width")
    return values


def attach_image_widths(data: pd.DataFrame, dataset_root: Path, split: str,
                        extension: str = ".png") -> pd.DataFrame:
    import cv2

    cache = {}
    for scene_id, img_id in data[["scene_id", "img_id"]].drop_duplicates().itertuples(index=False, name=None):
        path = dataset_root / split / f"{int(scene_id):06d}" / "depth" / f"{int(img_id):06d}.{extension.lstrip('.')}"
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Cannot measure evaluation image width: {path}")
        cache[(int(scene_id), int(img_id))] = depth.shape[1]
    output = data.copy()
    measured = np.array([cache[(int(s), int(i))] for s, i in zip(data.scene_id, data.img_id)])
    if "image_width" in data and not np.array_equal(data.image_width.to_numpy(), measured):
        raise ValueError("Saved image_width disagrees with dataset images")
    output["image_width"] = measured
    if "MSPD_px" in output:
        output["MSPD_normalized"] = normalized_mspd(output)
    return output


def metric_grids():
    yield "MSSD", np.nan, "MSSD_normalized", CORRECTNESS
    yield "MSPD", np.nan, "MSPD_normalized", MSPD_CORRECTNESS
    for tau in VSD_TAUS:
        yield "VSD", float(tau), f"VSD_tau_{tau:.2f}", CORRECTNESS


def row_recalls(data: pd.DataFrame) -> pd.DataFrame:
    mspd = normalized_mspd(data)
    mssd = data.MSSD_normalized.to_numpy(dtype=float)
    vsd = data[[f"VSD_tau_{tau:.2f}" for tau in VSD_TAUS]].to_numpy(dtype=float)
    if not np.isfinite(mssd).all() or (mssd < 0).any():
        raise ValueError("Invalid MSSD")
    if not np.isfinite(vsd).all() or (vsd < 0).any() or (vsd > 1 + 1e-9).any():
        raise ValueError("Invalid VSD")
    output = data.copy()
    output["MSPD_normalized"] = mspd
    output["AR_MSSD_row"] = (mssd[:, None] < CORRECTNESS).mean(axis=1)
    output["AR_MSPD_row"] = (mspd[:, None] < MSPD_CORRECTNESS).mean(axis=1)
    output["AR_VSD_row"] = (vsd[:, :, None] < CORRECTNESS).mean(axis=(1, 2))
    output["AR_mean_row"] = output[["AR_MSSD_row", "AR_MSPD_row", "AR_VSD_row"]].mean(axis=1)
    return output
