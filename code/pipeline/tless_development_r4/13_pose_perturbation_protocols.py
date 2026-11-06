"""Explicit public-pose noise protocols; no GT input or implicit default pivot."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

CAMERA="external_camera_origin_left_v1"
DECOUPLED="external_model_origin_decoupled_v1"


def noise(rotation_deg,translation_mm,object_id,selected_index,namespace_seed):
    if min(float(rotation_deg),float(translation_mm))<0 or not np.isfinite([rotation_deg,translation_mm]).all():
        raise ValueError("Noise magnitudes must be finite and nonnegative")
    seed=int(namespace_seed)+int(object_id)*1000003+int(selected_index)*9176
    rng=np.random.default_rng(seed)
    axis=rng.normal(size=3);axis/=max(np.linalg.norm(axis),1e-12)
    direction=rng.normal(size=3);direction/=max(np.linalg.norm(direction),1e-12)
    q=Rotation.from_rotvec(axis*np.deg2rad(rotation_deg)).as_matrix()
    u=direction*(float(translation_mm)/1000.)
    return q,u,dict(effective_rng_seed=seed,namespace_seed=int(namespace_seed),
                    noise_axis=axis.tolist(),translation_direction=direction.tolist(),
                    rotation_deg=float(rotation_deg),translation_mm=float(translation_mm))


def paired_poses(reference,q,u):
    reference=np.asarray(reference,dtype=np.float64)
    q=np.asarray(q,dtype=np.float64);u=np.asarray(u,dtype=np.float64)
    if reference.shape!=(4,4) or q.shape!=(3,3) or u.shape!=(3,) or not all(np.isfinite(v).all() for v in [reference,q,u]):
        raise ValueError("Invalid pose/noise shape or nonfinite input")
    if not np.allclose(q@q.T,np.eye(3),atol=1e-10) or not np.isclose(np.linalg.det(q),1.,atol=1e-10):
        raise ValueError("Noise rotation must be proper")
    delta=np.eye(4);delta[:3,:3]=q;delta[:3,3]=u
    camera=delta@reference
    decoupled=camera.copy()
    decoupled[:3,3]=reference[:3,3]+u
    return {CAMERA:camera,DECOUPLED:decoupled}
