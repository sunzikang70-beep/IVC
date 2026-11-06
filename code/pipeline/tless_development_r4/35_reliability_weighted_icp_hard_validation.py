# HARD_VALIDATION_EXPERIMENT
import os
import json
import copy
import csv
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


# ==================================================
# Reliability-Gated Local Refinement
# Multi-object Structured-Corruption Validation
# ==================================================
#
# Scientific question:
#
#   Geometry ICP gives strong convergence robustness.
#   Direct Local Reliability ICP can improve accuracy under
#   local outliers, but can reduce convergence robustness.
#   Unconditional Geometry -> Local refinement largely restores
#   Geometry success, but does not help every corruption type.
#
# This experiment therefore tests a GT-FREE reliability gate:
#
#   Observation-supported visible CAD
#           ↓
#       Geometry ICP
#           ↓
#   Reliability-map analysis
#        /         \
#   gate OFF      gate ON
#      ↓             ↓
#   keep T_G      Local refinement
#
# The gate is designed to detect AREA-LIKE local inconsistency:
#   - remove mask boundaries by erosion,
#   - threshold strongly low reliability,
#   - remove thin structures by morphological opening,
#   - require a sufficiently large connected low-reliability area.
#
# This is intended to activate on local random outliers while
# remaining mostly inactive for:
#   - clean depth,
#   - coherent local bias,
#   - missing-depth holes.
#
# IMPORTANT:
#   - Gate input uses observed depth only.
#   - Gate does NOT use T_gt, ADD-S, success labels, corruption
#     type, requested corruption fraction, or oracle visibility.
#   - The gated method does NOT add a new ICP run. It selects
#     between the already-computed Geometry result and the
#     Geometry->Local staged result.
#   - BOP mask_visib is still used as a segmentation proxy.
#
# ADD-S remains a controlled supplemental metric. Formal TLESS
# evaluation should later use BOP VSD / MSSD / MSPD.
#
# ==================================================
# Paths
# ==================================================
root = r"D:\6D_pose\bop_datasets\tless"

test_root = os.path.join(
    root,
    "test_primesense"
)

# ==================================================
# Multi-object validation objects
# ==================================================
#
# obj2 was the development object and is intentionally NOT
# included here, so this stage acts as a generalization test.
#
# Selected from TLESS models_info.json:
#   obj5  : discrete symmetry, medium size
#   obj13 : continuous axial symmetry, small size
#   obj18 : no listed symmetry, medium size
#   obj21 : no listed symmetry, medium size
#   obj27 : multiple discrete symmetries, large size
#
OBJECT_IDS = [
    5,
    13,
    18,
    21,
    27
]

# Current object globals.
# They are configured separately before processing each object.
obj_id = None
model_path = None
full_model_points = None
evaluation_model_points = None
diameter_m = None
success_threshold_m = None

models_info_path = os.path.join(
    root,
    "models",
    "models_info.json"
)

output_dir = r"D:\6D_pose\reliability_weighted_icp_hard_validation"

os.makedirs(
    output_dir,
    exist_ok=True
)

trial_csv = os.path.join(
    output_dir,
    "reliability_weighted_icp_hard_validation_trials.csv"
)

condition_summary_csv = os.path.join(
    output_dir,
    "reliability_weighted_icp_hard_validation_condition_summary.csv"
)

corruption_summary_csv = os.path.join(
    output_dir,
    "reliability_weighted_icp_hard_validation_corruption_summary.csv"
)

object_summary_csv = os.path.join(
    output_dir,
    "reliability_weighted_icp_hard_validation_object_summary.csv"
)

aggregate_csv = os.path.join(
    output_dir,
    "reliability_weighted_icp_hard_validation_aggregate.csv"
)

paired_csv = os.path.join(
    output_dir,
    "reliability_weighted_icp_hard_validation_paired_comparisons.csv"
)

transition_csv = os.path.join(
    output_dir,
    "reliability_weighted_icp_hard_validation_stage_transition_summary.csv"
)

gate_csv = os.path.join(
    output_dir,
    "reliability_weighted_icp_hard_validation_gate_diagnostics.csv"
)

gate_summary_csv = os.path.join(
    output_dir,
    "reliability_weighted_icp_hard_validation_gate_summary.csv"
)

visibility_csv = os.path.join(
    output_dir,
    "reliability_weighted_icp_hard_validation_visibility_diagnostics.csv"
)

checkpoint_csv = os.path.join(
    output_dir,
    "reliability_weighted_icp_hard_validation_checkpoint_trials.csv"
)


# ==================================================
# Multi-frame settings
# ==================================================
FRAMES_PER_OBJECT = 100
# Reproducible initial pose perturbation
POSE_SEED = 42
MIN_VISIB_FRACT = 0.30
MIN_VISIBLE_PIXELS = 500
MIN_VISIBLE_CAD_POINTS = 300

VISIBLE_DEPTH_TOLERANCE = 0.005

MAX_SOURCE_POINTS = 6000

# ==================================================
# Estimated visibility settings
# ==================================================
#
# This stage removes GT POSE from source visibility selection.
#
# The visible source is inferred from:
#   - current estimated pose
#   - CAD projection
#   - model self-occlusion (z-buffer)
#   - object segmentation mask
#
# IMPORTANT:
# The BOP mask_visib is still used as a proxy for a future
# segmentation network output. Therefore this is NOT yet a
# completely GT-free perception pipeline.
#
# GT pose is used only for:
#   - controlled T_init generation
#   - oracle visibility reference / diagnostics
#   - ADD-S evaluation
#
# It is NOT used by estimated visibility generation.
EST_VIS_ZBUFFER_TOLERANCE = 0.002
EST_VIS_MASK_DILATION_PIXELS = 2

# ==================================================
# Observation-supported visibility
# ==================================================
#
# After model-only z-buffer visibility, each remaining CAD point
# must also be supported by the observed object depth.
#
# Two GT-free checks are used:
#
#   1) 3D nearest-neighbor support in the observed object cloud.
#   2) Local 2D depth support in a 5x5 image neighborhood.
#
# The thresholds are intentionally fixed before this experiment.
# They are NOT tuned from the 5-object evaluation results.
#
# The controlled T_init perturbation is roughly millimeter-scale;
# 15 mm therefore leaves margin for pose error while rejecting
# model surfaces that are far from the observed surface.
OBS_SUPPORT_3D_MAX_DISTANCE = 0.015
OBS_SUPPORT_DEPTH_MAX_RESIDUAL = 0.015
OBS_SUPPORT_DEPTH_WINDOW_RADIUS = 2

# Diagnostic soft scales only. They are used to report a
# GT-free source-quality score and to guard the refresh step.
OBS_SUPPORT_3D_SOFT_SCALE = 0.008
OBS_SUPPORT_DEPTH_SOFT_SCALE = 0.008

# One conservative refresh after the first observation-supported
# Local Reliability ICP.
ITERATIVE_VISIBILITY_UPDATES = 1

# If an estimated visible source is too small, the frame is not
# silently repaired using GT. The method is recorded as failed.
MIN_ESTIMATED_VISIBLE_CAD_POINTS = 150

# Save long-run checkpoints.
CHECKPOINT_EVERY_PROCESSED_FRAMES = 5


# ==================================================
# Local depth corruption experiment settings
# ==================================================
#
# Three structured corruption types are tested:
#
# 1) local_bias
#    A spatially local patch receives a systematic depth shift.
#
# 2) local_outlier
#    A spatially local patch receives large signed depth errors.
#
# 3) local_hole
#    A spatially local patch is removed from the depth image.
#
# The corrupted region occupies approximately 10 / 20 / 30 %
# of the valid visible-object depth pixels.
#
# IMPORTANT:
# The SAME patch location is reused across the three corruption
# types for the same frame / fraction / trial. This improves
# fairness when comparing corruption mechanisms.
corruption_types = [
    "local_bias",
    "local_outlier",
    "local_hole"
]

corruption_fractions = [
    0.20,
    0.30,
    0.40
]

# Local systematic bias magnitude: +/- 5 mm.
# The sign is deterministic-random per trial.
LOCAL_BIAS_MAGNITUDE_M = 0.010

# Local outliers use random signed magnitudes between
# 5 mm and 20 mm.
LOCAL_OUTLIER_MIN_ABS_M = 0.010
LOCAL_OUTLIER_MAX_ABS_M = 0.050

# Diagnostic-stage repeats.
# Increase later only after the method behavior is understood.
REPEATS_PER_LEVEL = 3

# ==================================================
# Large-scale validation settings
# ==================================================
#
# Parameters are intentionally frozen from the previous
# development experiments. Do NOT tune them on these multi-object validation frames.
#
# Success-rate uncertainty is estimated with a frame-cluster
# bootstrap so repeated corruptions from the same frame are
# resampled together.
BOOTSTRAP_REPEATS = 2000
BOOTSTRAP_SEED = 20260826

# Periodic checkpointing reduces the risk of losing a long run.
CHECKPOINT_EVERY_PROCESSED_FRAMES = 5


# ==================================================
# ICP settings
# ==================================================
max_correspondence_distance = 0.05
max_iterations = 50

# Geometry robust weighting
robust_scale = 0.005
correspondence_scale = 0.02

# Local depth reliability
depth_consistency_scale = 0.003
depth_weight_floor = 0.05
edge_weight_floor = 0.25

# ==================================================
# Reliability gate - FROZEN BEFORE THIS EVALUATION
# ==================================================
#
# The gate looks only at the observed depth reliability map.
#
# Rationale:
#   * coherent bias:
#       low reliability mainly appears around a thin patch boundary;
#   * local hole:
#       missing pixels are invalid, and the hole boundary is removed
#       by the interior-mask erosion;
#   * local random outlier:
#       inconsistency occupies a 2-D area inside the valid mask.
#
# Therefore:
#   1) erode the valid mask by 1 pixel,
#   2) find strongly low-reliability interior pixels,
#   3) use 3x3 morphological opening to remove thin structures,
#   4) require both enough opened area and a connected component.
#
# These thresholds are NOT selected using ADD-S or success labels.
GATE_LOW_RELIABILITY_THRESHOLD = 0.25

GATE_INTERIOR_EROSION_SIZE = 3
GATE_OPEN_KERNEL_SIZE = 3

GATE_MIN_OPENED_LOW_FRACTION = 0.03
GATE_MIN_LARGEST_COMPONENT_FRACTION = 0.02

# Lower-tail check prevents a broad but only mildly reduced
# reliability region from triggering the Local stage.
GATE_MAX_Q10 = 0.35

# Model-observation depth consistency
#
# For each current CAD-to-target correspondence:
#
#   r_model = |z_model - z_observed|
#
#   w_model = exp(
#       -0.5 * (r_model / model_consistency_scale)^2
#   )
#
# The weight is softly floored to avoid completely deleting
# correspondences during early ICP iterations.
#
# IMPORTANT:
# This uses the CURRENT estimated pose only.
# It does NOT use T_gt, ADD-S, or the success label.
model_consistency_scale = 0.005
model_consistency_floor = 0.10

# Complementary gating
#
# The model-consistency term should be strong mainly when:
#
#   local reliability is HIGH
#   but
#   model-observation consistency is LOW.
#
# This targets smooth systematic local bias while avoiding
# repeated strong suppression of obvious local outliers/holes.
#
# g_local = sigmoid(
#     (w_local - complementary_local_threshold)
#     / complementary_temperature
# )
#
# effective_model_weight =
#     (1 - g_local) + g_local * w_model
#
# Final:
#
# w_final =
#     w_residual
#     * w_corr
#     * w_local
#     * effective_model_weight
#
# These are fixed diagnostic parameters; they are NOT tuned
# from the corruption-test results below.
complementary_local_threshold = 0.70
complementary_temperature = 0.10


# ==================================================
# Stable solver settings
# ==================================================

# LM damping
lm_lambda_initial = 1e-4
lm_lambda_min = 1e-8
lm_lambda_max = 1e8

# Retry the same outer iteration with larger damping
lm_max_attempts = 8
lm_increase = 10.0
lm_decrease = 0.3

# Backtracking line search
line_search_steps = [
    1.0,
    0.5,
    0.25,
    0.125,
    0.0625,
    0.03125
]

# Maximum accepted update magnitude
max_rotation_step = np.deg2rad(
    5.0
)

max_translation_step = 0.02

# Objective adds a small coverage penalty so that an update
# cannot look "better" only because correspondences disappear.
objective_coverage_penalty = 0.25

# Numerical convergence thresholds
cost_tolerance = 1e-10
rotation_step_tolerance = 1e-7
translation_step_tolerance = 1e-7

min_correspondences = 20


# ==================================================
# Initial-pose perturbation
# ==================================================
#
# Use the centered perturbation protocol introduced in the
# multi-frame script:
#
#   R_init = R_noise @ R_gt
#   t_init = t_gt + t_noise
#
# This avoids coupling rotation around the camera origin with
# a large apparent translation that depends on object distance.
# ==================================================
CENTERED_INITIALIZATION = True


# ==================================================
# Utility: transform points
# ==================================================
def transform_points(
        points,
        T
):

    return (
        points
        @ T[:3, :3].T
        + T[:3, 3]
    )


# ==================================================
# Pose errors
# ==================================================
def rotation_error_deg(
        T1,
        T2
):

    R = (
        T1[:3, :3]
        @ T2[:3, :3].T
    )

    c = np.clip(
        (
            np.trace(R)
            - 1.0
        )
        / 2.0,
        -1.0,
        1.0
    )

    return float(
        np.rad2deg(
            np.arccos(c)
        )
    )


def translation_error_mm(
        T1,
        T2
):

    return float(
        np.linalg.norm(
            T1[:3, 3]
            - T2[:3, 3]
        )
        * 1000.0
    )


# ==================================================
# ADD-S
# ==================================================
def compute_add_s(
        evaluation_model_points,
        T_pred,
        T_ref
):

    pred = transform_points(
        evaluation_model_points,
        T_pred
    )

    ref = transform_points(
        evaluation_model_points,
        T_ref
    )

    tree = cKDTree(
        ref
    )

    distances, _ = tree.query(
        pred,
        k=1
    )

    return float(
        np.mean(
            distances
        )
    )


# ==================================================
# Load CAD points
# ==================================================
def load_full_model_points(
        path
):

    pcd = o3d.io.read_point_cloud(
        path
    )

    points = np.asarray(
        pcd.points,
        dtype=np.float64
    )

    if len(
        points
    ) == 0:

        mesh = o3d.io.read_triangle_mesh(
            path
        )

        points = np.asarray(
            mesh.vertices,
            dtype=np.float64
        )

    if len(
        points
    ) == 0:

        raise RuntimeError(
            "Could not load CAD vertices: "
            + path
        )

    # TLESS BOP CAD is in millimeters.
    return (
        points
        / 1000.0
    )


# ==================================================
# Load models_info once
# ==================================================
with open(
    models_info_path,
    "r"
) as f:

    models_info = json.load(
        f
    )


def get_symmetry_class(
        object_id
):

    info = models_info[
        str(
            object_id
        )
    ]


    if "symmetries_continuous" in info:

        return "continuous"


    if "symmetries_discrete" in info:

        return "discrete"


    return "none"


def configure_object(
        object_id
):

    global obj_id
    global model_path
    global full_model_points
    global evaluation_model_points
    global diameter_m
    global success_threshold_m


    obj_id = int(
        object_id
    )


    model_path = os.path.join(
        root,
        "models",
        f"obj_{obj_id:06d}.ply"
    )


    full_model_points = load_full_model_points(
        model_path
    )


    # Continue using full-CAD ADD-S as the common controlled
    # diagnostic metric for continuity with the obj2 experiments.
    evaluation_model_points = full_model_points


    diameter_m = (
        float(
            models_info[
                str(
                    obj_id
                )
            ][
                "diameter"
            ]
        )
        / 1000.0
    )


    success_threshold_m = (
        0.1
        * diameter_m
    )


    return {
        "obj_id": obj_id,
        "diameter_mm": diameter_m * 1000.0,
        "success_threshold_mm": success_threshold_m * 1000.0,
        "symmetry_class": get_symmetry_class(
            obj_id
        ),
        "cad_points": len(
            full_model_points
        )
    }


# ==================================================
# Discover eligible candidates
# ==================================================
def discover_candidates():

    candidates = []

    if not os.path.isdir(
        test_root
    ):

        raise FileNotFoundError(
            "TLESS test_primesense folder not found: "
            + test_root
        )


    scene_names = sorted(
        [
            name
            for name in os.listdir(
                test_root
            )
            if name.isdigit()
            and os.path.isdir(
                os.path.join(
                    test_root,
                    name
                )
            )
        ],
        key=lambda x: int(
            x
        )
    )


    for scene_name in scene_names:

        scene_path = os.path.join(
            test_root,
            scene_name
        )

        gt_path = os.path.join(
            scene_path,
            "scene_gt.json"
        )

        camera_path = os.path.join(
            scene_path,
            "scene_camera.json"
        )

        info_path = os.path.join(
            scene_path,
            "scene_gt_info.json"
        )


        if (
            not os.path.exists(
                gt_path
            )
            or not os.path.exists(
                camera_path
            )
        ):

            continue


        with open(
            gt_path,
            "r"
        ) as f:

            scene_gt = json.load(
                f
            )


        if os.path.exists(
            info_path
        ):

            with open(
                info_path,
                "r"
            ) as f:

                scene_info = json.load(
                    f
                )

        else:

            scene_info = None


        for img_key in sorted(
            scene_gt.keys(),
            key=lambda x: int(
                x
            )
        ):

            frame_matches = []


            for inst_id, pose in enumerate(
                scene_gt[
                    img_key
                ]
            ):

                if int(
                    pose[
                        "obj_id"
                    ]
                ) != obj_id:

                    continue


                visib_fract = 1.0
                px_count_visib = -1


                if (
                    scene_info is not None
                    and img_key in scene_info
                    and inst_id
                    < len(
                        scene_info[
                            img_key
                        ]
                    )
                ):

                    info = scene_info[
                        img_key
                    ][
                        inst_id
                    ]

                    visib_fract = float(
                        info.get(
                            "visib_fract",
                            1.0
                        )
                    )

                    px_count_visib = int(
                        info.get(
                            "px_count_visib",
                            -1
                        )
                    )


                if (
                    visib_fract
                    < MIN_VISIB_FRACT
                ):

                    continue


                if (
                    px_count_visib >= 0
                    and px_count_visib
                    < MIN_VISIBLE_PIXELS
                ):

                    continue


                frame_matches.append({
                    "scene_id": int(
                        scene_name
                    ),
                    "scene_name": scene_name,
                    "img_id": int(
                        img_key
                    ),
                    "img_key": img_key,
                    "inst_id": inst_id,
                    "visib_fract": visib_fract,
                    "px_count_visib": px_count_visib
                })


            if len(
                frame_matches
            ) == 0:

                continue


            # If the object appears more than once, keep the most visible instance.
            best = max(
                frame_matches,
                key=lambda item: (
                    item[
                        "visib_fract"
                    ],
                    item[
                        "px_count_visib"
                    ]
                )
            )


            candidates.append(
                best
            )


    if len(
        candidates
    ) == 0:

        raise RuntimeError(
            f"No eligible frames found for obj_id={obj_id}."
        )


    return sorted(
        candidates,
        key=lambda item: (
            item[
                "scene_id"
            ],
            item[
                "img_id"
            ],
            item[
                "inst_id"
            ]
        )
    )


# ==================================================
# Even sampling
# ==================================================
def select_evenly(
        candidates,
        max_frames
):

    if len(
        candidates
    ) <= max_frames:

        return candidates.copy()


    indices = np.linspace(
        0,
        len(
            candidates
        ) - 1,
        max_frames
    )


    indices = np.round(
        indices
    ).astype(
        int
    )


    selected = []

    seen = set()


    for idx in indices:

        idx = int(
            idx
        )

        if idx in seen:

            continue


        seen.add(
            idx
        )

        selected.append(
            candidates[
                idx
            ]
        )


    return selected



# ==================================================
# Load one frame
# ==================================================
def load_frame_data(
        candidate
):

    scene_name = candidate[
        "scene_name"
    ]

    img_id = candidate[
        "img_id"
    ]

    img_key = candidate[
        "img_key"
    ]

    inst_id = candidate[
        "inst_id"
    ]


    scene_path = os.path.join(
        test_root,
        scene_name
    )


    with open(
        os.path.join(
            scene_path,
            "scene_camera.json"
        ),
        "r"
    ) as f:

        scene_camera = json.load(
            f
        )


    with open(
        os.path.join(
            scene_path,
            "scene_gt.json"
        ),
        "r"
    ) as f:

        scene_gt = json.load(
            f
        )


    K = np.array(
        scene_camera[
            img_key
        ][
            "cam_K"
        ],
        dtype=np.float64
    ).reshape(
        3,
        3
    )


    depth_scale = float(
        scene_camera[
            img_key
        ][
            "depth_scale"
        ]
    )


    img_name = (
        f"{img_id:06d}.png"
    )

    mask_name = (
        f"{img_id:06d}_{inst_id:06d}.png"
    )


    depth_raw = cv2.imread(
        os.path.join(
            scene_path,
            "depth",
            img_name
        ),
        cv2.IMREAD_UNCHANGED
    )


    mask = cv2.imread(
        os.path.join(
            scene_path,
            "mask_visib",
            mask_name
        ),
        cv2.IMREAD_GRAYSCALE
    )


    if depth_raw is None:

        raise FileNotFoundError(
            "Missing depth image."
        )


    if mask is None:

        raise FileNotFoundError(
            "Missing mask_visib image."
        )


    depth = (
        depth_raw.astype(
            np.float64
        )
        * depth_scale
        / 1000.0
    )


    mask_bool = (
        mask > 0
    )


    mask_pixels = int(
        np.count_nonzero(
            mask_bool
        )
    )


    if (
        mask_pixels
        < MIN_VISIBLE_PIXELS
    ):

        raise RuntimeError(
            "Visible mask too small."
        )


    pose = scene_gt[
        img_key
    ][
        inst_id
    ]


    if int(
        pose[
            "obj_id"
        ]
    ) != obj_id:

        raise RuntimeError(
            "Selected instance does not match obj_id."
        )


    R_gt = np.array(
        pose[
            "cam_R_m2c"
        ],
        dtype=np.float64
    ).reshape(
        3,
        3
    )


    t_gt = (
        np.array(
            pose[
                "cam_t_m2c"
            ],
            dtype=np.float64
        )
        / 1000.0
    )


    T_gt_bop = np.eye(
        4,
        dtype=np.float64
    )


    T_gt_bop[
        :3,
        :3
    ] = R_gt


    T_gt_bop[
        :3,
        3
    ] = t_gt


    flip = np.eye(
        4,
        dtype=np.float64
    )

    flip[
        1,
        1
    ] = -1.0


    T_gt = (
        flip
        @ T_gt_bop
    )


    return {
        "scene_name": scene_name,
        "scene_id": int(
            scene_name
        ),
        "img_id": img_id,
        "img_key": img_key,
        "inst_id": inst_id,
        "K": K,
        "depth_scale": depth_scale,
        "depth": depth,
        "mask_bool": mask_bool,
        "mask_pixels": mask_pixels,
        "T_gt": T_gt,
        "visib_fract": candidate[
            "visib_fract"
        ]
    }


# ==================================================
# Depth -> point cloud
# ==================================================
def depth_to_cloud(
        depth_img,
        mask_bool,
        K
):

    fx = K[
        0,
        0
    ]

    fy = K[
        1,
        1
    ]

    cx = K[
        0,
        2
    ]

    cy = K[
        1,
        2
    ]


    valid = (
        mask_bool
        & (
            depth_img
            > 0
        )
    )


    v, u = np.nonzero(
        valid
    )


    z = depth_img[
        v,
        u
    ]


    x = (
        (
            u
            - cx
        )
        * z
        / fx
    )


    y = -(
        (
            v
            - cy
        )
        * z
        / fy
    )


    points = np.stack(
        [
            x,
            y,
            z
        ],
        axis=1
    ).astype(
        np.float64
    )


    cloud = (
        o3d.geometry.PointCloud()
    )


    cloud.points = (
        o3d.utility.Vector3dVector(
            points
        )
    )


    uv = np.stack(
        [
            u,
            v
        ],
        axis=1
    ).astype(
        np.int32
    )


    return (
        cloud,
        uv
    )


# ==================================================
# Depth reliability
# ==================================================
def compute_depth_reliability(
        depth_img,
        mask_bool,
        uv
):

    h, w = depth_img.shape


    valid = (
        mask_bool
        & (
            depth_img
            > 0
        )
    )


    diff_sum = np.zeros(
        (
            h,
            w
        ),
        dtype=np.float64
    )


    neighbor_count = np.zeros(
        (
            h,
            w
        ),
        dtype=np.float64
    )


    # --------------------------------------------------
    # Left-right pairs
    # --------------------------------------------------
    pair_h = (
        valid[
            :,
            1:
        ]
        & valid[
            :,
            :-1
        ]
    )


    diff_h = (
        np.abs(
            depth_img[
                :,
                1:
            ]
            - depth_img[
                :,
                :-1
            ]
        )
        * pair_h
    )


    diff_sum[
        :,
        1:
    ] += diff_h


    diff_sum[
        :,
        :-1
    ] += diff_h


    neighbor_count[
        :,
        1:
    ] += pair_h.astype(
        np.float64
    )


    neighbor_count[
        :,
        :-1
    ] += pair_h.astype(
        np.float64
    )


    # --------------------------------------------------
    # Up-down pairs
    # --------------------------------------------------
    pair_v = (
        valid[
            1:,
            :
        ]
        & valid[
            :-1,
            :
        ]
    )


    diff_v = (
        np.abs(
            depth_img[
                1:,
                :
            ]
            - depth_img[
                :-1,
                :
            ]
        )
        * pair_v
    )


    diff_sum[
        1:,
        :
    ] += diff_v


    diff_sum[
        :-1,
        :
    ] += diff_v


    neighbor_count[
        1:,
        :
    ] += pair_v.astype(
        np.float64
    )


    neighbor_count[
        :-1,
        :
    ] += pair_v.astype(
        np.float64
    )


    local_diff = np.zeros(
        (
            h,
            w
        ),
        dtype=np.float64
    )


    has_neighbor = (
        neighbor_count
        > 0
    )


    local_diff[
        has_neighbor
    ] = (
        diff_sum[
            has_neighbor
        ]
        / neighbor_count[
            has_neighbor
        ]
    )


    local_diff[
        ~has_neighbor
    ] = (
        10.0
        * depth_consistency_scale
    )


    w_consistency = np.exp(
        -0.5
        * (
            local_diff
            / depth_consistency_scale
        ) ** 2
    )


    neighbor_ratio = np.clip(
        neighbor_count
        / 4.0,
        0.0,
        1.0
    )


    w_edge = (
        edge_weight_floor
        + (
            1.0
            - edge_weight_floor
        )
        * neighbor_ratio
    )


    reliability_map = np.clip(
        w_consistency
        * w_edge,
        depth_weight_floor,
        1.0
    )


    u = uv[
        :,
        0
    ]


    v = uv[
        :,
        1
    ]


    return reliability_map[
        v,
        u
    ].astype(
        np.float64
    )



# ==================================================
# Full reliability map for GT-free gate diagnostics
# ==================================================
def compute_depth_reliability_map(
        depth_img,
        mask_bool
):

    h, w = depth_img.shape


    valid = (
        mask_bool
        & (
            depth_img
            > 0
        )
    )


    diff_sum = np.zeros(
        (
            h,
            w
        ),
        dtype=np.float64
    )


    neighbor_count = np.zeros(
        (
            h,
            w
        ),
        dtype=np.float64
    )


    pair_h = (
        valid[
            :,
            1:
        ]
        & valid[
            :,
            :-1
        ]
    )


    diff_h = (
        np.abs(
            depth_img[
                :,
                1:
            ]
            - depth_img[
                :,
                :-1
            ]
        )
        * pair_h
    )


    diff_sum[
        :,
        1:
    ] += diff_h


    diff_sum[
        :,
        :-1
    ] += diff_h


    neighbor_count[
        :,
        1:
    ] += pair_h.astype(
        np.float64
    )


    neighbor_count[
        :,
        :-1
    ] += pair_h.astype(
        np.float64
    )


    pair_v = (
        valid[
            1:,
            :
        ]
        & valid[
            :-1,
            :
        ]
    )


    diff_v = (
        np.abs(
            depth_img[
                1:,
                :
            ]
            - depth_img[
                :-1,
                :
            ]
        )
        * pair_v
    )


    diff_sum[
        1:,
        :
    ] += diff_v


    diff_sum[
        :-1,
        :
    ] += diff_v


    neighbor_count[
        1:,
        :
    ] += pair_v.astype(
        np.float64
    )


    neighbor_count[
        :-1,
        :
    ] += pair_v.astype(
        np.float64
    )


    local_diff = np.zeros(
        (
            h,
            w
        ),
        dtype=np.float64
    )


    has_neighbor = (
        neighbor_count
        > 0
    )


    local_diff[
        has_neighbor
    ] = (
        diff_sum[
            has_neighbor
        ]
        / neighbor_count[
            has_neighbor
        ]
    )


    local_diff[
        ~has_neighbor
    ] = (
        10.0
        * depth_consistency_scale
    )


    w_consistency = np.exp(
        -0.5
        * (
            local_diff
            / depth_consistency_scale
        ) ** 2
    )


    neighbor_ratio = np.clip(
        neighbor_count
        / 4.0,
        0.0,
        1.0
    )


    w_edge = (
        edge_weight_floor
        + (
            1.0
            - edge_weight_floor
        )
        * neighbor_ratio
    )


    reliability_map = np.clip(
        w_consistency
        * w_edge,
        depth_weight_floor,
        1.0
    )


    reliability_map[
        ~valid
    ] = np.nan


    return (
        reliability_map,
        valid,
        neighbor_count,
        local_diff
    )


# ==================================================
# Reliability-gate features
# ==================================================
def compute_reliability_gate_features(
        depth_img,
        mask_bool
):

    (
        reliability_map,
        valid,
        neighbor_count,
        local_diff
    ) = compute_depth_reliability_map(
        depth_img,
        mask_bool
    )


    valid_count = int(
        np.count_nonzero(
            valid
        )
    )


    if valid_count == 0:

        return {
            "trigger": False,
            "reason": "no_valid_depth",
            "valid_count": 0,
            "interior_count": 0,
            "mean_reliability": np.nan,
            "median_reliability": np.nan,
            "q10_reliability": np.nan,
            "q25_reliability": np.nan,
            "strong_low_fraction": 0.0,
            "opened_low_fraction": 0.0,
            "largest_component_fraction": 0.0,
            "largest_component_share_of_low": 0.0,
            "mean_local_diff_mm": np.nan,
            "q90_local_diff_mm": np.nan
        }


    erosion_kernel = np.ones(
        (
            GATE_INTERIOR_EROSION_SIZE,
            GATE_INTERIOR_EROSION_SIZE
        ),
        dtype=np.uint8
    )


    interior = cv2.erode(
        valid.astype(
            np.uint8
        ),
        erosion_kernel,
        iterations=1
    ).astype(
        bool
    )


    interior_count = int(
        np.count_nonzero(
            interior
        )
    )


    if interior_count == 0:

        interior = valid.copy()

        interior_count = valid_count


    values = reliability_map[
        interior
    ]


    local_diff_values = local_diff[
        interior
    ]


    strong_low = (
        interior
        & (
            reliability_map
            <= GATE_LOW_RELIABILITY_THRESHOLD
        )
    )


    open_kernel = np.ones(
        (
            GATE_OPEN_KERNEL_SIZE,
            GATE_OPEN_KERNEL_SIZE
        ),
        dtype=np.uint8
    )


    opened_low = cv2.morphologyEx(
        strong_low.astype(
            np.uint8
        ),
        cv2.MORPH_OPEN,
        open_kernel,
        iterations=1
    ).astype(
        bool
    )


    strong_low_count = int(
        np.count_nonzero(
            strong_low
        )
    )


    opened_low_count = int(
        np.count_nonzero(
            opened_low
        )
    )


    strong_low_fraction = (
        strong_low_count
        / max(
            interior_count,
            1
        )
    )


    opened_low_fraction = (
        opened_low_count
        / max(
            interior_count,
            1
        )
    )


    largest_component = 0


    if opened_low_count > 0:

        (
            label_count,
            labels,
            stats,
            centroids
        ) = cv2.connectedComponentsWithStats(
            opened_low.astype(
                np.uint8
            ),
            connectivity=8
        )


        if label_count > 1:

            component_areas = stats[
                1:,
                cv2.CC_STAT_AREA
            ]


            largest_component = int(
                np.max(
                    component_areas
                )
            )


    largest_component_fraction = (
        largest_component
        / max(
            interior_count,
            1
        )
    )


    largest_component_share_of_low = (
        largest_component
        / max(
            opened_low_count,
            1
        )
    )


    q10 = float(
        np.quantile(
            values,
            0.10
        )
    )


    q25 = float(
        np.quantile(
            values,
            0.25
        )
    )


    mean_reliability = float(
        np.mean(
            values
        )
    )


    median_reliability = float(
        np.median(
            values
        )
    )


    mean_local_diff_mm = float(
        np.mean(
            local_diff_values
        )
        * 1000.0
    )


    q90_local_diff_mm = float(
        np.quantile(
            local_diff_values,
            0.90
        )
        * 1000.0
    )


    area_ok = bool(
        opened_low_fraction
        >= GATE_MIN_OPENED_LOW_FRACTION
    )


    component_ok = bool(
        largest_component_fraction
        >= GATE_MIN_LARGEST_COMPONENT_FRACTION
    )


    tail_ok = bool(
        q10
        <= GATE_MAX_Q10
    )


    trigger = bool(
        area_ok
        and component_ok
        and tail_ok
    )


    if trigger:

        reason = "area_component_tail"

    elif not area_ok:

        reason = "opened_area_too_small"

    elif not component_ok:

        reason = "component_too_small"

    else:

        reason = "lower_tail_not_low_enough"


    return {
        "trigger": trigger,
        "reason": reason,
        "valid_count": valid_count,
        "interior_count": interior_count,
        "mean_reliability": mean_reliability,
        "median_reliability": median_reliability,
        "q10_reliability": q10,
        "q25_reliability": q25,
        "strong_low_fraction": float(
            strong_low_fraction
        ),
        "opened_low_fraction": float(
            opened_low_fraction
        ),
        "largest_component_fraction": float(
            largest_component_fraction
        ),
        "largest_component_share_of_low": float(
            largest_component_share_of_low
        ),
        "mean_local_diff_mm": mean_local_diff_mm,
        "q90_local_diff_mm": q90_local_diff_mm
    }


# ==================================================
# Select Geometry or Geometry->Local using GT-free gate
# ==================================================
def select_reliability_gated_result(
        geometry_result,
        staged_result,
        gate_features
):

    if gate_features[
        "trigger"
    ]:

        result = dict(
            staged_result
        )

        selected_stage = "geometry_then_local"

    else:

        result = dict(
            geometry_result
        )

        selected_stage = "geometry_only"


    result.update({
        "gate_triggered": int(
            gate_features[
                "trigger"
            ]
        ),
        "gate_reason": gate_features[
            "reason"
        ],
        "gate_selected_stage": selected_stage
    })


    return result


# ==================================================
# Oracle visible CAD
# ==================================================
def generate_oracle_visible_model(
        model_points,
        T_gt,
        depth,
        mask_bool,
        K
):

    fx = K[
        0,
        0
    ]

    fy = K[
        1,
        1
    ]

    cx = K[
        0,
        2
    ]

    cy = K[
        1,
        2
    ]


    h, w = depth.shape


    points_cam = transform_points(
        model_points,
        T_gt
    )


    z = points_cam[
        :,
        2
    ]


    valid_z = (
        z > 1e-6
    )


    u_float = np.zeros_like(
        z
    )

    v_float = np.zeros_like(
        z
    )


    u_float[
        valid_z
    ] = (
        fx
        * points_cam[
            valid_z,
            0
        ]
        / z[
            valid_z
        ]
        + cx
    )


    v_float[
        valid_z
    ] = (
        cy
        - fy
        * points_cam[
            valid_z,
            1
        ]
        / z[
            valid_z
        ]
    )


    u = np.rint(
        u_float
    ).astype(
        np.int32
    )


    v = np.rint(
        v_float
    ).astype(
        np.int32
    )


    inside = (
        valid_z
        & (
            u >= 0
        )
        & (
            u < w
        )
        & (
            v >= 0
        )
        & (
            v < h
        )
    )


    candidate_indices = np.nonzero(
        inside
    )[0]


    if len(
        candidate_indices
    ) == 0:

        return np.empty(
            (
                0,
                3
            ),
            dtype=np.float64
        )


    uu = u[
        candidate_indices
    ]


    vv = v[
        candidate_indices
    ]


    zz = z[
        candidate_indices
    ]


    observed_depth = depth[
        vv,
        uu
    ]


    visible = (
        mask_bool[
            vv,
            uu
        ]
        & (
            observed_depth
            > 0
        )
        & (
            np.abs(
                observed_depth
                - zz
            )
            < VISIBLE_DEPTH_TOLERANCE
        )
    )


    visible_indices = candidate_indices[
        visible
    ]


    source_points = model_points[
        visible_indices
    ].copy()


    if (
        MAX_SOURCE_POINTS is not None
        and len(
            source_points
        )
        > MAX_SOURCE_POINTS
    ):

        idx = np.linspace(
            0,
            len(
                source_points
            ) - 1,
            MAX_SOURCE_POINTS
        )


        idx = np.round(
            idx
        ).astype(
            int
        )


        source_points = source_points[
            idx
        ]


    return source_points



# ==================================================
# Model-only z-buffer visibility from an estimated pose
# ==================================================
def generate_estimated_visible_model(
        model_points,
        T_est,
        mask_bool,
        K
):

    fx = K[
        0,
        0
    ]

    fy = K[
        1,
        1
    ]

    cx = K[
        0,
        2
    ]

    cy = K[
        1,
        2
    ]


    h, w = mask_bool.shape


    if EST_VIS_MASK_DILATION_PIXELS > 0:

        kernel_size = (
            2
            * EST_VIS_MASK_DILATION_PIXELS
            + 1
        )


        kernel = np.ones(
            (
                kernel_size,
                kernel_size
            ),
            dtype=np.uint8
        )


        support_mask = cv2.dilate(
            mask_bool.astype(
                np.uint8
            ),
            kernel,
            iterations=1
        ).astype(
            bool
        )

    else:

        support_mask = mask_bool.copy()


    points_cam = transform_points(
        model_points,
        T_est
    )


    z = points_cam[
        :,
        2
    ]


    valid_z = (
        z > 1e-6
    )


    u_float = np.zeros_like(
        z
    )

    v_float = np.zeros_like(
        z
    )


    u_float[
        valid_z
    ] = (
        fx
        * points_cam[
            valid_z,
            0
        ]
        / z[
            valid_z
        ]
        + cx
    )


    v_float[
        valid_z
    ] = (
        cy
        - fy
        * points_cam[
            valid_z,
            1
        ]
        / z[
            valid_z
        ]
    )


    u = np.rint(
        u_float
    ).astype(
        np.int32
    )


    v = np.rint(
        v_float
    ).astype(
        np.int32
    )


    inside = (
        valid_z
        & (
            u >= 0
        )
        & (
            u < w
        )
        & (
            v >= 0
        )
        & (
            v < h
        )
    )


    candidate_indices = np.nonzero(
        inside
    )[0]


    if len(
        candidate_indices
    ) == 0:

        return (
            np.empty(
                (
                    0,
                    3
                ),
                dtype=np.float64
            ),
            np.empty(
                0,
                dtype=np.int64
            ),
            {
                "projected_points": 0,
                "mask_supported_points": 0,
                "zbuffer_visible_points": 0
            }
        )


    uu = u[
        candidate_indices
    ]


    vv = v[
        candidate_indices
    ]


    zz = z[
        candidate_indices
    ]


    mask_supported = support_mask[
        vv,
        uu
    ]


    candidate_indices = candidate_indices[
        mask_supported
    ]


    uu = uu[
        mask_supported
    ]


    vv = vv[
        mask_supported
    ]


    zz = zz[
        mask_supported
    ]


    if len(
        candidate_indices
    ) == 0:

        return (
            np.empty(
                (
                    0,
                    3
                ),
                dtype=np.float64
            ),
            np.empty(
                0,
                dtype=np.int64
            ),
            {
                "projected_points": int(
                    np.count_nonzero(
                        inside
                    )
                ),
                "mask_supported_points": 0,
                "zbuffer_visible_points": 0
            }
        )


    # --------------------------------------------------
    # Pixel z-buffer.
    #
    # For each projected pixel, retain model points whose depth
    # is within a small tolerance of the nearest projected model
    # depth. This removes self-occluded back-surface points.
    # --------------------------------------------------
    linear_pixel = (
        vv.astype(
            np.int64
        )
        * int(
            w
        )
        + uu.astype(
            np.int64
        )
    )


    order = np.argsort(
        linear_pixel
    )


    pixel_sorted = linear_pixel[
        order
    ]


    z_sorted = zz[
        order
    ]


    idx_sorted = candidate_indices[
        order
    ]


    unique_pixels, start_indices = np.unique(
        pixel_sorted,
        return_index=True
    )


    z_min = np.minimum.reduceat(
        z_sorted,
        start_indices
    )


    group_index = np.searchsorted(
        unique_pixels,
        pixel_sorted
    )


    visible_sorted = (
        z_sorted
        <= (
            z_min[
                group_index
            ]
            + EST_VIS_ZBUFFER_TOLERANCE
        )
    )


    visible_indices = idx_sorted[
        visible_sorted
    ]


    # Stable ordering makes experiments reproducible.
    visible_indices = np.sort(
        visible_indices
    )


    if (
        MAX_SOURCE_POINTS is not None
        and len(
            visible_indices
        )
        > MAX_SOURCE_POINTS
    ):

        sample_idx = np.linspace(
            0,
            len(
                visible_indices
            )
            - 1,
            MAX_SOURCE_POINTS
        )


        sample_idx = np.round(
            sample_idx
        ).astype(
            int
        )


        visible_indices = visible_indices[
            sample_idx
        ]


    source_points = model_points[
        visible_indices
    ].copy()


    diagnostics = {
        "projected_points": int(
            np.count_nonzero(
                inside
            )
        ),
        "mask_supported_points": int(
            len(
                candidate_indices
            )
        ),
        "zbuffer_visible_points": int(
            len(
                visible_indices
            )
        )
    }


    return (
        source_points,
        visible_indices.astype(
            np.int64
        ),
        diagnostics
    )



# ==================================================
# Local image-depth support
# ==================================================
def compute_local_depth_support_residuals(
        points_cam,
        depth,
        mask_bool,
        K
):

    n = len(
        points_cam
    )


    if n == 0:

        return np.empty(
            0,
            dtype=np.float64
        )


    fx = K[
        0,
        0
    ]

    fy = K[
        1,
        1
    ]

    cx = K[
        0,
        2
    ]

    cy = K[
        1,
        2
    ]


    h, w = depth.shape


    z = points_cam[
        :,
        2
    ]


    valid_z = (
        z > 1e-6
    )


    safe_z = np.maximum(
        z,
        1e-9
    )


    u = np.rint(
        fx
        * points_cam[
            :,
            0
        ]
        / safe_z
        + cx
    ).astype(
        np.int32
    )


    v = np.rint(
        cy
        - fy
        * points_cam[
            :,
            1
        ]
        / safe_z
    ).astype(
        np.int32
    )


    best = np.full(
        n,
        np.inf,
        dtype=np.float64
    )


    radius = int(
        OBS_SUPPORT_DEPTH_WINDOW_RADIUS
    )


    for dv in range(
        -radius,
        radius + 1
    ):

        vv = (
            v
            + dv
        )


        valid_v = (
            vv >= 0
        ) & (
            vv < h
        )


        for du in range(
            -radius,
            radius + 1
        ):

            uu = (
                u
                + du
            )


            inside = (
                valid_z
                & valid_v
                & (
                    uu >= 0
                )
                & (
                    uu < w
                )
            )


            idx = np.nonzero(
                inside
            )[0]


            if len(
                idx
            ) == 0:

                continue


            observed = depth[
                vv[
                    idx
                ],
                uu[
                    idx
                ]
            ]


            observed_valid = (
                mask_bool[
                    vv[
                        idx
                    ],
                    uu[
                        idx
                    ]
                ]
                & (
                    observed
                    > 0
                )
            )


            idx = idx[
                observed_valid
            ]


            if len(
                idx
            ) == 0:

                continue


            observed = depth[
                vv[
                    idx
                ],
                uu[
                    idx
                ]
            ]


            residual = np.abs(
                observed
                - z[
                    idx
                ]
            )


            best[
                idx
            ] = np.minimum(
                best[
                    idx
                ],
                residual
            )


    return best


# ==================================================
# GT-free observation-supported estimated visibility
# ==================================================
def generate_observation_supported_visible_model(
        model_points,
        T_est,
        depth,
        mask_bool,
        K,
        target_cloud
):

    # Stage A:
    # model projection + self-occlusion z-buffer + mask support.
    (
        _,
        zbuffer_indices,
        zbuffer_diag
    ) = generate_estimated_visible_model(
        model_points,
        T_est,
        mask_bool,
        K
    )


    if len(
        zbuffer_indices
    ) == 0:

        diagnostics = dict(
            zbuffer_diag
        )


        diagnostics.update({
            "zbuffer_candidates": 0,
            "support_3d_points": 0,
            "support_depth_points": 0,
            "joint_supported_points": 0,
            "median_nn_distance_m": np.nan,
            "median_depth_residual_m": np.nan,
            "mean_soft_support": np.nan,
            "quality_score": np.inf
        })


        return (
            np.empty(
                (
                    0,
                    3
                ),
                dtype=np.float64
            ),
            np.empty(
                0,
                dtype=np.int64
            ),
            diagnostics
        )


    candidate_model = model_points[
        zbuffer_indices
    ]


    candidate_cam = transform_points(
        candidate_model,
        T_est
    )


    target_points = np.asarray(
        target_cloud.points,
        dtype=np.float64
    )


    if len(
        target_points
    ) == 0:

        diagnostics = dict(
            zbuffer_diag
        )


        diagnostics.update({
            "zbuffer_candidates": int(
                len(
                    zbuffer_indices
                )
            ),
            "support_3d_points": 0,
            "support_depth_points": 0,
            "joint_supported_points": 0,
            "median_nn_distance_m": np.nan,
            "median_depth_residual_m": np.nan,
            "mean_soft_support": np.nan,
            "quality_score": np.inf
        })


        return (
            np.empty(
                (
                    0,
                    3
                ),
                dtype=np.float64
            ),
            np.empty(
                0,
                dtype=np.int64
            ),
            diagnostics
        )


    tree = cKDTree(
        target_points
    )


    nn_distance, _ = tree.query(
        candidate_cam,
        k=1
    )


    depth_residual = compute_local_depth_support_residuals(
        candidate_cam,
        depth,
        mask_bool,
        K
    )


    support_3d = (
        nn_distance
        <= OBS_SUPPORT_3D_MAX_DISTANCE
    )


    support_depth = (
        depth_residual
        <= OBS_SUPPORT_DEPTH_MAX_RESIDUAL
    )


    joint_support = (
        support_3d
        & support_depth
    )


    kept_indices = zbuffer_indices[
        joint_support
    ]


    # Stable deterministic cap.
    if (
        MAX_SOURCE_POINTS is not None
        and len(
            kept_indices
        )
        > MAX_SOURCE_POINTS
    ):

        sample_idx = np.linspace(
            0,
            len(
                kept_indices
            )
            - 1,
            MAX_SOURCE_POINTS
        )


        sample_idx = np.round(
            sample_idx
        ).astype(
            int
        )


        kept_indices = kept_indices[
            sample_idx
        ]


    source_points = model_points[
        kept_indices
    ].copy()


    finite_depth = np.isfinite(
        depth_residual
    )


    if np.any(
        finite_depth
    ):

        median_depth = float(
            np.median(
                depth_residual[
                    finite_depth
                ]
            )
        )

    else:

        median_depth = np.inf


    median_nn = float(
        np.median(
            nn_distance
        )
    )


    soft_3d = np.exp(
        -0.5
        * (
            nn_distance
            / OBS_SUPPORT_3D_SOFT_SCALE
        ) ** 2
    )


    soft_depth = np.zeros_like(
        depth_residual,
        dtype=np.float64
    )


    soft_depth[
        finite_depth
    ] = np.exp(
        -0.5
        * (
            depth_residual[
                finite_depth
            ]
            / OBS_SUPPORT_DEPTH_SOFT_SCALE
        ) ** 2
    )


    soft_support = (
        soft_3d
        * soft_depth
    )


    mean_soft_support = float(
        np.mean(
            soft_support
        )
    )


    quality_score = float(
        median_nn
        / OBS_SUPPORT_3D_MAX_DISTANCE
        + median_depth
        / OBS_SUPPORT_DEPTH_MAX_RESIDUAL
    )


    diagnostics = dict(
        zbuffer_diag
    )


    diagnostics.update({
        "zbuffer_candidates": int(
            len(
                zbuffer_indices
            )
        ),
        "support_3d_points": int(
            np.count_nonzero(
                support_3d
            )
        ),
        "support_depth_points": int(
            np.count_nonzero(
                support_depth
            )
        ),
        "joint_supported_points": int(
            len(
                kept_indices
            )
        ),
        "median_nn_distance_m": median_nn,
        "median_depth_residual_m": median_depth,
        "mean_soft_support": mean_soft_support,
        "quality_score": quality_score
    })


    return (
        source_points,
        kept_indices.astype(
            np.int64
        ),
        diagnostics
    )


# ==================================================
# Conservative one-refresh observation-supported ICP
# ==================================================
def run_iterative_observation_supported_local_icp(
        model_points,
        depth,
        mask_bool,
        K,
        target_cloud,
        local_reliability,
        T_initial,
        T_gt
):

    (
        source_stage1,
        indices_stage1,
        diag_stage1
    ) = generate_observation_supported_visible_model(
        model_points,
        T_initial,
        depth,
        mask_bool,
        K,
        target_cloud
    )


    stage1 = run_local_with_source(
        source_stage1,
        target_cloud,
        T_initial,
        T_gt,
        local_reliability
    )


    T_stage1 = stage1[
        "T"
    ]


    (
        source_stage2,
        indices_stage2,
        diag_stage2
    ) = generate_observation_supported_visible_model(
        model_points,
        T_stage1,
        depth,
        mask_bool,
        K,
        target_cloud
    )


    # GT-free refresh gate:
    # refresh only if the newly inferred source has no-worse
    # observation support quality.
    refresh_eligible = bool(
        len(
            source_stage2
        )
        >= MIN_ESTIMATED_VISIBLE_CAD_POINTS
        and np.isfinite(
            diag_stage2[
                "quality_score"
            ]
        )
        and (
            not np.isfinite(
                diag_stage1[
                    "quality_score"
                ]
            )
            or diag_stage2[
                "quality_score"
            ]
            <= diag_stage1[
                "quality_score"
            ]
        )
    )


    if not refresh_eligible:

        result = dict(
            stage1
        )


        result.update({
            "stage1": stage1,
            "refresh_used": 0,
            "refresh_accepted": 0,
            "source_stage1_points": len(
                source_stage1
            ),
            "source_stage2_points": len(
                source_stage2
            ),
            "indices_stage1": indices_stage1,
            "indices_stage2": indices_stage2,
            "diag_stage1": diag_stage1,
            "diag_stage2": diag_stage2,
            "final_visibility_indices": indices_stage1
        })


        return result


    stage2 = run_local_with_source(
        source_stage2,
        target_cloud,
        T_stage1,
        T_gt,
        local_reliability
    )


    (
        _,
        indices_after_stage2,
        diag_after_stage2
    ) = generate_observation_supported_visible_model(
        model_points,
        stage2[
            "T"
        ],
        depth,
        mask_bool,
        K,
        target_cloud
    )


    # Final GT-free guard:
    # accept Stage 2 only when observation support quality does
    # not deteriorate after the refinement.
    refresh_accepted = bool(
        np.isfinite(
            diag_after_stage2[
                "quality_score"
            ]
        )
        and diag_after_stage2[
            "quality_score"
        ]
        <= diag_stage2[
            "quality_score"
        ]
    )


    if refresh_accepted:

        result = dict(
            stage2
        )


        final_indices = indices_after_stage2

    else:

        result = dict(
            stage1
        )


        final_indices = indices_stage1


    result.update({
        "stage1": stage1,
        "stage2": stage2,
        "refresh_used": 1,
        "refresh_accepted": int(
            refresh_accepted
        ),
        "source_stage1_points": len(
            source_stage1
        ),
        "source_stage2_points": len(
            source_stage2
        ),
        "indices_stage1": indices_stage1,
        "indices_stage2": indices_stage2,
        "diag_stage1": diag_stage1,
        "diag_stage2": diag_stage2,
        "diag_after_stage2": diag_after_stage2,
        "final_visibility_indices": final_indices
    })


    return result


# ==================================================
# Oracle visible indices for diagnostics only
# ==================================================
def generate_oracle_visible_model_with_indices(
        model_points,
        T_gt,
        depth,
        mask_bool,
        K
):

    fx = K[
        0,
        0
    ]

    fy = K[
        1,
        1
    ]

    cx = K[
        0,
        2
    ]

    cy = K[
        1,
        2
    ]


    h, w = depth.shape


    points_cam = transform_points(
        model_points,
        T_gt
    )


    z = points_cam[
        :,
        2
    ]


    valid_z = (
        z > 1e-6
    )


    u_float = np.zeros_like(
        z
    )

    v_float = np.zeros_like(
        z
    )


    u_float[
        valid_z
    ] = (
        fx
        * points_cam[
            valid_z,
            0
        ]
        / z[
            valid_z
        ]
        + cx
    )


    v_float[
        valid_z
    ] = (
        cy
        - fy
        * points_cam[
            valid_z,
            1
        ]
        / z[
            valid_z
        ]
    )


    u = np.rint(
        u_float
    ).astype(
        np.int32
    )


    v = np.rint(
        v_float
    ).astype(
        np.int32
    )


    inside = (
        valid_z
        & (
            u >= 0
        )
        & (
            u < w
        )
        & (
            v >= 0
        )
        & (
            v < h
        )
    )


    candidate_indices = np.nonzero(
        inside
    )[0]


    if len(
        candidate_indices
    ) == 0:

        return (
            np.empty(
                (
                    0,
                    3
                ),
                dtype=np.float64
            ),
            np.empty(
                0,
                dtype=np.int64
            )
        )


    uu = u[
        candidate_indices
    ]


    vv = v[
        candidate_indices
    ]


    zz = z[
        candidate_indices
    ]


    observed_depth = depth[
        vv,
        uu
    ]


    visible = (
        mask_bool[
            vv,
            uu
        ]
        & (
            observed_depth
            > 0
        )
        & (
            np.abs(
                observed_depth
                - zz
            )
            < VISIBLE_DEPTH_TOLERANCE
        )
    )


    visible_indices = candidate_indices[
        visible
    ]


    visible_indices = np.sort(
        visible_indices
    )


    if (
        MAX_SOURCE_POINTS is not None
        and len(
            visible_indices
        )
        > MAX_SOURCE_POINTS
    ):

        sample_idx = np.linspace(
            0,
            len(
                visible_indices
            )
            - 1,
            MAX_SOURCE_POINTS
        )


        sample_idx = np.round(
            sample_idx
        ).astype(
            int
        )


        visible_indices = visible_indices[
            sample_idx
        ]


    return (
        model_points[
            visible_indices
        ].copy(),
        visible_indices.astype(
            np.int64
        )
    )


# ==================================================
# Visibility-set diagnostics
# ==================================================
def visibility_set_metrics(
        estimated_indices,
        oracle_indices
):

    estimated_set = set(
        np.asarray(
            estimated_indices,
            dtype=np.int64
        ).tolist()
    )


    oracle_set = set(
        np.asarray(
            oracle_indices,
            dtype=np.int64
        ).tolist()
    )


    intersection = len(
        estimated_set
        & oracle_set
    )


    union = len(
        estimated_set
        | oracle_set
    )


    if len(
        estimated_set
    ) > 0:

        precision = (
            intersection
            / len(
                estimated_set
            )
        )

    else:

        precision = np.nan


    if len(
        oracle_set
    ) > 0:

        recall = (
            intersection
            / len(
                oracle_set
            )
        )

    else:

        recall = np.nan


    if union > 0:

        jaccard = (
            intersection
            / union
        )

    else:

        jaccard = np.nan


    return {
        "intersection": intersection,
        "precision": float(
            precision
        ),
        "recall": float(
            recall
        ),
        "jaccard": float(
            jaccard
        )
    }


# ==================================================
# Run Local Reliability ICP with an explicit source
# ==================================================
def run_local_with_source(
        source_points,
        target_cloud,
        T_initial,
        T_gt,
        local_reliability
):

    if len(
        source_points
    ) < MIN_ESTIMATED_VISIBLE_CAD_POINTS:

        return {
            "T": T_initial.copy(),
            "fitness": 0.0,
            "rmse": np.nan,
            "add_s_m": compute_add_s(
                evaluation_model_points,
                T_initial,
                T_gt
            ),
            "success": 0.0,
            "objective": np.inf,
            "accepted_steps": 0,
            "rejected_steps": 0,
            "final_lambda": np.nan,
            "termination_reason": "source_too_small"
        }


    return run_stable_weighted_icp(
        source_points,
        target_cloud,
        T_initial,
        T_gt,
        local_reliability
    )



# ==================================================
# Geometry-first Local Reliability refinement
# ==================================================
def run_geometry_then_local_refinement(
        source_points,
        target_cloud,
        geometry_result,
        T_gt,
        local_reliability
):

    # --------------------------------------------------
    # Stage 1 is NOT rerun here.
    # The already-computed geometry_result is reused so that:
    #
    #   geometry_icp
    # and
    #   geometry_then_local_refinement
    #
    # have exactly the same Stage-1 solution.
    # --------------------------------------------------
    stage2 = run_local_with_source(
        source_points,
        target_cloud,
        geometry_result[
            "T"
        ],
        T_gt,
        local_reliability
    )


    result = dict(
        stage2
    )


    result.update({
        "stage1_T": geometry_result[
            "T"
        ],
        "stage1_add_s_m": geometry_result[
            "add_s_m"
        ],
        "stage1_success": geometry_result[
            "success"
        ],
        "stage1_fitness": geometry_result[
            "fitness"
        ],
        "stage1_rmse": geometry_result[
            "rmse"
        ],
        "stage1_objective": geometry_result[
            "objective"
        ],
        "stage1_accepted_steps": geometry_result[
            "accepted_steps"
        ],
        "stage1_rejected_steps": geometry_result[
            "rejected_steps"
        ],
        "stage1_final_lambda": geometry_result[
            "final_lambda"
        ],
        "stage1_termination_reason": geometry_result[
            "termination_reason"
        ],
        "stage2_add_s_delta_mm": (
            stage2[
                "add_s_m"
            ]
            - geometry_result[
                "add_s_m"
            ]
        ) * 1000.0,
        "stage2_success_change": (
            stage2[
                "success"
            ]
            - geometry_result[
                "success"
            ]
        )
    })


    return result


# ==================================================
# One-refresh iterative estimated visibility
# ==================================================
def run_iterative_estimated_visibility_local_icp(
        model_points,
        mask_bool,
        K,
        target_cloud,
        local_reliability,
        T_initial,
        T_gt
):

    (
        source_stage1,
        indices_stage1,
        diag_stage1
    ) = generate_estimated_visible_model(
        model_points,
        T_initial,
        mask_bool,
        K
    )


    stage1 = run_local_with_source(
        source_stage1,
        target_cloud,
        T_initial,
        T_gt,
        local_reliability
    )


    T_current = stage1[
        "T"
    ]


    (
        source_stage2,
        indices_stage2,
        diag_stage2
    ) = generate_estimated_visible_model(
        model_points,
        T_current,
        mask_bool,
        K
    )


    stage2 = run_local_with_source(
        source_stage2,
        target_cloud,
        T_current,
        T_gt,
        local_reliability
    )


    result = dict(
        stage2
    )


    result[
        "stage1"
    ] = stage1


    result[
        "source_stage1_points"
    ] = len(
        source_stage1
    )


    result[
        "source_stage2_points"
    ] = len(
        source_stage2
    )


    result[
        "indices_stage1"
    ] = indices_stage1


    result[
        "indices_stage2"
    ] = indices_stage2


    result[
        "diag_stage1"
    ] = diag_stage1


    result[
        "diag_stage2"
    ] = diag_stage2


    return result


# ==================================================
# Initial pose
# ==================================================
def generate_initial_pose(
        T_gt
):

    rng = np.random.RandomState(
        POSE_SEED
    )


    angle = np.deg2rad(
        rng.uniform(
            -5.0,
            5.0
        )
    )


    axis = rng.randn(
        3
    )


    axis /= np.linalg.norm(
        axis
    )


    R_noise = (
        o3d.geometry.get_rotation_matrix_from_axis_angle(
            axis
            * angle
        )
    )


    t_noise = rng.uniform(
        -0.005,
        0.005,
        size=3
    )


    if CENTERED_INITIALIZATION:

        T_init = T_gt.copy()


        T_init[
            :3,
            :3
        ] = (
            R_noise
            @ T_gt[
                :3,
                :3
            ]
        )


        T_init[
            :3,
            3
        ] = (
            T_gt[
                :3,
                3
            ]
            + t_noise
        )


        return T_init


    T_noise = np.eye(
        4,
        dtype=np.float64
    )


    T_noise[
        :3,
        :3
    ] = R_noise


    T_noise[
        :3,
        3
    ] = t_noise


    return (
        T_noise
        @ T_gt
    )


# ==================================================
# Select a spatially local corruption patch
#
# We choose one valid object pixel as a center and then select
# the K nearest valid object pixels in image space.
#
# This produces a compact local region while preserving an
# approximately exact requested corruption fraction.
# ==================================================
def select_local_patch(
        clean_depth,
        mask_bool,
        fraction,
        seed
):

    valid = (
        mask_bool
        & (
            clean_depth
            > 0
        )
    )


    v_valid, u_valid = np.nonzero(
        valid
    )


    n_valid = len(
        v_valid
    )


    if n_valid == 0:

        raise RuntimeError(
            "No valid object depth pixels."
        )


    k = int(
        np.round(
            fraction
            * n_valid
        )
    )


    k = max(
        1,
        min(
            k,
            n_valid
        )
    )


    rng = np.random.default_rng(
        seed
    )


    center_idx = int(
        rng.integers(
            0,
            n_valid
        )
    )


    center_v = int(
        v_valid[
            center_idx
        ]
    )


    center_u = int(
        u_valid[
            center_idx
        ]
    )


    dist2 = (
        (
            v_valid
            - center_v
        ) ** 2
        + (
            u_valid
            - center_u
        ) ** 2
    )


    if k == n_valid:

        selected = np.arange(
            n_valid
        )

    else:

        selected = np.argpartition(
            dist2,
            k - 1
        )[
            :k
        ]


    patch_mask = np.zeros_like(
        mask_bool,
        dtype=bool
    )


    patch_mask[
        v_valid[
            selected
        ],
        u_valid[
            selected
        ]
    ] = True


    actual_fraction = float(
        np.count_nonzero(
            patch_mask
        )
        / n_valid
    )


    return (
        patch_mask,
        center_u,
        center_v,
        actual_fraction
    )


# ==================================================
# Apply structured local depth corruption
#
# Corruption is applied to the DEPTH IMAGE first.
# The point cloud is regenerated afterward, so x/y/z remain
# consistent with the camera model.
# ==================================================
def make_local_corrupted_target(
        clean_depth,
        mask_bool,
        K,
        patch_mask,
        corruption_type,
        value_seed
):

    corrupted_depth = clean_depth.copy()


    patch_valid = (
        patch_mask
        & mask_bool
        & (
            clean_depth
            > 0
        )
    )


    patch_pixels = int(
        np.count_nonzero(
            patch_valid
        )
    )


    if patch_pixels == 0:

        raise RuntimeError(
            "Local corruption patch is empty."
        )


    rng = np.random.default_rng(
        value_seed
    )


    signed_bias_mm = 0.0


    if corruption_type == "local_bias":

        sign = (
            1.0
            if rng.random() >= 0.5
            else -1.0
        )


        signed_bias = (
            sign
            * LOCAL_BIAS_MAGNITUDE_M
        )


        corrupted_depth[
            patch_valid
        ] += signed_bias


        signed_bias_mm = (
            signed_bias
            * 1000.0
        )


    elif corruption_type == "local_outlier":

        magnitudes = rng.uniform(
            LOCAL_OUTLIER_MIN_ABS_M,
            LOCAL_OUTLIER_MAX_ABS_M,
            size=patch_pixels
        )


        signs = rng.choice(
            np.array(
                [
                    -1.0,
                    1.0
                ],
                dtype=np.float64
            ),
            size=patch_pixels
        )


        corrupted_depth[
            patch_valid
        ] += (
            signs
            * magnitudes
        )


    elif corruption_type == "local_hole":

        corrupted_depth[
            patch_valid
        ] = 0.0


    else:

        raise ValueError(
            "Unknown corruption type: "
            + str(
                corruption_type
            )
        )


    corrupted_depth[
        corrupted_depth
        <= 0.001
    ] = 0.0


    target_cloud, target_uv = depth_to_cloud(
        corrupted_depth,
        mask_bool,
        K
    )


    depth_reliability = compute_depth_reliability(
        corrupted_depth,
        mask_bool,
        target_uv
    )


    if len(
        target_cloud.points
    ) == 0:

        raise RuntimeError(
            "Corruption removed the entire target cloud."
        )


    # --------------------------------------------------
    # Diagnostics:
    # how strongly did the current depth-reliability signal
    # react inside and outside the corrupted patch?
    #
    # For local_hole there may be no remaining points inside
    # the patch, so patch reliability can be NaN.
    # --------------------------------------------------
    u = target_uv[
        :,
        0
    ]


    v = target_uv[
        :,
        1
    ]


    target_in_patch = patch_mask[
        v,
        u
    ]


    if np.any(
        target_in_patch
    ):

        patch_reliability_mean = float(
            np.mean(
                depth_reliability[
                    target_in_patch
                ]
            )
        )

    else:

        patch_reliability_mean = np.nan


    target_outside_patch = (
        ~target_in_patch
    )


    if np.any(
        target_outside_patch
    ):

        outside_reliability_mean = float(
            np.mean(
                depth_reliability[
                    target_outside_patch
                ]
            )
        )

    else:

        outside_reliability_mean = np.nan


    return {
        "corrupted_depth": corrupted_depth,
        "target_cloud": target_cloud,
        "target_uv": target_uv,
        "depth_reliability": depth_reliability,
        "global_reliability_mean": float(
            np.mean(
                depth_reliability
            )
        ),
        "patch_reliability_mean": patch_reliability_mean,
        "outside_reliability_mean": outside_reliability_mean,
        "target_retention_ratio": float(
            len(
                target_cloud.points
            )
            / np.count_nonzero(
                mask_bool
                & (
                    clean_depth
                    > 0
                )
            )
        ),
        "signed_bias_mm": signed_bias_mm
    }


# ==================================================
# Ordinary Open3D ICP
# ==================================================
def run_open3d_icp(
        source_points,
        target_cloud,
        T_initial,
        T_gt
):

    source_cloud = (
        o3d.geometry.PointCloud()
    )


    source_cloud.points = (
        o3d.utility.Vector3dVector(
            source_points
        )
    )


    src = copy.deepcopy(
        source_cloud
    )


    tgt = copy.deepcopy(
        target_cloud
    )


    src.transform(
        T_initial
    )


    src.estimate_normals()
    tgt.estimate_normals()


    result = (
        o3d.pipelines.registration.registration_icp(
            src,
            tgt,
            max_correspondence_distance,
            np.eye(
                4
            ),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_iterations
            )
        )
    )


    T_final = (
        result.transformation
        @ T_initial
    )


    add_s = compute_add_s(
        evaluation_model_points,
        T_final,
        T_gt
    )


    return {
        "T": T_final,
        "fitness": float(
            result.fitness
        ),
        "rmse": float(
            result.inlier_rmse
        ),
        "add_s_m": add_s,
        "success": float(
            add_s
            < success_threshold_m
        )
    }


# ==================================================
# Prepare target for custom solver
# ==================================================
def prepare_target(
        target_cloud
):

    tgt = copy.deepcopy(
        target_cloud
    )


    tgt.estimate_normals()


    target_points = np.asarray(
        tgt.points,
        dtype=np.float64
    )


    target_normals = np.asarray(
        tgt.normals,
        dtype=np.float64
    ).copy()


    normal_norm = np.linalg.norm(
        target_normals,
        axis=1
    )


    valid_normal = (
        normal_norm
        > 1e-8
    )


    target_normals[
        valid_normal
    ] /= (
        normal_norm[
            valid_normal,
            None
        ]
    )


    tree = cKDTree(
        target_points
    )


    return (
        target_points,
        target_normals,
        tree
    )


# ==================================================
# Build weighted linear system
# ==================================================
def build_weighted_system(
        source_points,
        T,
        target_points,
        target_normals,
        tree,
        point_factor
):

    source_world = transform_points(
        source_points,
        T
    )


    distances, indices = tree.query(
        source_world,
        k=1
    )


    inlier = (
        distances
        < max_correspondence_distance
    )


    inlier_count = int(
        np.count_nonzero(
            inlier
        )
    )


    if (
        inlier_count
        < min_correspondences
    ):

        return None


    p = source_world[
        inlier
    ]


    corr_indices = indices[
        inlier
    ]


    q = target_points[
        corr_indices
    ]


    n = target_normals[
        corr_indices
    ]


    d = distances[
        inlier
    ]


    factor = point_factor[
        corr_indices
    ]


    n_ok = (
        np.linalg.norm(
            n,
            axis=1
        )
        > 1e-8
    )


    p = p[
        n_ok
    ]


    q = q[
        n_ok
    ]


    n = n[
        n_ok
    ]


    d = d[
        n_ok
    ]


    factor = factor[
        n_ok
    ]


    if len(
        p
    ) < min_correspondences:

        return None


    residual = np.sum(
        n
        * (
            p
            - q
        ),
        axis=1
    )


    w_residual = (
        1.0
        / (
            1.0
            + (
                residual
                / robust_scale
            ) ** 2
        )
    )


    w_corr = np.exp(
        -0.5
        * (
            d
            / correspondence_scale
        ) ** 2
    )


    factor = np.clip(
        factor,
        depth_weight_floor,
        1.0
    )


    weights = (
        w_residual
        * w_corr
        * factor
    )


    weights = np.clip(
        weights,
        1e-6,
        1.0
    )


    J_rot = np.cross(
        p,
        n
    )


    J = np.hstack(
        [
            J_rot,
            n
        ]
    )


    sqrt_w = np.sqrt(
        weights
    )


    Jw = (
        J
        * sqrt_w[
            :,
            None
        ]
    )


    rw = (
        residual
        * sqrt_w
    )


    return {
        "Jw": Jw,
        "rw": rw,
        "residual": residual,
        "weights": weights,
        "distances": d,
        "inlier_ratio": float(
            len(
                p
            )
            / len(
                source_points
            )
        )
    }


# ==================================================
# Stable objective
#
# This objective is only used to decide whether a proposed
# LM/line-search step should be accepted.
#
# Lower is better.
# ==================================================
def evaluate_weighted_objective(
        source_points,
        T,
        target_points,
        target_normals,
        tree,
        point_factor
):

    system = build_weighted_system(
        source_points,
        T,
        target_points,
        target_normals,
        tree,
        point_factor
    )


    if system is None:

        return np.inf


    weights = system[
        "weights"
    ]


    residual = system[
        "residual"
    ]


    weight_sum = float(
        np.sum(
            weights
        )
    )


    if (
        weight_sum
        <= 1e-12
    ):

        return np.inf


    weighted_mse = float(
        np.sum(
            weights
            * residual ** 2
        )
        / weight_sum
    )


    coverage_penalty = (
        objective_coverage_penalty
        * robust_scale ** 2
        * (
            1.0
            - system[
                "inlier_ratio"
            ]
        )
    )


    return (
        weighted_mse
        + coverage_penalty
    )


# ==================================================
# Small SE(3) increment
# ==================================================
def make_increment(
        delta,
        step_scale
):

    omega = (
        delta[
            :3
        ]
        * step_scale
    )


    trans = (
        delta[
            3:
        ]
        * step_scale
    )


    omega_norm = np.linalg.norm(
        omega
    )


    if (
        omega_norm
        > max_rotation_step
    ):

        omega *= (
            max_rotation_step
            / omega_norm
        )


    trans_norm = np.linalg.norm(
        trans
    )


    if (
        trans_norm
        > max_translation_step
    ):

        trans *= (
            max_translation_step
            / trans_norm
        )


    R_inc, _ = cv2.Rodrigues(
        omega.astype(
            np.float64
        )
    )


    T_inc = np.eye(
        4,
        dtype=np.float64
    )


    T_inc[
        :3,
        :3
    ] = R_inc


    T_inc[
        :3,
        3
    ] = trans


    return (
        T_inc,
        omega,
        trans
    )


# ==================================================
# Stable weighted ICP
#
# Levenberg-Marquardt + backtracking line search
# ==================================================
def run_stable_weighted_icp(
        source_points,
        target_cloud,
        T_initial,
        T_gt,
        point_factor
):

    (
        target_points,
        target_normals,
        tree
    ) = prepare_target(
        target_cloud
    )


    point_factor = np.asarray(
        point_factor,
        dtype=np.float64
    )


    if len(
        point_factor
    ) != len(
        target_points
    ):

        raise ValueError(
            "point_factor length does not match target points."
        )


    T = T_initial.copy()


    current_cost = evaluate_weighted_objective(
        source_points,
        T,
        target_points,
        target_normals,
        tree,
        point_factor
    )


    lm_lambda = (
        lm_lambda_initial
    )


    accepted_steps = 0
    rejected_steps = 0

    termination_reason = (
        "max_iterations"
    )


    for iteration in range(
        max_iterations
    ):

        system = build_weighted_system(
            source_points,
            T,
            target_points,
            target_normals,
            tree,
            point_factor
        )


        if system is None:

            termination_reason = (
                "too_few_correspondences"
            )

            break


        Jw = system[
            "Jw"
        ]


        rw = system[
            "rw"
        ]


        H = (
            Jw.T
            @ Jw
        )


        g = (
            Jw.T
            @ rw
        )


        diag_H = np.maximum(
            np.diag(
                H
            ),
            1e-9
        )


        accepted = False


        for _ in range(
            lm_max_attempts
        ):

            H_lm = (
                H
                + lm_lambda
                * np.diag(
                    diag_H
                )
            )


            try:

                delta = -np.linalg.solve(
                    H_lm,
                    g
                )


            except np.linalg.LinAlgError:

                lm_lambda = min(
                    lm_lambda
                    * lm_increase,
                    lm_lambda_max
                )

                rejected_steps += 1

                continue


            if not np.all(
                np.isfinite(
                    delta
                )
            ):

                lm_lambda = min(
                    lm_lambda
                    * lm_increase,
                    lm_lambda_max
                )

                rejected_steps += 1

                continue


            for step_scale in line_search_steps:

                (
                    T_inc,
                    omega_step,
                    trans_step
                ) = make_increment(
                    delta,
                    step_scale
                )


                T_candidate = (
                    T_inc
                    @ T
                )


                candidate_cost = (
                    evaluate_weighted_objective(
                        source_points,
                        T_candidate,
                        target_points,
                        target_normals,
                        tree,
                        point_factor
                    )
                )


                if (
                    np.isfinite(
                        candidate_cost
                    )
                    and candidate_cost
                    < current_cost
                    - cost_tolerance
                ):

                    cost_drop = (
                        current_cost
                        - candidate_cost
                    )


                    T = T_candidate

                    current_cost = (
                        candidate_cost
                    )


                    lm_lambda = max(
                        lm_lambda
                        * lm_decrease,
                        lm_lambda_min
                    )


                    accepted_steps += 1

                    accepted = True


                    if (
                        np.linalg.norm(
                            omega_step
                        )
                        < rotation_step_tolerance
                        and np.linalg.norm(
                            trans_step
                        )
                        < translation_step_tolerance
                    ):

                        termination_reason = (
                            "small_step"
                        )


                    elif (
                        cost_drop
                        < cost_tolerance
                    ):

                        termination_reason = (
                            "small_cost_change"
                        )


                    break


            if accepted:

                break


            lm_lambda = min(
                lm_lambda
                * lm_increase,
                lm_lambda_max
            )


            rejected_steps += 1


        if not accepted:

            termination_reason = (
                "no_descent_step"
            )

            break


        if termination_reason in (
            "small_step",
            "small_cost_change"
        ):

            break


    # ==================================================
    # Final correspondence statistics
    # ==================================================
    source_final = transform_points(
        source_points,
        T
    )


    final_distance, _ = tree.query(
        source_final,
        k=1
    )


    final_inlier = (
        final_distance
        < max_correspondence_distance
    )


    fitness = float(
        np.mean(
            final_inlier
        )
    )


    if np.any(
        final_inlier
    ):

        rmse = float(
            np.sqrt(
                np.mean(
                    final_distance[
                        final_inlier
                    ] ** 2
                )
            )
        )

    else:

        rmse = 0.0


    add_s = compute_add_s(
        evaluation_model_points,
        T,
        T_gt
    )


    return {
        "T": T,
        "fitness": fitness,
        "rmse": rmse,
        "add_s_m": add_s,
        "success": float(
            add_s
            < success_threshold_m
        ),
        "objective": float(
            current_cost
        ),
        "accepted_steps": accepted_steps,
        "rejected_steps": rejected_steps,
        "final_lambda": float(
            lm_lambda
        ),
        "termination_reason": termination_reason
    }



# ==================================================
# Model-observation consistency weight
#
# p: current transformed CAD correspondence point
# q: observed target correspondence point
#
# This specifically reacts to systematic depth disagreement
# that may remain locally smooth inside a corrupted patch.
# ==================================================
def compute_model_consistency_weights(
        p,
        q
):

    depth_residual = np.abs(
        p[
            :,
            2
        ]
        - q[
            :,
            2
        ]
    )


    raw_weight = np.exp(
        -0.5
        * (
            depth_residual
            / model_consistency_scale
        ) ** 2
    )


    model_weight = (
        model_consistency_floor
        + (
            1.0
            - model_consistency_floor
        )
        * raw_weight
    )


    model_weight = np.clip(
        model_weight,
        model_consistency_floor,
        1.0
    )


    return (
        model_weight.astype(
            np.float64
        ),
        depth_residual.astype(
            np.float64
        )
    )


# ==================================================
# Build weighted system:
#
# w_final =
#     w_residual
#     * w_corr
#     * w_local
#     * w_model
#
# w_local:
#     current four-neighborhood local depth reliability
#
# w_model:
#     current model-observation Z consistency
# ==================================================
def build_model_consistency_system(
        source_points,
        T,
        target_points,
        target_normals,
        tree,
        local_factor
):

    source_world = transform_points(
        source_points,
        T
    )


    distances, indices = tree.query(
        source_world,
        k=1
    )


    inlier = (
        distances
        < max_correspondence_distance
    )


    if np.count_nonzero(
        inlier
    ) < min_correspondences:

        return None


    p = source_world[
        inlier
    ]


    corr_indices = indices[
        inlier
    ]


    q = target_points[
        corr_indices
    ]


    n = target_normals[
        corr_indices
    ]


    d = distances[
        inlier
    ]


    local = local_factor[
        corr_indices
    ]


    n_ok = (
        np.linalg.norm(
            n,
            axis=1
        )
        > 1e-8
    )


    p = p[
        n_ok
    ]

    q = q[
        n_ok
    ]

    n = n[
        n_ok
    ]

    d = d[
        n_ok
    ]

    local = local[
        n_ok
    ]


    if len(
        p
    ) < min_correspondences:

        return None


    residual = np.sum(
        n
        * (
            p
            - q
        ),
        axis=1
    )


    w_residual = (
        1.0
        / (
            1.0
            + (
                residual
                / robust_scale
            ) ** 2
        )
    )


    w_corr = np.exp(
        -0.5
        * (
            d
            / correspondence_scale
        ) ** 2
    )


    local = np.clip(
        local,
        depth_weight_floor,
        1.0
    )


    (
        w_model,
        model_depth_residual
    ) = compute_model_consistency_weights(
        p,
        q
    )


    weights = (
        w_residual
        * w_corr
        * local
        * w_model
    )


    weights = np.clip(
        weights,
        1e-6,
        1.0
    )


    J_rot = np.cross(
        p,
        n
    )


    J = np.hstack(
        [
            J_rot,
            n
        ]
    )


    sqrt_w = np.sqrt(
        weights
    )


    Jw = (
        J
        * sqrt_w[
            :,
            None
        ]
    )


    rw = (
        residual
        * sqrt_w
    )


    return {
        "Jw": Jw,
        "rw": rw,
        "residual": residual,
        "weights": weights,
        "distances": d,
        "model_weights": w_model,
        "model_depth_residual": model_depth_residual,
        "inlier_ratio": float(
            len(
                p
            )
            / len(
                source_points
            )
        )
    }


# ==================================================
# Objective for Local + Model Consistency solver
# ==================================================
def evaluate_model_consistency_objective(
        source_points,
        T,
        target_points,
        target_normals,
        tree,
        local_factor
):

    system = build_model_consistency_system(
        source_points,
        T,
        target_points,
        target_normals,
        tree,
        local_factor
    )


    if system is None:

        return np.inf


    weights = system[
        "weights"
    ]


    residual = system[
        "residual"
    ]


    weight_sum = float(
        np.sum(
            weights
        )
    )


    if weight_sum <= 1e-12:

        return np.inf


    weighted_mse = float(
        np.sum(
            weights
            * residual ** 2
        )
        / weight_sum
    )


    coverage_penalty = (
        objective_coverage_penalty
        * robust_scale ** 2
        * (
            1.0
            - system[
                "inlier_ratio"
            ]
        )
    )


    return (
        weighted_mse
        + coverage_penalty
    )


# ==================================================
# Stable Local + Model Consistency Weighted ICP
#
# Same LM + line-search solver as the validated stable
# weighted ICP. Only the reliability factor differs.
# ==================================================
def run_model_consistency_weighted_icp(
        source_points,
        target_cloud,
        T_initial,
        T_gt,
        local_factor
):

    (
        target_points,
        target_normals,
        tree
    ) = prepare_target(
        target_cloud
    )


    local_factor = np.asarray(
        local_factor,
        dtype=np.float64
    )


    if len(
        local_factor
    ) != len(
        target_points
    ):

        raise ValueError(
            "local_factor length does not match target points."
        )


    T = T_initial.copy()


    current_cost = (
        evaluate_model_consistency_objective(
            source_points,
            T,
            target_points,
            target_normals,
            tree,
            local_factor
        )
    )


    lm_lambda = lm_lambda_initial

    accepted_steps = 0
    rejected_steps = 0

    termination_reason = (
        "max_iterations"
    )


    for iteration in range(
        max_iterations
    ):

        system = build_model_consistency_system(
            source_points,
            T,
            target_points,
            target_normals,
            tree,
            local_factor
        )


        if system is None:

            termination_reason = (
                "too_few_correspondences"
            )

            break


        Jw = system[
            "Jw"
        ]


        rw = system[
            "rw"
        ]


        H = (
            Jw.T
            @ Jw
        )


        g = (
            Jw.T
            @ rw
        )


        diag_H = np.maximum(
            np.diag(
                H
            ),
            1e-9
        )


        accepted = False


        for _ in range(
            lm_max_attempts
        ):

            H_lm = (
                H
                + lm_lambda
                * np.diag(
                    diag_H
                )
            )


            try:

                delta = -np.linalg.solve(
                    H_lm,
                    g
                )


            except np.linalg.LinAlgError:

                lm_lambda = min(
                    lm_lambda
                    * lm_increase,
                    lm_lambda_max
                )

                rejected_steps += 1

                continue


            if not np.all(
                np.isfinite(
                    delta
                )
            ):

                lm_lambda = min(
                    lm_lambda
                    * lm_increase,
                    lm_lambda_max
                )

                rejected_steps += 1

                continue


            for step_scale in line_search_steps:

                (
                    T_inc,
                    omega_step,
                    trans_step
                ) = make_increment(
                    delta,
                    step_scale
                )


                T_candidate = (
                    T_inc
                    @ T
                )


                candidate_cost = (
                    evaluate_model_consistency_objective(
                        source_points,
                        T_candidate,
                        target_points,
                        target_normals,
                        tree,
                        local_factor
                    )
                )


                if (
                    np.isfinite(
                        candidate_cost
                    )
                    and candidate_cost
                    < current_cost
                    - cost_tolerance
                ):

                    cost_drop = (
                        current_cost
                        - candidate_cost
                    )


                    T = T_candidate
                    current_cost = candidate_cost


                    lm_lambda = max(
                        lm_lambda
                        * lm_decrease,
                        lm_lambda_min
                    )


                    accepted_steps += 1
                    accepted = True


                    if (
                        np.linalg.norm(
                            omega_step
                        )
                        < rotation_step_tolerance
                        and np.linalg.norm(
                            trans_step
                        )
                        < translation_step_tolerance
                    ):

                        termination_reason = (
                            "small_step"
                        )


                    elif (
                        cost_drop
                        < cost_tolerance
                    ):

                        termination_reason = (
                            "small_cost_change"
                        )


                    break


            if accepted:

                break


            lm_lambda = min(
                lm_lambda
                * lm_increase,
                lm_lambda_max
            )


            rejected_steps += 1


        if not accepted:

            termination_reason = (
                "no_descent_step"
            )

            break


        if termination_reason in (
            "small_step",
            "small_cost_change"
        ):

            break


    source_final = transform_points(
        source_points,
        T
    )


    final_distance, _ = tree.query(
        source_final,
        k=1
    )


    final_inlier = (
        final_distance
        < max_correspondence_distance
    )


    fitness = float(
        np.mean(
            final_inlier
        )
    )


    if np.any(
        final_inlier
    ):

        rmse = float(
            np.sqrt(
                np.mean(
                    final_distance[
                        final_inlier
                    ] ** 2
                )
            )
        )

    else:

        rmse = 0.0


    final_system = build_model_consistency_system(
        source_points,
        T,
        target_points,
        target_normals,
        tree,
        local_factor
    )


    if final_system is None:

        final_model_weight_mean = np.nan
        final_model_depth_residual_mean_m = np.nan

    else:

        final_model_weight_mean = float(
            np.mean(
                final_system[
                    "model_weights"
                ]
            )
        )


        final_model_depth_residual_mean_m = float(
            np.mean(
                final_system[
                    "model_depth_residual"
                ]
            )
        )


    add_s = compute_add_s(
        evaluation_model_points,
        T,
        T_gt
    )


    return {
        "T": T,
        "fitness": fitness,
        "rmse": rmse,
        "add_s_m": add_s,
        "success": float(
            add_s
            < success_threshold_m
        ),
        "objective": float(
            current_cost
        ),
        "accepted_steps": accepted_steps,
        "rejected_steps": rejected_steps,
        "final_lambda": float(
            lm_lambda
        ),
        "termination_reason": termination_reason,
        "model_weight_mean": final_model_weight_mean,
        "model_depth_residual_mean_m": (
            final_model_depth_residual_mean_m
        )
    }


# ==================================================
# Diagnostic model-consistency statistics inside/outside
# the known synthetic corruption patch.
#
# The patch mask is used ONLY for diagnostics/evaluation.
# It is NOT passed into the optimizer.
# ==================================================
def compute_model_weight_patch_diagnostics(
        source_points,
        T,
        target_cloud,
        target_uv,
        patch_mask
):

    target_points = np.asarray(
        target_cloud.points,
        dtype=np.float64
    )


    if len(
        target_points
    ) == 0:

        return (
            np.nan,
            np.nan,
            np.nan
        )


    tree = cKDTree(
        target_points
    )


    source_world = transform_points(
        source_points,
        T
    )


    distances, indices = tree.query(
        source_world,
        k=1
    )


    inlier = (
        distances
        < max_correspondence_distance
    )


    if np.count_nonzero(
        inlier
    ) < min_correspondences:

        return (
            np.nan,
            np.nan,
            np.nan
        )


    p = source_world[
        inlier
    ]


    corr_indices = indices[
        inlier
    ]


    q = target_points[
        corr_indices
    ]


    model_weights, _ = compute_model_consistency_weights(
        p,
        q
    )


    corr_uv = target_uv[
        corr_indices
    ]


    u = corr_uv[
        :,
        0
    ]


    v = corr_uv[
        :,
        1
    ]


    in_patch = patch_mask[
        v,
        u
    ]


    overall_mean = float(
        np.mean(
            model_weights
        )
    )


    if np.any(
        in_patch
    ):

        patch_mean = float(
            np.mean(
                model_weights[
                    in_patch
                ]
            )
        )

    else:

        patch_mean = np.nan


    outside = (
        ~in_patch
    )


    if np.any(
        outside
    ):

        outside_mean = float(
            np.mean(
                model_weights[
                    outside
                ]
            )
        )

    else:

        outside_mean = np.nan


    return (
        overall_mean,
        patch_mean,
        outside_mean
    )



# ==================================================
# Complementary local-to-model gate
# ==================================================
def compute_complementary_gate(
        local_weight
):

    local_weight = np.asarray(
        local_weight,
        dtype=np.float64
    )


    x = (
        (
            local_weight
            - complementary_local_threshold
        )
        / complementary_temperature
    )


    x = np.clip(
        x,
        -30.0,
        30.0
    )


    gate = (
        1.0
        / (
            1.0
            + np.exp(
                -x
            )
        )
    )


    return gate.astype(
        np.float64
    )


# ==================================================
# Complementary effective model weight
#
# If local reliability is low:
#   gate -> 0
#   effective model weight -> 1
#   => avoid double suppression.
#
# If local reliability is high:
#   gate -> 1
#   effective model weight -> w_model
#   => detect smooth systematic model-depth disagreement.
# ==================================================
def compute_complementary_effective_model_weight(
        local_weight,
        model_weight
):

    gate = compute_complementary_gate(
        local_weight
    )


    effective_model_weight = (
        (
            1.0
            - gate
        )
        + gate
        * model_weight
    )


    effective_model_weight = np.clip(
        effective_model_weight,
        model_consistency_floor,
        1.0
    )


    return (
        effective_model_weight.astype(
            np.float64
        ),
        gate.astype(
            np.float64
        )
    )


# ==================================================
# Build complementary weighted system
# ==================================================
def build_complementary_consistency_system(
        source_points,
        T,
        target_points,
        target_normals,
        tree,
        local_factor
):

    source_world = transform_points(
        source_points,
        T
    )


    distances, indices = tree.query(
        source_world,
        k=1
    )


    inlier = (
        distances
        < max_correspondence_distance
    )


    if np.count_nonzero(
        inlier
    ) < min_correspondences:

        return None


    p = source_world[
        inlier
    ]


    corr_indices = indices[
        inlier
    ]


    q = target_points[
        corr_indices
    ]


    n = target_normals[
        corr_indices
    ]


    d = distances[
        inlier
    ]


    local = local_factor[
        corr_indices
    ]


    n_ok = (
        np.linalg.norm(
            n,
            axis=1
        )
        > 1e-8
    )


    p = p[
        n_ok
    ]

    q = q[
        n_ok
    ]

    n = n[
        n_ok
    ]

    d = d[
        n_ok
    ]

    local = local[
        n_ok
    ]


    if len(
        p
    ) < min_correspondences:

        return None


    residual = np.sum(
        n
        * (
            p
            - q
        ),
        axis=1
    )


    w_residual = (
        1.0
        / (
            1.0
            + (
                residual
                / robust_scale
            ) ** 2
        )
    )


    w_corr = np.exp(
        -0.5
        * (
            d
            / correspondence_scale
        ) ** 2
    )


    local = np.clip(
        local,
        depth_weight_floor,
        1.0
    )


    (
        w_model,
        model_depth_residual
    ) = compute_model_consistency_weights(
        p,
        q
    )


    (
        effective_model_weight,
        complementary_gate
    ) = compute_complementary_effective_model_weight(
        local,
        w_model
    )


    weights = (
        w_residual
        * w_corr
        * local
        * effective_model_weight
    )


    weights = np.clip(
        weights,
        1e-6,
        1.0
    )


    J_rot = np.cross(
        p,
        n
    )


    J = np.hstack(
        [
            J_rot,
            n
        ]
    )


    sqrt_w = np.sqrt(
        weights
    )


    Jw = (
        J
        * sqrt_w[
            :,
            None
        ]
    )


    rw = (
        residual
        * sqrt_w
    )


    return {
        "Jw": Jw,
        "rw": rw,
        "residual": residual,
        "weights": weights,
        "distances": d,
        "local_weights": local,
        "model_weights": w_model,
        "effective_model_weights": effective_model_weight,
        "complementary_gate": complementary_gate,
        "model_depth_residual": model_depth_residual,
        "corr_indices": corr_indices[
            n_ok
        ],
        "inlier_ratio": float(
            len(
                p
            )
            / len(
                source_points
            )
        )
    }


# ==================================================
# Complementary objective
# ==================================================
def evaluate_complementary_consistency_objective(
        source_points,
        T,
        target_points,
        target_normals,
        tree,
        local_factor
):

    system = build_complementary_consistency_system(
        source_points,
        T,
        target_points,
        target_normals,
        tree,
        local_factor
    )


    if system is None:

        return np.inf


    weights = system[
        "weights"
    ]


    residual = system[
        "residual"
    ]


    weight_sum = float(
        np.sum(
            weights
        )
    )


    if weight_sum <= 1e-12:

        return np.inf


    weighted_mse = float(
        np.sum(
            weights
            * residual ** 2
        )
        / weight_sum
    )


    coverage_penalty = (
        objective_coverage_penalty
        * robust_scale ** 2
        * (
            1.0
            - system[
                "inlier_ratio"
            ]
        )
    )


    return (
        weighted_mse
        + coverage_penalty
    )


# ==================================================
# Stable complementary ICP
#
# Same LM + backtracking solver as all validated stable
# weighted variants. Only the weight construction changes.
# ==================================================
def run_complementary_consistency_icp(
        source_points,
        target_cloud,
        T_initial,
        T_gt,
        local_factor
):

    (
        target_points,
        target_normals,
        tree
    ) = prepare_target(
        target_cloud
    )


    local_factor = np.asarray(
        local_factor,
        dtype=np.float64
    )


    if len(
        local_factor
    ) != len(
        target_points
    ):

        raise ValueError(
            "local_factor length does not match target points."
        )


    T = T_initial.copy()


    current_cost = (
        evaluate_complementary_consistency_objective(
            source_points,
            T,
            target_points,
            target_normals,
            tree,
            local_factor
        )
    )


    lm_lambda = lm_lambda_initial

    accepted_steps = 0
    rejected_steps = 0
    termination_reason = "max_iterations"


    for iteration in range(
        max_iterations
    ):

        system = build_complementary_consistency_system(
            source_points,
            T,
            target_points,
            target_normals,
            tree,
            local_factor
        )


        if system is None:

            termination_reason = (
                "too_few_correspondences"
            )

            break


        Jw = system[
            "Jw"
        ]


        rw = system[
            "rw"
        ]


        H = (
            Jw.T
            @ Jw
        )


        g = (
            Jw.T
            @ rw
        )


        diag_H = np.maximum(
            np.diag(
                H
            ),
            1e-9
        )


        accepted = False


        for _ in range(
            lm_max_attempts
        ):

            H_lm = (
                H
                + lm_lambda
                * np.diag(
                    diag_H
                )
            )


            try:

                delta = -np.linalg.solve(
                    H_lm,
                    g
                )


            except np.linalg.LinAlgError:

                lm_lambda = min(
                    lm_lambda
                    * lm_increase,
                    lm_lambda_max
                )

                rejected_steps += 1

                continue


            if not np.all(
                np.isfinite(
                    delta
                )
            ):

                lm_lambda = min(
                    lm_lambda
                    * lm_increase,
                    lm_lambda_max
                )

                rejected_steps += 1

                continue


            for step_scale in line_search_steps:

                (
                    T_inc,
                    omega_step,
                    trans_step
                ) = make_increment(
                    delta,
                    step_scale
                )


                T_candidate = (
                    T_inc
                    @ T
                )


                candidate_cost = (
                    evaluate_complementary_consistency_objective(
                        source_points,
                        T_candidate,
                        target_points,
                        target_normals,
                        tree,
                        local_factor
                    )
                )


                if (
                    np.isfinite(
                        candidate_cost
                    )
                    and candidate_cost
                    < current_cost
                    - cost_tolerance
                ):

                    cost_drop = (
                        current_cost
                        - candidate_cost
                    )


                    T = T_candidate
                    current_cost = candidate_cost


                    lm_lambda = max(
                        lm_lambda
                        * lm_decrease,
                        lm_lambda_min
                    )


                    accepted_steps += 1
                    accepted = True


                    if (
                        np.linalg.norm(
                            omega_step
                        )
                        < rotation_step_tolerance
                        and np.linalg.norm(
                            trans_step
                        )
                        < translation_step_tolerance
                    ):

                        termination_reason = (
                            "small_step"
                        )


                    elif (
                        cost_drop
                        < cost_tolerance
                    ):

                        termination_reason = (
                            "small_cost_change"
                        )


                    break


            if accepted:

                break


            lm_lambda = min(
                lm_lambda
                * lm_increase,
                lm_lambda_max
            )


            rejected_steps += 1


        if not accepted:

            termination_reason = (
                "no_descent_step"
            )

            break


        if termination_reason in (
            "small_step",
            "small_cost_change"
        ):

            break


    source_final = transform_points(
        source_points,
        T
    )


    final_distance, _ = tree.query(
        source_final,
        k=1
    )


    final_inlier = (
        final_distance
        < max_correspondence_distance
    )


    fitness = float(
        np.mean(
            final_inlier
        )
    )


    if np.any(
        final_inlier
    ):

        rmse = float(
            np.sqrt(
                np.mean(
                    final_distance[
                        final_inlier
                    ] ** 2
                )
            )
        )

    else:

        rmse = 0.0


    final_system = build_complementary_consistency_system(
        source_points,
        T,
        target_points,
        target_normals,
        tree,
        local_factor
    )


    if final_system is None:

        final_gate_mean = np.nan
        final_model_weight_mean = np.nan
        final_effective_model_weight_mean = np.nan

    else:

        final_gate_mean = float(
            np.mean(
                final_system[
                    "complementary_gate"
                ]
            )
        )


        final_model_weight_mean = float(
            np.mean(
                final_system[
                    "model_weights"
                ]
            )
        )


        final_effective_model_weight_mean = float(
            np.mean(
                final_system[
                    "effective_model_weights"
                ]
            )
        )


    add_s = compute_add_s(
        evaluation_model_points,
        T,
        T_gt
    )


    return {
        "T": T,
        "fitness": fitness,
        "rmse": rmse,
        "add_s_m": add_s,
        "success": float(
            add_s
            < success_threshold_m
        ),
        "objective": float(
            current_cost
        ),
        "accepted_steps": accepted_steps,
        "rejected_steps": rejected_steps,
        "final_lambda": float(
            lm_lambda
        ),
        "termination_reason": termination_reason,
        "complementary_gate_mean": final_gate_mean,
        "model_weight_mean": final_model_weight_mean,
        "effective_model_weight_mean": (
            final_effective_model_weight_mean
        )
    }


# ==================================================
# Complementary patch diagnostics
#
# Synthetic patch labels are used ONLY here for diagnostic
# reporting. The optimizer never sees the patch mask.
# ==================================================
def compute_complementary_patch_diagnostics(
        source_points,
        T,
        target_cloud,
        target_uv,
        patch_mask,
        local_factor
):

    (
        target_points,
        target_normals,
        tree
    ) = prepare_target(
        target_cloud
    )


    system = build_complementary_consistency_system(
        source_points,
        T,
        target_points,
        target_normals,
        tree,
        local_factor
    )


    if system is None:

        return {
            "gate_mean": np.nan,
            "gate_patch_mean": np.nan,
            "gate_outside_mean": np.nan,
            "model_patch_mean": np.nan,
            "model_outside_mean": np.nan,
            "effective_patch_mean": np.nan,
            "effective_outside_mean": np.nan
        }


    corr_indices = system[
        "corr_indices"
    ]


    corr_uv = target_uv[
        corr_indices
    ]


    u = corr_uv[
        :,
        0
    ]


    v = corr_uv[
        :,
        1
    ]


    in_patch = patch_mask[
        v,
        u
    ]


    outside = (
        ~in_patch
    )


    gate = system[
        "complementary_gate"
    ]


    model_weight = system[
        "model_weights"
    ]


    effective = system[
        "effective_model_weights"
    ]


    def safe_mean(
            values,
            selector
    ):

        if np.any(
            selector
        ):

            return float(
                np.mean(
                    values[
                        selector
                    ]
                )
            )

        return np.nan


    return {
        "gate_mean": float(
            np.mean(
                gate
            )
        ),
        "gate_patch_mean": safe_mean(
            gate,
            in_patch
        ),
        "gate_outside_mean": safe_mean(
            gate,
            outside
        ),
        "model_patch_mean": safe_mean(
            model_weight,
            in_patch
        ),
        "model_outside_mean": safe_mean(
            model_weight,
            outside
        ),
        "effective_patch_mean": safe_mean(
            effective,
            in_patch
        ),
        "effective_outside_mean": safe_mean(
            effective,
            outside
        )
    }


# ==================================================
# Statistical helpers
# ==================================================
BOOTSTRAP_REPEATS = 2000
BOOTSTRAP_SEED = 20260828


def frame_cluster_success_ci(
        records,
        bootstrap_repeats=BOOTSTRAP_REPEATS,
        seed=BOOTSTRAP_SEED
):

    if len(
        records
    ) == 0:

        return (
            np.nan,
            np.nan
        )


    by_frame = {}


    for record in records:

        by_frame.setdefault(
            record[
                "frame_id"
            ],
            []
        ).append(
            float(
                record[
                    "success"
                ]
            )
        )


    frame_rates = np.asarray(
        [
            np.mean(
                values
            )
            for values in by_frame.values()
        ],
        dtype=np.float64
    )


    if len(
        frame_rates
    ) <= 1:

        value = float(
            frame_rates[
                0
            ]
        )

        return (
            value,
            value
        )


    rng = np.random.default_rng(
        seed
        + len(
            records
        )
        + len(
            frame_rates
        )
    )


    sampled = rng.integers(
        0,
        len(
            frame_rates
        ),
        size=(
            bootstrap_repeats,
            len(
                frame_rates
            )
        )
    )


    values = np.mean(
        frame_rates[
            sampled
        ],
        axis=1
    )


    low, high = np.percentile(
        values,
        [
            2.5,
            97.5
        ]
    )


    return (
        float(
            low
        ),
        float(
            high
        )
    )


def summarize(
        records
):

    if len(
        records
    ) == 0:

        return {
            "N": 0,
            "objects": 0,
            "frames": 0,
            "success_count": 0,
            "failure_count": 0,
            "success_rate": np.nan,
            "success_ci_low": np.nan,
            "success_ci_high": np.nan,
            "mean_add_s_mm": np.nan,
            "median_add_s_mm": np.nan,
            "success_add_s_mm": np.nan,
            "fitness_mean": np.nan,
            "mean_source_points": np.nan
        }


    success = np.asarray(
        [
            r[
                "success"
            ]
            for r in records
        ],
        dtype=np.float64
    )


    add_s = np.asarray(
        [
            r[
                "add_s_mm"
            ]
            for r in records
        ],
        dtype=np.float64
    )


    fitness = np.asarray(
        [
            r[
                "fitness"
            ]
            for r in records
        ],
        dtype=np.float64
    )


    source_points = np.asarray(
        [
            r[
                "source_points"
            ]
            for r in records
        ],
        dtype=np.float64
    )


    success_mask = (
        success
        > 0.5
    )


    if np.any(
        success_mask
    ):

        success_add_s = float(
            np.mean(
                add_s[
                    success_mask
                ]
            )
        )

    else:

        success_add_s = np.nan


    ci_low, ci_high = frame_cluster_success_ci(
        records
    )


    success_count = int(
        np.count_nonzero(
            success_mask
        )
    )


    return {
        "N": len(
            records
        ),
        "objects": len(
            {
                r[
                    "object_id"
                ]
                for r in records
            }
        ),
        "frames": len(
            {
                r[
                    "frame_id"
                ]
                for r in records
            }
        ),
        "success_count": success_count,
        "failure_count": int(
            len(
                records
            )
            - success_count
        ),
        "success_rate": float(
            np.mean(
                success
            )
        ),
        "success_ci_low": ci_low,
        "success_ci_high": ci_high,
        "mean_add_s_mm": float(
            np.mean(
                add_s
            )
        ),
        "median_add_s_mm": float(
            np.median(
                add_s
            )
        ),
        "success_add_s_mm": success_add_s,
        "fitness_mean": float(
            np.mean(
                fitness
            )
        ),
        "mean_source_points": float(
            np.mean(
                source_points
            )
        )
    }


def paired_cluster_delta_ci(
        rows,
        comparison_key
):

    if len(
        rows
    ) == 0:

        return (
            np.nan,
            np.nan
        )


    by_frame = {}


    for row in rows:

        by_frame.setdefault(
            row[
                "frame_id"
            ],
            []
        ).append(
            float(
                row[
                    comparison_key
                ]
            )
        )


    frame_delta = np.asarray(
        [
            np.mean(
                values
            )
            for values in by_frame.values()
        ],
        dtype=np.float64
    )


    if len(
        frame_delta
    ) <= 1:

        value = float(
            frame_delta[
                0
            ]
        )

        return (
            value,
            value
        )


    rng = np.random.default_rng(
        BOOTSTRAP_SEED
        + 700000
        + len(
            rows
        )
        + len(
            frame_delta
        )
        + sum(
            ord(
                c
            )
            for c in comparison_key
        )
    )


    sampled = rng.integers(
        0,
        len(
            frame_delta
        ),
        size=(
            BOOTSTRAP_REPEATS,
            len(
                frame_delta
            )
        )
    )


    values = np.mean(
        frame_delta[
            sampled
        ],
        axis=1
    )


    low, high = np.percentile(
        values,
        [
            2.5,
            97.5
        ]
    )


    return (
        float(
            low
        ),
        float(
            high
        )
    )


def summarize_pair(
        rows,
        reference_prefix,
        candidate_prefix
):

    if len(
        rows
    ) == 0:

        return {
            "N": 0,
            "objects": 0,
            "frames": 0,
            "reference_only_success": 0,
            "candidate_only_success": 0,
            "both_success": 0,
            "both_fail": 0,
            "success_delta": np.nan,
            "success_delta_ci_low": np.nan,
            "success_delta_ci_high": np.nan,
            "both_success_median_add_delta_mm": np.nan,
            "candidate_lower_add_fraction": np.nan
        }


    ref_success = np.asarray(
        [
            row[
                reference_prefix
                + "_success"
            ]
            for row in rows
        ],
        dtype=np.float64
    )


    cand_success = np.asarray(
        [
            row[
                candidate_prefix
                + "_success"
            ]
            for row in rows
        ],
        dtype=np.float64
    )


    ref_ok = (
        ref_success
        > 0.5
    )


    cand_ok = (
        cand_success
        > 0.5
    )


    both_success_mask = (
        ref_ok
        & cand_ok
    )


    reference_only = int(
        np.count_nonzero(
            ref_ok
            & (
                ~cand_ok
            )
        )
    )


    candidate_only = int(
        np.count_nonzero(
            cand_ok
            & (
                ~ref_ok
            )
        )
    )


    both_success = int(
        np.count_nonzero(
            both_success_mask
        )
    )


    both_fail = int(
        np.count_nonzero(
            (
                ~ref_ok
            )
            & (
                ~cand_ok
            )
        )
    )


    success_delta_values = (
        cand_success
        - ref_success
    )


    success_delta = float(
        np.mean(
            success_delta_values
        )
    )


    comparison_key = (
        candidate_prefix
        + "_minus_"
        + reference_prefix
        + "_success"
    )


    ci_rows = []


    for row, delta in zip(
        rows,
        success_delta_values
    ):

        new_row = dict(
            row
        )


        new_row[
            comparison_key
        ] = float(
            delta
        )


        ci_rows.append(
            new_row
        )


    ci_low, ci_high = paired_cluster_delta_ci(
        ci_rows,
        comparison_key
    )


    if both_success > 0:

        ref_add = np.asarray(
            [
                row[
                    reference_prefix
                    + "_add_s_mm"
                ]
                for row in rows
            ],
            dtype=np.float64
        )[
            both_success_mask
        ]


        cand_add = np.asarray(
            [
                row[
                    candidate_prefix
                    + "_add_s_mm"
                ]
                for row in rows
            ],
            dtype=np.float64
        )[
            both_success_mask
        ]


        add_delta = (
            cand_add
            - ref_add
        )


        median_add_delta = float(
            np.median(
                add_delta
            )
        )


        candidate_lower_fraction = float(
            np.mean(
                cand_add
                < ref_add
            )
        )

    else:

        median_add_delta = np.nan
        candidate_lower_fraction = np.nan


    return {
        "N": len(
            rows
        ),
        "objects": len(
            {
                row[
                    "object_id"
                ]
                for row in rows
            }
        ),
        "frames": len(
            {
                row[
                    "frame_id"
                ]
                for row in rows
            }
        ),
        "reference_only_success": reference_only,
        "candidate_only_success": candidate_only,
        "both_success": both_success,
        "both_fail": both_fail,
        "success_delta": success_delta,
        "success_delta_ci_low": ci_low,
        "success_delta_ci_high": ci_high,
        "both_success_median_add_delta_mm": median_add_delta,
        "candidate_lower_add_fraction": candidate_lower_fraction
    }


def safe_nanmean(
        values
):

    arr = np.asarray(
        values,
        dtype=np.float64
    )


    finite = np.isfinite(
        arr
    )


    if not np.any(
        finite
    ):

        return np.nan


    return float(
        np.mean(
            arr[
                finite
            ]
        )
    )


def write_csv(
        path,
        header,
        rows
):

    with open(
        path,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow(
            header
        )

        writer.writerows(
            rows
        )


# ==================================================
# Methods / protocols
# ==================================================
#
# For each protocol we compare:
#
#   Geometry
#       T_init -> Geometry ICP
#
#   Direct Local
#       T_init -> Local Reliability ICP
#
#   Geometry -> Local
#       T_init -> Geometry ICP -> Local Reliability refinement
#
# IMPORTANT:
# Stage 1 of Geometry->Local reuses the EXACT Geometry result.
#
method_names = [
    "oracle_geometry_icp",
    "oracle_direct_local_icp",
    "oracle_geometry_then_local_icp",
    "oracle_reliability_gated_icp",

    "protocolA_fixedclean_geometry_icp",
    "protocolA_fixedclean_direct_local_icp",
    "protocolA_fixedclean_geometry_then_local_icp",
    "protocolA_fixedclean_reliability_gated_icp",

    "protocolB_e2e_geometry_icp",
    "protocolB_e2e_direct_local_icp",
    "protocolB_e2e_geometry_then_local_icp",
    "protocolB_e2e_reliability_gated_icp"
]


protocol_methods = {
    "oracle_reference": {
        "geometry": "oracle_geometry_icp",
        "direct_local": "oracle_direct_local_icp",
        "staged": "oracle_geometry_then_local_icp",
        "gated": "oracle_reliability_gated_icp"
    },
    "protocolA_fixed_clean_visibility": {
        "geometry": "protocolA_fixedclean_geometry_icp",
        "direct_local": "protocolA_fixedclean_direct_local_icp",
        "staged": "protocolA_fixedclean_geometry_then_local_icp",
        "gated": "protocolA_fixedclean_reliability_gated_icp"
    },
    "protocolB_end_to_end_corrupted_visibility": {
        "geometry": "protocolB_e2e_geometry_icp",
        "direct_local": "protocolB_e2e_direct_local_icp",
        "staged": "protocolB_e2e_geometry_then_local_icp",
        "gated": "protocolB_e2e_reliability_gated_icp"
    }
}


# ==================================================
# Record containers
# ==================================================
clean_records_all = {
    method: []
    for method in method_names
}


clean_records_qualified = {
    method: []
    for method in method_names
}


aggregate_all = {
    method: []
    for method in method_names
}


aggregate_qualified = {
    method: []
    for method in method_names
}


records_by_condition_all = {
    (
        corruption_type,
        fraction
    ): {
        method: []
        for method in method_names
    }
    for corruption_type in corruption_types
    for fraction in corruption_fractions
}


records_by_condition_qualified = {
    (
        corruption_type,
        fraction
    ): {
        method: []
        for method in method_names
    }
    for corruption_type in corruption_types
    for fraction in corruption_fractions
}


records_by_corruption_all = {
    corruption_type: {
        method: []
        for method in method_names
    }
    for corruption_type in corruption_types
}


records_by_corruption_qualified = {
    corruption_type: {
        method: []
        for method in method_names
    }
    for corruption_type in corruption_types
}


object_records_all = {
    object_id: {
        method: []
        for method in method_names
    }
    for object_id in OBJECT_IDS
}


object_records_qualified = {
    object_id: {
        method: []
        for method in method_names
    }
    for object_id in OBJECT_IDS
}


paired_rows = {
    protocol_name: []
    for protocol_name in protocol_methods
}


trial_rows = []
visibility_rows = []
frame_rows = []

gate_rows = []
clean_gate_rows = []

processed_frames = 0
skipped_frames = 0

object_processed = {
    object_id: 0
    for object_id in OBJECT_IDS
}

object_skipped = {
    object_id: 0
    for object_id in OBJECT_IDS
}

object_oracle_qualified = {
    object_id: 0
    for object_id in OBJECT_IDS
}


trial_header = [
    "object_id",
    "symmetry_class",
    "diameter_mm",
    "frame_id",
    "scene_id",
    "img_id",
    "inst_id",
    "visib_fract",
    "oracle_reference_qualified",
    "corruption_type",
    "requested_corruption_fraction",
    "actual_corruption_fraction",
    "patch_pixels",
    "patch_center_u",
    "patch_center_v",
    "trial",
    "method",
    "source_points",
    "ADD_S_mm",
    "RMSE_mm",
    "fitness",
    "success",
    "local_reliability_mean",
    "local_patch_reliability_mean",
    "local_outside_reliability_mean",
    "target_retention_ratio",
    "signed_bias_mm",

    "gate_triggered",
    "gate_reason",
    "gate_mean_reliability",
    "gate_median_reliability",
    "gate_q10_reliability",
    "gate_q25_reliability",
    "gate_strong_low_fraction",
    "gate_opened_low_fraction",
    "gate_largest_component_fraction",
    "gate_largest_component_share_of_low",
    "gate_mean_local_diff_mm",
    "gate_q90_local_diff_mm",

    "stage1_ADD_S_mm",
    "stage1_success",
    "stage2_ADD_S_delta_mm",
    "stage2_success_change",
    "termination_reason"
]


estimated_trials_per_frame = (
    len(
        corruption_types
    )
    * len(
        corruption_fractions
    )
    * REPEATS_PER_LEVEL
)


# 9 unique ICP calls per corruption trial:
# 3 protocols x (Geometry + Direct Local + Local refinement).
estimated_icp_runs = (
    len(
        OBJECT_IDS
    )
    * FRAMES_PER_OBJECT
    * (
        6
        + estimated_trials_per_frame
        * 9
    )
)


print(
    "=================================================="
)

print(
    "Reliability-Gated Local Refinement Corruption Multi-object"
)

print(
    "=================================================="
)

print(
    "Objects:",
    OBJECT_IDS
)

print(
    "Frames/object:",
    FRAMES_PER_OBJECT
)

print(
    "Repeats/condition:",
    REPEATS_PER_LEVEL
)

print(
    "Corruption types:",
    corruption_types
)

print(
    "Corruption fractions:",
    corruption_fractions
)

print(
    "Approx. ICP runs:",
    estimated_icp_runs
)

print(
    "Stage order:",
    "Geometry -> Local Reliability"
)

print(
    "Stage-1 result reused:",
    True
)

print(
    "Reliability gate:",
    "interior-area + connected-component + q10"
)

print(
    "Gate low-reliability threshold:",
    GATE_LOW_RELIABILITY_THRESHOLD
)

print(
    "Gate minimum opened-low fraction:",
    GATE_MIN_OPENED_LOW_FRACTION
)

print(
    "Gate minimum largest-component fraction:",
    GATE_MIN_LARGEST_COMPONENT_FRACTION
)

print(
    "Gate maximum q10:",
    GATE_MAX_Q10
)

print(
    "Protocol A source visibility uses corrupted depth:",
    False
)

print(
    "Protocol B source visibility uses corrupted depth:",
    True
)

print(
    "GT pose used in Protocol A/B source selection:",
    False
)

print(
    "BOP mask_visib used as segmentation proxy:",
    True
)

print(
    "Output directory:",
    output_dir
)


# ==================================================
# Object loop
# ==================================================
for object_index, object_id in enumerate(
    OBJECT_IDS
):

    print(
        "\n\n=================================================="
    )

    print(
        f"OBJECT {object_id} "
        f"({object_index + 1}/{len(OBJECT_IDS)})"
    )

    print(
        "=================================================="
    )


    meta = configure_object(
        object_id
    )


    try:

        all_candidates = discover_candidates()

    except Exception as e:

        print(
            "OBJECT SKIP:",
            repr(
                e
            )
        )

        continue


    selected_candidates = select_evenly(
        all_candidates,
        FRAMES_PER_OBJECT
    )


    print(
        "Eligible candidates:",
        len(
            all_candidates
        )
    )

    print(
        "Selected:",
        len(
            selected_candidates
        )
    )


    # ==================================================
    # Frame loop
    # ==================================================
    for frame_index, candidate in enumerate(
        selected_candidates
    ):

        try:

            frame = load_frame_data(
                candidate
            )


            clean_target, clean_uv = depth_to_cloud(
                frame[
                    "depth"
                ],
                frame[
                    "mask_bool"
                ],
                frame[
                    "K"
                ]
            )


            if len(
                clean_target.points
            ) < MIN_VISIBLE_PIXELS:

                raise RuntimeError(
                    "Clean target point cloud too small."
                )


            clean_reliability = compute_depth_reliability(
                frame[
                    "depth"
                ],
                frame[
                    "mask_bool"
                ],
                clean_uv
            )


            T_init = generate_initial_pose(
                frame[
                    "T_gt"
                ]
            )


            (
                oracle_source,
                oracle_indices
            ) = generate_oracle_visible_model_with_indices(
                full_model_points,
                frame[
                    "T_gt"
                ],
                frame[
                    "depth"
                ],
                frame[
                    "mask_bool"
                ],
                frame[
                    "K"
                ]
            )


            (
                fixed_clean_source,
                fixed_clean_indices,
                fixed_clean_diag
            ) = generate_observation_supported_visible_model(
                full_model_points,
                T_init,
                frame[
                    "depth"
                ],
                frame[
                    "mask_bool"
                ],
                frame[
                    "K"
                ],
                clean_target
            )


        except Exception as e:

            skipped_frames += 1

            object_skipped[
                object_id
            ] += 1


            print(
                "SKIP:",
                repr(
                    e
                )
            )

            continue


        processed_frames += 1

        object_processed[
            object_id
        ] += 1


        frame_id = (
            f"O{object_id:03d}_"
            f"S{frame['scene_id']:06d}_"
            f"I{frame['img_id']:06d}_"
            f"N{frame['inst_id']:02d}"
        )


        print(
            "\n----------------------------------------------"
        )

        print(
            f"{frame_id} | "
            f"Frame {frame_index + 1}/"
            f"{len(selected_candidates)}"
        )


        # ==================================================
        # CLEAN gate diagnostics
        # ==================================================
        clean_gate_features = compute_reliability_gate_features(
            frame[
                "depth"
            ],
            frame[
                "mask_bool"
            ]
        )


        clean_gate_rows.append({
            "object_id": object_id,
            "frame_id": frame_id,
            "trigger": int(
                clean_gate_features[
                    "trigger"
                ]
            ),
            "reason": clean_gate_features[
                "reason"
            ],
            "mean_reliability": clean_gate_features[
                "mean_reliability"
            ],
            "median_reliability": clean_gate_features[
                "median_reliability"
            ],
            "q10_reliability": clean_gate_features[
                "q10_reliability"
            ],
            "q25_reliability": clean_gate_features[
                "q25_reliability"
            ],
            "strong_low_fraction": clean_gate_features[
                "strong_low_fraction"
            ],
            "opened_low_fraction": clean_gate_features[
                "opened_low_fraction"
            ],
            "largest_component_fraction": clean_gate_features[
                "largest_component_fraction"
            ],
            "largest_component_share_of_low": clean_gate_features[
                "largest_component_share_of_low"
            ],
            "mean_local_diff_mm": clean_gate_features[
                "mean_local_diff_mm"
            ],
            "q90_local_diff_mm": clean_gate_features[
                "q90_local_diff_mm"
            ]
        })


        # ==================================================
        # Clean baselines
        # ==================================================
        clean_geometry_factor = np.ones(
            len(
                clean_target.points
            ),
            dtype=np.float64
        )


        # Oracle
        clean_oracle_geometry = run_stable_weighted_icp(
            oracle_source,
            clean_target,
            T_init,
            frame[
                "T_gt"
            ],
            clean_geometry_factor
        )


        clean_oracle_direct_local = run_stable_weighted_icp(
            oracle_source,
            clean_target,
            T_init,
            frame[
                "T_gt"
            ],
            clean_reliability
        )


        clean_oracle_staged = run_geometry_then_local_refinement(
            oracle_source,
            clean_target,
            clean_oracle_geometry,
            frame[
                "T_gt"
            ],
            clean_reliability
        )


        clean_oracle_gated = select_reliability_gated_result(
            clean_oracle_geometry,
            clean_oracle_staged,
            clean_gate_features
        )


        # Protocol A
        clean_A_geometry = run_local_with_source(
            fixed_clean_source,
            clean_target,
            T_init,
            frame[
                "T_gt"
            ],
            clean_geometry_factor
        )


        clean_A_direct_local = run_local_with_source(
            fixed_clean_source,
            clean_target,
            T_init,
            frame[
                "T_gt"
            ],
            clean_reliability
        )


        clean_A_staged = run_geometry_then_local_refinement(
            fixed_clean_source,
            clean_target,
            clean_A_geometry,
            frame[
                "T_gt"
            ],
            clean_reliability
        )


        clean_A_gated = select_reliability_gated_result(
            clean_A_geometry,
            clean_A_staged,
            clean_gate_features
        )


        # On clean depth, Protocol B has exactly the same source.
        # Copy results instead of rerunning duplicate ICP.
        clean_B_geometry = dict(
            clean_A_geometry
        )

        clean_B_direct_local = dict(
            clean_A_direct_local
        )

        clean_B_staged = dict(
            clean_A_staged
        )


        clean_B_gated = dict(
            clean_A_gated
        )


        clean_outputs = {
            "oracle_geometry_icp": (
                clean_oracle_geometry,
                len(
                    oracle_source
                )
            ),
            "oracle_direct_local_icp": (
                clean_oracle_direct_local,
                len(
                    oracle_source
                )
            ),
            "oracle_geometry_then_local_icp": (
                clean_oracle_staged,
                len(
                    oracle_source
                )
            ),
            "oracle_reliability_gated_icp": (
                clean_oracle_gated,
                len(
                    oracle_source
                )
            ),
            "protocolA_fixedclean_geometry_icp": (
                clean_A_geometry,
                len(
                    fixed_clean_source
                )
            ),
            "protocolA_fixedclean_direct_local_icp": (
                clean_A_direct_local,
                len(
                    fixed_clean_source
                )
            ),
            "protocolA_fixedclean_geometry_then_local_icp": (
                clean_A_staged,
                len(
                    fixed_clean_source
                )
            ),
            "protocolA_fixedclean_reliability_gated_icp": (
                clean_A_gated,
                len(
                    fixed_clean_source
                )
            ),
            "protocolB_e2e_geometry_icp": (
                clean_B_geometry,
                len(
                    fixed_clean_source
                )
            ),
            "protocolB_e2e_direct_local_icp": (
                clean_B_direct_local,
                len(
                    fixed_clean_source
                )
            ),
            "protocolB_e2e_geometry_then_local_icp": (
                clean_B_staged,
                len(
                    fixed_clean_source
                )
            ),
            "protocolB_e2e_reliability_gated_icp": (
                clean_B_gated,
                len(
                    fixed_clean_source
                )
            )
        }


        # Preserve the previous protocol exactly:
        # clean oracle direct-local success defines qualification.
        oracle_reference_qualified = bool(
            clean_oracle_direct_local[
                "success"
            ]
            > 0.5
        )


        if oracle_reference_qualified:

            object_oracle_qualified[
                object_id
            ] += 1


        for method_name, (
            output,
            source_count
        ) in clean_outputs.items():

            record = {
                "object_id": object_id,
                "frame_id": frame_id,
                "add_s_mm": (
                    output[
                        "add_s_m"
                    ]
                    * 1000.0
                ),
                "success": output[
                    "success"
                ],
                "fitness": output[
                    "fitness"
                ],
                "source_points": int(
                    source_count
                )
            }


            clean_records_all[
                method_name
            ].append(
                record
            )


            if oracle_reference_qualified:

                clean_records_qualified[
                    method_name
                ].append(
                    record
                )


        fixed_metrics = visibility_set_metrics(
            fixed_clean_indices,
            oracle_indices
        )


        print(
            f"Clean source | "
            f"Oracle={len(oracle_source)} | "
            f"ObsFixed={len(fixed_clean_source)} | "
            f"P/R/J="
            f"{fixed_metrics['precision']:.3f}/"
            f"{fixed_metrics['recall']:.3f}/"
            f"{fixed_metrics['jaccard']:.3f} | "
            f"oracle-qualified={oracle_reference_qualified}"
        )


        print(
            f"Clean staged ADD-S mm | "
            f"Oracle G->L="
            f"{clean_oracle_staged['add_s_m'] * 1000.0:.3f} | "
            f"A G->L="
            f"{clean_A_staged['add_s_m'] * 1000.0:.3f}"
        )


        frame_method_records = {
            method: []
            for method in method_names
        }


        # ==================================================
        # Corruption trials
        # ==================================================
        for fraction_index, fraction in enumerate(
            corruption_fractions
        ):

            for trial in range(
                REPEATS_PER_LEVEL
            ):

                patch_seed = (
                    200000
                    + object_id * 1000000
                    + frame_index * 10000
                    + fraction_index * 100
                    + trial
                )


                (
                    patch_mask,
                    patch_center_u,
                    patch_center_v,
                    actual_fraction
                ) = select_local_patch(
                    frame[
                        "depth"
                    ],
                    frame[
                        "mask_bool"
                    ],
                    fraction,
                    patch_seed
                )


                patch_pixels = int(
                    np.count_nonzero(
                        patch_mask
                    )
                )


                for corruption_index, corruption_type in enumerate(
                    corruption_types
                ):

                    value_seed = (
                        500000
                        + object_id * 1000000
                        + frame_index * 10000
                        + fraction_index * 1000
                        + trial * 10
                        + corruption_index
                    )


                    corrupted = make_local_corrupted_target(
                        frame[
                            "depth"
                        ],
                        frame[
                            "mask_bool"
                        ],
                        frame[
                            "K"
                        ],
                        patch_mask,
                        corruption_type,
                        value_seed
                    )


                    corrupted_depth = corrupted[
                        "corrupted_depth"
                    ]


                    target_cloud = corrupted[
                        "target_cloud"
                    ]


                    local_reliability = corrupted[
                        "depth_reliability"
                    ]


                    geometry_factor = np.ones(
                        len(
                            target_cloud.points
                        ),
                        dtype=np.float64
                    )


                    # ==================================================
                    # GT-free reliability gate
                    # ==================================================
                    gate_features = compute_reliability_gate_features(
                        corrupted_depth,
                        frame[
                            "mask_bool"
                        ]
                    )


                    # Protocol B source from corrupted observation.
                    (
                        e2e_source,
                        e2e_indices,
                        e2e_diag
                    ) = generate_observation_supported_visible_model(
                        full_model_points,
                        T_init,
                        corrupted_depth,
                        frame[
                            "mask_bool"
                        ],
                        frame[
                            "K"
                        ],
                        target_cloud
                    )


                    # ==================================================
                    # Oracle
                    # ==================================================
                    oracle_geometry = run_stable_weighted_icp(
                        oracle_source,
                        target_cloud,
                        T_init,
                        frame[
                            "T_gt"
                        ],
                        geometry_factor
                    )


                    oracle_direct_local = run_stable_weighted_icp(
                        oracle_source,
                        target_cloud,
                        T_init,
                        frame[
                            "T_gt"
                        ],
                        local_reliability
                    )


                    oracle_staged = run_geometry_then_local_refinement(
                        oracle_source,
                        target_cloud,
                        oracle_geometry,
                        frame[
                            "T_gt"
                        ],
                        local_reliability
                    )


                    oracle_gated = select_reliability_gated_result(
                        oracle_geometry,
                        oracle_staged,
                        gate_features
                    )


                    # ==================================================
                    # Protocol A - fixed clean estimated visibility
                    # ==================================================
                    A_geometry = run_local_with_source(
                        fixed_clean_source,
                        target_cloud,
                        T_init,
                        frame[
                            "T_gt"
                        ],
                        geometry_factor
                    )


                    A_direct_local = run_local_with_source(
                        fixed_clean_source,
                        target_cloud,
                        T_init,
                        frame[
                            "T_gt"
                        ],
                        local_reliability
                    )


                    A_staged = run_geometry_then_local_refinement(
                        fixed_clean_source,
                        target_cloud,
                        A_geometry,
                        frame[
                            "T_gt"
                        ],
                        local_reliability
                    )


                    A_gated = select_reliability_gated_result(
                        A_geometry,
                        A_staged,
                        gate_features
                    )


                    # ==================================================
                    # Protocol B - end-to-end corrupted visibility
                    # ==================================================
                    B_geometry = run_local_with_source(
                        e2e_source,
                        target_cloud,
                        T_init,
                        frame[
                            "T_gt"
                        ],
                        geometry_factor
                    )


                    B_direct_local = run_local_with_source(
                        e2e_source,
                        target_cloud,
                        T_init,
                        frame[
                            "T_gt"
                        ],
                        local_reliability
                    )


                    B_staged = run_geometry_then_local_refinement(
                        e2e_source,
                        target_cloud,
                        B_geometry,
                        frame[
                            "T_gt"
                        ],
                        local_reliability
                    )


                    B_gated = select_reliability_gated_result(
                        B_geometry,
                        B_staged,
                        gate_features
                    )


                    outputs = {
                        "oracle_geometry_icp": (
                            oracle_geometry,
                            len(
                                oracle_source
                            )
                        ),
                        "oracle_direct_local_icp": (
                            oracle_direct_local,
                            len(
                                oracle_source
                            )
                        ),
                        "oracle_geometry_then_local_icp": (
                            oracle_staged,
                            len(
                                oracle_source
                            )
                        ),
                        "oracle_reliability_gated_icp": (
                            oracle_gated,
                            len(
                                oracle_source
                            )
                        ),

                        "protocolA_fixedclean_geometry_icp": (
                            A_geometry,
                            len(
                                fixed_clean_source
                            )
                        ),
                        "protocolA_fixedclean_direct_local_icp": (
                            A_direct_local,
                            len(
                                fixed_clean_source
                            )
                        ),
                        "protocolA_fixedclean_geometry_then_local_icp": (
                            A_staged,
                            len(
                                fixed_clean_source
                            )
                        ),
                        "protocolA_fixedclean_reliability_gated_icp": (
                            A_gated,
                            len(
                                fixed_clean_source
                            )
                        ),

                        "protocolB_e2e_geometry_icp": (
                            B_geometry,
                            len(
                                e2e_source
                            )
                        ),
                        "protocolB_e2e_direct_local_icp": (
                            B_direct_local,
                            len(
                                e2e_source
                            )
                        ),
                        "protocolB_e2e_geometry_then_local_icp": (
                            B_staged,
                            len(
                                e2e_source
                            )
                        ),
                        "protocolB_e2e_reliability_gated_icp": (
                            B_gated,
                            len(
                                e2e_source
                            )
                        )
                    }


                    trial_record_by_method = {}


                    for method_name, (
                        output,
                        source_count
                    ) in outputs.items():

                        record = {
                            "object_id": object_id,
                            "frame_id": frame_id,
                            "add_s_mm": (
                                output[
                                    "add_s_m"
                                ]
                                * 1000.0
                            ),
                            "success": output[
                                "success"
                            ],
                            "fitness": output[
                                "fitness"
                            ],
                            "source_points": int(
                                source_count
                            )
                        }


                        trial_record_by_method[
                            method_name
                        ] = record


                        frame_method_records[
                            method_name
                        ].append(
                            record
                        )


                        aggregate_all[
                            method_name
                        ].append(
                            record
                        )


                        object_records_all[
                            object_id
                        ][
                            method_name
                        ].append(
                            record
                        )


                        records_by_condition_all[
                            (
                                corruption_type,
                                fraction
                            )
                        ][
                            method_name
                        ].append(
                            record
                        )


                        records_by_corruption_all[
                            corruption_type
                        ][
                            method_name
                        ].append(
                            record
                        )


                        if oracle_reference_qualified:

                            aggregate_qualified[
                                method_name
                            ].append(
                                record
                            )


                            object_records_qualified[
                                object_id
                            ][
                                method_name
                            ].append(
                                record
                            )


                            records_by_condition_qualified[
                                (
                                    corruption_type,
                                    fraction
                                )
                            ][
                                method_name
                            ].append(
                                record
                            )


                            records_by_corruption_qualified[
                                corruption_type
                            ][
                                method_name
                            ].append(
                                record
                            )


                        if (
                            "geometry_then_local"
                            in method_name
                        ):

                            stage1_add_s_mm = (
                                output[
                                    "stage1_add_s_m"
                                ]
                                * 1000.0
                            )

                            stage1_success = output[
                                "stage1_success"
                            ]

                            stage2_add_s_delta_mm = output[
                                "stage2_add_s_delta_mm"
                            ]

                            stage2_success_change = output[
                                "stage2_success_change"
                            ]

                        else:

                            stage1_add_s_mm = ""
                            stage1_success = ""
                            stage2_add_s_delta_mm = ""
                            stage2_success_change = ""


                        trial_rows.append([
                            object_id,
                            meta[
                                "symmetry_class"
                            ],
                            meta[
                                "diameter_mm"
                            ],
                            frame_id,
                            frame[
                                "scene_id"
                            ],
                            frame[
                                "img_id"
                            ],
                            frame[
                                "inst_id"
                            ],
                            frame[
                                "visib_fract"
                            ],
                            int(
                                oracle_reference_qualified
                            ),
                            corruption_type,
                            fraction,
                            actual_fraction,
                            patch_pixels,
                            patch_center_u,
                            patch_center_v,
                            trial,
                            method_name,
                            int(
                                source_count
                            ),
                            output[
                                "add_s_m"
                            ] * 1000.0,
                            (
                                output[
                                    "rmse"
                                ]
                                * 1000.0
                                if np.isfinite(
                                    output[
                                        "rmse"
                                    ]
                                )
                                else np.nan
                            ),
                            output[
                                "fitness"
                            ],
                            output[
                                "success"
                            ],
                            corrupted[
                                "global_reliability_mean"
                            ],
                            corrupted[
                                "patch_reliability_mean"
                            ],
                            corrupted[
                                "outside_reliability_mean"
                            ],
                            corrupted[
                                "target_retention_ratio"
                            ],
                            corrupted[
                                "signed_bias_mm"
                            ],

                            int(
                                gate_features[
                                    "trigger"
                                ]
                            ),
                            gate_features[
                                "reason"
                            ],
                            gate_features[
                                "mean_reliability"
                            ],
                            gate_features[
                                "median_reliability"
                            ],
                            gate_features[
                                "q10_reliability"
                            ],
                            gate_features[
                                "q25_reliability"
                            ],
                            gate_features[
                                "strong_low_fraction"
                            ],
                            gate_features[
                                "opened_low_fraction"
                            ],
                            gate_features[
                                "largest_component_fraction"
                            ],
                            gate_features[
                                "largest_component_share_of_low"
                            ],
                            gate_features[
                                "mean_local_diff_mm"
                            ],
                            gate_features[
                                "q90_local_diff_mm"
                            ],

                            stage1_add_s_mm,
                            stage1_success,
                            stage2_add_s_delta_mm,
                            stage2_success_change,
                            output[
                                "termination_reason"
                            ]
                        ])


                    # ==================================================
                    # Paired protocol records
                    # ==================================================
                    for protocol_name, method_map in protocol_methods.items():

                        g = trial_record_by_method[
                            method_map[
                                "geometry"
                            ]
                        ]


                        d = trial_record_by_method[
                            method_map[
                                "direct_local"
                            ]
                        ]


                        s = trial_record_by_method[
                            method_map[
                                "staged"
                            ]
                        ]


                        q = trial_record_by_method[
                            method_map[
                                "gated"
                            ]
                        ]


                        paired_rows[
                            protocol_name
                        ].append({
                            "object_id": object_id,
                            "frame_id": frame_id,
                            "qualified": bool(
                                oracle_reference_qualified
                            ),
                            "corruption_type": corruption_type,
                            "fraction": fraction,
                            "trial": trial,

                            "geometry_success": g[
                                "success"
                            ],
                            "geometry_add_s_mm": g[
                                "add_s_mm"
                            ],

                            "direct_local_success": d[
                                "success"
                            ],
                            "direct_local_add_s_mm": d[
                                "add_s_mm"
                            ],

                            "staged_success": s[
                                "success"
                            ],
                            "staged_add_s_mm": s[
                                "add_s_mm"
                            ],

                            "gated_success": q[
                                "success"
                            ],
                            "gated_add_s_mm": q[
                                "add_s_mm"
                            ]
                        })


                    # ==================================================
                    # Gate diagnostics
                    # ==================================================
                    gate_rows.append({
                        "object_id": object_id,
                        "frame_id": frame_id,
                        "qualified": int(
                            oracle_reference_qualified
                        ),
                        "corruption_type": corruption_type,
                        "fraction": fraction,
                        "trial": trial,
                        "trigger": int(
                            gate_features[
                                "trigger"
                            ]
                        ),
                        "reason": gate_features[
                            "reason"
                        ],
                        "valid_count": gate_features[
                            "valid_count"
                        ],
                        "interior_count": gate_features[
                            "interior_count"
                        ],
                        "mean_reliability": gate_features[
                            "mean_reliability"
                        ],
                        "median_reliability": gate_features[
                            "median_reliability"
                        ],
                        "q10_reliability": gate_features[
                            "q10_reliability"
                        ],
                        "q25_reliability": gate_features[
                            "q25_reliability"
                        ],
                        "strong_low_fraction": gate_features[
                            "strong_low_fraction"
                        ],
                        "opened_low_fraction": gate_features[
                            "opened_low_fraction"
                        ],
                        "largest_component_fraction": gate_features[
                            "largest_component_fraction"
                        ],
                        "largest_component_share_of_low": gate_features[
                            "largest_component_share_of_low"
                        ],
                        "mean_local_diff_mm": gate_features[
                            "mean_local_diff_mm"
                        ],
                        "q90_local_diff_mm": gate_features[
                            "q90_local_diff_mm"
                        ]
                    })


                    # ==================================================
                    # Visibility diagnostics
                    # ==================================================
                    e2e_metrics = visibility_set_metrics(
                        e2e_indices,
                        oracle_indices
                    )


                    visibility_rows.append([
                        object_id,
                        frame_id,
                        int(
                            oracle_reference_qualified
                        ),
                        corruption_type,
                        fraction,
                        trial,
                        len(
                            oracle_indices
                        ),
                        len(
                            fixed_clean_indices
                        ),
                        len(
                            e2e_indices
                        ),
                        fixed_metrics[
                            "precision"
                        ],
                        fixed_metrics[
                            "recall"
                        ],
                        fixed_metrics[
                            "jaccard"
                        ],
                        e2e_metrics[
                            "precision"
                        ],
                        e2e_metrics[
                            "recall"
                        ],
                        e2e_metrics[
                            "jaccard"
                        ],
                        (
                            len(
                                e2e_indices
                            )
                            / max(
                                len(
                                    fixed_clean_indices
                                ),
                                1
                            )
                        )
                    ])


                    print(
                        f"{corruption_type} | "
                        f"f={fraction:.2f} | "
                        f"trial={trial} | "
                        f"Gate={int(gate_features['trigger'])} "
                        f"(open={gate_features['opened_low_fraction']:.3f}, "
                        f"comp={gate_features['largest_component_fraction']:.3f}, "
                        f"q10={gate_features['q10_reliability']:.3f}) | "
                        f"Oracle G/D/S/Q="
                        f"{oracle_geometry['success']:.0f}/"
                        f"{oracle_direct_local['success']:.0f}/"
                        f"{oracle_staged['success']:.0f}/"
                        f"{oracle_gated['success']:.0f} | "
                        f"A G/D/S/Q="
                        f"{A_geometry['success']:.0f}/"
                        f"{A_direct_local['success']:.0f}/"
                        f"{A_staged['success']:.0f}/"
                        f"{A_gated['success']:.0f} | "
                        f"B G/D/S/Q="
                        f"{B_geometry['success']:.0f}/"
                        f"{B_direct_local['success']:.0f}/"
                        f"{B_staged['success']:.0f}/"
                        f"{B_gated['success']:.0f}"
                    )


        # ==================================================
        # Per-frame summary
        # ==================================================
        for method_name in method_names:

            stats = summarize(
                frame_method_records[
                    method_name
                ]
            )


            frame_rows.append([
                object_id,
                frame_id,
                int(
                    oracle_reference_qualified
                ),
                method_name,
                stats[
                    "N"
                ],
                stats[
                    "success_count"
                ],
                stats[
                    "failure_count"
                ],
                stats[
                    "success_rate"
                ],
                stats[
                    "median_add_s_mm"
                ],
                stats[
                    "success_add_s_mm"
                ],
                stats[
                    "mean_source_points"
                ]
            ])


        # ==================================================
        # Checkpoint
        # ==================================================
        if (
            processed_frames
            % CHECKPOINT_EVERY_PROCESSED_FRAMES
            == 0
        ):

            write_csv(
                checkpoint_csv,
                trial_header,
                trial_rows
            )


            print(
                "Checkpoint saved after",
                processed_frames,
                "processed frames."
            )


# ==================================================
# Guard
# ==================================================
if processed_frames == 0:

    raise RuntimeError(
        "No frames processed successfully."
    )


# ==================================================
# Clean summary
# ==================================================
print(
    "\n\n=================================================="
)

print(
    "[Clean Baseline Summary - Reliability Gated]"
)

print(
    "Processed frames:",
    processed_frames
)

print(
    "Skipped frames:",
    skipped_frames
)


for object_id in OBJECT_IDS:

    print(
        f"Obj {object_id}: "
        f"processed={object_processed[object_id]}, "
        f"oracle-qualified="
        f"{object_oracle_qualified[object_id]}"
    )


for subset_name, source in [
    (
        "all_frames",
        clean_records_all
    ),
    (
        "oracle_reference_qualified",
        clean_records_qualified
    )
]:

    print(
        f"\n[{subset_name}]"
    )


    for method_name in method_names:

        stats = summarize(
            source[
                method_name
            ]
        )


        print(
            f"{method_name} | "
            f"Frames={stats['frames']} | "
            f"Success={stats['success_rate']:.4f} "
            f"[95% CI "
            f"{stats['success_ci_low']:.4f}, "
            f"{stats['success_ci_high']:.4f}] | "
            f"Median ADD-S="
            f"{stats['median_add_s_mm']:.4f} mm | "
            f"Mean source="
            f"{stats['mean_source_points']:.1f}"
        )


# ==================================================
# Summary by corruption condition
# ==================================================
condition_summary_rows = []


print(
    "\n=================================================="
)

print(
    "[Summary by Corruption Condition - Reliability Gated]"
)


for corruption_type in corruption_types:

    print(
        f"\n######## {corruption_type} ########"
    )


    for fraction in corruption_fractions:

        print(
            f"\nCorruption fraction: {fraction:.2f}"
        )


        for subset_name, source in [
            (
                "all_frames",
                records_by_condition_all
            ),
            (
                "oracle_reference_qualified",
                records_by_condition_qualified
            )
        ]:

            print(
                f"[{subset_name}]"
            )


            for method_name in method_names:

                stats = summarize(
                    source[
                        (
                            corruption_type,
                            fraction
                        )
                    ][
                        method_name
                    ]
                )


                print(
                    f"{method_name} | "
                    f"Objects={stats['objects']} | "
                    f"Frames={stats['frames']} | "
                    f"N={stats['N']} | "
                    f"Success={stats['success_rate']:.4f} "
                    f"[{stats['success_ci_low']:.4f}, "
                    f"{stats['success_ci_high']:.4f}] | "
                    f"Median ADD-S="
                    f"{stats['median_add_s_mm']:.4f} mm"
                )


                condition_summary_rows.append([
                    corruption_type,
                    fraction,
                    subset_name,
                    method_name,
                    stats[
                        "objects"
                    ],
                    stats[
                        "frames"
                    ],
                    stats[
                        "N"
                    ],
                    stats[
                        "success_count"
                    ],
                    stats[
                        "failure_count"
                    ],
                    stats[
                        "success_rate"
                    ],
                    stats[
                        "success_ci_low"
                    ],
                    stats[
                        "success_ci_high"
                    ],
                    stats[
                        "mean_add_s_mm"
                    ],
                    stats[
                        "median_add_s_mm"
                    ],
                    stats[
                        "success_add_s_mm"
                    ],
                    stats[
                        "fitness_mean"
                    ],
                    stats[
                        "mean_source_points"
                    ]
                ])


# ==================================================
# Aggregate by corruption type
# ==================================================
corruption_summary_rows = []


print(
    "\n=================================================="
)

print(
    "[Aggregate by Corruption Type - Reliability Gated]"
)


for corruption_type in corruption_types:

    print(
        f"\n[{corruption_type}]"
    )


    for subset_name, source in [
        (
            "all_frames",
            records_by_corruption_all
        ),
        (
            "oracle_reference_qualified",
            records_by_corruption_qualified
        )
    ]:

        print(
            f"[{subset_name}]"
        )


        for method_name in method_names:

            stats = summarize(
                source[
                    corruption_type
                ][
                    method_name
                ]
            )


            print(
                f"{method_name} | "
                f"Objects={stats['objects']} | "
                f"Frames={stats['frames']} | "
                f"N={stats['N']} | "
                f"Success={stats['success_rate']:.4f} "
                f"[{stats['success_ci_low']:.4f}, "
                f"{stats['success_ci_high']:.4f}] | "
                f"Median ADD-S="
                f"{stats['median_add_s_mm']:.4f} mm"
            )


            corruption_summary_rows.append([
                corruption_type,
                subset_name,
                method_name,
                stats[
                    "objects"
                ],
                stats[
                    "frames"
                ],
                stats[
                    "N"
                ],
                stats[
                    "success_count"
                ],
                stats[
                    "failure_count"
                ],
                stats[
                    "success_rate"
                ],
                stats[
                    "success_ci_low"
                ],
                stats[
                    "success_ci_high"
                ],
                stats[
                    "mean_add_s_mm"
                ],
                stats[
                    "median_add_s_mm"
                ],
                stats[
                    "success_add_s_mm"
                ],
                stats[
                    "fitness_mean"
                ],
                stats[
                    "mean_source_points"
                ]
            ])


# ==================================================
# Per-object aggregate
# ==================================================
object_summary_rows = []


print(
    "\n=================================================="
)

print(
    "[Per-object Corruption Aggregate - Reliability Gated]"
)


for object_id in OBJECT_IDS:

    print(
        f"\n######## OBJECT {object_id} ########"
    )


    for subset_name, source in [
        (
            "all_frames",
            object_records_all
        ),
        (
            "oracle_reference_qualified",
            object_records_qualified
        )
    ]:

        print(
            f"[{subset_name}]"
        )


        for method_name in method_names:

            stats = summarize(
                source[
                    object_id
                ][
                    method_name
                ]
            )


            print(
                f"{method_name} | "
                f"Frames={stats['frames']} | "
                f"N={stats['N']} | "
                f"Success={stats['success_rate']:.4f} | "
                f"Median ADD-S="
                f"{stats['median_add_s_mm']:.4f} mm"
            )


            object_summary_rows.append([
                object_id,
                subset_name,
                method_name,
                stats[
                    "frames"
                ],
                stats[
                    "N"
                ],
                stats[
                    "success_count"
                ],
                stats[
                    "failure_count"
                ],
                stats[
                    "success_rate"
                ],
                stats[
                    "success_ci_low"
                ],
                stats[
                    "success_ci_high"
                ],
                stats[
                    "mean_add_s_mm"
                ],
                stats[
                    "median_add_s_mm"
                ],
                stats[
                    "success_add_s_mm"
                ],
                stats[
                    "fitness_mean"
                ],
                stats[
                    "mean_source_points"
                ]
            ])


# ==================================================
# Overall aggregate
# ==================================================
aggregate_rows = []


print(
    "\n=================================================="
)

print(
    "[Aggregate over ALL Objects and Corruption Trials - Reliability Gated]"
)


for subset_name, source in [
    (
        "all_frames",
        aggregate_all
    ),
    (
        "oracle_reference_qualified",
        aggregate_qualified
    )
]:

    print(
        f"\n[{subset_name}]"
    )


    for method_name in method_names:

        stats = summarize(
            source[
                method_name
            ]
        )


        print(
            f"{method_name} | "
            f"Objects={stats['objects']} | "
            f"Frames={stats['frames']} | "
            f"N={stats['N']} | "
            f"Success={stats['success_rate']:.4f} "
            f"[95% CI "
            f"{stats['success_ci_low']:.4f}, "
            f"{stats['success_ci_high']:.4f}] | "
            f"Mean ADD-S="
            f"{stats['mean_add_s_mm']:.4f} mm | "
            f"Median ADD-S="
            f"{stats['median_add_s_mm']:.4f} mm | "
            f"Success ADD-S="
            f"{stats['success_add_s_mm']:.4f} mm"
        )


        aggregate_rows.append([
            subset_name,
            method_name,
            stats[
                "objects"
            ],
            stats[
                "frames"
            ],
            stats[
                "N"
            ],
            stats[
                "success_count"
            ],
            stats[
                "failure_count"
            ],
            stats[
                "success_rate"
            ],
            stats[
                "success_ci_low"
            ],
            stats[
                "success_ci_high"
            ],
            stats[
                "mean_add_s_mm"
            ],
            stats[
                "median_add_s_mm"
            ],
            stats[
                "success_add_s_mm"
            ],
            stats[
                "fitness_mean"
            ],
            stats[
                "mean_source_points"
            ]
        ])


# ==================================================
# Paired comparisons
# ==================================================
paired_summary_rows = []
transition_rows = []


def append_comparison(
        analysis_level,
        protocol_name,
        comparison_name,
        corruption_type,
        fraction,
        subset_name,
        rows,
        reference_prefix,
        candidate_prefix
):

    stats = summarize_pair(
        rows,
        reference_prefix,
        candidate_prefix
    )


    print(
        f"{analysis_level} | "
        f"protocol={protocol_name} | "
        f"compare={comparison_name} | "
        f"type={corruption_type} | "
        f"fraction={fraction} | "
        f"subset={subset_name} | "
        f"N={stats['N']} | "
        f"Success delta="
        f"{stats['success_delta']:+.4f} "
        f"[95% CI "
        f"{stats['success_delta_ci_low']:+.4f}, "
        f"{stats['success_delta_ci_high']:+.4f}] | "
        f"Candidate-only="
        f"{stats['candidate_only_success']} | "
        f"Reference-only="
        f"{stats['reference_only_success']} | "
        f"Both-success median ADD-S delta="
        f"{stats['both_success_median_add_delta_mm']:+.4f} mm"
    )


    paired_summary_rows.append([
        analysis_level,
        protocol_name,
        comparison_name,
        corruption_type,
        fraction,
        subset_name,
        stats[
            "objects"
        ],
        stats[
            "frames"
        ],
        stats[
            "N"
        ],
        stats[
            "reference_only_success"
        ],
        stats[
            "candidate_only_success"
        ],
        stats[
            "both_success"
        ],
        stats[
            "both_fail"
        ],
        stats[
            "success_delta"
        ],
        stats[
            "success_delta_ci_low"
        ],
        stats[
            "success_delta_ci_high"
        ],
        stats[
            "both_success_median_add_delta_mm"
        ],
        stats[
            "candidate_lower_add_fraction"
        ]
    ])


print(
    "\n=================================================="
)

print(
    "[Paired Reliability-Gated Comparisons]"
)

print(
    "Positive success delta means candidate is better."
)

print(
    "Negative ADD-S delta means candidate is more accurate."
)


for protocol_name, rows_all in paired_rows.items():

    qualified_rows = [
        row
        for row in rows_all
        if row[
            "qualified"
        ]
    ]


    # --------------------------------------------------
    # Overall:
    # staged vs geometry
    # --------------------------------------------------
    append_comparison(
        "overall",
        protocol_name,
        "staged_vs_geometry",
        "ALL",
        "ALL",
        "oracle_reference_qualified",
        qualified_rows,
        "geometry",
        "staged"
    )


    # --------------------------------------------------
    # Overall:
    # staged vs direct Local
    # --------------------------------------------------
    append_comparison(
        "overall",
        protocol_name,
        "staged_vs_direct_local",
        "ALL",
        "ALL",
        "oracle_reference_qualified",
        qualified_rows,
        "direct_local",
        "staged"
    )


    # Primary gated comparisons.
    append_comparison(
        "overall",
        protocol_name,
        "gated_vs_geometry",
        "ALL",
        "ALL",
        "oracle_reference_qualified",
        qualified_rows,
        "geometry",
        "gated"
    )


    append_comparison(
        "overall",
        protocol_name,
        "gated_vs_unconditional_staged",
        "ALL",
        "ALL",
        "oracle_reference_qualified",
        qualified_rows,
        "staged",
        "gated"
    )


    append_comparison(
        "overall",
        protocol_name,
        "gated_vs_direct_local",
        "ALL",
        "ALL",
        "oracle_reference_qualified",
        qualified_rows,
        "direct_local",
        "gated"
    )


    for corruption_type in corruption_types:

        type_rows = [
            row
            for row in qualified_rows
            if row[
                "corruption_type"
            ] == corruption_type
        ]


        append_comparison(
            "corruption_type",
            protocol_name,
            "staged_vs_geometry",
            corruption_type,
            "ALL",
            "oracle_reference_qualified",
            type_rows,
            "geometry",
            "staged"
        )


        append_comparison(
            "corruption_type",
            protocol_name,
            "staged_vs_direct_local",
            corruption_type,
            "ALL",
            "oracle_reference_qualified",
            type_rows,
            "direct_local",
            "staged"
        )


        append_comparison(
            "corruption_type",
            protocol_name,
            "gated_vs_geometry",
            corruption_type,
            "ALL",
            "oracle_reference_qualified",
            type_rows,
            "geometry",
            "gated"
        )


        append_comparison(
            "corruption_type",
            protocol_name,
            "gated_vs_unconditional_staged",
            corruption_type,
            "ALL",
            "oracle_reference_qualified",
            type_rows,
            "staged",
            "gated"
        )


        strong_rows = [
            row
            for row in type_rows
            if abs(
                row[
                    "fraction"
                ]
                - 0.30
            ) < 1e-12
        ]


        append_comparison(
            "strong_condition",
            protocol_name,
            "staged_vs_geometry",
            corruption_type,
            0.30,
            "oracle_reference_qualified",
            strong_rows,
            "geometry",
            "staged"
        )


        append_comparison(
            "strong_condition",
            protocol_name,
            "staged_vs_direct_local",
            corruption_type,
            0.30,
            "oracle_reference_qualified",
            strong_rows,
            "direct_local",
            "staged"
        )


        append_comparison(
            "strong_condition",
            protocol_name,
            "gated_vs_geometry",
            corruption_type,
            0.30,
            "oracle_reference_qualified",
            strong_rows,
            "geometry",
            "gated"
        )


        append_comparison(
            "strong_condition",
            protocol_name,
            "gated_vs_unconditional_staged",
            corruption_type,
            0.30,
            "oracle_reference_qualified",
            strong_rows,
            "staged",
            "gated"
        )


    # --------------------------------------------------
    # Stage transition summary:
    # specifically counts what Stage 2 does to Geometry.
    # --------------------------------------------------
    for corruption_type in corruption_types:

        for fraction in corruption_fractions:

            rows = [
                row
                for row in qualified_rows
                if (
                    row[
                        "corruption_type"
                    ] == corruption_type
                    and abs(
                        row[
                            "fraction"
                        ]
                        - fraction
                    ) < 1e-12
                )
            ]


            stats = summarize_pair(
                rows,
                "geometry",
                "staged"
            )


            transition_rows.append([
                protocol_name,
                corruption_type,
                fraction,
                stats[
                    "frames"
                ],
                stats[
                    "N"
                ],
                stats[
                    "reference_only_success"
                ],
                stats[
                    "candidate_only_success"
                ],
                stats[
                    "both_success"
                ],
                stats[
                    "both_fail"
                ],
                stats[
                    "success_delta"
                ],
                stats[
                    "success_delta_ci_low"
                ],
                stats[
                    "success_delta_ci_high"
                ],
                stats[
                    "both_success_median_add_delta_mm"
                ],
                stats[
                    "candidate_lower_add_fraction"
                ]
            ])


# ==================================================
# Reliability gate diagnostics
# ==================================================
print(
    "\n=================================================="
)

print(
    "[Reliability Gate Diagnostics]"
)

print(
    "Gate is GT-free. Corruption labels below are used only"
)

print(
    "for post-hoc controlled evaluation of trigger behavior."
)


gate_summary_rows = []


def summarize_gate_rows(
        rows
):

    if len(
        rows
    ) == 0:

        return {
            "N": 0,
            "frames": 0,
            "trigger_rate": np.nan,
            "mean_reliability": np.nan,
            "q10_reliability": np.nan,
            "strong_low_fraction": np.nan,
            "opened_low_fraction": np.nan,
            "largest_component_fraction": np.nan,
            "mean_local_diff_mm": np.nan,
            "q90_local_diff_mm": np.nan
        }


    return {
        "N": len(
            rows
        ),
        "frames": len(
            {
                row[
                    "frame_id"
                ]
                for row in rows
            }
        ),
        "trigger_rate": float(
            np.mean(
                [
                    row[
                        "trigger"
                    ]
                    for row in rows
                ]
            )
        ),
        "mean_reliability": safe_nanmean(
            [
                row[
                    "mean_reliability"
                ]
                for row in rows
            ]
        ),
        "q10_reliability": safe_nanmean(
            [
                row[
                    "q10_reliability"
                ]
                for row in rows
            ]
        ),
        "strong_low_fraction": safe_nanmean(
            [
                row[
                    "strong_low_fraction"
                ]
                for row in rows
            ]
        ),
        "opened_low_fraction": safe_nanmean(
            [
                row[
                    "opened_low_fraction"
                ]
                for row in rows
            ]
        ),
        "largest_component_fraction": safe_nanmean(
            [
                row[
                    "largest_component_fraction"
                ]
                for row in rows
            ]
        ),
        "mean_local_diff_mm": safe_nanmean(
            [
                row[
                    "mean_local_diff_mm"
                ]
                for row in rows
            ]
        ),
        "q90_local_diff_mm": safe_nanmean(
            [
                row[
                    "q90_local_diff_mm"
                ]
                for row in rows
            ]
        )
    }


clean_gate_stats = summarize_gate_rows(
    clean_gate_rows
)


print(
    f"CLEAN | "
    f"Frames={clean_gate_stats['frames']} | "
    f"Trigger rate={clean_gate_stats['trigger_rate']:.4f} | "
    f"q10={clean_gate_stats['q10_reliability']:.4f} | "
    f"Opened-low={clean_gate_stats['opened_low_fraction']:.4f} | "
    f"Largest-component="
    f"{clean_gate_stats['largest_component_fraction']:.4f}"
)


gate_summary_rows.append([
    "clean",
    "clean",
    0.0,
    "all_frames",
    clean_gate_stats[
        "frames"
    ],
    clean_gate_stats[
        "N"
    ],
    clean_gate_stats[
        "trigger_rate"
    ],
    clean_gate_stats[
        "mean_reliability"
    ],
    clean_gate_stats[
        "q10_reliability"
    ],
    clean_gate_stats[
        "strong_low_fraction"
    ],
    clean_gate_stats[
        "opened_low_fraction"
    ],
    clean_gate_stats[
        "largest_component_fraction"
    ],
    clean_gate_stats[
        "mean_local_diff_mm"
    ],
    clean_gate_stats[
        "q90_local_diff_mm"
    ]
])


for corruption_type in corruption_types:

    for fraction in corruption_fractions:

        condition_rows = [
            row
            for row in gate_rows
            if (
                row[
                    "corruption_type"
                ]
                == corruption_type
                and abs(
                    row[
                        "fraction"
                    ]
                    - fraction
                ) < 1e-12
            )
        ]


        for subset_name, subset_rows in [
            (
                "all_frames",
                condition_rows
            ),
            (
                "oracle_reference_qualified",
                [
                    row
                    for row in condition_rows
                    if row[
                        "qualified"
                    ]
                ]
            )
        ]:

            stats = summarize_gate_rows(
                subset_rows
            )


            print(
                f"{corruption_type} | "
                f"fraction={fraction:.2f} | "
                f"subset={subset_name} | "
                f"N={stats['N']} | "
                f"Trigger={stats['trigger_rate']:.4f} | "
                f"q10={stats['q10_reliability']:.4f} | "
                f"Strong-low="
                f"{stats['strong_low_fraction']:.4f} | "
                f"Opened-low="
                f"{stats['opened_low_fraction']:.4f} | "
                f"Largest-comp="
                f"{stats['largest_component_fraction']:.4f}"
            )


            gate_summary_rows.append([
                "corruption",
                corruption_type,
                fraction,
                subset_name,
                stats[
                    "frames"
                ],
                stats[
                    "N"
                ],
                stats[
                    "trigger_rate"
                ],
                stats[
                    "mean_reliability"
                ],
                stats[
                    "q10_reliability"
                ],
                stats[
                    "strong_low_fraction"
                ],
                stats[
                    "opened_low_fraction"
                ],
                stats[
                    "largest_component_fraction"
                ],
                stats[
                    "mean_local_diff_mm"
                ],
                stats[
                    "q90_local_diff_mm"
                ]
            ])


# ==================================================
# Visibility robustness diagnostics
# ==================================================
print(
    "\n=================================================="
)

print(
    "[Visibility Robustness under Corruption]"
)


for corruption_type in corruption_types:

    for fraction in corruption_fractions:

        rows = [
            row
            for row in visibility_rows
            if (
                row[
                    3
                ] == corruption_type
                and abs(
                    row[
                        4
                    ]
                    - fraction
                ) < 1e-12
            )
        ]


        print(
            f"{corruption_type} | "
            f"fraction={fraction:.2f} | "
            f"Fixed-clean J="
            f"{safe_nanmean([row[11] for row in rows]):.4f} | "
            f"E2E J="
            f"{safe_nanmean([row[14] for row in rows]):.4f} | "
            f"E2E/fixed source ratio="
            f"{safe_nanmean([row[15] for row in rows]):.4f}"
        )


# ==================================================
# Save CSV files
# ==================================================
write_csv(
    trial_csv,
    trial_header,
    trial_rows
)


write_csv(
    condition_summary_csv,
    [
        "corruption_type",
        "corruption_fraction",
        "subset",
        "method",
        "object_count",
        "frame_count",
        "N",
        "success_count",
        "failure_count",
        "success_rate",
        "success_rate_CI_low",
        "success_rate_CI_high",
        "ADD_S_mean_mm",
        "ADD_S_median_mm",
        "success_ADD_S_mean_mm",
        "fitness_mean",
        "mean_source_points"
    ],
    condition_summary_rows
)


write_csv(
    corruption_summary_csv,
    [
        "corruption_type",
        "subset",
        "method",
        "object_count",
        "frame_count",
        "N",
        "success_count",
        "failure_count",
        "success_rate",
        "success_rate_CI_low",
        "success_rate_CI_high",
        "ADD_S_mean_mm",
        "ADD_S_median_mm",
        "success_ADD_S_mean_mm",
        "fitness_mean",
        "mean_source_points"
    ],
    corruption_summary_rows
)


write_csv(
    object_summary_csv,
    [
        "object_id",
        "subset",
        "method",
        "frame_count",
        "N",
        "success_count",
        "failure_count",
        "success_rate",
        "success_rate_CI_low",
        "success_rate_CI_high",
        "ADD_S_mean_mm",
        "ADD_S_median_mm",
        "success_ADD_S_mean_mm",
        "fitness_mean",
        "mean_source_points"
    ],
    object_summary_rows
)


write_csv(
    aggregate_csv,
    [
        "subset",
        "method",
        "object_count",
        "frame_count",
        "N",
        "success_count",
        "failure_count",
        "success_rate",
        "success_rate_CI_low",
        "success_rate_CI_high",
        "ADD_S_mean_mm",
        "ADD_S_median_mm",
        "success_ADD_S_mean_mm",
        "fitness_mean",
        "mean_source_points"
    ],
    aggregate_rows
)


write_csv(
    paired_csv,
    [
        "analysis_level",
        "protocol",
        "comparison",
        "corruption_type",
        "corruption_fraction",
        "subset",
        "object_count",
        "frame_count",
        "N_paired_trials",
        "reference_only_success",
        "candidate_only_success",
        "both_success",
        "both_fail",
        "candidate_minus_reference_success_rate",
        "success_delta_CI_low",
        "success_delta_CI_high",
        "both_success_median_candidate_minus_reference_ADD_S_mm",
        "fraction_candidate_lower_ADD_S_among_both_success"
    ],
    paired_summary_rows
)


write_csv(
    transition_csv,
    [
        "protocol",
        "corruption_type",
        "corruption_fraction",
        "frame_count",
        "N",
        "geometry_success_staged_fail",
        "geometry_fail_staged_success",
        "both_success",
        "both_fail",
        "staged_minus_geometry_success_rate",
        "success_delta_CI_low",
        "success_delta_CI_high",
        "both_success_median_staged_minus_geometry_ADD_S_mm",
        "fraction_staged_lower_ADD_S_among_both_success"
    ],
    transition_rows
)


write_csv(
    gate_csv,
    [
        "object_id",
        "frame_id",
        "oracle_reference_qualified",
        "corruption_type",
        "corruption_fraction",
        "trial",
        "gate_triggered",
        "gate_reason",
        "valid_count",
        "interior_count",
        "mean_reliability",
        "median_reliability",
        "q10_reliability",
        "q25_reliability",
        "strong_low_fraction",
        "opened_low_fraction",
        "largest_component_fraction",
        "largest_component_share_of_low",
        "mean_local_diff_mm",
        "q90_local_diff_mm"
    ],
    [
        [
            row[
                "object_id"
            ],
            row[
                "frame_id"
            ],
            row[
                "qualified"
            ],
            row[
                "corruption_type"
            ],
            row[
                "fraction"
            ],
            row[
                "trial"
            ],
            row[
                "trigger"
            ],
            row[
                "reason"
            ],
            row[
                "valid_count"
            ],
            row[
                "interior_count"
            ],
            row[
                "mean_reliability"
            ],
            row[
                "median_reliability"
            ],
            row[
                "q10_reliability"
            ],
            row[
                "q25_reliability"
            ],
            row[
                "strong_low_fraction"
            ],
            row[
                "opened_low_fraction"
            ],
            row[
                "largest_component_fraction"
            ],
            row[
                "largest_component_share_of_low"
            ],
            row[
                "mean_local_diff_mm"
            ],
            row[
                "q90_local_diff_mm"
            ]
        ]
        for row in gate_rows
    ]
)


write_csv(
    gate_summary_csv,
    [
        "analysis_type",
        "corruption_type",
        "corruption_fraction",
        "subset",
        "frame_count",
        "N",
        "trigger_rate",
        "mean_reliability",
        "mean_q10_reliability",
        "mean_strong_low_fraction",
        "mean_opened_low_fraction",
        "mean_largest_component_fraction",
        "mean_local_diff_mm",
        "mean_q90_local_diff_mm"
    ],
    gate_summary_rows
)


write_csv(
    visibility_csv,
    [
        "object_id",
        "frame_id",
        "oracle_reference_qualified",
        "corruption_type",
        "corruption_fraction",
        "trial",
        "oracle_visible_points",
        "fixed_clean_observation_visible_points",
        "e2e_corrupted_observation_visible_points",
        "fixed_clean_precision_vs_oracle",
        "fixed_clean_recall_vs_oracle",
        "fixed_clean_jaccard_vs_oracle",
        "e2e_precision_vs_oracle",
        "e2e_recall_vs_oracle",
        "e2e_jaccard_vs_oracle",
        "e2e_to_fixed_source_count_ratio"
    ],
    visibility_rows
)


write_csv(
    checkpoint_csv,
    trial_header,
    trial_rows
)


print(
    "\nSaved trials:",
    trial_csv
)

print(
    "Saved condition summary:",
    condition_summary_csv
)

print(
    "Saved corruption summary:",
    corruption_summary_csv
)

print(
    "Saved object summary:",
    object_summary_csv
)

print(
    "Saved aggregate:",
    aggregate_csv
)

print(
    "Saved paired comparisons:",
    paired_csv
)

print(
    "Saved stage-transition summary:",
    transition_csv
)

print(
    "Saved gate diagnostics:",
    gate_csv
)

print(
    "Saved gate summary:",
    gate_summary_csv
)

print(
    "Saved visibility diagnostics:",
    visibility_csv
)

print(
    "Saved checkpoint:",
    checkpoint_csv
)

print(
    "\nSCIENTIFIC TARGET:"
)

print(
    "Test whether a GT-free reliability-map gate can preserve"
)

print(
    "Geometry ICP on clean/bias/hole cases while selectively"
)

print(
    "activating Local refinement for area-like local outliers."
)

print(
    "\nPRIMARY CHECKS:"
)

print(
    "1) CLEAN gate trigger rate should be low."
)

print(
    "2) Outlier trigger rate should rise with corruption fraction."
)

print(
    "3) Bias/hole trigger rates should remain substantially lower."
)

print(
    "4) Protocol-B gated success should stay near Geometry."
)

print(
    "5) Under strong outlier, gated ADD-S should improve over Geometry."
)

print(
    "\nIMPORTANT:"
)

print(
    "The gate itself does not use corruption labels, T_gt, ADD-S,"
)

print(
    "success labels, or oracle visibility. Corruption labels are"
)

print(
    "used only for post-hoc controlled analysis."
)

print(
    "\nNOTE:"
)

print(
    "GT pose is still used for controlled T_init generation,"
)

print(
    "oracle-reference visibility, and ADD-S evaluation."
)

print(
    "Protocol A/B visibility selection and the reliability gate"
)

print(
    "do not use GT pose."
)

print(
    "BOP mask_visib is still used as a segmentation proxy."
)
