"""
Evaluates the reconstruction of 3D keypoints by a desired 3D pose lifter method.
"""

# --- External ---
import matplotlib
matplotlib.use('Agg')

import argparse
import datetime
import glob
import json
import os
import typing
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from argparse import Namespace
from tqdm import tqdm

# --- Internal ---
from data.data_config import FPS, YOLO_PARENTS
from anim.data.amass import YOLO_LOWER_JOINTS, YOLO_UPPER_JOINTS, YoloJoints, preprocess_keypoints
from utils import utils_filters, utils_fix, utils_metric, utils_print, utils_plot
from utils.utils_metric import EvaluationMetrics
from utils.utils_print import (
    print_summary,
    print_summary_by_joint,
    print_summary_o,
    print_summary_by_joint_o,
    print_summary_by_joint_v_m,
)
from utils.utils_plot import (
    animate,
    plot_distribution_of_error_for_joint,
    plot_error_heatmap,
    plot_joint_errors,
    plot_joint_errors_with_time_conf,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXCLUDED_JOINTS: typing.FrozenSet[YoloJoints] = frozenset({
    YoloJoints.NOSE,
    YoloJoints.LEFT_EYE,
    YoloJoints.RIGHT_EYE,
    YoloJoints.LEFT_EAR,
    YoloJoints.RIGHT_EAR,
})

OCCLUSION_CONF_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Misc / helpers
# ---------------------------------------------------------------------------

def to_scientific_notation(value: typing.Optional[float]) -> typing.Optional[str]:
    """Convert a float to a 2-significant-figure scientific notation string."""
    return None if value is None else f"{value:.2e}"


def override_json_parameters(args: argparse.Namespace, joint_params: dict) -> None:
    """Override per-joint Kalman parameters from CLI arguments (in-place)."""
    if args.joint_to_filter is None:
        return
    joint = joint_params[args.joint_to_filter]
    if args.measurement_noise is not None:
        joint['measurement_noise'] = args.measurement_noise
    if args.process_noise is not None:
        joint['process_noise'] = args.process_noise


def normalize_path(path: str) -> str:
    """
    Strip dataset-specific directory components and file extension so that
    paths from different subdirectories can be compared by identity.
    """
    parts = [p for p in path.split('/') if p not in {'triang', 'preprocessed', 'openmpl'}]
    if not parts:
        return ''
    parts[-1] = os.path.splitext(parts[-1])[0]
    return '/'.join(parts)


def compare_lists(files_gt: typing.List[str], files_pred: typing.List[str]) -> None:
    """Assert that two file lists refer to the same sequences after normalisation."""
    gt_norm   = sorted(normalize_path(f) for f in files_gt)
    pred_norm = sorted(normalize_path(f) for f in files_pred)

    for gt, pred in zip(gt_norm, pred_norm):
        if gt != pred:
            print(f"Mismatch:\n  GT:   {gt}\n  Pred: {pred}")
            raise AssertionError("File lists contain mismatched entries.")

    if len(pred_norm) > len(gt_norm):
        print("Item present in pred but missing from gt:", pred_norm[-1])


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_gt(path: str) -> np.lib.npyio.NpzFile:
    return np.load(path, allow_pickle=True)


def load_kp(path: str) -> np.lib.npyio.NpzFile:
    """Load the .npz prediction file corresponding to *path* (any extension)."""
    npz_path = Path(path).with_suffix('.npz')
    return np.load(npz_path, allow_pickle=True)


def save_eval_metrics_to_json(
    eval_metrics,
    args: argparse.Namespace,
    process_noise_value: typing.Optional[float] = None,
    measurement_noise_value: typing.Optional[float] = None,
    output_dir: str = './data/results/',
) -> str:
    """Serialise evaluation metrics to a JSON file with a descriptive name."""

    pn_str    = f"pn{to_scientific_notation(process_noise_value)}" if process_noise_value    is not None else "pnNone"
    mn_str    = f"mn{to_scientific_notation(measurement_noise_value)}" if measurement_noise_value is not None else "mnNone"
    joint_str = f"joint{args.joint_to_filter}" if args.joint_to_filter is not None else "jointAll"
    filename  = os.path.join(output_dir, f"eval_metrics_{pn_str}_{mn_str}_{joint_str}.json")

    payload: dict = {
        'parameters': {
            'method':             args.method,
            'model':              args.model,
            'use_kalman_filter':  args.use_kalman_filter,
            'joint_to_filter':    args.joint_to_filter,
            'process_noise':      to_scientific_notation(process_noise_value),
            'measurement_noise':  to_scientific_notation(measurement_noise_value),
        },
        'metrics':   _build_metrics_dict(eval_metrics, args),
        'timestamp': datetime.datetime.now().isoformat(),
    }

    with open(filename, 'w') as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    print(f"Evaluation metrics saved to: {filename}")
    return filename


def _build_metrics_dict(eval_metrics, args: argparse.Namespace) -> dict:
    """Return the 'metrics' sub-dict depending on whether a single joint is targeted."""

    def sci(arr):
        return [to_scientific_notation(x) for x in arr]

    def summary(**kwargs) -> dict:
        return {k: to_scientific_notation(np.mean(v) if hasattr(v, '__len__') else v)
                for k, v in kwargs.items()}

    if args.joint_to_filter is None:
        mpjpes, mpjves, mpjpes_o, mpjves_o = eval_metrics
        return {
            'mpjpes':   sci(mpjpes),
            'mpjves':   sci(mpjves),
            'mpjpes_o': sci(mpjpes_o),
            'mpjves_o': sci(mpjves_o),
            'summary': {
                **summary(mean_mpjpe=np.mean(mpjpes), std_mpjpe=np.std(mpjpes)),
                **summary(mean_mpjve=np.mean(mpjves), std_mpjve=np.std(mpjves)),
                **summary(mean_mpjpe_o=np.mean(mpjpes_o), std_mpjpe_o=np.std(mpjpes_o)),
                **summary(mean_mpjve_o=np.mean(mpjves_o), std_mpjve_o=np.std(mpjves_o)),
            },
        }

    pjpes, pjves, pjpes_o, pjves_o, jitters_o = eval_metrics
    return {
        'summary': {
            **summary(mean_pjpe=np.mean(pjpes),     std_pjpe=np.std(pjpes)),
            **summary(mean_pjve=np.mean(pjves),     std_pjve=np.std(pjves)),
            **summary(mean_pjpe_o=np.mean(pjpes_o), std_pjpe_o=np.std(pjpes_o)),
            **summary(mean_pjve_o=np.mean(pjves_o), std_pjve_o=np.std(pjves_o)),
            **summary(mean_jitter_o=np.mean(jitters_o), std_jitter_o=np.std(jitters_o)),
        }
    }


# ---------------------------------------------------------------------------
# Core evaluation helpers
# ---------------------------------------------------------------------------

def _build_preprocess_config(args: argparse.Namespace) -> Namespace:
    if args.use_kalman_filter:
        assert args.filter_name is not None, (
            f"--filter_name is required when --use_kalman_filter is set, got {args.filter_name!r}"
        )
        return Namespace(USE_KALMAN_FILTER=True, FILTER_NAME=args.filter_name)
    return Namespace(USE_KALMAN_FILTER=False)


def _apply_occlusion_mask(
    points3d_gt: np.ndarray,
    points3d_pred: np.ndarray,
    conf_pred: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove frames where any included joint is occluded (conf < threshold in
    any camera view).  Returns the filtered triple (gt, pred, conf).
    """
    included_idx   = [i for i in range(conf_pred.shape[1]) if i not in EXCLUDED_JOINTS]
    conf_included  = conf_pred[:, included_idx, :]
    frame_occluded = (conf_included < OCCLUSION_CONF_THRESHOLD).any(axis=-1).any(axis=-1)
    valid_idx      = np.where(~frame_occluded)[0]

    return points3d_gt[valid_idx], points3d_pred[valid_idx], conf_pred[valid_idx]


def _process_single_sequence(
    file_gt: str,
    file_pred: str,
    file_yolo: str,
    args: argparse.Namespace,
    preprocess_config: Namespace) -> EvaluationMetrics:
    """Load one GT / prediction pair, optionally filter, and return metrics."""

    data_gt   = load_gt(file_gt)
    data_pred = load_kp(file_pred)

    # Ground-truth 3-D keypoints
    raw_gt = data_gt.get('yolo_keypoints') or data_gt.get('pose_estimation_keypoints')
    assert raw_gt is not None and 'ground_truth' in raw_gt, (
        f"'ground_truth' key missing in {file_gt}"
    )
    points3d_gt = raw_gt['ground_truth']

    conf_pred = data_pred.get('conf')

    points3d_pred = preprocess_keypoints(
        torch.tensor(data_pred['points3d'], dtype=torch.float32),
        preprocess_config,
        conf_scores=torch.tensor(conf_pred, dtype=torch.float32) if conf_pred is not None else None,
        use_kalman_filter=args.use_kalman_filter,
        args=args,
        threshold=OCCLUSION_CONF_THRESHOLD,
        occlusion_aware=args.occlusion_aware,
        kalman_params_path='utils/kalman_parameters_evaluate.json',
    )
    # Fall back to YOLO triangulation confidence when the method has none
    if conf_pred is None:
        conf_pred = load_kp(file_yolo)['conf']

    return EvaluationMetrics(
        points3d_gt, points3d_pred, conf_pred,
        excluded=EXCLUDED_JOINTS,
        files=[file_gt, file_pred],
    )


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def one_method_vs_gt(
    args: argparse.Namespace,
    output_dir: str) -> None:
    """Iterate over all test sequences, compute metrics, and print summaries."""

    files_pred = sorted(glob.glob(os.path.join(args.model, "**", "test", args.method,       "*.npz")))
    files_gt   = sorted(glob.glob(os.path.join(args.model, "**", "test", "preprocessed",    "*.pkl")))
    files_yolo = sorted(glob.glob(os.path.join(args.model, "**", "test", "triang",          "*.npz")))

    if len(files_gt) != len(files_pred) or len(files_gt) != len(files_yolo):
        compare_lists(files_gt, files_pred)
        raise ValueError(
            f"File list length mismatch — GT: {len(files_gt)}, "
            f"pred: {len(files_pred)}, yolo: {len(files_yolo)}"
        )

    preprocess_config = _build_preprocess_config(args)

    # Accumulators
    mpjpes,  mpjves,  jitters                            = [], [], []
    mpjpes_o, mpjves_o                                   = [], []
    pjpes_by_joint,   pjves_by_joint                     = [], []
    pjpes_by_joint_m, pjves_by_joint_m                   = [], []
    pjpes_by_joint_o, pjves_by_joint_o, jitters_by_joint_o = [], [], []
    jitters_by_joint_v, jitters_by_joint_m               = [], []
    
    # Occlusion counts for micro-averages
    global_total_frames = 0
    global_occ_counts_by_joint = None
    global_total_included_frames = 0
    global_total_included_occ = 0

    for file_gt, file_pred, file_yolo in tqdm(
        zip(files_gt, files_pred, files_yolo), total=len(files_gt)
    ):
        metrics = _process_single_sequence(
            file_gt, file_pred, file_yolo, args, preprocess_config
        )

        # Sequence-level aggregates
        mpjpes.append(metrics.mpjpe())
        mpjves.append(metrics.mpjve())
        jitters.append(metrics.jitter())

        conf = metrics.conf_pred  # expose if EvaluationMetrics stores it
        if conf is not None:
            # Micro-average occlusion rates
            occ_all = (conf < 0.5).any(axis=-1)
            if global_occ_counts_by_joint is None:
                global_occ_counts_by_joint = occ_all.sum(axis=0)
            else:
                global_occ_counts_by_joint += occ_all.sum(axis=0)
            global_total_frames += conf.shape[0]

            conf_included = conf[:, metrics.included_joints, :]
            occ_included = (conf_included < 0.5).any(axis=-1)
            global_total_included_occ += occ_included.sum()
            global_total_included_frames += np.prod(occ_included.shape)

        # Per-joint
        pjpes_by_joint.append(metrics.pjpe_by_joint())
        pjves_by_joint.append(metrics.pjve_by_joint())
        pjpes_by_joint_m.append(metrics.pjpe_by_joint_m())
        pjves_by_joint_m.append(metrics.pjve_by_joint_m())

        # Occlusion-filtered
        mpjpes_o.append(metrics.mpjpe_o())
        mpjves_o.append(metrics.mpjve_o())
        pjpes_by_joint_o.append(metrics.pjpe_by_joint_o())
        pjves_by_joint_o.append(metrics.pjve_by_joint_o())
        jitters_by_joint_v.append(metrics.jitter_by_joint_v())
        jitters_by_joint_m.append(metrics.jitter_by_joint_m())
        jitters_by_joint_o.append(metrics.jitter_by_joint_o())

    # --- Compute occlusion rates ---
    occlusion_rates = None
    occlusion_rates_by_joint = None
    if global_total_frames > 0:
        occlusion_rates = [global_total_included_occ / global_total_included_frames]
        # Wrapping in a list to mimic the structure print_summary_by_joint expects (1 x njoints array)
        occlusion_rates_by_joint = [global_occ_counts_by_joint / global_total_frames]

    # --- Print summaries ---
    kwargs = {'method': args.method, 'model': args.model, 'excluded': EXCLUDED_JOINTS}

    list_of_metrics          = [mpjpes, mpjves, jitters] + ([occlusion_rates] if occlusion_rates else [])
    list_of_metrics_by_joint = [pjpes_by_joint, pjves_by_joint] + ([occlusion_rates_by_joint] if occlusion_rates_by_joint else [])
    list_of_metrics_o        = [mpjpes_o, mpjves_o]
    list_of_metrics_by_joint_o = [pjpes_by_joint_o, pjves_by_joint_o, jitters_by_joint_o]

    print_summary(*list_of_metrics, **kwargs)
    print_summary_by_joint(*list_of_metrics_by_joint, **kwargs)
    print_summary_o(*list_of_metrics_o, **kwargs)
    print_summary_by_joint_o(*list_of_metrics_by_joint_o, **kwargs)

    # New Visible vs Occluded Summary
    print_summary_by_joint_v_m(pjpes_by_joint, pjpes_by_joint_m, 
                               pjves_by_joint, pjves_by_joint_m, 
                               jitters_by_joint_v, jitters_by_joint_m, 
                               **kwargs)


# ---------------------------------------------------------------------------
# Edge-case detection
# ---------------------------------------------------------------------------

def detect_edge_case(eval_metrics, topn: int = 10) -> None:
    """Print the sequences with the highest per-metric error values."""
    assert topn > 0, f"topn must be positive, got {topn}"

    mpjpes, mpjves, mpjpes_o, mpjves_o = eval_metrics

    def top_n(values: list, n: int) -> typing.Tuple[np.ndarray, np.ndarray]:
        arr = np.array(values)
        idx = np.argsort(arr)[-n:][::-1]
        return idx, arr[idx]

    metrics_to_report = [
        ('MPJPE',   *top_n(mpjpes,   topn)),
        ('MPJVE',   *top_n(mpjves,   topn)),
        ('MPJPE_O', *top_n(mpjpes_o, topn)),
        ('MPJVE_O', *top_n(mpjves_o, topn)),
    ]

    print("=" * 80)
    print(f"{'TOP N HIGHEST METRIC VALUES':^80}")
    print("=" * 80)
    print(f"{'Metric':<15} {'Rank':<6} {'Index':<8} {'Value':<12}")
    print("-" * 80)

    for label, indices, values in metrics_to_report:
        for rank, (idx, val) in enumerate(zip(indices, values), start=1):
            print(f"{label:<15} {f'#{rank}':<6} {idx:<8} {val:<12.6f}")
        print("-" * 80)

    print("=" * 80)


# ---------------------------------------------------------------------------
# Visualisation helper
# ---------------------------------------------------------------------------

def load_and_vis(args: argparse.Namespace, file_id: int = 0) -> None:
    assert file_id > 0, "file_id must be a positive integer"

    file_pred = os.path.join(args.model, "CMU", "train", args.method,       f"{file_id}.npz")
    file_gt   = os.path.join(args.model, "CMU", "train", "preprocessed",    f"{file_id}.pkl")

    data_gt   = load_gt(file_gt)
    data_pred = load_kp(file_pred)

    raw_gt = data_gt.get('yolo_keypoints') or data_gt.get('pose_estimation_keypoints')
    assert raw_gt is not None and 'ground_truth' in raw_gt
    points3d_gt   = raw_gt['ground_truth']
    points3d_pred = data_pred['points3d']

    conf = data_pred['conf'] if args.method == 'triang' else None

    nframes, njoints, _ = points3d_gt.shape
    included_joints = [j for j in range(njoints) if j not in EXCLUDED_JOINTS]

    animate(points3d_pred, points3d_gt, f'anim_file_id_{file_id}')

    pred   = points3d_pred[:, included_joints]
    gt     = points3d_gt[:, included_joints].numpy()
    errors = np.linalg.norm(pred - gt, axis=-1) * 100  # m → cm

    if conf is not None:
        conf = conf[:, included_joints]
        plot_joint_errors_with_time_conf(errors, conf, f"anim_{file_id}_plot.png")
    else:
        plot_joint_errors(errors, f"anim_{file_id}_plot.png")

    plot_error_heatmap(errors, f"anim_{file_id}_hm.png")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate 3D keypoint reconstruction against ground truth."
    )
    parser.add_argument(
        '--method', type=str, required=True, choices=('triang', 'openmpl'),
        help="Source of predicted keypoints [triang|openmpl]",
    )
    parser.add_argument(
        '--model', type=str, default='./data/keypoints/yolov8n-pose_protocol_1',
        help="Path to the keypoints folder",
    )
    parser.add_argument(
        '--use_kalman_filter', 
        action='store_true'
    )
    parser.add_argument(
        '--joint_to_filter', type=int, default=None,
        help="Joint ID to evaluate / filter (None → all joints)",
    )
    parser.add_argument(
        '--measurement_noise', type=float, default=None,
        help="Override measurement noise from JSON config",
    )
    parser.add_argument(
        '--process_noise', type=float, default=None,
        help="Override process noise from JSON config",
    )
    parser.add_argument(
        '--output_dir', type=str,
        default='./data/results/two_dimensional_grid_search_results/',
    )
    parser.add_argument(
        '--occlusion_aware', action='store_true',
        help="Enable occlusion-aware filtering",
    )
    parser.add_argument(
        '--overwrite', action='store_true',
        help="Overwrite existing result files",
    )
    parser.add_argument(
        '--filter_name', 
        type=str, 
        default=None
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    assert os.path.exists(args.model), f"Model path does not exist: {args.model}"
    os.makedirs(args.output_dir, exist_ok=True)

    one_method_vs_gt(args, args.output_dir)


if __name__ == '__main__':
    main()