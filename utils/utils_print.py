import typing
import numpy as np
from anim.data.amass import YoloJoints

#Pretty print
def print_summary(mpjpes:typing.List, mpjves:typing.List, jitters:typing.List, occlusion_rates:typing.List=None, **kwargs):

    method = kwargs["method"]
    model = kwargs["model"]

    def format_metric(name, values):
        mean = np.nanmean(values)
        std = np.nanstd(values)
        return f"{name:<20}: {mean:.2f} ± {std:.2f}"

    print(f"\n📊 Evaluation Summary for {method} with 2D data from {model}")
    print("-" * 40)
    print(format_metric("MPJPE [cm]", mpjpes))
    print(format_metric("MPJVE [cm/s]", mpjves))
    print(format_metric("Jitter ratio", jitters))
    if occlusion_rates is not None: print(format_metric("Occlusion Rate [%]", [v * 100 for v in occlusion_rates]))
    print("-" * 40)

def print_summary_by_joint(pjpes_by_joint: typing.List, pjves_by_joint: typing.List, occlusion_rates_by_joint:typing.List=None,  excluded:dict={}, **kwargs):    
    method = kwargs["method"]
    model = kwargs["model"]

    pjpes_by_joint           = np.array(pjpes_by_joint)
    pjves_by_joint           = np.array(pjves_by_joint)
    if occlusion_rates_by_joint is not None:    occlusion_rates_by_joint = np.array(occlusion_rates_by_joint)

    # Joint names and indices to keep (exclude eyes and ears)
    joint_indices = [j for j in range(len(YoloJoints)) if j not in excluded and j != YoloJoints.NUM_JTS]
    joint_names = [YoloJoints(j).name.replace("_", " ").title() for j in joint_indices]


    print(f"\n📊 Evaluation Summary for {method} with 2D data from {model}")
    print("-" * 40)
    print(f"{'Joint':<20} {'PJPE [cm]':>15} {'PJVE [cm/s]':>15} {'Occlusion rate [%]':>15}")
    print("-" * 40)


    for j,jname in zip(joint_indices, joint_names):
        pjpe_mean, pjpe_std = np.nanmean(pjpes_by_joint[:, j]), np.nanstd(pjpes_by_joint[:, j])
        pjve_mean, pjve_std = np.nanmean(pjves_by_joint[:, j]), np.nanstd(pjves_by_joint[:, j])

        if occlusion_rates_by_joint is not None:
            occl_rate_mean, occl_rate_std = np.mean(occlusion_rates_by_joint[:, j] * 100), np.std(occlusion_rates_by_joint[:, j] * 100)
            print(f"{jname:<20} {pjpe_mean:>7.2f} ± {pjpe_std:<5.2f} {pjve_mean:>7.2f} ± {pjve_std:<5.2f} {occl_rate_mean:>7.2f} ± {occl_rate_std:<5.2f}")
        else:
            print(f"{jname:<20} {pjpe_mean:>7.2f} ± {pjpe_std:<5.2f} {pjve_mean:>7.2f} ± {pjve_std:<5.2f}")

    print("-" * 40)

def print_summary_o(mpjpes_o:typing.List, mpjves_o:typing.List, **kwargs):
    method = kwargs["method"]
    model = kwargs["model"]

    def format_metric(name, values):
        mean = np.nanmean(values)
        std = np.nanstd(values)
        return f"{name:<20}: {mean:.2f} ± {std:.2f}"

    print(f"\n📊 Evaluation Summary for {method} with 2D data from {model} without discarding occlusion")
    print("-" * 40)
    print(format_metric("MPJPE [cm]", mpjpes_o))
    print(format_metric("MPJVE [cm/s]", mpjves_o))
    #print(format_metric("Jitter ratio", jitters))
    print("-" * 40)

def print_summary_by_joint_o(pjpes_by_joint_o: typing.List, pjves_by_joint_o: typing.List, jitter_by_joint_o:typing.List, excluded:dict={}, **kwargs):
    method = kwargs["method"]
    model = kwargs["model"]

    pjpes_by_joint           = np.array(pjpes_by_joint_o)
    pjves_by_joint           = np.array(pjves_by_joint_o)
    jitter_by_joint          = np.array(jitter_by_joint_o)

    # Joint names and indices to keep (exclude eyes and ears)
    joint_indices = [j for j in range(len(YoloJoints)) if j not in excluded and j != YoloJoints.NUM_JTS]
    joint_names = [YoloJoints(j).name.replace("_", " ").title() for j in joint_indices]


    print(f"\n📊 Evaluation Summary for {method} with 2D data from {model} without discard occlusion")
    print("-" * 40)
    print(f"{'Joint':<20} {'PJPE [cm]':>15} {'PJVE [cm/s]':>15} {'Jitter ratio':>15}")
    print("-" * 40)

    print("-" * 40)
    for j,jname in zip(joint_indices, joint_names):
        pjpe_mean, pjpe_std = np.nanmean(pjpes_by_joint[:, j]), np.nanstd(pjpes_by_joint[:, j])
        pjve_mean, pjve_std = np.nanmean(pjves_by_joint[:, j]), np.nanstd(pjves_by_joint[:, j])
        jitter_mean, jitter_std = np.nanmean(jitter_by_joint[:, j]), np.nanstd(jitter_by_joint[:, j])
        print(f"{jname:<20} {pjpe_mean:>7.2f} ± {pjpe_std:<5.2f} {pjve_mean:>7.2f} ± {pjve_std:<5.2f} {jitter_mean:>7.2f} ± {jitter_std:<5.2f}")

    print("-" * 40)

def print_summary_by_joint_v_m(pjpes_v: np.ndarray, pjpes_m: np.ndarray, 
                               pjves_v: np.ndarray, pjves_m: np.ndarray, 
                               jitters_v: np.ndarray, jitters_m: np.ndarray, 
                               excluded: dict = {}, **kwargs):
    """Pretty print comparing visible (v) and missing/occluded (m) metrics per joint."""
    method = kwargs["method"]
    model = kwargs["model"]

    pjpes_v = np.array(pjpes_v)
    pjpes_m = np.array(pjpes_m)
    pjves_v = np.array(pjves_v)
    pjves_m = np.array(pjves_m)
    jitters_v = np.array(jitters_v)
    jitters_m = np.array(jitters_m)

    # Index 1 is njoints (shape is [num_files, njoints])
    n_joints_available = pjpes_v.shape[1] if pjpes_v.ndim > 1 else len(YoloJoints)
    joint_indices = [j for j in range(n_joints_available) if j not in excluded and j != YoloJoints.NUM_JTS]
    joint_names = [YoloJoints(j).name.replace("_", " ").title() for j in joint_indices]

    print(f"\n📊 Evaluation Summary (Visible vs Occluded) for {method} | Model: {model}")
    print("=" * 110)
    print(f"{'Joint':<20} | {'PJPE [cm] (V/M)':^25} | {'PJVE [cm/s] (V/M)':^25} | {'Jitter (V/M)':^25}")
    print("-" * 110)

    for j, jname in zip(joint_indices, joint_names):
        pe_v_mean, pe_m_mean = np.nanmean(pjpes_v[:, j]), np.nanmean(pjpes_m[:, j])
        ve_v_mean, ve_m_mean = np.nanmean(pjves_v[:, j]), np.nanmean(pjves_m[:, j])
        ji_v_mean, ji_m_mean = np.nanmean(jitters_v[:, j]), np.nanmean(jitters_m[:, j])

        print(f"{jname:<20} | {pe_v_mean:>8.2f} / {pe_m_mean:<8.2f} | {ve_v_mean:>8.2f} / {ve_m_mean:<8.2f} | {ji_v_mean:>8.2f} / {ji_m_mean:<8.2f}")

    print("=" * 110)
