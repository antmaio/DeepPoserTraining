"""
Training entry point.
"""
# External
import argparse
import json
import logging
import os
import shutil

import numpy as np
import tomli
import torch
import torch.nn as nn
from typing import Optional, Union
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

torch.autograd.set_detect_anomaly(True)

# Internal
import data.data_config as bm_C
import anim.data.amass as amass
import anim.models as models
from anim.models.parallel_wrapper import ParallelWrapper
from human_body_prior.body_model.body_model import BodyModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RUN_CONFIG_NAME = 'run_config.json'
_DEFAULT_CONFIG_PATH = 'configs/train.toml'
_DEFAULT_DATAROOT = './data/keypoints/'

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training entry point.

    The config file is given as a plain positional argument named
    ``config`` (e.g. ``python train.py configs/train.toml``), not as a
    ``--config`` option, and defaults to ``_DEFAULT_CONFIG_PATH`` when
    omitted.

    Returns:
        Parsed ``argparse.Namespace`` with a single ``config`` attribute
        holding the path to the TOML config file to load.
    """
    parser = argparse.ArgumentParser(description='Train a motion model.')
    parser.add_argument(
        'config',
        type=str,
        nargs='?',
        default=_DEFAULT_CONFIG_PATH,
        help=f"Path to the training TOML config file (default: '{_DEFAULT_CONFIG_PATH}').",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

class Config:
    """Attribute-based view over a plain dict — mirrors TOML / JSON configs."""

    def __init__(self, dictionary: dict):
        """Copy every key/value pair from ``dictionary`` onto this instance.

        Args:
            dictionary: Flat mapping of config keys to values. Each key
                becomes an attribute of the same name on this object.
        """
        for k, v in dictionary.items():
            setattr(self, k, v)


def get_model_run_config(model_dir: str) -> Optional[dict]:
    """Load the run_config.json saved alongside a model, or return None.

    Args:
        model_dir: Directory containing a previously saved model, expected
            to hold a ``run_config.json`` file (see ``_RUN_CONFIG_NAME``).

    Returns:
        The parsed JSON config as a dict, or ``None`` if the file doesn't
        exist under ``model_dir``.
    """
    path = os.path.join(model_dir, _RUN_CONFIG_NAME)
    if not os.path.exists(path):
        logging.info(f'{path} not found – ignored.')
        return None
    with open(path, 'r') as fp:
        return json.load(fp)


def log_config(cfg_dict: dict, is_train: bool = True, model_cfg_dict: dict = None):
    """Log the configuration state in a graceful, cyan-colored summary.

    Groups known keys under readable headers (Data / Training / Model /
    System / Kalman), logs any remaining keys under "Other", and optionally
    logs a separate model-specific configuration dict.

    Args:
        cfg_dict: The full run configuration to log.
        is_train: If False, omits the "Training" group entirely (useful
            when logging config for inference/evaluation-only runs).
        model_cfg_dict: Optional secondary config (e.g. parsed model
            architecture config) to log under its own "Model Configuration"
            section. Nested dict values are logged under their own
            sub-headers.
    """
    CYAN  = "\033[36m"
    BOLD  = "\033[1m"
    RESET = "\033[0m"

    logging.info(f"{CYAN}--- Configuration Summary ---{RESET}")
    # Group keys for better readability
    groups = {
        "Data":     ["dataset", "yolo_model", "mode", "data_dir", "dataroot", "cache_dir", "skinned_mesh_topology"],
        "Training": ["batch_size", "win_len", "win_overlap", "zero_betas", "max_epoch", "epochs_per_save", "save_dir"],
        "Model":    ["model_config", "resume_dir"],
        "System":   ["device_str", "dtype_str", "dataloader_num_workers"],
        "Kalman":   ["with_kalman_filter", "filter_name"]
    }

    shown_keys = set()
    if not is_train:
        shown_keys.update(groups["Training"])
        del groups["Training"]

    for group_name, keys in groups.items():
        # Only show group if at least one key is present
        relevant_keys = [k for k in keys if k in cfg_dict]
        if not relevant_keys:
            continue

        logging.info(f"{CYAN}{BOLD}[{group_name}]{RESET}")
        for k in relevant_keys:
            val = cfg_dict[k]
            logging.info(f"{CYAN}  {k:25}: {val}{RESET}")
            shown_keys.add(k)

    # Show any remaining keys that weren't in groups
    other_keys = [k for k in cfg_dict if k not in shown_keys and not k.startswith('_')]
    if other_keys:
        logging.info(f"{CYAN}{BOLD}[Other]{RESET}")
        for k in other_keys:
            logging.info(f"{CYAN}  {k:25}: {cfg_dict[k]}{RESET}")

    logging.info(f"{CYAN}----------------------------{RESET}")

    if model_cfg_dict:
        logging.info(f"{CYAN}--- Model Configuration ---{RESET}")
        for k, v in model_cfg_dict.items():
            if isinstance(v, dict):
                logging.info(f"{CYAN}{BOLD}[{k}]{RESET}")
                for sub_k, sub_v in v.items():
                    logging.info(f"{CYAN}  {sub_k:25}: {sub_v}{RESET}")
            else:
                logging.info(f"{CYAN}{BOLD}[Model Info]{RESET}")
                logging.info(f"{CYAN}  {k:25}: {v}{RESET}")
        logging.info(f"{CYAN}----------------------------{RESET}")


def _build_data_dir(cfg_dict: dict) -> str:
    """Construct the dataset root directory from protocol and topology settings.

    The directory is built as ``{dataroot}/{yolo_model}_protocol_{N}`` where
    ``N`` is 1 (for protocols 1 and 2) or 3, with a ``_smplx`` suffix added
    when the configured skinned-mesh topology is SMPLX.

    Args:
        cfg_dict: Run configuration dict. Reads ``yolo_model`` (default
            ``'yolov8n-pose'``), ``protocol`` (default ``1``), ``dataroot``
            (default ``_DEFAULT_DATAROOT``), and ``skinned_mesh_topology``.

    Returns:
        The constructed data directory path (not verified to exist here;
        callers are expected to validate it, e.g. via
        ``os.path.isdir(cfg.data_dir)``).
    """
    yolo_model = cfg_dict.get('yolo_model', 'yolov8n-pose')
    protocol   = cfg_dict.get('protocol', 1)
    str_prot   = 1 if protocol in (1, 2) else 3
    dataroot   = cfg_dict.get('dataroot', _DEFAULT_DATAROOT)
    data_dir   = os.path.join(dataroot, f"{yolo_model}_protocol_{str_prot}")
    if cfg_dict.get('skinned_mesh_topology') == 'smplx':
        data_dir += '_smplx'
    return data_dir


def _build_body_models(cfg: Config, device) -> list:
    """Instantiate the body model(s) needed for the configured topology.

    Args:
        cfg: Run configuration. Reads ``skinned_mesh_topology`` (default
            ``'smpl'``).
        device: Torch device to move the body model(s) to.

    Returns:
        For SMPL topology: a two-element list ``[bm_male, bm_female]`` of
        ``BodyModel`` instances. For any other topology (SMPLX): a
        single-element list holding a frozen neutral SMPLX layer.
    """
    topology = getattr(cfg, 'skinned_mesh_topology', 'smpl')
    if topology == 'smpl':
        bm_male   = BodyModel(bm_fname=bm_C._BM_FNAME_MALE_,   num_betas=bm_C._NUM_BETAS_,
                              num_dmpls=bm_C._NUM_DMPLS_,  dmpl_fname=bm_C._DMPL_FNAME_MALE_).to(device)
        bm_female = BodyModel(bm_fname=bm_C._BM_FNAME_FEMALE_,  num_betas=bm_C._NUM_BETAS_,
                              num_dmpls=bm_C._NUM_DMPLS_,  dmpl_fname=bm_C._DMPL_FNAME_FEMALE_).to(device)
        return [bm_male, bm_female]
    return [amass.get_frozen_smplx_layer(gender='neutral', num_betas=bm_C._NUM_BETAS_)]


def _ensure_scheduler(model, n_batches: int):
    """Guarantee the model has a valid scheduler, creating one if necessary.

    Args:
        model: The (unwrapped) model instance, expected to expose
            ``nbatch``/``set_nbatch`` and ``lr_scheduler``/``set_scheduler``.
        n_batches: Number of batches per epoch, needed to size the
            scheduler if one must be created.
    """
    if not getattr(model, 'nbatch', None):
        model.set_nbatch(n_batches)
    if not getattr(model, 'lr_scheduler', None):
        model.set_scheduler()


# ---------------------------------------------------------------------------
# Training loop helpers
# ---------------------------------------------------------------------------

def _accumulate_losses(accumulator: Optional[dict], batch_losses: dict) -> dict:
    """Append one batch's per-key losses onto a running per-epoch accumulator.

    Args:
        accumulator: Existing dict mapping loss name to a list of per-batch
            numpy values, or ``None`` to start a fresh accumulator.
        batch_losses: Dict mapping loss name to a scalar torch tensor for
            the current batch.

    Returns:
        The (possibly newly created) accumulator dict, with this batch's
        losses appended under each key.
    """
    if accumulator is None:
        return {k: [v.detach().cpu().numpy()] for k, v in batch_losses.items()}
    for k, v in batch_losses.items():
        accumulator[k].append(v.detach().cpu().numpy())
    return accumulator


def _mean_losses(accumulator: dict) -> dict:
    """Reduce a per-epoch loss accumulator to per-key means.

    Args:
        accumulator: Dict mapping loss name to a list of per-batch values.

    Returns:
        Dict mapping loss name to the float mean over all accumulated batches.
    """
    return {k: float(np.mean(vs)) for k, vs in accumulator.items()}


def _log_losses(epoch: int, train_losses: dict, val_losses: dict, writer: SummaryWriter):
    """Log and write to TensorBoard the train/val loss values for one epoch.

    Args:
        epoch: Current epoch index.
        train_losses: Dict mapping loss name to its mean training value.
        val_losses: Dict mapping loss name to its mean validation value.
            Expected to share the same keys as ``train_losses``.
        writer: TensorBoard ``SummaryWriter`` to log scalar pairs to. Any
            ``OSError`` raised while writing (e.g. a transient filesystem
            issue) is silently ignored so training can continue.
    """
    for key in train_losses:
        t, v = train_losses[key], val_losses[key]
        logging.info(f"Epoch={epoch} | {key}: train={t:.6f} | val={v:.6f}")
        try:
            writer.add_scalars(key, {'train': t, 'val': v}, epoch)
        except OSError:
            pass


def _log_gradients(model, epoch: int, writer: SummaryWriter):
    """Log per-parameter weight and gradient histograms to TensorBoard.

    Args:
        model: The (possibly wrapped) model whose ``named_parameters()``
            will be iterated. Only parameters with ``requires_grad=True``
            are logged.
        epoch: Current epoch index, used as the TensorBoard step.
        writer: TensorBoard ``SummaryWriter`` to log histograms/scalars to.
    """
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        writer.add_histogram(f'Weights/{name}', param.data.cpu().numpy(), epoch)
        if param.grad is not None:
            writer.add_histogram(f'Gradients/{name}', param.grad.data.cpu().numpy(), epoch)
            writer.add_scalar(f'GradNorm/{name}', param.grad.data.norm().item(), epoch)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Run the full training entry point: config loading, setup, and the training loop.

    Behavior overview:
      1. Parse the ``config`` CLI argument and load the TOML config.
      2. If ``resume_dir`` is set, merge the saved run config with the new
         TOML (new values take priority) and resume from the last checkpoint;
         otherwise create a new model/run directory and start from scratch.
      3. Build datasets/dataloaders, body models, optimizer scheduler, and
         (optionally) multi-GPU wrapping.
      4. Run the train/validate loop up to ``max_epoch``, periodically
         checkpointing and logging losses/gradients to TensorBoard.
      5. On ``KeyboardInterrupt`` or ``models.StopTrainingException``, stop
         gracefully; a final checkpoint is always saved before returning.
    """
    args = parse_args()
    config_path = args.config
    with open(config_path, 'rb') as fp:
        cfg_dict = tomli.load(fp)

    # If the user provides `resume_dir` in the TOML, we resume. Else train from scratch.
    resume_dir = cfg_dict.get('resume_dir')
    training_from_scratch = resume_dir is None

    if not training_from_scratch:
        assert os.path.isdir(resume_dir), f"resume_dir '{resume_dir}' is not a valid directory."
        model_dir  = resume_dir
        last_epoch = models.get_last_model_epoch(model_dir)
        run_cfg = get_model_run_config(model_dir)
        if run_cfg is None:
            raise FileNotFoundError(f"Missing {_RUN_CONFIG_NAME} in '{model_dir}'.")
        # Merge new parameters onto existing ones, with TOML having priority
        cfg_dict = {**run_cfg, **cfg_dict}
        cfg_dict['model_config'] = os.path.join(model_dir, 'model_config.toml')

    # ---- system args ----
    device_str  = cfg_dict.get('device_str', 'cuda')
    dtype_str   = cfg_dict.get('dtype_str', 'float32')
    num_workers = cfg_dict.get('dataloader_num_workers', 0)

    cfg_dict['device_str']             = device_str
    cfg_dict['dtype_str']              = dtype_str
    cfg_dict['dataloader_num_workers'] = num_workers

    device = torch.device(device_str)
    dtype  = torch.float32  # Only supported float32 for now

    if training_from_scratch:
        # Derive protocol and data_dir from dataset string
        dataset_name = cfg_dict.get('dataset', 'amass-p1')
        if 'amass' in dataset_name:
            cfg_dict['protocol'] = int(dataset_name.split('-p')[-1])
        assert cfg_dict.get('protocol') in (1, 2, 3), "Protocol must be 1, 2, or 3."
        cfg_dict.setdefault('data_dir', _build_data_dir(cfg_dict))

        model, model_dir = models.create_new_model_and_dir(cfg_dict['model_config'],
                                                           cfg_dict.get('save_dir', './saves'))
        last_epoch = -1
        logging.info(f"Training '{model_dir}' from scratch.")

        cfg_dict['model_config'] = os.path.join(model_dir, 'model_config.toml')

        with open(os.path.join(model_dir, _RUN_CONFIG_NAME), 'w') as fp:
            json.dump(cfg_dict, fp, indent=4)

    else:
        # nbatch will be reset once the DataLoader is ready
        model = models.load_model(model_dir, last_epoch, 0, device=device)
        logging.info(f"Resuming '{model_dir}' from epoch {last_epoch}.")

    cfg = Config(cfg_dict)
    assert os.path.isdir(cfg.data_dir), f"data_dir '{cfg.data_dir}' does not exist."

    log_config(cfg_dict)

    # ---- body models ----
    body_models = _build_body_models(cfg, device)

    # ---- datasets ----
    common = dict(
        cfg=cfg,
        dataset_str=getattr(cfg, 'dataset', 'amass-p1'),
        topology=getattr(cfg, 'skinned_mesh_topology', 'smpl'),
        ratio=getattr(cfg, 'data_ratio', None),
        win_len=getattr(cfg, 'win_len', 40),
        win_overlap=getattr(cfg, 'win_overlap', 5),
        zero_betas=getattr(cfg, 'zero_betas', False),
        dtype=dtype,
    )
    train_dataset = amass.get_dataset(**common, split='train')
    val_dataset   = amass.get_dataset(**common, split='valid')

    batch_size       = getattr(cfg, 'batch_size', 200)
    loader_kwargs    = dict(batch_size=batch_size, num_workers=num_workers, drop_last=True)
    train_dataloader = DataLoader(train_dataset, shuffle=True,  **loader_kwargs)
    val_dataloader   = DataLoader(val_dataset,   shuffle=True,  **loader_kwargs)

    # ---- scheduler (needs n_batches) ----
    _ensure_scheduler(model, len(train_dataloader))

    # ---- optional Kalman filter file ----
    if getattr(cfg, 'with_kalman_filter', False):
        src = 'utils/kalman_parameters.json'
        dst = os.path.join(model_dir, 'kalman_parameters.json')
        if os.path.exists(src):
            shutil.copy(src, dst)
            logging.info(f'Copied Kalman parameters → {dst}')
        else:
            logging.warning(f'Kalman parameters not found at {src}')

    # ---- optional body-model injection ----
    gender = 'male' if getattr(cfg, 'skinned_mesh_topology', 'smpl') == 'smpl' else 'neutral'
    try:
        model.set_body_model(body_models, gender)
    except AttributeError:
        pass

    # ---- multi-GPU ----
    if torch.cuda.device_count() > 1:
        logging.info(f'Using {torch.cuda.device_count()} GPUs.')
        model = nn.DataParallel(model)

    model = ParallelWrapper(model).to(device, dtype)

    models.load_opt(model_dir, model, last_epoch)
    models.load_sched(model_dir, model, last_epoch)

    logging.info(f'Batches per epoch — train: {len(train_dataloader)} | val: {len(val_dataloader)}')

    # ---- training loop ----
    writer        = SummaryWriter(os.path.join(model_dir, 'logs'))
    epoch         = last_epoch + 1
    step          = 0
    max_epoch     = getattr(cfg, 'max_epoch', 400)
    epochs_per_save = getattr(cfg, 'epochs_per_save', 10)
    error_last_save = False

    try:
        while epoch < max_epoch:
            logging.info(f'========== Epoch {epoch} ==========')

            # -- train --
            model.train()
            train_losses = None
            for batch in train_dataloader:
                with torch.no_grad():
                    model_input, model_target = models.batch_to_model_input_and_target(
                        batch, device, dtype, mode3d=model.mode3d)
                step += 1
                model.reset()
                train_losses = _accumulate_losses(
                    train_losses,
                    model.forward_pass(model_input, model_target, optimise=True),
                )

            # -- validate --
            model.eval()
            val_losses = None
            for batch in val_dataloader:
                with torch.no_grad():
                    model_input, model_target = models.batch_to_model_input_and_target(
                        batch, device, dtype, mode3d=model.mode3d)
                    model.reset()
                    val_losses = _accumulate_losses(
                        val_losses,
                        model.forward_pass(model_input, model_target),
                    )

            train_losses = _mean_losses(train_losses)
            val_losses   = _mean_losses(val_losses)
            _log_losses(epoch, train_losses, val_losses, writer)
            _log_gradients(model, epoch, writer)

            # -- checkpoint --
            if error_last_save or (epoch % epochs_per_save == 0):
                error_last_save = False
                try:
                    models.save_model(model, model_dir, epoch)
                except OSError:
                    logging.warning(f'OSError while saving at epoch {epoch}; will retry next epoch.')
                    error_last_save = True

            model.epoch_end(epoch, train_losses, val_losses)
            epoch += 1

    except KeyboardInterrupt:
        logging.info('KeyboardInterrupt — saving before exit.')
    except models.StopTrainingException as exc:
        logging.info(f'StopTrainingException: {exc}')

    models.save_model(model, model_dir, epoch)
    logging.info(f"Finished training '{model_dir}' at epoch {epoch}.")


if __name__ == '__main__':
    main()