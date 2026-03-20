"""
Inspired from https://github.com/georgedf1/sfbpe/tree/main
"""
# External
import logging
import os
import shutil

import tomli
import torch

# Internal
from .base import * #TODO remove * for safety 
from .hmd_poser_ext_hmr_head_centered import HMDPoserExtHeadCentered
from utils.utils_option import generate_time_str

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: tuple[type[BaseModel], ...] = (
    HMDPoserExtHeadCentered,
    # Add new models here
)
_MODEL_DICT: dict[str, type[BaseModel]] = {
    m.model_str(): m for m in _MODEL_REGISTRY
}

_MODEL_CONFIG_NAME       = 'model_config.toml'
_STATE_DICTS_DIR         = 'model_state_dicts'


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _create_model(model_cfg_path: str) -> BaseModel:
    assert os.path.isfile(model_cfg_path), f"Config not found: {model_cfg_path}"
    with open(model_cfg_path, 'rb') as fp:
        cfg = tomli.load(fp)
    assert 'model_str'  in cfg, "Model TOML must have a 'model_str' entry."
    assert 'model_args' in cfg, "Model TOML must have a '[model_args]' section."
    model_str = cfg['model_str']
    assert '_' not in model_str, "model_str must not contain underscores."
    assert model_str in _MODEL_DICT, f"Unknown model '{model_str}'."
    return _MODEL_DICT[model_str](**cfg['model_args'])


def _state_dict_dir(model_dir: str) -> str:
    return os.path.join(model_dir, _STATE_DICTS_DIR)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_new_model_and_dir(
    model_cfg_path: str,
    save_dir: str,
) -> tuple[BaseModel, str]:
    """Create a fresh model and its associated directory."""
    model = _create_model(model_cfg_path)
    model_dir = os.path.join(save_dir, f"{model.model_str()}_{generate_time_str()}")
    os.makedirs(model_dir)
    shutil.copyfile(model_cfg_path, os.path.join(model_dir, _MODEL_CONFIG_NAME))
    os.mkdir(_state_dict_dir(model_dir))
    return model, model_dir


def save_model(model: BaseModel, model_dir: str, epoch: int):
    """Save model weights, optimiser state, and scheduler state for *epoch*."""
    sd_dir = _state_dict_dir(model_dir)
    torch.save(model.state_dict(),         os.path.join(sd_dir, f'{epoch}.pt'))
    torch.save(model.optim.state_dict(),   os.path.join(sd_dir, f'optimizer_{epoch}.pt'))
    torch.save(model.lr_scheduler.state_dict(), os.path.join(sd_dir, f'scheduler_{epoch}.pt'))


def get_last_model_epoch(model_dir: str) -> int:
    """Return the highest epoch number for which a checkpoint exists."""
    sd_dir = _state_dict_dir(model_dir)
    epochs = [
        int(f[:-3])
        for f in os.listdir(sd_dir)
        if f.endswith('.pt') and f[:-3].isdigit()
    ]
    assert epochs, f"No model checkpoints found in '{sd_dir}'."
    return max(epochs)


def load_model(model_dir: str, epoch: int, nbatch: int, device) -> BaseModel:
    """
    Load a model from a checkpoint, trying multiple key-adaptation strategies.
    """
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

    checkpoint_path = os.path.join(_state_dict_dir(model_dir), f'{epoch}.pt')
    assert os.path.isfile(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"

    model = _create_model(os.path.join(model_dir, _MODEL_CONFIG_NAME))
    model.set_nbatch(nbatch)
    model.set_scheduler()

    strategies = [
        ('direct',            lambda sd: sd),
        ('old-pytorch-keys',  lambda sd: adapt_ckpt_to_model_key(model, sd)),
        ('remove-module',     lambda sd: adapt_ckpt_module_to_model_key(model, sd)),
    ]

    checkpoint = torch.load(checkpoint_path, weights_only=True)
    for name, transform_fn in strategies:
        try:
            logging.info(f'Loading weights ({name})…')
            model.load_state_dict(transform_fn(checkpoint))
            logging.info(f'Success with strategy: {name}.')
            return model
        except Exception:
            pass

    raise RuntimeError(f"All loading strategies failed for checkpoint: {checkpoint_path}")


def load_opt(model_dir: str, model, epoch: int):
    """Load optimiser state into *model*, moving tensors to the model's device."""
    sd_dir = _state_dict_dir(model_dir)
    opt_path = os.path.join(sd_dir, f'optimizer_{epoch}.pt')
    try:
        assert os.path.isfile(opt_path)
        actual = model.mmodel if hasattr(model, 'mmodel') else model
        device = next(actual.parameters()).device
        state  = torch.load(opt_path, map_location='cpu')
        for s in state['state'].values():
            for k, v in s.items():
                if torch.is_tensor(v):
                    s[k] = v.to(device)
        actual.optim.load_state_dict(state)
        logging.info('Optimiser loaded.')
    except Exception as exc:
        logging.info(f'Could not load optimiser: {exc}')


def load_sched(model_dir: str, model, epoch: int):
    """Load scheduler state into *model*."""
    sd_dir   = _state_dict_dir(model_dir)
    sch_path = os.path.join(sd_dir, f'scheduler_{epoch}.pt')
    try:
        assert os.path.isfile(sch_path)
        actual = model.mmodel if hasattr(model, 'mmodel') else model
        actual.lr_scheduler.load_state_dict(
            torch.load(sch_path, map_location='cpu')
        )
        logging.info('Scheduler loaded.')
    except Exception as exc:
        logging.info(f'Could not load scheduler: {exc}')


# ---------------------------------------------------------------------------
# Checkpoint key-adaptation utilities
# ---------------------------------------------------------------------------

def adapt_ckpt_module_to_model_key(model: BaseModel, state_dict: dict) -> dict:
    """Remove 'module.' prefixes added by DataParallel wrapping."""
    return {k.replace('module.', ''): v for k, v in state_dict.items()}


def adapt_ckpt_to_model_key(model: BaseModel, checkpoint: dict) -> dict:
    """
    Translate PyTorch 2.3.x weight_norm parametrization keys to 2.0.x convention.
    """
    new_sd = {}
    for k, v in checkpoint.items():
        if '.parametrizations.' in k and (k.endswith('.original0') or k.endswith('.original1')):
            suffix = '_g' if k.endswith('.original0') else '_v'
            for weight_name in ('weight_hh_l0', 'weight_ih_l0'):
                old_pattern = f'.parametrizations.{weight_name}.original{"0" if suffix == "_g" else "1"}'
                if old_pattern in k:
                    new_k = k.replace(old_pattern, f'.{weight_name}{suffix}')
                    break
            else:
                new_k = k  # fallback: keep as-is
            new_sd[new_k] = v
        else:
            new_sd[k] = v
    return new_sd




# ---------------------------------------------------------------------------
# Device-transfer helpers (used when loading from CPU checkpoints)
# ---------------------------------------------------------------------------

def _move_optimizer_to_device(model, device):
    for state in model.optim.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    return model


def _move_scheduler_to_device(model, device):
    for state in model.lr_scheduler.optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    return model
