"""
Inspired from https://github.com/georgedf1/sfbpe/tree/main
"""
# External
import os
import shutil
import tomli
#Internal
from .base import *
from .hmd_poser_ext import HMDPoserExt
from utils.utils_option import generate_time_str
# --- Add new models here ---
__MODEL_REGISTRY = (HMDPoserExt,)
__MODEL_DICT = {model.model_str(): model for model in __MODEL_REGISTRY}

__MODEL_CONFIG_NAME = 'model_config.toml'
__MODEL_STATE_DICTS_DIR_NAME = 'model_state_dicts'


def __create_model(model_cfg_path: str) -> BaseModel:
    assert os.path.isfile(model_cfg_path)
    with open(model_cfg_path, 'rb') as fp:
        model_cfg = tomli.load(fp)
    assert 'model_str' in model_cfg, "Model TOML must have 'model_str' entry"
    assert 'model_args' in model_cfg, "Model TOML must have '[model_args]'"
    model_str = model_cfg['model_str']
    model_args = model_cfg['model_args']
    assert '_' not in model_str, 'model_str containing underscore is forbidden'
    assert model_str in __MODEL_DICT, f"Could not find model with name '{model_str}'"
    return __MODEL_DICT[model_str](**model_args)


def create_new_model_and_dir(model_cfg_path: str, save_dir: str) -> tuple[BaseModel, str]:
    model = __create_model(model_cfg_path)
    time_str = generate_time_str()
    model_dir = os.path.join(save_dir, f"{model.model_str()}_{time_str}")
    os.makedirs(model_dir)
    model_cfg_copy_path = os.path.join(model_dir, __MODEL_CONFIG_NAME)
    shutil.copyfile(model_cfg_path, model_cfg_copy_path)
    model_state_dict_dir = os.path.join(model_dir, __MODEL_STATE_DICTS_DIR_NAME)
    os.mkdir(model_state_dict_dir)
    return model, model_dir


def save_model(model: BaseModel, model_dir: str, epoch: int):
    model_state_dict_save_path = os.path.join(model_dir, __MODEL_STATE_DICTS_DIR_NAME, str(epoch) + '.pt')
    torch.save(model.state_dict(), model_state_dict_save_path)


def get_last_model_epoch(model_dir) -> int:
    model_state_dict_dir = os.path.join(model_dir, __MODEL_STATE_DICTS_DIR_NAME)
    checkpoints = list(map(lambda s: int(s[:-3]),
                           filter(lambda f: f.endswith('.pt'),
                                  os.listdir(model_state_dict_dir))))
    assert len(checkpoints) > 0, f"No model saves available for '{model_dir}'"
    checkpoints.sort()  # sort by ascending epoch
    # Fetch newest if epoch unspecified
    return checkpoints[-1]


def load_model(model_dir: str, epoch: int):
    model_state_dict_dir = os.path.join(model_dir, __MODEL_STATE_DICTS_DIR_NAME)
    checkpoint_path = os.path.join(model_state_dict_dir, str(epoch) + '.pt')
    assert os.path.isfile(checkpoint_path)
    model_cfg_path = os.path.join(model_dir, __MODEL_CONFIG_NAME)
    model = __create_model(model_cfg_path)
    state_dict = _fix_key_names(torch.load(checkpoint_path, weights_only=True))
    
    model.load_state_dict(state_dict)
    return model

# fix key names for hmdnemo
def _fix_key_names(state_dict: dict):
    return {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
