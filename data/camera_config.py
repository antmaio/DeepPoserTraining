"""
Camera-directory configuration loading.

Resolves the virtual-camera directories used by each of the three capture
"protocols" from an optional TOML config file, falling back to
``DEFAULT_CAMERA_PATHS`` for any protocol not specified. All resolved paths
are converted to absolute paths and verified to exist on disk.

For backward compatibility, the resolved paths are also exposed as the
module-level constants ``CAMERA_PATH_PROTOCOL_1``, ``CAMERA_PATH_PROTOCOL_2``,
and ``CAMERA_PATH_PROTOCOL_3``, computed once at import time using the
defaults (i.e. without reading any config file).
"""

import os
from typing import Optional

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

# Default camera directories, keyed by protocol number, used whenever a
# config file is not supplied or omits a given protocol.
DEFAULT_CAMERA_PATHS = {
    1: './data/virtual_cameras/protocol_1',
    2: './data/virtual_cameras/protocol_2',
    3: './data/virtual_cameras/protocol_3',
}

CAMERA_PATH_PROTOCOL_1 = DEFAULT_CAMERA_PATHS[1]
CAMERA_PATH_PROTOCOL_2 = DEFAULT_CAMERA_PATHS[2]
CAMERA_PATH_PROTOCOL_3 = DEFAULT_CAMERA_PATHS[3]


def _normalize_path(path: str) -> str:
    """Return an absolute path for the provided camera directory.

    Args:
        path: A relative or absolute directory path.

    Returns:
        The absolute-path form of ``path`` (does not resolve symlinks or
        verify existence).
    """
    return os.path.abspath(path)


def _validate_camera_paths(camera_paths: dict[int, str]) -> dict[int, str]:
    """Normalize and verify that each protocol's camera directory exists.

    Args:
        camera_paths: Mapping of protocol number to a candidate directory
            path. Mutated in place: each value is replaced with its
            normalized absolute path.

    Returns:
        The same ``camera_paths`` dict, with every path normalized to an
        absolute path.

    Raises:
        FileNotFoundError: If any protocol's (normalized) camera directory
            does not exist on disk.
    """
    for protocol, path in camera_paths.items():
        normalized = _normalize_path(path)
        if not os.path.exists(normalized):
            raise FileNotFoundError(f"Camera path for protocol {protocol} does not exist: {normalized}")
        camera_paths[protocol] = normalized
    return camera_paths


def load_camera_paths(config_path: Optional[str] = None, toml_data: Optional[dict] = None) -> dict[int, str]:
    """Load camera directories from a TOML config, falling back to defaults.

    Looks for a ``[camera]`` table with optional ``protocol_1``,
    ``protocol_2``, ``protocol_3`` keys, either supplied directly via
    ``toml_data`` or read from the file at ``config_path``. Any protocol not
    present in the config falls back to ``DEFAULT_CAMERA_PATHS``. All
    resulting paths are normalized to absolute paths and verified to exist.

    Args:
        config_path: Path to a TOML config file to read, if ``toml_data`` is
            not already supplied. Ignored when ``toml_data`` is given.
        toml_data: Already-parsed TOML data (e.g. from a previous
            ``tomllib.load`` call) to read the ``[camera]`` table from,
            taking priority over ``config_path``.

    Returns:
        Mapping of protocol number (1, 2, 3) to its resolved, absolute,
        existing camera directory path.

    Raises:
        FileNotFoundError: If any resolved camera directory does not exist.
        OSError: If ``config_path`` is given but cannot be opened.
        tomllib.TOMLDecodeError: If the config file is not valid TOML.
    """
    camera_cfg: dict = {}

    if toml_data is not None:
        camera_cfg = toml_data.get('camera', {})
    elif config_path is not None:
        with open(config_path, 'rb') as f:
            camera_cfg = tomllib.load(f).get('camera', {})

    camera_paths = {
        1: camera_cfg.get('protocol_1', DEFAULT_CAMERA_PATHS[1]),
        2: camera_cfg.get('protocol_2', DEFAULT_CAMERA_PATHS[2]),
        3: camera_cfg.get('protocol_3', DEFAULT_CAMERA_PATHS[3]),
    }

    return _validate_camera_paths(camera_paths)


# Backward-compatible module-level constants.
# These are the built-in defaults; the actual config-driven values are
# loaded through ``load_camera_paths()`` when a TOML file is available.
