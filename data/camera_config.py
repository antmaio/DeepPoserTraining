import os

#Path of virtual cameras
CAMERA_PATH_PROTOCOL_1 = './data/virtual_cameras/protocol_1'
CAMERA_PATH_PROTOCOL_2 = './data/virtual_cameras/protocol_2'
CAMERA_PATH_PROTOCOL_3 = './data/virtual_cameras/protocol_3'

assert os.path.exists(CAMERA_PATH_PROTOCOL_1), f"Path not found: {CAMERA_PATH_PROTOCOL_1}"
assert os.path.exists(CAMERA_PATH_PROTOCOL_2), f"Path not found: {CAMERA_PATH_PROTOCOL_2}"
assert os.path.exists(CAMERA_PATH_PROTOCOL_3), f"Path not found: {CAMERA_PATH_PROTOCOL_3}"