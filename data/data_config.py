import enum
import os

_SUPPORT_DATA_ = os.environ.get('SUPPORT_DATA_DIR', './data/support_data')
_SMPLX_DIR_    = os.environ.get('SMPLX_DIR',        './data/body_models/smplx')
_AMASS_DIR_    = os.environ.get('AMASS_DIR',        '../AMASS')

_NUM_BETAS_ = 16   # number of shape parameters
_NUM_DMPLS_ = 8    # number of DMPL parameters

_BM_FNAME_MALE_    = os.path.join(_SUPPORT_DATA_, 'body_models/smplh/male/model.npz')
_BM_FNAME_FEMALE_  = os.path.join(_SUPPORT_DATA_, 'body_models/smplh/female/model.npz')
_DMPL_FNAME_MALE_  = os.path.join(_SUPPORT_DATA_, 'body_models/dmpls/male/model.npz')
_DMPL_FNAME_FEMALE_= os.path.join(_SUPPORT_DATA_, 'body_models/dmpls/female/model.npz')

# ---------------------------------------------------------------------------
# SMPL joint definitions
# ---------------------------------------------------------------------------

SMPL_JOINTS = {
    'hips': 0, 'leftUpLeg': 1, 'rightUpLeg': 2, 'spine': 3,
    'leftLeg': 4, 'rightLeg': 5, 'spine1': 6, 'leftFoot': 7,
    'rightFoot': 8, 'spine2': 9, 'leftToeBase': 10, 'rightToeBase': 11,
    'neck': 12, 'leftShoulder': 13, 'rightShoulder': 14, 'head': 15,
    'leftArm': 16, 'rightArm': 17, 'leftForeArm': 18, 'rightForeArm': 19,
    'leftHand': 20, 'rightHand': 21,
}

SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 12, 12, 13, 14, 16, 17, 18, 19]

# ---------------------------------------------------------------------------
# SMPL-X joint definitions
# ---------------------------------------------------------------------------

class SmplxJoints(enum.IntEnum):
    PELVIS        = 0
    LEFT_HIP      = 1
    RIGHT_HIP     = 2
    SPINE_1       = 3
    LEFT_KNEE     = 4
    RIGHT_KNEE    = 5
    SPINE_2       = 6
    LEFT_ANKLE    = 7
    RIGHT_ANKLE   = 8
    SPINE_3       = 9
    LEFT_FOOT     = 10
    RIGHT_FOOT    = 11
    NECK          = 12
    LEFT_COLLAR   = 13
    RIGHT_COLLAR  = 14
    HEAD          = 15
    LEFT_SHOULDER = 16
    RIGHT_SHOULDER= 17
    LEFT_ELBOW    = 18
    RIGHT_ELBOW   = 19
    LEFT_WRIST    = 20
    RIGHT_WRIST   = 21
    NUM_JTS       = 22

SMPLX_BODY_HIERARCHY = (
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19
)

SMPLX_UPPER_JOINTS = [
    SmplxJoints.SPINE_1,   SmplxJoints.SPINE_2,   SmplxJoints.SPINE_3,
    SmplxJoints.NECK,      SmplxJoints.LEFT_COLLAR, SmplxJoints.RIGHT_COLLAR,
    SmplxJoints.HEAD,
    SmplxJoints.LEFT_SHOULDER,  SmplxJoints.RIGHT_SHOULDER,
    SmplxJoints.LEFT_ELBOW,     SmplxJoints.RIGHT_ELBOW,
    SmplxJoints.LEFT_WRIST,     SmplxJoints.RIGHT_WRIST,
]

SMPLX_LOWER_JOINTS = [
    SmplxJoints.PELVIS,
    SmplxJoints.LEFT_HIP,    SmplxJoints.RIGHT_HIP,
    SmplxJoints.LEFT_KNEE,   SmplxJoints.RIGHT_KNEE,
    SmplxJoints.LEFT_ANKLE,  SmplxJoints.RIGHT_ANKLE,
    SmplxJoints.LEFT_FOOT,   SmplxJoints.RIGHT_FOOT,
]


FPS = 60.0

# ---------------------------------------------------------------------------
# YOLO joint definitions
# ---------------------------------------------------------------------------

class YoloJoints(enum.IntEnum):
    """Enum mapping YOLO pose-estimation joint names to their indices."""
    NOSE           = 0
    LEFT_EYE       = 1
    RIGHT_EYE      = 2
    LEFT_EAR       = 3
    RIGHT_EAR      = 4
    LEFT_SHOULDER  = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW     = 7
    RIGHT_ELBOW    = 8
    LEFT_WRIST     = 9
    RIGHT_WRIST    = 10
    LEFT_HIP       = 11
    RIGHT_HIP      = 12
    LEFT_KNEE      = 13
    RIGHT_KNEE     = 14
    LEFT_ANKLE     = 15
    RIGHT_ANKLE    = 16
    NUM_JTS        = 17


YOLO_PARENTS = (
    -1,  # NOSE
     0,  # LEFT_EYE
     0,  # RIGHT_EYE
     1,  # LEFT_EAR
     2,  # RIGHT_EAR
     0,  # LEFT_SHOULDER
     0,  # RIGHT_SHOULDER
     5,  # LEFT_ELBOW
     6,  # RIGHT_ELBOW
     7,  # LEFT_WRIST
     8,  # RIGHT_WRIST
     5,  # LEFT_HIP
     6,  # RIGHT_HIP
    11,  # LEFT_KNEE
    12,  # RIGHT_KNEE
    13,  # LEFT_ANKLE
    14,  # RIGHT_ANKLE
)

YOLO_UPPER_JOINTS = [
    YoloJoints.NOSE,
    YoloJoints.LEFT_SHOULDER,
    YoloJoints.RIGHT_SHOULDER,
    YoloJoints.LEFT_ELBOW,
    YoloJoints.RIGHT_ELBOW,
    YoloJoints.LEFT_WRIST,
    YoloJoints.RIGHT_WRIST,
]

YOLO_LOWER_JOINTS = [
    YoloJoints.LEFT_HIP,
    YoloJoints.RIGHT_HIP,
    YoloJoints.LEFT_KNEE,
    YoloJoints.RIGHT_KNEE,
    YoloJoints.LEFT_ANKLE,
    YoloJoints.RIGHT_ANKLE,
]
