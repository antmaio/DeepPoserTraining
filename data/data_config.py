import enum

OUTPUT_DIR  = "./data/keypoints/"

# SMPL 
SMPL_JOINTS = {'hips' : 0, 'leftUpLeg' : 1, 'rightUpLeg' : 2, 'spine' : 3, 'leftLeg' : 4, 'rightLeg' : 5,
                'spine1' : 6, 'leftFoot' : 7, 'rightFoot' : 8, 'spine2' : 9, 'leftToeBase' : 10, 'rightToeBase' : 11, 
                'neck' : 12, 'leftShoulder' : 13, 'rightShoulder' : 14, 'head' : 15, 'leftArm' : 16, 'rightArm' : 17,
                'leftForeArm' : 18, 'rightForeArm' : 19, 'leftHand' : 20, 'rightHand' : 21}
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 12, 12, 13, 14, 16, 17, 18, 19]

#YOLO
class YoloJoints(enum.IntEnum):
    """Enum mapping YOLO pose estimation joint names to their indices"""
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
    NUM_JTS = 17

YOLO_PARENTS = (
    -1,  # NOSE
    0,   # LEFT_EYE
    0,   # RIGHT_EYE
    1,   # LEFT_EAR
    2,   # RIGHT_EAR
    0,   # LEFT_SHOULDER (connected to NOSE)
    0,   # RIGHT_SHOULDER
    5,   # LEFT_ELBOW
    6,   # RIGHT_ELBOW
    7,   # LEFT_WRIST
    8,   # RIGHT_WRIST
    5,   # LEFT_HIP (usually connected to LEFT_SHOULDER)
    6,   # RIGHT_HIP
    11,  # LEFT_KNEE
    12,  # RIGHT_KNEE
    13,  # LEFT_ANKLE
    14   # RIGHT_ANKLE
)