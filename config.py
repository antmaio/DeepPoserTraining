# Modify to point to datasets
EGOBODY_DIR = "./data/egobody"
SMPL_DIR = "./data/smpl"
SMPLX_DIR = "./data/smplx"
AMASS_DIR = "./data/amass/smplx-n"
DATA_SPLIT_DIR = "../AGRoL/prepare_data/data_split"

# Modify
AS_TESTSET = "cmu" #only used when PROTOCOL = 2

MODE = 'triang'
YOLO_MODEL = 'yolov8n-pose'
# Modify to specify where sfbpe will write its cache
CACHE_DIR = ".cache"