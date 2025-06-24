import os

_SUPPORT_DATA_ = "../AGRoL/support_data"

_NUM_BETAS_ = 16 # number of body parameters
_NUM_DMPLS_ = 8 # number of DMPL parameters
_BM_FNAME_MALE_ = os.path.join(_SUPPORT_DATA_, 'body_models/smplh/{}/model.npz'.format('male'))
_BM_FNAME_FEMALE_ = os.path.join(_SUPPORT_DATA_, 'body_models/smplh/{}/model.npz'.format('female'))

_DMPL_FNAME_MALE_ = os.path.join(_SUPPORT_DATA_, 'body_models/dmpls/{}/model.npz'.format('male'))
_DMPL_FNAME_FEMALE_ = os.path.join(_SUPPORT_DATA_, 'body_models/dmpls/{}/model.npz'.format('female'))