from .encoder import EEGEncoder
from .augmentations import (
    EEGAugmentation,
    GaussianNoise,
    AmplitudeScale,
    ChannelDropout,
    TemporalShift,
    FrequencyMask,
    TemporalCrop,
    SignFlip,
    get_default_augmentation,
    get_strong_augmentation,
    get_mae_mask,
)
from .vicreg import VICReg, vicreg_loss
from .byol import BYOL
from .mae import EEGMaskedAutoencoder, EEGMaskedEncoder
from .finetune import FinetunedEEGModel, SupervisedEEGModel
