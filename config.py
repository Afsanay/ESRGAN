from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

CHECKPOINT_GEN = "gen.pth"
DEVICE = "cpu"
LEARNING_RATE = 1e-4


test_transform = A.Compose(
    [
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)