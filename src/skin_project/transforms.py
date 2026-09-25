"""Training, validation and brightness-test image transforms.

Training transforms may include augmentation; development / evaluation / inference
transforms must be **deterministic** (no randomness) so that a dev score is
reproducible and comparable across runs. Both pipelines end with the same resize,
tensor conversion, and ImageNet-style normalisation so a model sees identical pixel
statistics at train and eval time (a mismatch here is a classic silent fault).
"""

from __future__ import annotations

from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

# ImageNet statistics — also the correct stats for the torchvision pretrained
# backbones used later, so we standardise on them everywhere.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class AdjustBrightness:
    """Apply a fixed, deterministic brightness factor to a PIL image.

    A top-level callable is used instead of a lambda so the transform remains
    picklable when a Windows ``DataLoader`` uses worker processes.
    """

    def __init__(self, factor: float = 1.0) -> None:
        factor = float(factor)
        if factor <= 0:
            raise ValueError("brightness factor must be greater than zero")
        self.factor = factor

    def __call__(self, image: Image.Image) -> Image.Image:
        return TF.adjust_brightness(image, self.factor)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(factor={self.factor:g})"


def to_rgb(image: Image.Image) -> Image.Image:
    """Force any image (grayscale, RGBA, palette, CMYK) to 3-channel RGB.

    This also makes inference robust to grayscale, RGBA and palette images.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def build_train_transforms(image_size: int, augment: dict | None = None):
    """Return the *training* transform pipeline.

    `augment` is a dict of augmentation switches from the config, e.g.
    ``{"horizontal_flip": true, "rotation": 15, "brightness_jitter": 0.3}``.

    ``brightness_jitter`` changes brightness only. ``color_jitter`` changes
    brightness, contrast and saturation together.
    """
    augment = augment or {}
    ops = [transforms.Lambda(to_rgb)]
    if augment.get("random_resized_crop", False):
        scale_min = float(augment.get("crop_scale_min", 0.80))
        scale_max = float(augment.get("crop_scale_max", 1.00))
        if not 0 < scale_min <= scale_max <= 1:
            raise ValueError("crop scale must satisfy 0 < min <= max <= 1")
        ops.append(
            transforms.RandomResizedCrop(
                image_size,
                scale=(scale_min, scale_max),
                ratio=(0.90, 1.10),
            )
        )
    else:
        ops.append(transforms.Resize((image_size, image_size)))

    if augment.get("horizontal_flip", False):
        ops.append(transforms.RandomHorizontalFlip())
    if augment.get("vertical_flip", False):
        ops.append(transforms.RandomVerticalFlip())

    rotation = augment.get("rotation", 0)
    if rotation:
        ops.append(transforms.RandomRotation(rotation))

    brightness_jitter = float(augment.get("brightness_jitter", 0))
    if brightness_jitter < 0:
        raise ValueError("brightness_jitter must be non-negative")
    if brightness_jitter:
        brightness_probability = float(augment.get("brightness_probability", 1.0))
        if not 0 <= brightness_probability <= 1:
            raise ValueError("brightness_probability must be between zero and one")
        brightness_transform = transforms.ColorJitter(brightness=brightness_jitter)
        if brightness_probability < 1.0:
            ops.append(
                transforms.RandomApply([brightness_transform], p=brightness_probability)
            )
        else:
            ops.append(brightness_transform)

    color_jitter = augment.get("color_jitter", 0)
    if color_jitter:
        ops.append(
            transforms.ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=color_jitter,
            )
        )
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    return transforms.Compose(ops)


def build_eval_transforms(image_size: int, brightness_factor: float = 1.0):
    """Return the deterministic *evaluation* transform pipeline.

    Randomised augmentation at development time is inappropriate because it makes
    the validation score noisy and non-reproducible.
    ``brightness_factor`` is a controlled corruption applied before tensor
    conversion and normalisation; 1.0 leaves the resized RGB image unchanged.
    """
    brightness_factor = float(brightness_factor)
    if brightness_factor <= 0:
        raise ValueError("brightness_factor must be greater than zero")

    ops = [
        transforms.Lambda(to_rgb),
        transforms.Resize((image_size, image_size)),
    ]
    if brightness_factor != 1.0:
        ops.append(AdjustBrightness(brightness_factor))
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(ops)


def build_transforms(
    image_size: int,
    train: bool,
    augment: dict | None = None,
    brightness_factor: float = 1.0,
):
    """Convenience dispatcher used by the dataset/dataloader builders."""
    return (
        build_train_transforms(image_size, augment)
        if train
        else build_eval_transforms(image_size, brightness_factor=brightness_factor)
    )
